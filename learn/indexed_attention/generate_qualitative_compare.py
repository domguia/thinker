"""
Qualitative side-by-side generation: Thinker checkpoint vs a small reference
LLM, on a SMALL, FIXED sample of the retrieval val set (--val_data, same
format as train_prompt_response.py --dataset_type retrieval).

Why this exists: the CE metric alone (val_answer) can hide real behavioral
differences (repetition, incoherence, plausible-but-wrong hallucination) that
only show up by actually reading generated text -- requested by the user
(2026-09-22) and independently by a peer session (analyst-agent) wanting a
Thinker-vs-small-Qwen manual comparison. "Fixed sample" is a deliberate
choice (user, 2026-09-22): the same --n_samples examples (first N rows of
--val_data, deterministic, no shuffling) are used every time this script is
run, so outputs from different checkpoints/runs stay directly comparable
over time instead of each report reading a different random slice.

Thinker has no built-in generate() (core/indexed_thinker_model.py's OutputStream
is trained via teacher forcing only, spec 14.3) -- greedy decoding here calls
Thinker.forward() once per generated token, feeding the growing "answer"
target_input each time (same shift-by-one-with-leading-pad convention as
data/prompt_response_dataset.py's _teacher_forced_target) and taking the
argmax at the current position. This recomputes the KB/core register (and
every earlier decoded position) from scratch at each step -- wasteful, but
simple and correct, and fine for a small qualitative sample (not meant to
benchmark decoding speed).

The reference LLM is generated with HF's own model.generate() (real KV-cache,
efficient) on the SAME textual prompt (row["text"] truncated right before the
assistant marker, prepare_retrieval_data.py's CHATML_TEMPLATE) so both models
see an equivalent (if not token-identical, given different tokenizers)
context/question.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import ASSISTANT_MARKER, RetrievalPromptDataset


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """logits: (B, vocab). temperature<=0 -> greedy argmax (default). Otherwise
    temperature-scaled softmax with optional nucleus (top-p) filtering before
    sampling -- requested (analyst-agent, 2026-09-22) to check whether the
    greedy '<think>'-collapse is a pure argmax artifact or persists under
    sampling (see dev_notes/experiments/prompt_response_pipeline.md)."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        cutoff = (cum - sorted_probs) > top_p  # keep first token that crosses top_p
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_rank = torch.multinomial(sorted_probs, 1).squeeze(-1)
        return sorted_idx.gather(1, sampled_rank.unsqueeze(-1)).squeeze(-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.no_grad()
def generate_thinker(model, ds: RetrievalPromptDataset, indices: list[int], device,
                      n_step: int, block_size: int, max_answer_len: int, tokenizer,
                      temperature: float = 0.0, top_p: float = 1.0, seed: int = 0,
                      log_first_step_topk: int = 0) -> tuple[list[str], list[dict]]:
    items = [ds[i] for i in indices]
    kb_tokens = torch.stack([it["kb_tokens"] for it in items]).to(device)
    kb_source_ids = torch.stack([it["kb_source_ids"] for it in items]).to(device)
    kb_leaf_mask = torch.stack([it["kb_leaf_mask"] for it in items]).to(device)
    query_tokens = kb_tokens[:, -block_size:]

    B = kb_tokens.shape[0]
    pad_id = ds.pad_id
    eos_id = tokenizer.eos_token_id
    target_input = torch.full((B, max_answer_len), pad_id, dtype=torch.long, device=device)
    generated = torch.full((B, max_answer_len), pad_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    if temperature > 0:
        torch.manual_seed(seed)

    first_step_diag = []
    for t in range(max_answer_len):
        _, stream_outputs = model(kb_tokens=kb_tokens, kb_source_ids=kb_source_ids,
                                   query_tokens=query_tokens, n_step=n_step,
                                   kb_leaf_mask=kb_leaf_mask, target_input={"answer": target_input})
        logits_t = stream_outputs["answer"][:, t, :]
        if t == 0 and log_first_step_topk > 0:
            probs0 = torch.softmax(logits_t.float(), dim=-1)
            topk = probs0.topk(log_first_step_topk, dim=-1)
            for b in range(B):
                toks = [tokenizer.decode([tid]) for tid in topk.indices[b].tolist()]
                first_step_diag.append({"example": b,
                                         "top_tokens": list(zip(toks, [round(v, 4) for v in topk.values[b].tolist()]))})
        next_token = _sample_next(logits_t, temperature, top_p)
        next_token = torch.where(done, torch.full_like(next_token, pad_id), next_token)
        generated[:, t] = next_token
        if eos_id is not None:
            done = done | (next_token == eos_id)
        if t + 1 < max_answer_len:
            target_input[:, t + 1] = next_token
        if bool(done.all()):
            break

    texts = []
    for b in range(B):
        ids = generated[b].tolist()
        if eos_id is not None and eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
    return texts, first_step_diag


@torch.no_grad()
def generate_reference(model, tokenizer, prompts: list[str], device, max_new_tokens: int) -> list[str]:
    texts = []
    for prompt in prompts:  # one at a time -- different prompt lengths, avoids left-padding complexity for a small sample
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_ids = out[0, enc["input_ids"].shape[1]:]
        texts.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint2", default=None,
                     help="optional second Thinker checkpoint -- if given, compares checkpoint vs "
                          "checkpoint2 (same architecture flags applied to both) INSTEAD OF checkpoint "
                          "vs --ref_model. Same decoding (--temperature/--top_p) applied to both.")
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--n_samples", type=int, default=30,
                     help="fixed sample size -- always the FIRST n_samples rows of --val_data "
                          "(deterministic, no shuffling) so results are comparable across runs")
    ap.add_argument("--ref_model", default="qwen35",
                     help="HF repo id or a family alias from core/model_families.py -- default "
                          "qwen35 (Qwen3.5-0.8B) is the smallest model in the same tokenizer "
                          "family as the Thinker checkpoint's qwen35 --tokenizer")
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--use_ff", action="store_true", default=True)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--answer_n_layers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0 (default) = greedy argmax. >0 = temperature-scaled sampling "
                          "(e.g. 0.7-1.0), combine with --top_p for nucleus sampling.")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, ignored if --temperature=0")
    ap.add_argument("--answer_head_lora_rank", type=int, default=0,
                     help="if >0, wrap the answer head in a LoRAHead (rank r) before loading "
                          "--checkpoint/--checkpoint2 -- required to reload a checkpoint trained "
                          "with train_prompt_response.py's --answer_head_lora_rank")
    ap.add_argument("--log_first_step_topk", type=int, default=0,
                     help="if >0, print the top-K (token, prob) pairs at generation position 0 "
                          "for every example -- diagnostic for whether a degenerate output is "
                          "the model being confidently wrong or just flat/undecided (analyst-agent, "
                          "2026-09-22)")
    ap.add_argument("--out", required=True, help="markdown report path")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    ds = RetrievalPromptDataset(args.val_data, tok, args.block_size, args.n_docs_max, args.max_answer_len)
    indices = list(range(min(args.n_samples, len(ds))))
    print(f"fixed sample: first {len(indices)} rows of {args.val_data}", flush=True)

    rows = []
    with open(args.val_data) as f:
        for i, line in enumerate(f):
            if i > indices[-1]:
                break
            rows.append(json.loads(line))
    rows = [rows[i] for i in indices]

    def load_and_generate(checkpoint_path: str):
        print(f"loading Thinker checkpoint {checkpoint_path} ...", flush=True)
        m = Thinker(
            vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
            block_size=args.block_size, depth=0, n_slots=1, n_head=args.n_head,
            use_ff=args.use_ff, stream_dims={"answer": vocab_size},
            stream_sequence={"answer": True}, max_target_len=args.max_answer_len,
            stream_n_layers={"answer": args.answer_n_layers},
        ).to(device)
        if args.answer_head_lora_rank > 0:
            from learn.indexed_attention.train_prompt_response import LoRAHead
            m.streams["answer"].head = LoRAHead(m.streams["answer"].head, args.answer_head_lora_rank).to(device)
        m.load_state_dict(torch.load(checkpoint_path, map_location=device))
        m.eval()
        answers, diag = generate_thinker(
            m, ds, indices, device, args.n_step, args.block_size, args.max_answer_len, tok,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            log_first_step_topk=args.log_first_step_topk)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return answers, diag

    thinker_answers, first_step_diag = load_and_generate(args.checkpoint)

    if args.checkpoint2:
        other_answers, _ = load_and_generate(args.checkpoint2)
        other_name = args.checkpoint2
    else:
        other_name = resolve_model_name(args.ref_model)
        print(f"loading reference LLM {other_name} ...", flush=True)
        ref_tok = AutoTokenizer.from_pretrained(other_name)
        ref_model = AutoModelForCausalLM.from_pretrained(other_name, torch_dtype=dtype).to(device).eval()
        prompts = []
        for row in rows:
            text = row.get("text") or ""
            pos = text.find(ASSISTANT_MARKER)
            prompts.append(text[:pos + len(ASSISTANT_MARKER)] if pos != -1 else text)
        other_answers = generate_reference(ref_model, ref_tok, prompts, device, args.max_answer_len)
        del ref_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    lines = [
        f"# Qualitative comparison: Thinker vs {other_name}",
        "",
        f"- Thinker checkpoint: `{args.checkpoint}`",
        f"- {'Second Thinker checkpoint' if args.checkpoint2 else 'Reference model'}: `{other_name}`",
        f"- Sample: first {len(indices)} rows of `{args.val_data}` (fixed, deterministic)",
        f"- Decoding: {'greedy' if args.temperature <= 0 else f'temperature={args.temperature}, top_p={args.top_p}, seed={args.seed}'}",
        "",
    ]
    if first_step_diag:
        lines += ["## First-step top-K logits (position 0, before any generated token)", ""]
        for d in first_step_diag:
            lines.append(f"- example {d['example']}: {d['top_tokens']}")
        lines.append("")
    for i, (row, t_ans, r_ans) in enumerate(zip(rows, thinker_answers, other_answers)):
        lines += [
            f"## Example {i} (num_hops={row.get('num_hops')})",
            f"**Question**: {row['question']}",
            f"**Gold answer**: {row['answer']}",
            f"**Thinker ({args.checkpoint})**: {t_ans!r}",
            f"**{other_name}**: {r_ans!r}",
            "",
        ]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
