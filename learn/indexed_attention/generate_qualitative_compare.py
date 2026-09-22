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


@torch.no_grad()
def generate_thinker(model, ds: RetrievalPromptDataset, indices: list[int], device,
                      n_step: int, block_size: int, max_answer_len: int, tokenizer) -> list[str]:
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

    for t in range(max_answer_len):
        _, stream_outputs = model(kb_tokens=kb_tokens, kb_source_ids=kb_source_ids,
                                   query_tokens=query_tokens, n_step=n_step,
                                   kb_leaf_mask=kb_leaf_mask, target_input={"answer": target_input})
        next_token = stream_outputs["answer"][:, t, :].argmax(dim=-1)
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
    return texts


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

    print(f"loading Thinker checkpoint {args.checkpoint} ...", flush=True)
    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=0, n_slots=1, n_head=args.n_head,
        use_ff=args.use_ff, stream_dims={"answer": vocab_size},
        stream_sequence={"answer": True}, max_target_len=args.max_answer_len,
        stream_n_layers={"answer": args.answer_n_layers},
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    thinker_answers = generate_thinker(model, ds, indices, device, args.n_step, args.block_size,
                                        args.max_answer_len, tok)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ref_name = resolve_model_name(args.ref_model)
    print(f"loading reference LLM {ref_name} ...", flush=True)
    ref_tok = AutoTokenizer.from_pretrained(ref_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, torch_dtype=dtype).to(device).eval()
    prompts = []
    for row in rows:
        text = row.get("text") or ""
        pos = text.find(ASSISTANT_MARKER)
        prompts.append(text[:pos + len(ASSISTANT_MARKER)] if pos != -1 else text)
    ref_answers = generate_reference(ref_model, ref_tok, prompts, device, args.max_answer_len)
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lines = [
        f"# Qualitative comparison: Thinker vs {ref_name}",
        "",
        f"- Thinker checkpoint: `{args.checkpoint}`",
        f"- Reference model: `{ref_name}`",
        f"- Sample: first {len(indices)} rows of `{args.val_data}` (fixed, deterministic)",
        "",
    ]
    for i, (row, t_ans, r_ans) in enumerate(zip(rows, thinker_answers, ref_answers)):
        lines += [
            f"## Example {i} (num_hops={row.get('num_hops')})",
            f"**Question**: {row['question']}",
            f"**Gold answer**: {row['answer']}",
            f"**Thinker**: {t_ans!r}",
            f"**{ref_name}**: {r_ans!r}",
            "",
        ]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
