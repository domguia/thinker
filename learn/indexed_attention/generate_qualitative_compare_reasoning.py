"""
Qualitative generation check for --dataset_type reasoning (openr1_math etc.),
mirrors generate_qualitative_compare.py's retrieval version but for
ReasoningPromptDataset's two independent sequence streams (thinking, answer).

Why a separate script rather than extending the retrieval one: the two
dataset classes build very different tensors (kb_tokens/query_tokens shape,
--n_ctx vs --block_size/--n_docs_max) and reasoning has TWO target streams
instead of one. core/indexed_thinker_model.py's OutputStream has no
self-attention across positions and both streams only cross-attend to the
SAME shared sm_k/sm_v from the core register (computed once from kb_tokens,
independent of either target) -- so `thinking` and `answer` can be decoded
independently, in any order, each via the same per-step teacher-forcing-
convention loop as generate_qualitative_compare.py's generate_thinker.

Requested by analyst-agent (2026-09-22): a good val_answer number alone
doesn't establish that free-running generation is coherent (see the
retrieval <think>-collapse investigation) -- apply the same qualitative
check to the math CE-only vs KD checkpoints before treating that CE-vs-KD
result as solid.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import ReasoningPromptDataset
from learn.indexed_attention.generate_qualitative_compare import _sample_next


@torch.no_grad()
def generate_stream(model, kb_tokens, kb_source_ids, kb_leaf_mask, query_tokens, device,
                     n_step: int, max_len: int, stream_name: str, other_len: int, pad_id: int, eos_id,
                     tokenizer, temperature: float, top_p: float, seed: int) -> list[str]:
    """Thinker.forward() requires a target_input entry for EVERY sequence_mode stream
    (core/indexed_thinker_model.py's stream_query_input), even though thinking/answer
    are otherwise decoded independently (both only cross-attend to the same sm_k/sm_v,
    no self-attention between positions or across streams) -- a pad-filled placeholder
    for the stream we're not currently decoding has no effect on the one we care about."""
    B = kb_tokens.shape[0]
    other_name = "answer" if stream_name == "thinking" else "thinking"
    other_placeholder = torch.full((B, other_len), pad_id, dtype=torch.long, device=device)
    target_input = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    generated = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    if temperature > 0:
        torch.manual_seed(seed)
    for t in range(max_len):
        _, stream_outputs = model(kb_tokens=kb_tokens, kb_source_ids=kb_source_ids,
                                   query_tokens=query_tokens, n_step=n_step, kb_leaf_mask=kb_leaf_mask,
                                   target_input={stream_name: target_input, other_name: other_placeholder})
        logits_t = stream_outputs[stream_name][:, t, :]
        next_token = _sample_next(logits_t, temperature, top_p)
        next_token = torch.where(done, torch.full_like(next_token, pad_id), next_token)
        generated[:, t] = next_token
        if eos_id is not None:
            done = done | (next_token == eos_id)
        if t + 1 < max_len:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--tokenizer", default="lfm2")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--use_ff", action="store_true", default=True)
    ap.add_argument("--n_ctx", type=int, default=256)
    ap.add_argument("--block_size", type=int, default=16,
                     help="Thinker's own constructor block_size -- NOT the same as --n_ctx (dataset "
                          "prompt length); train_prompt_response.py always passes --block_size here "
                          "regardless of dataset_type, default 16, must match the checkpoint's training run")
    ap.add_argument("--max_thinking_len", type=int, default=1024,
                     help="the checkpoint's own training max_thinking_len (for model construction)")
    ap.add_argument("--gen_thinking_len", type=int, default=200,
                     help="cap on how many 'thinking' tokens to actually GENERATE (each token is a "
                          "full forward pass, no KV cache -- 1024 sequential steps would be slow for "
                          "a qualitative check; 200 is enough to see whether the stream collapses)")
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--thinking_n_layers", type=int, default=1)
    ap.add_argument("--answer_n_layers", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    ds = ReasoningPromptDataset(args.val_data, tok, n_ctx=args.n_ctx,
                                 max_thinking_len=args.max_thinking_len, max_answer_len=args.max_answer_len)
    indices = list(range(min(args.n_samples, len(ds))))
    print(f"fixed sample: first {len(indices)} rows of {args.val_data}", flush=True)
    rows = [ds.examples[i] for i in indices]

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=0, n_slots=1, n_head=args.n_head, use_ff=args.use_ff,
        stream_dims={"thinking": vocab_size, "answer": vocab_size},
        stream_sequence={"thinking": True, "answer": True},
        max_target_len=max(args.max_thinking_len, args.max_answer_len),
        stream_n_layers={"thinking": args.thinking_n_layers, "answer": args.answer_n_layers},
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    items = [ds[i] for i in indices]
    kb_tokens = torch.stack([it["kb_tokens"] for it in items]).to(device)
    kb_source_ids = torch.stack([it["kb_source_ids"] for it in items]).to(device)
    kb_leaf_mask = torch.stack([it["kb_leaf_mask"] for it in items]).to(device)
    query_tokens = kb_tokens  # reasoning: whole prompt seeds the register (query_tokens_for in train_prompt_response.py)

    thinking_answers = generate_stream(model, kb_tokens, kb_source_ids, kb_leaf_mask, query_tokens, device,
                                        args.n_step, args.gen_thinking_len, "thinking", args.max_answer_len,
                                        ds.pad_id, tok.eos_token_id, tok, args.temperature, args.top_p, args.seed)
    answer_answers = generate_stream(model, kb_tokens, kb_source_ids, kb_leaf_mask, query_tokens, device,
                                      args.n_step, args.max_answer_len, "answer", args.max_thinking_len,
                                      ds.pad_id, tok.eos_token_id, tok, args.temperature, args.top_p, args.seed)

    lines = [
        f"# Qualitative eval (reasoning) -- {args.checkpoint}",
        f"- Sample: first {len(indices)} rows of {args.val_data} (fixed, deterministic)",
        f"- Decoding: {'greedy' if args.temperature <= 0 else f'temperature={args.temperature}, top_p={args.top_p}'}",
        "",
    ]
    for i, (row, th, an) in enumerate(zip(rows, thinking_answers, answer_answers)):
        lines += [
            f"## Example {i}",
            f"**Problem**: {row['problem'][:200]}",
            f"**Gold answer**: {row['answer']}",
            f"**Generated thinking**: {th[:300]!r}",
            f"**Generated answer**: {an!r}",
            "",
        ]
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
