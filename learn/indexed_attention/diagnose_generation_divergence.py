"""
Diagnose WHERE free-running generation first diverges from what the model itself
would predict under teacher forcing (ground-truth prefix), on a saved Thinker
checkpoint -- supervisor-agent request (2026-09-22), priority 1 of the
generation-collapse investigation (dev_notes/experiments/prompt_response_pipeline.md,
"KD-vs-CE-only isolé + LoRA + embed-KD" entry: all 4 variants collapse in free
generation despite normal-looking CE loss).

Two forward passes per example, both argmax (temperature=0, matching the collapse
seen at greedy decoding in qualitative_eval_at_end):
  1. teacher-forced: target_input = ground-truth answer_target_input (fixed prefix
     at every position, exactly like evaluate()'s CE computation) -> argmax_t is
     what the model would predict at position t IF given the true prefix.
  2. free-run: target_input built autoregressively from the model's own previous
     argmax (exactly like generate_qualitative_compare.py's generate_thinker).

Comparing argmax_t (teacher-forced) vs free_t (free-run) at each position isolates
exposure bias from everything else: same model, same weights, same position -- the
only difference is whether the CONDITIONING history is the true answer or the
model's own prior guesses. The first position where they disagree is the
divergence point. "diverges immediately" (position 0 or 1) points to a
start-of-generation calibration bias; "diverges after several correct tokens"
points to cumulative drift compounding across the recurrent n_step iterations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from learn.indexed_attention.train_prompt_response import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--val_teacher_targets", default=None)
    ap.add_argument("--teacher_name", default=None)
    ap.add_argument("--teacher_max_length", type=int, default=4096)
    ap.add_argument("--repr_teacher_layer", type=int, default=None)
    ap.add_argument("--repr_proj_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_type", default="retrieval", choices=["retrieval"])
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--answer_head_lora_rank", type=int, default=0)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_slots", type=int, default=1)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--pool_n_head", type=int, default=1)
    ap.add_argument("--k_dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--disable_kb", action="store_true")
    ap.add_argument("--use_ff", action="store_true")
    ap.add_argument("--ff_hidden_mult", type=int, default=4)
    ap.add_argument("--answer_n_layers", type=int, default=1)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    val_ds = build_dataset(args.dataset_type, args.val_data, tok, args,
                            teacher_targets=args.val_teacher_targets, repr_teacher_hidden=None)
    print(f"loaded val: {len(val_ds)} examples from {args.val_data}", flush=True)

    stream_dims = {"answer": vocab_size}
    stream_sequence = {"answer": True}
    stream_n_layers = {"answer": args.answer_n_layers}

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=args.max_answer_len,
        stream_n_layers=stream_n_layers,
    ).to(device)
    if args.answer_head_lora_rank > 0:
        from learn.indexed_attention.train_prompt_response import LoRAHead
        model.streams["answer"].head = LoRAHead(model.streams["answer"].head, args.answer_head_lora_rank).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded {args.checkpoint}", flush=True)

    n = min(args.n_samples, len(val_ds))
    items = [val_ds[i] for i in range(n)]
    kb_tokens = torch.stack([it["kb_tokens"] for it in items]).to(device)
    kb_source_ids = torch.stack([it["kb_source_ids"] for it in items]).to(device)
    kb_leaf_mask = torch.stack([it["kb_leaf_mask"] for it in items]).to(device)
    query_tokens = kb_tokens[:, -args.block_size:]
    gt_target_input = torch.stack([it["answer_target_input"] for it in items]).to(device)
    gt_labels = torch.stack([it["answer_labels"] for it in items]).to(device)
    pad_id = val_ds.pad_id
    B = kb_tokens.shape[0]
    L = args.max_answer_len

    with torch.no_grad():
        # (1) teacher-forced: single forward pass, true prefix at every position.
        _, out_tf = model(kb_tokens=kb_tokens, kb_source_ids=kb_source_ids, query_tokens=query_tokens,
                           n_step=args.n_step, kb_leaf_mask=kb_leaf_mask,
                           target_input={"answer": gt_target_input})
        tf_argmax = out_tf["answer"].argmax(dim=-1)  # (B, L): what the model predicts at t given TRUE prefix

        # (2) free-run: autoregressive, one forward per token, own argmax feeds next step.
        free_target_input = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        free_argmax = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        for t in range(L):
            _, out_fr = model(kb_tokens=kb_tokens, kb_source_ids=kb_source_ids, query_tokens=query_tokens,
                               n_step=args.n_step, kb_leaf_mask=kb_leaf_mask,
                               target_input={"answer": free_target_input})
            next_tok = out_fr["answer"][:, t, :].argmax(dim=-1)
            free_argmax[:, t] = next_tok
            if t + 1 < L:
                free_target_input[:, t + 1] = next_tok

    valid_mask = gt_labels != pad_id  # only score positions the ground-truth answer actually covers

    # teacher-forced ARGMAX ACCURACY vs ground truth: does the model predict the right token at
    # all when given the true prefix (not just "does free-run match teacher-forced", which can
    # both be wrong)? Low CE (soft distribution) does not imply the argmax/mode is correct.
    tf_correct = (tf_argmax == gt_labels) & valid_mask
    n_valid_tok = int(valid_mask.sum().item())
    tf_acc = float(tf_correct.sum().item()) / n_valid_tok if n_valid_tok else float("nan")
    tf_acc_pos0 = None
    pos0_mask = valid_mask[:, 0]
    if int(pos0_mask.sum().item()) > 0:
        tf_acc_pos0 = float((tf_argmax[:, 0] == gt_labels[:, 0])[pos0_mask].float().mean().item())
    print(f"\nteacher-forced argmax accuracy (does the model predict the TRUE next token when given "
          f"the TRUE prefix, i.e. is CE-low == argmax-correct?): {tf_acc*100:.1f}% ({int(tf_correct.sum().item())}/{n_valid_tok} tokens)")
    if tf_acc_pos0 is not None:
        print(f"teacher-forced argmax accuracy at position 0 only: {tf_acc_pos0*100:.1f}%")

    divergence_pos = []
    for b in range(B):
        valid_len = int(valid_mask[b].sum().item())
        if valid_len == 0:
            continue
        div = None
        for t in range(valid_len):
            if int(tf_argmax[b, t]) != int(free_argmax[b, t]):
                div = t
                break
        divergence_pos.append(div if div is not None else valid_len)  # None -> "never diverged within valid_len"

    counts = Counter(divergence_pos)
    n_scored = len(divergence_pos)
    n_immediate = sum(1 for d in divergence_pos if d == 0)
    n_within_3 = sum(1 for d in divergence_pos if d <= 2)
    mean_div = sum(divergence_pos) / n_scored if n_scored else float("nan")

    print(f"\nscored {n_scored} examples (valid answer_labels length > 0)")
    print(f"divergence position histogram (0 = model already disagrees with itself at the FIRST token): "
          f"{dict(sorted(counts.items()))}")
    print(f"immediate divergence (t=0): {n_immediate}/{n_scored} ({100*n_immediate/n_scored:.1f}%)")
    print(f"divergence within first 3 tokens (t<=2): {n_within_3}/{n_scored} ({100*n_within_3/n_scored:.1f}%)")
    print(f"mean divergence position: {mean_div:.2f}")

    # a few concrete examples for qualitative inspection
    def _safe_decode(ids: list[int]) -> str:
        ids = [i for i in ids if 0 <= i < vocab_size]
        return tok.decode(ids) if ids else ""

    examples_out = []
    for b in range(min(10, B)):
        valid_len = int(valid_mask[b].sum().item())
        tf_toks = _safe_decode(tf_argmax[b, :valid_len].tolist())
        fr_toks = _safe_decode(free_argmax[b, :valid_len].tolist())
        gt_toks = _safe_decode(gt_labels[b, :valid_len].tolist())
        examples_out.append({"example": b, "divergence_pos": divergence_pos[b] if b < len(divergence_pos) else None,
                              "ground_truth": gt_toks, "teacher_forced_argmax": tf_toks, "free_run": fr_toks})

    result = {"checkpoint": args.checkpoint, "n_scored": n_scored,
              "divergence_histogram": {str(k): v for k, v in sorted(counts.items())},
              "pct_immediate_t0": 100 * n_immediate / n_scored if n_scored else None,
              "pct_within_3": 100 * n_within_3 / n_scored if n_scored else None,
              "mean_divergence_position": mean_div,
              "teacher_forced_argmax_accuracy_pct": tf_acc * 100,
              "teacher_forced_argmax_accuracy_pos0_pct": tf_acc_pos0 * 100 if tf_acc_pos0 is not None else None,
              "examples": examples_out}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
