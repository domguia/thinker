"""
Same diagnostic as diagnose_generation_divergence.py (teacher-forced argmax
accuracy + safe-bet token concentration), adapted for a plain HF
AutoModelForCausalLM checkpoint (train_sft.py's Baseline C, flat dense
transformer, JsonlTextDataset -- input_ids only, no kb_tokens/retrieval
structure). Kept as a separate script rather than branching the retrieval
one because the two data/model interfaces don't share enough surface to
be worth unifying.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model_families import resolve_model_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint dir (config.json + weights), "
                                                          "or a state_dict .pt with --config_dir for architecture")
    ap.add_argument("--config_dir", default=None, help="if --checkpoint is a .pt state_dict, the HF config dir")
    ap.add_argument("--val_data", required=True, help="jsonl with a 'text' field per line")
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--n_samples", type=int, default=25)
    ap.add_argument("--max_eval_len", type=int, default=64, help="only score the LAST this-many tokens of "
                                                                   "each window, to mimic answer-length scale")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    if args.config_dir:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.config_dir)
        model = AutoModelForCausalLM.from_config(config)
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model = model.to(device).eval()
    print(f"loaded {args.checkpoint}", flush=True)

    examples = []
    with open(args.val_data) as f:
        for i, line in enumerate(f):
            if len(examples) >= args.n_samples:
                break
            row = json.loads(line)
            ids = tok(row["text"], truncation=True, max_length=args.block_size)["input_ids"]
            if len(ids) >= 8:
                examples.append(ids)
    print(f"loaded {len(examples)} examples from {args.val_data}", flush=True)

    def _safe_decode(ids: list[int]) -> str:
        ids = [i for i in ids if 0 <= i < vocab_size]
        return tok.decode(ids) if ids else ""

    all_tf_correct = []
    all_tf_argmax_ids = []
    divergence_pos = []

    with torch.no_grad():
        for ids in examples:
            full = torch.tensor([ids], dtype=torch.long, device=device)
            L = full.shape[1]
            score_start = max(1, L - args.max_eval_len)  # skip position 0 (no context to condition on)

            # teacher-forced: single forward on the TRUE sequence, argmax at t predicts token t+1
            logits_tf = model(full).logits[0]  # (L, vocab)
            tf_argmax = logits_tf.argmax(dim=-1)  # tf_argmax[t] predicts true token at t+1

            for t in range(score_start - 1, L - 1):
                pred = int(tf_argmax[t])
                true_next = int(full[0, t + 1])
                all_tf_correct.append(pred == true_next)
                all_tf_argmax_ids.append(pred)

            # free-run: regenerate the same tail autoregressively from the true prefix up to score_start
            gen = full[:, :score_start].clone()
            free_preds = []
            for t in range(score_start - 1, L - 1):
                logits = model(gen).logits[0, -1]
                nxt = int(logits.argmax(-1))
                free_preds.append(nxt)
                gen = torch.cat([gen, torch.tensor([[nxt]], device=device)], dim=1)

            tf_preds_window = [int(tf_argmax[t]) for t in range(score_start - 1, L - 1)]
            div = None
            for i, (a, b) in enumerate(zip(tf_preds_window, free_preds)):
                if a != b:
                    div = i
                    break
            divergence_pos.append(div if div is not None else len(tf_preds_window))

    n_flat = len(all_tf_correct)
    tf_acc = sum(all_tf_correct) / n_flat if n_flat else float("nan")
    tok_counts = Counter(all_tf_argmax_ids)
    top_tokens = tok_counts.most_common(15)
    top5_mass = sum(c for _, c in tok_counts.most_common(5)) / n_flat if n_flat else float("nan")
    n_unique = len(tok_counts)

    n_immediate = sum(1 for d in divergence_pos if d == 0)
    n_within_3 = sum(1 for d in divergence_pos if d <= 2)
    mean_div = sum(divergence_pos) / len(divergence_pos) if divergence_pos else float("nan")

    print(f"\nteacher-forced argmax accuracy: {tf_acc*100:.1f}% ({sum(all_tf_correct)}/{n_flat} tokens)")
    print(f"divergence (free-run vs teacher-forced): immediate t=0 {n_immediate}/{len(divergence_pos)}, "
          f"within 3 {n_within_3}/{len(divergence_pos)}, mean {mean_div:.2f}")
    print(f"\n--- safe-bet analysis: {n_flat} positions, {n_unique} unique tokens, "
          f"top-5 cover {top5_mass*100:.1f}% ---")
    for tid, c in top_tokens:
        print(f"  {tid}\t{_safe_decode([tid])!r}\t{c}\t{100*c/n_flat:.1f}%")

    result = {"checkpoint": args.checkpoint, "n_flat_positions": n_flat,
              "teacher_forced_argmax_accuracy_pct": tf_acc * 100,
              "n_unique_tokens": n_unique, "top5_mass_pct": top5_mass * 100,
              "top_tokens": [{"id": tid, "decoded": _safe_decode([tid]), "count": c, "pct": 100 * c / n_flat}
                             for tid, c in top_tokens],
              "mean_divergence_position": mean_div,
              "pct_immediate_t0": 100 * n_immediate / len(divergence_pos) if divergence_pos else None,
              "pct_within_3": 100 * n_within_3 / len(divergence_pos) if divergence_pos else None}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
