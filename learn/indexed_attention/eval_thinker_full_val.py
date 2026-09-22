"""
Evaluate a saved Thinker checkpoint (train_prompt_response.py's
--save_best_checkpoint_path) on the FULL held-out val set, not the
--val_batches-limited subset (default 20 batches) used for the val_answer
printed during training.

Why this exists: train_prompt_response.py's own training-time val_answer is
computed on only the first --val_batches batches (default 20, e.g. 2560/9000
examples at batch_size=128) for speed -- fine for picking a best checkpoint
mid-run, but NOT directly comparable to a full-val-set number (e.g. the
reference-LLM baselines from eval_llm_baseline_retrieval.py, which evaluate
all 9000 examples by default). This script reuses train_prompt_response.py's
own evaluate()/build_dataset()/Thinker construction so the metric definition
is identical -- only the number of batches (and no training step at all)
differs.

Pass the SAME architecture/data flags used to train the checkpoint
(--d_model, --n_head, --n_step, --use_ff, --block_size, --n_docs_max,
--tokenizer, etc.) -- this script does not persist or infer them.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from learn.indexed_attention.train_prompt_response import build_dataset, evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path saved by --save_best_checkpoint_path")
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--val_teacher_targets", default=None)
    ap.add_argument("--val_repr_teacher_hidden", default=None)
    ap.add_argument("--teacher_name", default=None)
    ap.add_argument("--teacher_max_length", type=int, default=4096)
    ap.add_argument("--repr_teacher_layer", type=int, default=None)
    ap.add_argument("--repr_proj_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset_type", default="retrieval", choices=["reasoning", "retrieval"])
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--n_ctx", type=int, default=256, help="reasoning only: prompt length (flat, depth=0)")
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--max_thinking_len", type=int, default=1024)
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
    ap.add_argument("--thinking_n_layers", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--extrapolate_n_steps", default=None,
                    help="comma-separated n_step_test values (e.g. '1,2,4,8,12,16') -- evaluates the SAME "
                         "loaded checkpoint at each, on the full val set, to probe whether the model "
                         "generalizes to a different number of recurrent iterations than it was trained "
                         "with (--n_step). Relevant here specifically because the register-update weights "
                         "are reused identically across every iteration (see train_prompt_response.py's own "
                         "--extrapolate_n_steps, same mechanism, just against a saved checkpoint instead of "
                         "right after training).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    val_ds = build_dataset(args.dataset_type, args.val_data, tok, args,
                            teacher_targets=args.val_teacher_targets,
                            repr_teacher_hidden=args.val_repr_teacher_hidden)
    print(f"loaded full val: {len(val_ds)} examples from {args.val_data}", flush=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    if args.dataset_type == "reasoning":
        stream_dims = {"thinking": vocab_size, "answer": vocab_size}
        stream_sequence = {"thinking": True, "answer": True}
        stream_n_layers = {"thinking": args.thinking_n_layers, "answer": args.answer_n_layers}
        max_target_len = max(args.max_thinking_len, args.max_answer_len)
    else:
        stream_dims = {"answer": vocab_size}
        stream_sequence = {"answer": True}
        stream_n_layers = {"answer": args.answer_n_layers}
        max_target_len = args.max_answer_len

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
        stream_n_layers=stream_n_layers,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = sum(t.numel() for t in model.parameters())
    print(f"loaded {args.checkpoint} ({n_params/1e6:.2f}M params)", flush=True)

    # n_batches has no cap here (unlike train_prompt_response.py's --val_batches default of 20) --
    # a value >= len(val_loader) makes evaluate()'s "if i >= n_batches: break" a no-op, covering
    # every example in val_ds exactly once.
    n_batches = len(val_loader)
    result = evaluate(model, val_loader, device, args.dataset_type, args.n_step, args.block_size,
                       n_batches=n_batches, teacher_enabled=val_ds.teacher is not None)
    print(f"\n[Thinker full-val] {args.checkpoint}: {result}", flush=True)
    print(f"answer_ce (full {len(val_ds)} examples) = {result['answer']:.4f}", flush=True)

    extrapolation_results = {}
    if args.extrapolate_n_steps:
        print(f"\n--- extrapolation probe (n_step_test vs training n_step={args.n_step}), full val ---", flush=True)
        for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
            r = evaluate(model, val_loader, device, args.dataset_type, n_step_test, args.block_size,
                         n_batches=n_batches, teacher_enabled=val_ds.teacher is not None)
            extrapolation_results[n_step_test] = r
            marker = " <- training n_step" if n_step_test == args.n_step else ""
            print(f"  n_step_test={n_step_test:3d} answer={r['answer']:.4f}{marker}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, **result,
                       "extrapolation": extrapolation_results or None}, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
