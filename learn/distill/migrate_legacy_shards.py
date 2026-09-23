"""One-off migration: take a legacy precompute_teacher_targets.py output
(the pre-2026-09-22 `<out>.topk_shards/` + `<out>.hidden_shards/` shard-pair
format, whether or not it was ever merged into a single .npz/memmap) and
finalize it directly into the new named-subset storage tree, WITHOUT
regenerating anything -- just reads the existing shard_*.npz files and
writes the same final artifacts precompute_teacher_targets.py itself would
have written for a fresh run under that --subset_name/--teacher_name.

Only run this once the legacy run is fully finished and nothing else still
has those shard files open for reading/writing (check with whoever produced
them first -- see dev_notes/grid5000_usage.log.md for why this matters:
this project has twice lost data to scripts touching shard files that
looked "done" but weren't).

The legacy shard dirs are left untouched (not deleted) -- remove them by
hand once the new-tree output is verified.

Example (openr1_math_full/val, bf16 checkpoint, full val pool = identity
subset 0..3889, top_k=32 + last-layer hidden):
    python learn/distill/migrate_legacy_shards.py \\
      --legacy_topk_shards .../val_topk32_hidden_bf16.topk_shards \\
      --legacy_hidden_shards .../val_topk32_hidden_bf16.hidden_shards \\
      --dataset_root data/distill/openr1_math_full --split val \\
      --subset_name lastLayer_n3890 --n_examples 3890 \\
      --teacher_name qwen_big --top_k 32 --hidden_layers 64 \\
      --dtype bfloat16 --checkpoint Qwen3.8-27B-bf16 --max_length 4096
"""
import argparse
import os

import numpy as np

from precompute_teacher_targets import finalize_hidden, finalize_topk


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy_topk_shards", required=True)
    p.add_argument("--legacy_hidden_shards", default=None, help="omit if this legacy run had --hidden_layers none")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--subset_name", required=True)
    p.add_argument("--n_examples", type=int, required=True,
                    help="total examples covered -- an identity subset (0..n-1) is assumed unless --indices_file is given")
    p.add_argument("--indices_file", default=None, help="override: explicit .npy of original row indices, if this legacy run wasn't a plain 0..n-1 prefix of <split>.jsonl")
    p.add_argument("--teacher_name", required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--hidden_layers", default="", help="comma-separated hidden_<L> indices actually present in the legacy shards, e.g. '64' or '0,8,64' -- empty if topk-only")
    p.add_argument("--dtype", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_length", type=int, required=True)
    p.add_argument("--attn_implementation", default="unknown",
                    help="record what the legacy run actually used if known (e.g. flash_attention_2, sdpa) -- "
                         "'unknown' if not tracked at the time, still recorded so the manifest doesn't silently omit it")
    p.add_argument("--quantization", default="unknown",
                    help="record what the legacy run actually used if known (e.g. none, bnb-4bit) -- 'unknown' if not tracked")
    args = p.parse_args()

    subset_dir = os.path.join(args.dataset_root, "subsets", args.split)
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = os.path.join(subset_dir, f"{args.subset_name}.indices.npy")
    if os.path.exists(subset_path):
        raise SystemExit(f"{subset_path} already exists -- pick a different --subset_name, or remove it first if this is a genuine re-migration")
    if args.indices_file:
        indices = np.load(args.indices_file)
        assert len(indices) == args.n_examples, f"--indices_file has {len(indices)} entries, expected --n_examples {args.n_examples}"
    else:
        indices = np.arange(args.n_examples, dtype=np.int64)
    np.save(subset_path, indices)
    print(f"Wrote {subset_path} ({len(indices)} examples)", flush=True)

    hidden_layer_indices = [int(x) for x in args.hidden_layers.split(",") if x != ""]
    teacher_cfg = {
        "dtype": args.dtype, "checkpoint": args.checkpoint, "max_length": args.max_length,
        "attn_implementation": args.attn_implementation, "quantization": args.quantization,
    }

    finalize_topk(args.dataset_root, args.split, args.subset_name, args.teacher_name,
                  args.legacy_topk_shards, args.top_k, teacher_cfg)
    if hidden_layer_indices:
        if not args.legacy_hidden_shards:
            raise SystemExit("--hidden_layers given but --legacy_hidden_shards is missing")
        finalize_hidden(args.dataset_root, args.split, args.subset_name, args.teacher_name,
                         args.legacy_hidden_shards, hidden_layer_indices, args.n_examples, teacher_cfg,
                         args.legacy_topk_shards)

    print("Migration finalize complete -- legacy shard dirs left untouched, remove manually once the new-tree output is verified.", flush=True)


if __name__ == "__main__":
    main()
