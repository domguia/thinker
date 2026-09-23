"""Merge the shards already written by a STILL-RUNNING precompute_teacher_targets.py
run into an intermediate, immediately-usable top-K (+ hidden) checkpoint, without
touching or interrupting the live process.

Safe because shards are written atomically (os.replace) and never modified after
creation -- this only reads shards that already exist at call time, symlinks the
ones within the requested prefix into a scratch dir, and reuses the exact same
finalize_topk/finalize_hidden merge logic as the main script's own completion path.

Only meaningful when --n_examples lands exactly on a shard boundary (a multiple of
the run's --shard_size) -- pass one of the boundaries visible in the shard
filenames (shard_<start>_<end>.npz) under
<dataset_root>/.precompute_tmp/<subset_name>.<teacher_name>/topk_shards/.

Usage:
  python learn/distill/merge_partial_topk.py \
    --dataset_root .../hotpotqa --split train --subset_name thinkfix_full \
    --teacher_name qwen_big --n_examples 20000 --checkpoint_name thinkfix_p20000 \
    --top_k 64 --hidden_layer_indices 64 \
    --checkpoint_dir /tmp/Qwen3.8-27B-bf16 --dtype bfloat16
"""
import argparse
import glob
import os
import re
import shutil
import tempfile

import numpy as np

from precompute_teacher_targets import SHARD_RE, finalize_hidden, finalize_topk


def symlink_shards_up_to(src_dir, dst_dir, n_examples):
    os.makedirs(dst_dir, exist_ok=True)
    included = 0
    for path in sorted(glob.glob(os.path.join(src_dir, "shard_*.npz"))):
        m = SHARD_RE.search(path)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if end > n_examples:
            continue
        os.symlink(os.path.abspath(path), os.path.join(dst_dir, os.path.basename(path)))
        included = max(included, end)
    return included


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--subset_name", required=True, help="the RUNNING job's --subset_name (e.g. thinkfix_full)")
    ap.add_argument("--teacher_name", required=True)
    ap.add_argument("--n_examples", type=int, required=True, help="must match a shard boundary exactly")
    ap.add_argument("--checkpoint_name", required=True, help="new subset_name for this intermediate checkpoint, e.g. thinkfix_p20000")
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--hidden_layer_indices", default="",
                     help="comma-separated, ALREADY-RESOLVED hidden_states indices (e.g. '64') -- "
                          "check the running job's log line 'Also extracting hidden states for layers: [...]' "
                          "for the exact value(s), since this script never loads the model to resolve 'last' itself")
    ap.add_argument("--checkpoint_dir", required=True, help="Teacher model_dir, for the manifest's teacher_cfg only")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    work_dir = os.path.join(args.dataset_root, ".precompute_tmp", f"{args.subset_name}.{args.teacher_name}")
    topk_shards = os.path.join(work_dir, "topk_shards")
    hidden_shards = os.path.join(work_dir, "hidden_shards")

    full_indices_path = os.path.join(args.dataset_root, "subsets", args.split, f"{args.subset_name}.indices.npy")
    full_indices = np.load(full_indices_path)
    if args.n_examples > len(full_indices):
        raise ValueError(f"--n_examples {args.n_examples} > full subset size {len(full_indices)}")

    ckpt_subset_dir = os.path.join(args.dataset_root, "subsets", args.split)
    os.makedirs(ckpt_subset_dir, exist_ok=True)
    ckpt_indices_path = os.path.join(ckpt_subset_dir, f"{args.checkpoint_name}.indices.npy")
    # Prefix truncation is only valid because load_or_create_subset stores
    # indices sorted ascending and thinkfix_full's --n_examples == full pool
    # size (a full sweep, not a random sample) -- processed order 0..N-1
    # matches dataset row order 0..N-1 exactly in that case only.
    np.save(ckpt_indices_path, full_indices[: args.n_examples])

    teacher_cfg = {
        "checkpoint": os.path.basename(args.checkpoint_dir.rstrip("/")),
        "dtype": args.dtype,
        "quantization": "none",
        "attn_implementation": "unknown",
        "max_length": None,
    }

    with tempfile.TemporaryDirectory() as scratch:
        topk_link_dir = os.path.join(scratch, "topk")
        covered = symlink_shards_up_to(topk_shards, topk_link_dir, args.n_examples)
        if covered < args.n_examples:
            raise ValueError(
                f"only {covered} examples covered by complete shards so far, "
                f"requested checkpoint at {args.n_examples} -- wait for more shards or lower --n_examples"
            )
        if covered != args.n_examples:
            raise ValueError(
                f"--n_examples {args.n_examples} does not land on a shard boundary "
                f"(closest available: {covered}) -- pick an exact boundary from the shard filenames"
            )
        finalize_topk(args.dataset_root, args.split, args.checkpoint_name, args.teacher_name,
                       topk_link_dir, args.top_k, teacher_cfg)

        hidden_layer_indices = [int(x) for x in args.hidden_layer_indices.split(",") if x.strip()]
        if hidden_layer_indices:
            hidden_link_dir = os.path.join(scratch, "hidden")
            symlink_shards_up_to(hidden_shards, hidden_link_dir, args.n_examples)
            finalize_hidden(args.dataset_root, args.split, args.checkpoint_name, args.teacher_name,
                             hidden_link_dir, hidden_layer_indices, args.n_examples, teacher_cfg, topk_link_dir)

    print(f"Checkpoint {args.checkpoint_name!r} ({args.n_examples} examples) merged successfully -- "
          f"safe to consume now, the source run keeps writing new shards independently.", flush=True)


if __name__ == "__main__":
    main()
