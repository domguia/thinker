"""CPU-only merge driver for precompute_teacher_targets.py shards.

Reuses merge_shards() directly instead of going through main(), so it never
touches the GPU and never requires full example-range contiguity: whatever
shard files exist get merged, and a manifest records exactly which example
ranges are covered so a later run can backfill a gap without re-merging
from scratch.

Usage:
    python -m learn.distill.merge_partial \
        --topk_dir DIR --hidden_dir DIR --out_file OUT.npz \
        --hidden_out_file OUT._hidden.npz --hidden_layer 64 --k 64 \
        --total_examples 4998
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_teacher_targets import SHARD_RE, merge_shards


def covered_ranges(shard_dir):
    ranges = sorted(
        (int(m.group(1)), int(m.group(2)))
        for p in glob.glob(os.path.join(shard_dir, "shard_*.npz"))
        for m in [SHARD_RE.search(p)] if m
    )
    # merge contiguous adjacent ranges for a compact manifest
    merged = []
    for start, end in ranges:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def shard_basenames(shard_dir):
    return {
        os.path.basename(p)
        for p in glob.glob(os.path.join(shard_dir, "shard_*.npz"))
        if SHARD_RE.search(p)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_dir", required=True)
    ap.add_argument("--hidden_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--hidden_out_file", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--hidden_layer", type=int, required=True)
    ap.add_argument("--total_examples", type=int, required=True,
                     help="Nominal full dataset size, for the manifest's missing-ranges report.")
    ap.add_argument("--manifest_out", required=True)
    args = ap.parse_args()

    # A shard is only truly "complete" if BOTH its topk and hidden files
    # exist -- an independent async-write crash can leave one done and the
    # other missing for the same example range (real incident, 2026-09-21:
    # ec_a's topk shard 3000-3500 finished but the hidden one didn't).
    # merge_shards() only globs topk_dir, so any topk-only shard here would
    # crash it trying to open a hidden file that isn't there. Temporarily
    # move such orphan topk shards out of the way so the glob only sees
    # shards that are genuinely complete in both directories, then restore
    # them afterward -- nothing is deleted, this run just skips them.
    topk_names = shard_basenames(args.topk_dir)
    hidden_names = shard_basenames(args.hidden_dir)
    orphan_topk = topk_names - hidden_names
    pending_dir = args.topk_dir + ".pending_hidden"
    if orphan_topk:
        os.makedirs(pending_dir, exist_ok=True)
        for name in orphan_topk:
            os.rename(os.path.join(args.topk_dir, name), os.path.join(pending_dir, name))
        print(f"Set aside {len(orphan_topk)} topk shard(s) with no matching hidden "
              f"shard yet (moved to {pending_dir}, not deleted): {sorted(orphan_topk)}", flush=True)

    ranges = covered_ranges(args.topk_dir)
    covered = sum(e - s for s, e in ranges)
    missing = []
    cursor = 0
    for s, e in ranges:
        if s > cursor:
            missing.append((cursor, s))
        cursor = e
    if cursor < args.total_examples:
        missing.append((cursor, args.total_examples))

    print(f"Covered ranges: {ranges} ({covered}/{args.total_examples} examples)", flush=True)
    print(f"Missing ranges: {missing}", flush=True)

    merge_shards(
        args.topk_dir, args.hidden_dir, args.out_file, args.hidden_out_file,
        args.k, [args.hidden_layer],
    )

    with open(args.manifest_out, "w") as f:
        json.dump({
            "total_examples_nominal": args.total_examples,
            "covered_ranges": ranges,
            "covered_count": covered,
            "missing_ranges": missing,
            "topk_file": args.out_file,
            "hidden_dir": (args.hidden_out_file[:-4] if args.hidden_out_file.endswith(".npz")
                            else args.hidden_out_file),
        }, f, indent=2)
    print(f"Manifest written to {args.manifest_out}", flush=True)


if __name__ == "__main__":
    main()
