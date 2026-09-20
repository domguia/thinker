"""
Merge Top-K Teacher-target .npz shards produced by running
precompute_teacher_targets.py on line-ordered splits of one input file
(e.g. `split -d -n l/3 train.jsonl train_shard_`) back into one .npz
equivalent to a single run on the full file.

Each shard's `topk_indices`/`topk_values`/`residual` are flat (total_tokens,
...) arrays; `offsets` marks per-example token-row boundaries within that
shard (offsets[i]:offsets[i+1] = example i's rows). Merging concatenates the
flat arrays in shard order and re-bases each shard's offsets by the running
total token count so far -- offsets[0] stays 0 only for the first shard,
every later shard's own leading 0 is dropped (it duplicates the previous
shard's final offset).

Usage:
    python learn/distill/merge_teacher_shards.py \
      --shards a_shard00.npz a_shard01.npz a_shard02.npz \
      --out a_topk32.npz
Shard order MUST match the original line order of `split`'s output
(shard_00, shard_01, ... in that order) -- this is not re-derived from the
filenames, pass them in the correct order explicitly.
"""
import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", nargs="+", required=True, help="shard .npz files, IN LINE ORDER")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    all_indices, all_values, all_residual = [], [], []
    all_offsets = [0]
    hidden_arrays = {}  # layer_key -> list of shard arrays, if hidden states were extracted
    k = None

    token_total = 0
    for shard_path in args.shards:
        d = np.load(shard_path)
        all_indices.append(d["indices"])
        all_values.append(d["values"])
        all_residual.append(d["residual"])
        shard_offsets = d["offsets"]
        # drop the shard's own leading 0 (duplicates the running total), rebase the rest
        all_offsets.extend((shard_offsets[1:] + token_total).tolist())
        token_total += shard_offsets[-1]
        for key in d.files:
            if key.startswith("hidden_"):
                hidden_arrays.setdefault(key, []).append(d[key])
        if k is None:
            k = int(d["k"])
        else:
            assert k == int(d["k"]), f"shard top_k mismatch: {k} vs {d['k']} in {shard_path}"
        print(f"{shard_path}: {len(shard_offsets) - 1} examples, {shard_offsets[-1]} tokens")

    merged = {
        "indices": np.concatenate(all_indices, axis=0),
        "values": np.concatenate(all_values, axis=0),
        "residual": np.concatenate(all_residual, axis=0),
        "offsets": np.array(all_offsets, dtype=np.int64),
        "k": k,
    }
    for key, arrs in hidden_arrays.items():
        merged[key] = np.concatenate(arrs, axis=0)

    np.savez_compressed(args.out, **merged)
    n_examples = len(all_offsets) - 1
    print(f"merged {len(args.shards)} shards -> {args.out}: {n_examples} examples, {token_total} tokens")


if __name__ == "__main__":
    main()
