#!/usr/bin/env python3
"""Drop a contiguous range of examples [exclude_lo, exclude_hi) from a
retrieval-style dataset (jsonl + top-K npz + hidden memmap dir), keeping
everything else. Used to strip corrupted examples (e.g. the hydra gap-fill
that produced garbage top-K/hidden data, see dev_notes/grid5000_usage.log.md
2026-09-21) before merging with another shard.
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--npz", required=True)
    p.add_argument("--hidden_dir", required=True)
    p.add_argument("--exclude_lo", type=int, required=True)
    p.add_argument("--exclude_hi", type=int, required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_hidden_dir", required=True)
    p.add_argument("--hidden_key", default="hidden_64")
    p.add_argument("--chunk_tokens", type=int, default=500_000)
    args = p.parse_args()

    lo, hi = args.exclude_lo, args.exclude_hi

    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.jsonl) as f, open(args.out_jsonl, "w") as out:
        for i, line in enumerate(f):
            if lo <= i < hi:
                continue
            out.write(line if line.endswith("\n") else line + "\n")
    print(f"[filter] wrote {args.out_jsonl}", flush=True)

    npz = np.load(args.npz)
    offsets = npz["offsets"]
    n_ex = len(offsets) - 1
    assert 0 <= lo < hi <= n_ex, f"exclude range [{lo},{hi}) out of bounds for {n_ex} examples"

    tok_lo, tok_hi = int(offsets[lo]), int(offsets[hi])
    keep_token_mask_ranges = [(0, tok_lo), (tok_hi, int(offsets[-1]))]

    indices = np.concatenate([npz["indices"][a:b] for a, b in keep_token_mask_ranges], axis=0)
    values = np.concatenate([npz["values"][a:b] for a, b in keep_token_mask_ranges], axis=0)
    residual = np.concatenate([npz["residual"][a:b] for a, b in keep_token_mask_ranges], axis=0)

    # offsets[:lo+1] ends at offsets[lo] (== tok_lo); offsets[hi:] shifted starts at
    # offsets[hi]-(tok_hi-tok_lo) == tok_lo again -- drop that duplicated boundary.
    kept_ex_offsets = np.concatenate([offsets[:lo + 1], (offsets[hi:] - (tok_hi - tok_lo))[1:]])
    assert kept_ex_offsets[-1] == indices.shape[0], (kept_ex_offsets[-1], indices.shape[0])

    np.savez(args.out_npz, indices=indices, values=values, residual=residual,
              offsets=kept_ex_offsets, k=npz["k"])
    print(f"[filter] wrote {args.out_npz} -- {len(kept_ex_offsets) - 1} examples "
          f"(dropped {hi - lo}), {kept_ex_offsets[-1]} tokens", flush=True)
    del npz, indices, values, residual

    hidden = np.load(os.path.join(args.hidden_dir, f"{args.hidden_key}.npy"), mmap_mode="r")
    hoffsets = np.load(os.path.join(args.hidden_dir, "offsets.npy"))
    assert len(hoffsets) - 1 == n_ex, f"hidden offsets have {len(hoffsets) - 1} examples, npz has {n_ex}"
    h_tok_lo, h_tok_hi = int(hoffsets[lo]), int(hoffsets[hi])
    h_total = int(hoffsets[-1])
    dim = hidden.shape[1]
    out_total = h_total - (h_tok_hi - h_tok_lo)

    os.makedirs(args.out_hidden_dir, exist_ok=True)
    out_path = os.path.join(args.out_hidden_dir, f"{args.hidden_key}.npy")
    out_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=hidden.dtype, shape=(out_total, dim))

    chunk = args.chunk_tokens
    write_pos = 0
    for a, b in [(0, h_tok_lo), (h_tok_hi, h_total)]:
        for start in range(a, b, chunk):
            end = min(start + chunk, b)
            n = end - start
            out_mm[write_pos:write_pos + n] = hidden[start:end]
            write_pos += n
            print(f"[filter] hidden {write_pos}/{out_total}", flush=True)
    out_mm.flush()
    assert write_pos == out_total

    out_hoffsets = np.concatenate([hoffsets[:lo + 1], (hoffsets[hi:] - (h_tok_hi - h_tok_lo))[1:]])
    np.save(os.path.join(args.out_hidden_dir, "offsets.npy"), out_hoffsets)
    print(f"[filter] wrote {args.out_hidden_dir} -- {out_total} tokens total", flush=True)
    print("[filter] DONE", flush=True)


if __name__ == "__main__":
    main()
