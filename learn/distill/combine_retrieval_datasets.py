#!/usr/bin/env python3
"""Combine two retrieval1 shard datasets (jsonl + top-K npz + hidden memmap dir)
into a single dataset, for the ecotaxe_a / ecotaxe_b split of retrieval1.

A was precomputed at K=64, B at K=32. Top-K entries are sorted descending by
value (verified directly on ecotaxe_a), so A's indices/values are truncated to
B's K on the fly -- lossless in the sense that the discarded entries are the
lowest-probability ones, matching what B already only has.

Each hidden dir's offsets.npy is treated as authoritative for how many tokens
of hidden_<layer>.npy are valid -- any trailing bytes beyond offsets[-1] are
ignored (see dev_notes/grid5000_usage.log.md, 2026-09-21, for why ecotaxe_b's
hidden_64.npy has a garbage tail past its true token count).
"""
import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a_jsonl", required=True)
    p.add_argument("--b_jsonl", required=True)
    p.add_argument("--a_npz", required=True)
    p.add_argument("--b_npz", required=True)
    p.add_argument("--a_hidden_dir", required=True)
    p.add_argument("--b_hidden_dir", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_hidden_dir", required=True)
    p.add_argument("--hidden_key", default="hidden_64")
    p.add_argument("--chunk_tokens", type=int, default=500_000)
    args = p.parse_args()

    out_dir = os.path.dirname(args.out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_jsonl, "w") as out:
        for src in (args.a_jsonl, args.b_jsonl):
            with open(src) as f:
                for line in f:
                    out.write(line if line.endswith("\n") else line + "\n")
    print(f"[combine] wrote {args.out_jsonl}", flush=True)

    a = np.load(args.a_npz)
    b = np.load(args.b_npz)
    k_b = int(b["k"])
    assert a["indices"].shape[1] >= k_b, (
        f"A has K={a['indices'].shape[1]} < B's K={k_b} -- cannot truncate A down to B's K"
    )
    indices = np.concatenate([a["indices"][:, :k_b], b["indices"]], axis=0)
    values = np.concatenate([a["values"][:, :k_b], b["values"]], axis=0)
    residual = np.concatenate([a["residual"], b["residual"]], axis=0)
    a_tokens = int(a["offsets"][-1])
    offsets = np.concatenate([a["offsets"], b["offsets"][1:] + a_tokens], axis=0)
    np.savez(args.out_npz, indices=indices, values=values, residual=residual,
             offsets=offsets, k=np.array(k_b, dtype=np.int64))
    print(f"[combine] wrote {args.out_npz} -- {len(offsets) - 1} examples, "
          f"{offsets[-1]} tokens, K={k_b}", flush=True)
    del a, b, indices, values, residual, offsets

    a_hidden = np.load(os.path.join(args.a_hidden_dir, f"{args.hidden_key}.npy"), mmap_mode="r")
    a_hoffsets = np.load(os.path.join(args.a_hidden_dir, "offsets.npy"))
    b_hidden = np.load(os.path.join(args.b_hidden_dir, f"{args.hidden_key}.npy"), mmap_mode="r")
    b_hoffsets = np.load(os.path.join(args.b_hidden_dir, "offsets.npy"))

    a_valid = int(a_hoffsets[-1])
    b_valid = int(b_hoffsets[-1])
    assert a_valid <= a_hidden.shape[0], f"A hidden offsets claim {a_valid} tokens but array only has {a_hidden.shape[0]}"
    assert b_valid <= b_hidden.shape[0], f"B hidden offsets claim {b_valid} tokens but array only has {b_hidden.shape[0]}"
    assert a_valid == a_tokens, f"A hidden token count {a_valid} != A top-K token count {a_tokens}"

    total_tokens = a_valid + b_valid
    dim = a_hidden.shape[1]
    assert b_hidden.shape[1] == dim, f"hidden dim mismatch A={dim} B={b_hidden.shape[1]}"

    os.makedirs(args.out_hidden_dir, exist_ok=True)
    out_path = os.path.join(args.out_hidden_dir, f"{args.hidden_key}.npy")
    out_mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=a_hidden.dtype,
                                        shape=(total_tokens, dim))

    chunk = args.chunk_tokens
    for start in range(0, a_valid, chunk):
        end = min(start + chunk, a_valid)
        out_mm[start:end] = a_hidden[start:end]
        print(f"[combine] A hidden {end}/{a_valid}", flush=True)
    for start in range(0, b_valid, chunk):
        end = min(start + chunk, b_valid)
        out_mm[a_valid + start:a_valid + end] = b_hidden[start:end]
        print(f"[combine] B hidden {end}/{b_valid}", flush=True)
    out_mm.flush()

    out_hoffsets = np.concatenate([a_hoffsets, b_hoffsets[1:] + a_valid], axis=0)
    np.save(os.path.join(args.out_hidden_dir, "offsets.npy"), out_hoffsets)
    print(f"[combine] wrote {args.out_hidden_dir} -- {total_tokens} tokens total", flush=True)
    print("[combine] DONE", flush=True)


if __name__ == "__main__":
    main()
