"""Precompute Top-K Teacher logits for offline logit-KD training.

Reads a JSONL produced by any prepare_*_data.py script (uses its "text"
field), runs each example through the Teacher, and stores per-token Top-K
logit indices + values plus a residual log-sum-exp scalar (the aggregated
mass of every non-Top-K token) so the full softmax denominator can still be
reconstructed exactly for the KL loss -- this is the "renorm scalar" from
the storage formula in learn/distill/README.md (~194 bytes/token at K=32:
int32 indices + fp16 values + fp16 residual).

Output is a single compressed .npz per input file with flat (total_tokens, K)
arrays plus an `offsets` array marking example boundaries (offsets[i] :
offsets[i+1] are the token rows for example i) -- a ragged-array-friendly
layout instead of one array per example.

Needs a GPU with enough VRAM for the Teacher (see bench_teacher.py first to
confirm the checkpoint loads and measure real throughput before committing
to a full precompute run, which is much slower: one forward pass per
example, no batching yet -- see the note in main() below).

Example:
    python learn/distill/precompute_topk_logits.py \
      --input_file /tmp/distill_data/reasoning/train.jsonl \
      --model_dir /path/to/Qwen3.8-27B-FP8 \
      --top_k 32 --out_file /tmp/distill_data/reasoning/train_topk32.npz
"""
import argparse
import json
import time

import numpy as np
import torch

from bench_teacher import load_model_and_tokenizer


def topk_with_residual(logits, k):
    """Top-K indices/values plus the log-sum-exp of everything else.

    exp(residual) + sum(exp(topk_values)) reconstructs the exact softmax
    denominator, so the KL loss can still be computed correctly against the
    full-vocabulary teacher distribution without storing all V logits.
    """
    topk = torch.topk(logits, k=k, dim=-1)
    masked = logits.scatter(-1, topk.indices, float("-inf"))
    residual = torch.logsumexp(masked, dim=-1)
    return topk.indices, topk.values, residual


def process_file(model, tokenizer, examples, k, max_length):
    all_indices, all_values, all_residual, offsets = [], [], [], [0]
    start_time = time.time()
    progress_every = max(1, len(examples) // 100)

    for i, ex in enumerate(examples, 1):
        inputs = tokenizer(ex["text"], truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]  # (seq_len, vocab)

        indices, values, residual = topk_with_residual(logits, k)
        all_indices.append(indices.to(torch.int32).cpu().numpy())
        all_values.append(values.to(torch.float16).cpu().numpy())
        all_residual.append(residual.to(torch.float16).cpu().numpy())
        offsets.append(offsets[-1] + logits.shape[0])

        if i % progress_every == 0 or i == len(examples):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"[progress] processed={i}/{len(examples)} tokens={offsets[-1]} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s",
                flush=True,
            )

    return (
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_values, axis=0),
        np.concatenate(all_residual, axis=0),
        np.array(offsets, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="JSONL from a prepare_*_data.py script (needs a 'text' field)")
    parser.add_argument("--model_dir", required=True, help="local Teacher snapshot dir, or a Hub repo id")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    with open(args.input_file) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} examples from {args.input_file}", flush=True)

    dtype = getattr(torch, args.dtype)
    print(f"Loading Teacher {args.model_dir} in {args.dtype} ...", flush=True)
    print("  (shard-loading progress is printed by transformers itself below)", flush=True)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_dir, dtype)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    # One forward pass per example (no batching): simplest correct version
    # first, per the project's staged-workflow habit -- batch later only if
    # bench_teacher.py's throughput numbers show it's actually the bottleneck.
    indices, values, residual, offsets = process_file(model, tokenizer, examples, args.top_k, args.max_length)

    np.savez_compressed(args.out_file, indices=indices, values=values, residual=residual, offsets=offsets, k=args.top_k)

    total_tokens = offsets[-1]
    measured_bytes_per_token = (indices.nbytes + values.nbytes + residual.nbytes) / total_tokens if total_tokens else float("nan")
    print(f"Wrote {args.out_file}: {total_tokens} tokens across {len(examples)} examples", flush=True)
    print(
        f"Measured storage: {measured_bytes_per_token:.1f} bytes/token at K={args.top_k} "
        f"(formula estimate: {args.top_k * 6 + 2} bytes/token)",
        flush=True,
    )


if __name__ == "__main__":
    main()
