"""Precompute Teacher targets (Top-K logits and/or hidden states) for offline KD.

Reads a JSONL produced by any prepare_*_data.py script (uses its "text"
field), runs each example through the Teacher, and stores per-token targets:

- Top-K logits: indices + values, plus a residual log-sum-exp scalar (the
  aggregated mass of every non-Top-K token) so the full softmax denominator
  can still be reconstructed exactly for the KL loss -- this is the "renorm
  scalar" from the storage formula in learn/distill/README.md (~194
  bytes/token at K=32: int32 indices + fp16 values + fp16 residual).
- Hidden states (--hidden_layers): for feature/embedding-level distillation,
  meant for small samples only (much heavier than Top-K logits: one full
  hidden_dim vector per token per selected layer, vs. K values). Which
  layer(s) to extract is a runtime choice ("none" / "last" / "all" / an
  explicit comma-separated list of indices), decided at inference time --
  the exact use in training is still open, this just makes the extraction
  available to experiment with later.

Output is a single compressed .npz per input file with flat (total_tokens, ...)
arrays plus an `offsets` array marking example boundaries (offsets[i] :
offsets[i+1] are the token rows for example i) -- a ragged-array-friendly
layout instead of one array per example.

Needs a GPU with enough VRAM for the Teacher (see bench_teacher.py first to
confirm the checkpoint loads and measure real throughput before committing
to a full precompute run, which is much slower: one forward pass per
example, no batching yet -- see the note in main() below).

Examples:
    # Top-K logits only (default)
    python learn/distill/precompute_teacher_targets.py \
      --input_file /tmp/distill_data/reasoning/train.jsonl \
      --model_dir /path/to/Qwen3.8-27B-FP8 \
      --top_k 32 --out_file /tmp/distill_data/reasoning/train_topk32.npz

    # Top-K logits + last hidden layer, on a small sample
    python learn/distill/precompute_teacher_targets.py \
      --input_file /tmp/distill_data/reasoning/sample.jsonl \
      --model_dir /path/to/Qwen3.8-27B-FP8 \
      --top_k 32 --hidden_layers last \
      --out_file /tmp/distill_data/reasoning/sample_targets.npz
"""
import argparse
import json
import time

import numpy as np
import torch

from bench_teacher import describe_gpus, load_model_and_tokenizer


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


def parse_hidden_layers(spec, num_layers):
    """"none" -> [], "last" -> [num_layers], "all" -> [0..num_layers], or a
    comma-separated list where each entry is either an explicit hidden_states
    index (0 is the embedding output, num_layers is the final layer's output)
    or the literal "last" -- e.g. "last,8" always gets the final layer plus
    layer 8, regardless of how many layers this particular Teacher has.

    2026-09-21 (model-design): the last layer is the default-useful one for
    almost any downstream reuse of these embeddings, so it should be captured
    on every repr-KD precompute, not just picked as a middle-ish-looking
    integer that happens to coincide with "last" for a 16-layer model (as
    happened here, undetected, for the first repr-KD run: LFM2-1.2B has
    num_hidden_layers=16, so "--hidden_layers 16" WAS already "last" -- but
    that only held because this particular Teacher has 16 layers, not by
    design). Use the "last" keyword explicitly (alone or combined with an
    intermediate index) instead of a hardcoded number, so the same flag stays
    correct across Teachers of different depths (LFM2/OLMo/Qwen)."""
    if spec == "none":
        return []
    if spec == "last":
        return [num_layers]
    if spec == "all":
        return list(range(num_layers + 1))
    return sorted({num_layers if x == "last" else int(x) for x in spec.split(",")})


def process_file(model, tokenizer, examples, k, max_length, hidden_layer_indices):
    all_indices, all_values, all_residual, offsets = [], [], [], [0]
    all_hidden = {layer: [] for layer in hidden_layer_indices}
    want_hidden = bool(hidden_layer_indices)
    start_time = time.time()
    progress_every = max(1, len(examples) // 100)

    for i, ex in enumerate(examples, 1):
        inputs = tokenizer(ex["text"], truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=want_hidden)
        logits = out.logits[0]  # (seq_len, vocab)

        indices, values, residual = topk_with_residual(logits, k)
        all_indices.append(indices.to(torch.int32).cpu().numpy())
        all_values.append(values.to(torch.float16).cpu().numpy())
        all_residual.append(residual.to(torch.float16).cpu().numpy())
        offsets.append(offsets[-1] + logits.shape[0])

        for layer in hidden_layer_indices:
            all_hidden[layer].append(out.hidden_states[layer][0].to(torch.float16).cpu().numpy())

        if i % progress_every == 0 or i == len(examples):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"[progress] processed={i}/{len(examples)} tokens={offsets[-1]} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s",
                flush=True,
            )

    hidden_arrays = {layer: np.concatenate(arrs, axis=0) for layer, arrs in all_hidden.items()}
    return (
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_values, axis=0),
        np.concatenate(all_residual, axis=0),
        np.array(offsets, dtype=np.int64),
        hidden_arrays,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True, help="JSONL from a prepare_*_data.py script (needs a 'text' field)")
    parser.add_argument("--model_dir", required=True, help="local Teacher snapshot dir, or a Hub repo id")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
        help="\"auto\" (default) keeps the checkpoint's own stored dtype -- required for the FP8 checkpoint: "
             "forcing bfloat16 dequantizes the whole model before it touches VRAM (~55.6 GB vs. ~30.9 GB "
             "native FP8), which silently triggers CPU offload and a ~50-100x throughput collapse on GPUs "
             "with less than ~56 GB VRAM (found 2026-09-04 on an A100 40GB; see qwen3.8-27b-notes.md). Only "
             "override this to a concrete dtype when using --quantization (bnb needs a real compute dtype).",
    )
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument(
        "--hidden_layers", default="none",
        help="'none' (default), 'last', 'all', or a comma-separated list of hidden_states "
             "indices (0=embedding output, num_layers=final layer) or the literal 'last' mixed "
             "in, e.g. 'last,8' -- always capture the last layer plus an intermediate one of your "
             "choice, correct regardless of this Teacher's actual depth. Only use on small samples "
             "-- one full hidden_dim vector per token per layer is much heavier than Top-K logits.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="limit to the first N visible GPUs for device_map=\"auto\" sharding "
             "(default: use every visible GPU)",
    )
    parser.add_argument(
        "--attn_implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="\"auto\" tries flash_attention_2 then falls back to sdpa",
    )
    parser.add_argument(
        "--quantization", default="none", choices=["none", "bnb-4bit", "bnb-8bit"],
        help="on-the-fly bitsandbytes quantization (needs pip install bitsandbytes); "
             "use with a bf16 repo (vendor or Unsloth), not with the already-quantized "
             "FP8 checkpoint -- see qwen3.8-27b-notes.md",
    )
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    with open(args.input_file) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} examples from {args.input_file}", flush=True)

    if args.dtype == "auto":
        if args.quantization != "none":
            raise ValueError("--dtype auto is only valid with --quantization none (bnb needs a concrete compute dtype, e.g. --dtype bfloat16)")
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)
    print("Detected GPU(s):", flush=True)
    describe_gpus()
    print(f"Loading Teacher {args.model_dir} in {args.dtype} ...", flush=True)
    print("  (shard-loading progress is printed by transformers itself below)", flush=True)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, dtype, num_gpus=args.num_gpus, attn_implementation=args.attn_implementation,
        quantization=args.quantization,
    )
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    # VLM wrapper configs (e.g. Qwen3_5Config) nest the LM's own config under
    # text_config -- num_hidden_layers lives there, not on the top-level config.
    text_config = getattr(model.config, "text_config", model.config)
    hidden_layer_indices = parse_hidden_layers(args.hidden_layers, text_config.num_hidden_layers)
    if hidden_layer_indices:
        print(f"Also extracting hidden states for layers: {hidden_layer_indices}", flush=True)

    # One forward pass per example (no batching): simplest correct version
    # first, per the project's staged-workflow habit -- batch later only if
    # bench_teacher.py's throughput numbers show it's actually the bottleneck.
    indices, values, residual, offsets, hidden_arrays = process_file(
        model, tokenizer, examples, args.top_k, args.max_length, hidden_layer_indices
    )

    save_kwargs = {"indices": indices, "values": values, "residual": residual, "offsets": offsets, "k": args.top_k}
    for layer, arr in hidden_arrays.items():
        save_kwargs[f"hidden_{layer}"] = arr
    np.savez_compressed(args.out_file, **save_kwargs)

    total_tokens = offsets[-1]
    measured_bytes_per_token = (indices.nbytes + values.nbytes + residual.nbytes) / total_tokens if total_tokens else float("nan")
    print(f"Wrote {args.out_file}: {total_tokens} tokens across {len(examples)} examples", flush=True)
    print(
        f"Measured Top-K storage: {measured_bytes_per_token:.1f} bytes/token at K={args.top_k} "
        f"(formula estimate: {args.top_k * 6 + 2} bytes/token)",
        flush=True,
    )
    for layer, arr in hidden_arrays.items():
        print(f"Hidden layer {layer}: shape={arr.shape} ({arr.nbytes / total_tokens:.1f} bytes/token)", flush=True)


if __name__ == "__main__":
    main()
