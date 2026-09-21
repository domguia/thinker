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

Output is TWO compressed .npz files -- top-K logits and hidden states are
kept separate so training code that only needs top-K (the common case) never
pays the cost of loading the much heavier embedding arrays:
    <out_file>            top-K logits: indices/values/residual/offsets/k
    <out_file>._hidden.npz  hidden states: hidden_<layer>/offsets (only
                             written when --hidden_layers != none)

2026-09-21 (critical fix, "on a perdu les 90%"): this script used to hold
everything in memory and call np.savez_compressed exactly once, at the very
end. Any interruption before completion (besteffort eviction, network
outage, crash -- all routine on this project's infra) lost 100% of the work
already done, twice in the same evening at ~90% completion. It now flushes
a shard to a persistent directory every --shard_size examples, in a
background thread so the GPU loop is not stalled by disk I/O, and resumes
automatically from the last complete shard if relaunched with the same
--out_file after being interrupted. Apply this same pattern to any other
precompute-style script in this project that runs for more than a few
minutes per example.

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
import glob
import json
import os
import re
import threading
import time

import numpy as np
import torch

from bench_teacher import describe_gpus, load_model_and_tokenizer

SHARD_RE = re.compile(r"shard_(\d+)_(\d+)\.npz$")


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


def shard_dir(out_file, suffix):
    base = out_file[:-4] if out_file.endswith(".npz") else out_file
    d = f"{base}.{suffix}_shards"
    os.makedirs(d, exist_ok=True)
    return d


def find_resume_point(topk_dir):
    """Scan for complete contiguous shards starting at 0; return the example
    index to resume from (0 if none found, or a gap/corruption is detected)."""
    shards = []
    for path in glob.glob(os.path.join(topk_dir, "shard_*.npz")):
        m = SHARD_RE.search(path)
        if m:
            shards.append((int(m.group(1)), int(m.group(2)), path))
    shards.sort()
    resume_at = 0
    for start, end, path in shards:
        if start != resume_at:
            break  # gap: stop trusting shards from here on
        try:
            np.load(path)  # cheap integrity check (raises on truncated file)
        except Exception:
            break
        resume_at = end
    return resume_at


def save_shard_async(path, arrays):
    """Write a shard in a background thread (numpy I/O releases the GIL) so
    the GPU loop is never stalled waiting for disk, with an atomic rename so
    a crash mid-write never leaves a corrupt shard that find_resume_point
    would trust."""
    def _write():
        # np.savez_compressed silently appends ".npz" when the given name
        # doesn't already end with it -- the tmp name must end in ".npz" too,
        # or the file actually written won't match what os.replace expects.
        tmp = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    t = threading.Thread(target=_write, daemon=False)
    t.start()
    return t


def merge_shards(topk_dir, hidden_dir, out_file, hidden_out_file, k, hidden_layer_indices):
    """Concatenate all shards into the final two .npz files and fix up the
    per-example token offsets, which are shard-relative (each shard starts
    its own offsets at 0) and need a running token total added back in.

    2026-09-21 (OOM fix): the hidden-states array (last layer, every token)
    can be tens of GB -- building it as a Python list of shard arrays then
    np.concatenate'ing doubles peak RAM (list + concatenated copy) and was
    observed to silently OOM-kill the process mid-merge on a node with
    otherwise plenty of GPU VRAM but limited host RAM (no traceback, since
    the OOM killer sends SIGKILL -- looked identical to the other silent
    deaths tonight until diagnosed). Top-K arrays stay small (K=32) so a
    plain concatenate is fine for those; the hidden array is written
    directly into a disk-backed memmap sized up front, shard by shard, so
    at most one shard's worth of hidden states is ever resident in RAM.
    """
    shards = sorted(
        (int(SHARD_RE.search(p).group(1)), p)
        for p in glob.glob(os.path.join(topk_dir, "shard_*.npz"))
    )
    all_indices, all_values, all_residual, offsets = [], [], [], [0]
    token_total = 0
    for _, path in shards:
        d = np.load(path)
        all_indices.append(d["indices"])
        all_values.append(d["values"])
        all_residual.append(d["residual"])
        offsets.extend((d["offsets"][1:] + token_total).tolist())
        token_total += int(d["offsets"][-1])

    indices = np.concatenate(all_indices, axis=0)
    values = np.concatenate(all_values, axis=0)
    residual = np.concatenate(all_residual, axis=0)
    offsets = np.array(offsets, dtype=np.int64)

    np.savez_compressed(out_file, indices=indices, values=values, residual=residual, offsets=offsets, k=k)

    hidden_arrays = {}
    if hidden_layer_indices:
        # Pass 1 (cheap, headers only): total tokens + per-layer dtype/dim.
        shapes = {}
        for layer in hidden_layer_indices:
            total = 0
            dim = dtype = None
            for _, path in shards:
                hd = np.load(os.path.join(hidden_dir, os.path.basename(path)))
                arr = hd[f"hidden_{layer}"]
                total += arr.shape[0]
                dim, dtype = arr.shape[1], arr.dtype
            shapes[layer] = (total, dim, dtype)

        hidden_npy_dir = hidden_out_file[:-4] if hidden_out_file.endswith(".npz") else hidden_out_file
        hidden_npy_dir += ".memmap_tmp"
        os.makedirs(hidden_npy_dir, exist_ok=True)
        memmaps = {}
        for layer, (total, dim, dtype) in shapes.items():
            mmap_path = os.path.join(hidden_npy_dir, f"hidden_{layer}.npy")
            memmaps[layer] = np.lib.format.open_memmap(mmap_path, mode="w+", dtype=dtype, shape=(total, dim))

        # Pass 2: copy each shard directly into its slice of the memmap.
        cursor = {layer: 0 for layer in hidden_layer_indices}
        for _, path in shards:
            hd = np.load(os.path.join(hidden_dir, os.path.basename(path)))
            for layer in hidden_layer_indices:
                arr = hd[f"hidden_{layer}"]
                n = arr.shape[0]
                memmaps[layer][cursor[layer]:cursor[layer] + n] = arr
                cursor[layer] += n

        for m in memmaps.values():
            m.flush()
        hidden_arrays = memmaps  # memmap arrays, not loaded fully in RAM
        # np.savez can't stream from memmaps without loading them, and the
        # whole point here is to never hold the full array in RAM -- so the
        # final hidden-states artifact IS the memmap directory (one .npy per
        # layer) rather than a single .npz. Ship an offsets.npy alongside so
        # a loader can still resolve token boundaries.
        np.save(os.path.join(hidden_npy_dir, "offsets.npy"), offsets)
        if os.path.isdir(hidden_out_file):
            pass  # already a dir from a prior partial run
        elif os.path.exists(hidden_out_file):
            os.remove(hidden_out_file)
        final_dir = hidden_out_file[:-4] if hidden_out_file.endswith(".npz") else hidden_out_file
        if os.path.isdir(final_dir):
            import shutil
            shutil.rmtree(final_dir)
        os.rename(hidden_npy_dir, final_dir)
        print(f"Hidden states written as memmap directory (not .npz, too large to hold in RAM): {final_dir}", flush=True)

    return indices, values, residual, offsets, hidden_arrays


def process_file(model, tokenizer, examples, k, max_length, hidden_layer_indices,
                  topk_dir, hidden_dir, shard_size, resume_at):
    pending_threads = []
    idx_buf, val_buf, res_buf, off_buf = [], [], [], [0]
    hidden_buf = {layer: [] for layer in hidden_layer_indices}
    shard_start = resume_at
    start_time = time.time()
    progress_every = max(1, len(examples) // 100)

    def flush(end_ex):
        nonlocal idx_buf, val_buf, res_buf, off_buf, hidden_buf, shard_start
        if end_ex == shard_start:
            return
        tag = f"shard_{shard_start:07d}_{end_ex:07d}.npz"
        topk_arrays = {
            "indices": np.concatenate(idx_buf, axis=0),
            "values": np.concatenate(val_buf, axis=0),
            "residual": np.concatenate(res_buf, axis=0),
            "offsets": np.array(off_buf, dtype=np.int64),
        }
        pending_threads.append(save_shard_async(os.path.join(topk_dir, tag), topk_arrays))
        if hidden_layer_indices:
            hidden_arrays = {f"hidden_{layer}": np.concatenate(arrs, axis=0) for layer, arrs in hidden_buf.items()}
            pending_threads.append(save_shard_async(os.path.join(hidden_dir, tag), hidden_arrays))
        idx_buf, val_buf, res_buf, off_buf = [], [], [], [0]
        hidden_buf = {layer: [] for layer in hidden_layer_indices}
        shard_start = end_ex

    for i, ex in enumerate(examples[resume_at:], resume_at + 1):
        inputs = tokenizer(ex["text"], truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=bool(hidden_layer_indices))
        logits = out.logits[0]  # (seq_len, vocab)

        indices, values, residual = topk_with_residual(logits, k)
        idx_buf.append(indices.to(torch.int32).cpu().numpy())
        val_buf.append(values.to(torch.float16).cpu().numpy())
        res_buf.append(residual.to(torch.float16).cpu().numpy())
        off_buf.append(off_buf[-1] + logits.shape[0])

        for layer in hidden_layer_indices:
            hidden_buf[layer].append(out.hidden_states[layer][0].to(torch.float16).cpu().numpy())

        if i % shard_size == 0 or i == len(examples):
            flush(i)

        if i % progress_every == 0 or i == len(examples):
            elapsed = time.time() - start_time
            done = i - resume_at
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"[progress] processed={i}/{len(examples)} tokens_this_run={off_buf[-1] if off_buf[-1] else ''} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} ex/s",
                flush=True,
            )

    for t in pending_threads:
        t.join()


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
             "-- one full hidden_dim vector per token per layer is much heavier than Top-K logits. "
             "Written to a SEPARATE .npz file from the top-K logits (see module docstring).",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--shard_size", type=int, default=1000,
        help="flush a persistent shard to disk every N examples (async, non-blocking) -- also the "
             "resume granularity if this run gets interrupted and relaunched with the same --out_file",
    )
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
    parser.add_argument("--out_file", required=True, help="top-K logits output (.npz); hidden states go to <out_file minus .npz>._hidden.npz")
    args = parser.parse_args()

    with open(args.input_file) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} examples from {args.input_file}", flush=True)

    topk_dir = shard_dir(args.out_file, "topk")
    hidden_dir = shard_dir(args.out_file, "hidden")
    resume_at = find_resume_point(topk_dir)
    if resume_at:
        print(f"Resuming from example {resume_at}/{len(examples)} (found complete shards up to there)", flush=True)
    if resume_at >= len(examples):
        print("All examples already covered by existing shards, skipping straight to merge.", flush=True)

    if args.dtype == "auto":
        if args.quantization != "none":
            raise ValueError("--dtype auto is only valid with --quantization none (bnb needs a concrete compute dtype, e.g. --dtype bfloat16)")
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)

    hidden_out_file = (args.out_file[:-4] if args.out_file.endswith(".npz") else args.out_file) + "._hidden.npz"

    if resume_at < len(examples):
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
        process_file(
            model, tokenizer, examples, args.top_k, args.max_length, hidden_layer_indices,
            topk_dir, hidden_dir, args.shard_size, resume_at,
        )
    else:
        # Nothing left to run (e.g. re-launched purely to retry a merge that
        # OOM'd) -- no need to reload the Teacher just to resolve "last" into
        # a layer number: the shards already have it baked into their key
        # names (hidden_<N>), so read it straight off an existing shard.
        print("Skipping model load (nothing left to process) -- reading hidden-layer indices off existing shards.", flush=True)
        any_hidden_shard = next(iter(glob.glob(os.path.join(hidden_dir, "shard_*.npz"))), None)
        if any_hidden_shard:
            d = np.load(any_hidden_shard)
            hidden_layer_indices = sorted(int(k.split("_", 1)[1]) for k in d.files if k.startswith("hidden_"))
        else:
            hidden_layer_indices = []

    indices, values, residual, offsets, hidden_arrays = merge_shards(
        topk_dir, hidden_dir, args.out_file, hidden_out_file, args.top_k, hidden_layer_indices,
    )

    total_tokens = offsets[-1]
    measured_bytes_per_token = (indices.nbytes + values.nbytes + residual.nbytes) / total_tokens if total_tokens else float("nan")
    print(f"Wrote {args.out_file}: {total_tokens} tokens across {len(examples)} examples", flush=True)
    print(
        f"Measured Top-K storage: {measured_bytes_per_token:.1f} bytes/token at K={args.top_k} "
        f"(formula estimate: {args.top_k * 6 + 2} bytes/token)",
        flush=True,
    )
    for layer, arr in hidden_arrays.items():
        print(f"Hidden layer {layer}: shape={arr.shape} ({arr.nbytes / total_tokens:.1f} bytes/token) -> {hidden_out_file}", flush=True)


if __name__ == "__main__":
    main()
