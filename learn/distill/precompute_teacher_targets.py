"""Precompute Teacher targets (Top-K logits and/or hidden states) for offline KD.

Reads <dataset_root>/<split>.jsonl (a prepare_*_data.py output, needs a
"text" field), runs a NAMED SUBSET of its rows through a Teacher, and writes
into the project's per-dataset storage tree (2026-09-22 redesign, replaces
the old single-.npz-plus-shards-to-merge layout):

    <dataset_root>/subsets/<split>/<subset_name>.indices.npy
        original row indices (into <split>.jsonl) this subset covers --
        created once (random sample without replacement, sorted ascending),
        reused/extended-by-new-subset from then on. Different subsets are
        independent named samples, NOT required to nest -- pick a fresh
        --subset_name for a different draw rather than mutating one in place.
    <dataset_root>/topk/<split>/<subset_name>.<teacher_name>.npz
        indices/values/residual/offsets/k -- Top-K logits for this subset.
    <dataset_root>/topk/<split>/manifest.json
        {<subset_name>: {"subset_file": "subsets/<split>/<subset_name>.indices.npy",
                          "teachers": {<teacher_name>: {"top_k", "dtype", "checkpoint",
                                                          "max_length", "file"}}}}
    <dataset_root>/embedding/<split>/layer_<L>/n<N>.<teacher_name>.npy(+.offsets.npy)
        hidden states for layer L, ONE FILE PER LAYER (not bundled), named
        by subset SIZE only (not the full subset_name -- the layer folder
        already gives context, see manifest for the canonical subset name).
    <dataset_root>/embedding/<split>/layer_<L>/manifest.json
        {"n<N>": {"subset": <subset_name>, "teachers": {<teacher_name>: {...}}}}

Multiple Teachers (different tokenizers/vocabs) can cover the same subset --
they get separate files (`<subset_name>.<teacher_name>.npz`, `n<N>.<teacher_name>.npy`)
since token counts/offsets differ per tokenizer; the manifest keys them by
teacher name so a loader picks the right one.

No shard-to-single-growing-file merge across separate runs anymore: shards
are purely this RUN's crash-recovery mechanism (written under a temp work
dir, cleaned up after a successful finalize) -- the final artifact for a
given (subset_name, teacher_name) is written once and never appended to.
Want more coverage later? Create a new, larger subset under a new name;
don't try to grow an existing one in place.

Needs a GPU with enough VRAM for the Teacher (see bench_teacher.py first to
confirm the checkpoint loads and measure real throughput before committing
to a full run, which is much slower: one forward pass per example, no
batching yet -- see the note in main() below).

2026-09-22: the FP8 checkpoint (Qwen3.8-27B-FP8) has a confirmed, still-
unfixed (checked against transformers main/5.18.0.dev0) `gate_proj`
dequantization bug -- silently produces garbage logits/hidden states with no
error. Use the bf16 checkpoint (`--dtype bfloat16`, e.g.
killerdroid/thinker-distill/Qwen3.8-27B-bf16) until this is fixed upstream;
see dev_notes/grid5000_usage.log.md for the root-cause writeup. Always
sanity-check a fresh Teacher/environment combo with a trivial greedy
`model.generate()` prompt before trusting a large run (see bench_teacher.py).

Examples:
    # Top-K only, on the (already extracted) full training pool
    python learn/distill/precompute_teacher_targets.py \\
      --dataset_root data/distill/openr1_math_full --split train \\
      --subset_name topk_n38901 --n_examples 38901 --seed 0 \\
      --teacher_name qwen_big --model_dir .../Qwen3.8-27B-bf16 --dtype bfloat16 \\
      --top_k 64

    # Top-K + every hidden layer, on a tiny diagnostic subset
    python learn/distill/precompute_teacher_targets.py \\
      --dataset_root data/distill/openr1_math_full --split train \\
      --subset_name allLayers_n4 --n_examples 4 --seed 0 \\
      --teacher_name qwen_big --model_dir .../Qwen3.8-27B-bf16 --dtype bfloat16 \\
      --top_k 64 --hidden_layers all
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
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
    correct across Teachers of different depths (LFM2/OLMo/Qwen).

    2026-09-22 (model-design): layer 0 (the raw token-embedding output,
    before any transformer block runs) is the one layer genuinely free of
    positional encoding for a RoPE model (Qwen3.5/3.8) -- RoPE rotates Q/K
    inside attention, it's never added to the residual stream, so every
    layer past the first block already has it folded in. Include "0"
    explicitly in --hidden_layers whenever a position-free embedding is
    wanted; it's just a normal explicit index here, nothing special-cased."""
    if spec == "none":
        return []
    if spec == "last":
        return [num_layers]
    if spec == "all":
        return list(range(num_layers + 1))
    return sorted({num_layers if x == "last" else int(x) for x in spec.split(",")})


def load_or_create_subset(dataset_root, split, subset_name, n_examples, seed, pool_size):
    """Resolve subsets/<split>/<subset_name>.indices.npy -- reuse it if it
    already exists (ignoring --n_examples/--seed, which only matter for
    CREATING a subset), otherwise draw a fresh random sample without
    replacement from range(pool_size) and persist it. Indices are sorted
    ascending purely for sequential-read locality; sampling itself is what
    makes this a random (not "first N") subset."""
    subset_dir = os.path.join(dataset_root, "subsets", split)
    os.makedirs(subset_dir, exist_ok=True)
    path = os.path.join(subset_dir, f"{subset_name}.indices.npy")
    if os.path.exists(path):
        indices = np.load(path)
        print(f"Reusing existing subset {path} ({len(indices)} examples) -- "
              f"--n_examples/--seed ignored for an existing subset.", flush=True)
        return indices
    if n_examples is None:
        raise ValueError(f"{path} doesn't exist yet -- pass --n_examples (and optionally --seed) to create it")
    if n_examples > pool_size:
        raise ValueError(f"--n_examples {n_examples} > pool size {pool_size} for {split}")
    indices = np.array(sorted(random.Random(seed).sample(range(pool_size), n_examples)), dtype=np.int64)
    tmp = path + ".tmp.npy"
    np.save(tmp, indices)
    os.replace(tmp, path)
    print(f"Created new subset {path}: {n_examples} examples (seed={seed})", flush=True)
    return indices


def shard_dir(work_dir, suffix):
    d = os.path.join(work_dir, f"{suffix}_shards")
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
        tmp = path[:-4] + ".tmp.npz" if path.endswith(".npz") else path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    t = threading.Thread(target=_write, daemon=False)
    t.start()
    return t


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


def _load_manifest(dir_path):
    path = os.path.join(dir_path, "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(dir_path, manifest):
    path = os.path.join(dir_path, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _check_no_collision(existing_cfg, new_cfg, where):
    """Refuse to silently overwrite a manifest entry with a differently-
    configured run under the same name -- same rationale as the shard
    resume/merge integrity checks above: a silent mismatch here is exactly
    the kind of misalignment (see model-families skill: qwen/qwen_big vocab
    mismatch, eval_causal_control.py tokenization bug) this project has
    already been bitten by twice."""
    if existing_cfg is None:
        return
    mismatched = {k: (existing_cfg.get(k), v) for k, v in new_cfg.items() if existing_cfg.get(k) != v}
    if mismatched:
        raise ValueError(
            f"{where}: existing manifest entry has different config than this run -- {mismatched}. "
            "Use a different --subset_name/--teacher_name if this is genuinely a different run."
        )


def finalize_topk(dataset_root, split, subset_name, teacher_name, topk_dir_shards, k,
                   teacher_cfg):
    shards = sorted(
        (int(m.group(1)), p)
        for p in glob.glob(os.path.join(topk_dir_shards, "shard_*.npz"))
        for m in [SHARD_RE.search(p)] if m
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

    out_dir = os.path.join(dataset_root, "topk", split)
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{subset_name}.{teacher_name}.npz"
    np.savez_compressed(os.path.join(out_dir, out_name), indices=indices, values=values,
                         residual=residual, offsets=offsets, k=k)

    manifest = _load_manifest(out_dir)
    entry = manifest.setdefault(subset_name, {"subset_file": f"subsets/{split}/{subset_name}.indices.npy",
                                               "teachers": {}})
    new_cfg = {**teacher_cfg, "top_k": k, "file": out_name}
    _check_no_collision(entry["teachers"].get(teacher_name), new_cfg, f"{out_dir}/manifest.json[{subset_name}][{teacher_name}]")
    entry["teachers"][teacher_name] = new_cfg
    _save_manifest(out_dir, manifest)

    total_tokens = int(offsets[-1])
    measured_bytes_per_token = (indices.nbytes + values.nbytes + residual.nbytes) / total_tokens if total_tokens else float("nan")
    print(f"Wrote {os.path.join(out_dir, out_name)}: {total_tokens} tokens across {len(shards) and offsets.shape[0] - 1} examples "
          f"({measured_bytes_per_token:.1f} bytes/token at K={k})", flush=True)


def finalize_hidden(dataset_root, split, subset_name, teacher_name, hidden_dir_shards,
                     hidden_layer_indices, n_examples, teacher_cfg, topk_dir_shards):
    """Per-layer files, named by subset SIZE only (n<N>) -- the layer_<L>/
    folder already gives the layer, manifest gives the canonical subset
    name. Same OOM-avoidance as the old merge_shards(): the hidden array can
    be tens of GB, so it's written directly into a disk-backed memmap sized
    up front, shard by shard, never held fully in RAM.

    Per-DOCUMENT offsets (not per-shard) are required so slice_span can find
    a given doc_id's token span -- hidden shards don't carry their own
    per-doc boundaries (only topk shards do, via the "offsets" key written
    by process_file's flush()), so topk_dir_shards is read here purely for
    those boundaries, matched to each hidden shard by its identical
    shard_<start>_<end>.npz filename."""
    if not hidden_layer_indices:
        return
    shards = sorted(
        (int(m.group(1)), p)
        for p in glob.glob(os.path.join(hidden_dir_shards, "shard_*.npz"))
        for m in [SHARD_RE.search(p)] if m
    )
    short_name = f"n{n_examples}"

    global_doc_offsets = [0]
    running = 0
    for _, path in shards:
        shard_name = os.path.basename(path)
        topk_shard = np.load(os.path.join(topk_dir_shards, shard_name))
        shard_doc_offsets = topk_shard["offsets"]  # per-doc boundaries WITHIN this shard
        global_doc_offsets.extend((shard_doc_offsets[1:] + running).tolist())
        running += int(shard_doc_offsets[-1])
    global_doc_offsets = np.array(global_doc_offsets, dtype=np.int64)
    assert len(global_doc_offsets) - 1 == n_examples, (
        f"topk shard offsets cover {len(global_doc_offsets) - 1} docs, expected n_examples={n_examples}"
    )

    shapes = {}
    for layer in hidden_layer_indices:
        total = 0
        dim = dtype = None
        for _, path in shards:
            hd = np.load(path)
            arr = hd[f"hidden_{layer}"]
            total += arr.shape[0]
            dim, dtype = arr.shape[1], arr.dtype
        shapes[layer] = (total, dim, dtype)

    offsets = global_doc_offsets
    for layer, (total, dim, dtype) in shapes.items():
        layer_dir = os.path.join(dataset_root, "embedding", split, f"layer_{layer}")
        os.makedirs(layer_dir, exist_ok=True)
        tmp_path = os.path.join(layer_dir, f"{short_name}.{teacher_name}.tmp.npy")
        mmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=(total, dim))
        cursor = 0
        for _, path in shards:
            hd = np.load(path)
            arr = hd[f"hidden_{layer}"]
            n = arr.shape[0]
            mmap[cursor:cursor + n] = arr
            cursor += n
        assert cursor == int(offsets[-1]), f"layer {layer}: hidden token count {cursor} != topk token count {int(offsets[-1])}"
        mmap.flush()
        final_path = os.path.join(layer_dir, f"{short_name}.{teacher_name}.npy")
        os.replace(tmp_path, final_path)
        if offsets is None:
            offsets = np.array(local_offsets, dtype=np.int64)
        np.save(os.path.join(layer_dir, f"{short_name}.{teacher_name}.offsets.npy"), offsets)

        manifest = _load_manifest(layer_dir)
        entry = manifest.setdefault(short_name, {"subset": subset_name, "teachers": {}})
        new_cfg = {**teacher_cfg, "file": f"{short_name}.{teacher_name}.npy"}
        _check_no_collision(entry["teachers"].get(teacher_name), new_cfg,
                             f"{layer_dir}/manifest.json[{short_name}][{teacher_name}]")
        entry["teachers"][teacher_name] = new_cfg
        _save_manifest(layer_dir, manifest)
        print(f"Hidden layer {layer}: shape=({total},{dim}) -> {final_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", required=True, help="e.g. data/distill/openr1_math_full -- must contain <split>.jsonl")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--subset_name", required=True, help="e.g. allLayers_n4, layerSubset_n18, lastLayer_n270, topk_n38901")
    parser.add_argument("--n_examples", type=int, default=None,
                         help="required only the FIRST time --subset_name is used (creates subsets/<split>/<subset_name>.indices.npy); "
                              "ignored (with a warning) if the subset already exists")
    parser.add_argument("--seed", type=int, default=0, help="only used when creating a new subset")
    parser.add_argument("--teacher_name", required=True, help="short id for this Teacher, e.g. qwen_big, lfm2_1_2b -- namespaces output files/manifest entries")
    parser.add_argument("--model_dir", required=True, help="local Teacher snapshot dir, or a Hub repo id")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
        help="\"auto\" (default) keeps the checkpoint's own stored dtype -- required for the FP8 checkpoint: "
             "forcing bfloat16 dequantizes the whole model before it touches VRAM (~55.6 GB vs. ~30.9 GB "
             "native FP8), which silently triggers CPU offload and a ~50-100x throughput collapse on GPUs "
             "with less than ~56 GB VRAM. 2026-09-22: the FP8 checkpoint has a confirmed, still-unfixed "
             "gate_proj dequant bug (garbage output, no error) -- use --dtype bfloat16 with the bf16 "
             "checkpoint until that's fixed upstream, see dev_notes/grid5000_usage.log.md.",
    )
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument(
        "--hidden_layers", default="none",
        help="'none' (default), 'last', 'all', or a comma-separated list of hidden_states "
             "indices (0=embedding output, num_layers=final layer) or the literal 'last' mixed "
             "in, e.g. 'last,0,8' -- include 0 explicitly for a positional-encoding-free "
             "embedding on a RoPE Teacher (see parse_hidden_layers docstring). Only use on small "
             "subsets -- one full hidden_dim vector per token per layer is much heavier than Top-K.",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--shard_size", type=int, default=1000,
        help="flush a persistent shard to disk every N examples (async, non-blocking) -- also the "
             "resume granularity if this run gets interrupted and relaunched with the same "
             "--dataset_root/--split/--subset_name/--teacher_name",
    )
    parser.add_argument("--num_gpus", type=int, default=None,
                         help="limit to the first N visible GPUs for device_map=\"auto\" sharding (default: every visible GPU)")
    parser.add_argument("--attn_implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--quantization", default="none", choices=["none", "bnb-4bit", "bnb-8bit"])
    args = parser.parse_args()

    split_path = os.path.join(args.dataset_root, f"{args.split}.jsonl")
    with open(split_path) as f:
        all_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(all_examples)} examples (full pool) from {split_path}", flush=True)

    indices = load_or_create_subset(args.dataset_root, args.split, args.subset_name,
                                     args.n_examples, args.seed, len(all_examples))
    examples = [all_examples[i] for i in indices.tolist()]
    n_examples = len(examples)
    print(f"Subset {args.subset_name!r}: {n_examples} examples", flush=True)

    work_dir = os.path.join(args.dataset_root, ".precompute_tmp", f"{args.subset_name}.{args.teacher_name}")
    topk_dir = shard_dir(work_dir, "topk")
    hidden_dir = shard_dir(work_dir, "hidden")
    resume_at = find_resume_point(topk_dir)
    if resume_at:
        print(f"Resuming from example {resume_at}/{n_examples} (found complete shards up to there)", flush=True)

    if args.dtype == "auto":
        if args.quantization != "none":
            raise ValueError("--dtype auto is only valid with --quantization none (bnb needs a concrete compute dtype, e.g. --dtype bfloat16)")
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)

    hidden_layer_indices = []
    if resume_at < n_examples:
        print("Detected GPU(s):", flush=True)
        describe_gpus()
        print(f"Loading Teacher {args.model_dir} in {args.dtype} ...", flush=True)
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

        process_file(
            model, tokenizer, examples, args.top_k, args.max_length, hidden_layer_indices,
            topk_dir, hidden_dir, args.shard_size, resume_at,
        )
    else:
        print("Nothing left to process -- reading hidden-layer indices off existing shards for finalize.", flush=True)
        any_hidden_shard = next(iter(glob.glob(os.path.join(hidden_dir, "shard_*.npz"))), None)
        if any_hidden_shard:
            d = np.load(any_hidden_shard)
            hidden_layer_indices = sorted(int(k.split("_", 1)[1]) for k in d.files if k.startswith("hidden_"))

    teacher_cfg = {
        "dtype": args.dtype, "checkpoint": args.model_dir, "max_length": args.max_length,
        "attn_implementation": args.attn_implementation, "quantization": args.quantization,
    }
    finalize_topk(args.dataset_root, args.split, args.subset_name, args.teacher_name,
                  topk_dir, args.top_k, teacher_cfg)
    finalize_hidden(args.dataset_root, args.split, args.subset_name, args.teacher_name,
                     hidden_dir, hidden_layer_indices, n_examples, teacher_cfg, topk_dir)

    shutil.rmtree(work_dir)
    print(f"Done -- cleaned up {work_dir}", flush=True)


if __name__ == "__main__":
    main()
