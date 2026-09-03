# Teacher benchmark plan (Qwen3.8-27B-FP8 on Grid'5000)

Concrete experiment matrix to validate `bench_teacher.py`'s configuration
knobs (`--num_gpus`, `--attn_implementation`, `--reasoning_effort`,
`--quantization`) against real hardware, using the reference numbers and
GPU-adaptive code added per [`qwen3.8-27b-notes.md`](./qwen3.8-27b-notes.md).
Run manually on Grid'5000 -- nothing here executes automatically.

**GPU availability changes constantly** -- the site/GPU table below reflects
the check from 2026-09-03 (see `qwen3.8-27b-notes.md#gpu-site-choice`);
re-check with the availability command there before reserving anything.

## Goals

1. Confirm the Teacher actually loads and runs correctly on each GPU
   generation present on Grid'5000 (Hopper, Ada, Ampere).
2. Catch configuration errors early by comparing our measured numbers
   against the video's reference points (order of magnitude and ratios,
   not absolute values -- see `qwen3.8-27b-notes.md`'s reference table).
3. Decide, per GPU generation, the config to actually use for the full
   `precompute_teacher_targets.py` run (FP8 vs. bnb-4bit, num_gpus, attn
   implementation).

## Phase 0 -- environment sanity (any single GPU, cheapest reservation)

Before spending real GPU time on the FP8 checkpoint, confirm the stack
loads at all and `describe_gpus()` matches the reserved hardware.

```bash
oarsub -I -n "distill-bench-p0" -l gpu=1,walltime=0:30:00 -q default
source ~/venvs/<env>/bin/activate  # transformers + accelerate installed
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --max_new_tokens 8 --reasoning_effort none --num_gpus 1 \
  --out_file /tmp/p0_sanity.json
```

Check:
- `describe_gpus()` output matches the reserved node's GPU/VRAM/compute
  capability.
- `Loaded with attn_implementation=...` line -- confirm it's not silently
  `eager` (that would mean both `flash_attention_2` and `sdpa` failed to
  load, worth investigating before any throughput comparison).
- Measured Top-K bytes/token matches the `K*6+2` formula regardless of
  GPU -- if it doesn't, that's a code bug, not a hardware effect.

## Phase 1 -- single-GPU baseline per GPU generation

Run once per GPU generation actually available, 1 GPU each, same command:

```bash
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --attn_implementation auto --reasoning_effort none \
  --max_new_tokens 64 --out_file /tmp/p1_<site>.json
```

| Generation | Example site (2026-09-03) | Native FP8 | What to check |
|---|---|---|---|
| Hopper | `hydra` (H200, 96 GB), `abacus27` (H100×4, 94 GB) | yes | this should be the fastest tok/s of the set |
| Ada | `abacus26` (L40S, 45 GB) | yes | fastest per-GPU-dollar; tighter headroom than Hopper (31 GB used of 45 GB) |
| Ampere | `sirius` (A100×8, 40 GB), `grouille` (A100×2, 40 GB), `abacus21` (A100, 40 GB), `chuc` (A100×4, 40 GB) | no (dequantizes) | tok/s should be noticeably lower than Hopper/Ada at the same precision -- if it's *not* lower, `describe_gpus()`'s native-FP8 detection or the dequantize path may be wrong |

Record for each: load time, `tokens_per_sec`, VRAM headroom (from
`describe_gpus()`'s total minus what `nvidia-smi` shows in use during the
run -- not printed by the script itself, check manually with `nvidia-smi`
in a second shell on the same job).

## Phase 2 -- reasoning_effort sanity check (fastest available GPU only)

Expensive in wall-clock (xhigh reasoning can run long) -- run once, on
whichever Hopper node is free, not on every GPU generation.

```bash
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --reasoning_effort xhigh --max_new_tokens 512 \
  --out_file /tmp/p2_xhigh.json

python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --reasoning_effort none --max_new_tokens 512 \
  --out_file /tmp/p2_none.json
```

Compare total wall-clock time for the same prompts. The video's reference
point is a **~9x** speedup (21 min → 137 s) on a similarly short-answer
task (SVG generation) -- expect a broadly similar order of magnitude on a
comparable prompt (the sample prompts here are short-answer math/trivia,
not open-ended generation, so the ratio may differ, but a ratio close to 1x
would mean `--reasoning_effort` isn't actually taking effect -- check the
"reasoning_effort=xhigh failed, falling back" case isn't silently happening
via the `TypeError` fallback in `apply_chat_template_with_reasoning_effort`).

## Phase 3 -- multi-GPU scaling (multi-GPU sites only)

The FP8 checkpoint (~31 GB) fits on a **single** GPU with ≥40 GB VRAM, so
`device_map="auto"` sharding across multiple GPUs adds cross-GPU
communication without a VRAM need -- expect single-GPU to be at least as
fast per-token as multi-GPU for this checkpoint. This phase exists to
*confirm* that, and to size headroom for a future bf16-vs-FP8 comparison
(Phase 5) or heavier hidden-state extraction runs.

Use a multi-GPU site: `abacus27` (H100×4), `sirius` (A100×8), `chuc`
(A100×4), or `grouille` (A100×2).

```bash
for n in 1 2 4; do
  python learn/distill/bench_teacher.py \
    --model_dir /path/to/Qwen3.8-27B-FP8 \
    --num_gpus $n --reasoning_effort none --max_new_tokens 64 \
    --out_file /tmp/p3_ngpu${n}.json
done
```

Expected: `tokens_per_sec` roughly flat or slightly *down* as `--num_gpus`
increases (communication overhead, no compute parallelism benefit for a
single-request generation). If throughput instead scales *up* noticeably
with `--num_gpus`, `max_memory` may not be constraining sharding the way
intended -- double check `build_max_memory()`'s output for that node
(GPUs beyond N should show `0GiB`).

## Phase 4 -- attn_implementation comparison (optional, needs flash-attn)

`flash-attn` needs compilation from source unless a prebuilt wheel matches
the exact CUDA/torch/Python version on the node -- check wheel availability
before attempting a source build (can take 20-40 min and burn walltime for
no result if it fails). Skip this phase if no prebuilt wheel is available
for the reserved node's environment; `sdpa` (the automatic fallback) is a
reasonable default regardless.

```bash
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --attn_implementation flash_attention_2 --reasoning_effort none \
  --out_file /tmp/p4_flash.json

python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --attn_implementation sdpa --reasoning_effort none \
  --out_file /tmp/p4_sdpa.json
```

Only 16/64 layers are regular attention (the rest is gated DeltaNet, which
flash-attn doesn't accelerate) -- expect a smaller flash-attn speedup here
than on a fully-standard-attention model of similar size.

## Phase 5 -- bnb-4bit / bnb-8bit vs. FP8 (Ampere site priority)

Since Ampere can't use FP8 tensor cores natively (it dequantizes), bnb-4bit
might be competitive or better there specifically. Needs `pip install
bitsandbytes` and a bf16 repo (`Qwen/Qwen3.8-27B` or an Unsloth bf16
reupload) rather than the FP8 checkpoint.

Run on an Ampere site (`sirius`, `grouille`, `abacus21`, or `chuc`):

```bash
python learn/distill/bench_teacher.py \
  --model_dir Qwen/Qwen3.8-27B --dtype bfloat16 \
  --num_gpus 1 --quantization bnb-4bit --reasoning_effort none \
  --out_file /tmp/p5_bnb4bit_ampere.json

python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --reasoning_effort none \
  --out_file /tmp/p5_fp8_ampere.json
```

Compare `tokens_per_sec` and load time. If bnb-4bit throughput is
comparable to or better than FP8 on Ampere, prefer bnb-4bit there for the
full precompute run (frees VRAM headroom too: 4-bit of a 27B model is
roughly half the FP8 checkpoint's footprint). On Hopper/Ada, stick with FP8
-- native tensor cores should keep it ahead.

## Phase 6 -- FP8 vs. bf16 divergence check (H200 preferred: single-GPU 96 GB fits both)

The general "broken quant looks like a working one" lesson from
`qwen3.8-27b-notes.md` applies to the FP8 checkpoint too, even though it's
vendor-official. `hydra` (H200, 96 GB) is the only single-GPU option large
enough to hold the bf16 checkpoint (~55.6 GB) alongside comfortable
headroom, avoiding a multi-GPU confound in this specific comparison.

```bash
# FP8 (default)
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --num_gpus 1 --reasoning_effort none --top_k 32 \
  --out_file /tmp/p6_fp8.json

# bf16, same sample prompts
python learn/distill/bench_teacher.py \
  --model_dir Qwen/Qwen3.8-27B --dtype bfloat16 \
  --num_gpus 1 --reasoning_effort none --top_k 32 \
  --out_file /tmp/p6_bf16.json
```

`bench_teacher.py` doesn't currently diff the two `Top-K` outputs itself --
this phase just collects both; comparing the recorded indices/values (not
serialized to the JSON summary, only aggregate bytes/token are) would need
a small ad hoc script reading the two runs' logits if a full divergence
check is wanted. Treat this phase as optional confirmation, not a blocker
for starting the full precompute run with FP8.

## Summary: recommended default per GPU generation

| GPU generation | Recommended Teacher config for the full precompute run |
|---|---|
| Hopper (H100, H200) | FP8 checkpoint, `--num_gpus 1`, `--attn_implementation auto` |
| Ada (L40S) | FP8 checkpoint, `--num_gpus 1` (tighter headroom -- watch VRAM) |
| Ampere (A100) | FP8 by default; switch to `--quantization bnb-4bit` with the bf16 repo if Phase 5 shows it's faster or the extra headroom is needed for `--hidden_layers` extraction |

This table is a starting hypothesis to confirm with Phases 1 and 5, not a
conclusion -- update it once real numbers come back.
