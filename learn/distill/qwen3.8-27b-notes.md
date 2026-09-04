# Qwen3.8-27B — Teacher-specific notes

Notes specific to the Teacher model chosen for distillation
(`Qwen/Qwen3.8-27B-FP8`), kept separate from the general pipeline
`README.md` (`prepare_*`, `train_sft.py`, `precompute_teacher_targets.py`).
Referenced from the README.

## Model identity

**`Qwen/Qwen3.8-27B-FP8`** — the **official Qwen** FP8-quantized checkpoint
(not a third-party GGUF quant) of a Qwen3.5 vision-language model
(`image-text-to-text`, `Qwen3_5ForConditionalGeneration`), used here purely
as a text Teacher (image/video inputs are never passed).

- **~30.9 GB** total (FP8). The unquantized bf16 checkpoint
  (`Qwen/Qwen3.8-27B`) is ~55.6 GB for the same ~27-28B params.
- Needs a GPU with **≥ ~31 GB free VRAM**. Native FP8 tensor-core compute
  (the actual speed benefit, not just the VRAM saving) requires Hopper/Ada
  (H100, H200, L40S — `abacus27`, `hydra`, `abacus26` on Grid'5000); Ampere
  (A100 and older) can still load and run the checkpoint but dequantizes to
  bf16/fp16 for compute.

**Real bottleneck found on Ampere (verified 2026-09-04, `abacus21`/A100 40GB)**:
`bench_teacher.py`'s current `load_model_and_tokenizer` loads with `dtype=bfloat16`
regardless of the checkpoint's native FP8 format, which **dequantizes the
whole model to ~55.6 GB bf16 before it even touches VRAM** — too big for a
40 GB A100. `transformers` silently offloads the overflow to CPU ("Some
parameters are on the meta device because they were offloaded to the cpu"),
and that CPU↔GPU traffic per token is the actual cause of the ~0.3-0.6 tok/s
throughput seen — **not** missing `causal-conv1d`/`flash-linear-attention`
kernels (installing both changed nothing: same throughput, just without the
"falling back to reference implementation" warnings). On Ampere, loading the
FP8 checkpoint *without* forcing bf16 (so it dequantizes only as needed per
layer, or stays FP8-stored with on-the-fly upcast) should avoid the CPU
offload entirely — worth fixing in `bench_teacher.py` before trusting any
Ampere throughput number. On Hopper (94-96 GB), this isn't an issue: the
55.6 GB bf16 model fits with headroom to spare.
- Released 2026-08-14.
- Within the Qwen3.8 line, 27B is actually the *smallest* dense-ish option —
  the alternatives are `Qwen3.8-Flash-Next` (360 GB) and
  `Qwen3.8-2.4T-A95B` (4.9 TB). The separate Qwen3.5 generation has real
  small dense sizes (0.8B/2B/4B/9B) if a lighter Teacher is ever needed for
  faster iteration.

## Architecture: hybrid attention (relevant to hidden-state extraction)

The model has **64 layers** total, of which only **16 are regular
attention** (the kind every transformer has) and **48 are gated DeltaNet**,
a form of linear attention that carries a running internal state (a
numeric memory of everything read in the prompt so far).

Practical consequence for `precompute_teacher_targets.py`: if hidden-state
extraction targets specific layers (`--hidden_layers <indices>`), keep in
mind that 3/4 of the layers are not "standard" attention but stateful
linear-attention — their semantics differ from a regular attention layer's.

This architecture makes a naive quantization strategy dangerous in a way
that doesn't announce itself: compressing the DeltaNet internal state too
aggressively breaks nothing visible (the model loads, generates tokens,
normal throughput) but progressively degrades coherence on long answers and
context tracking — no error log.

## Official FP8 vs third-party GGUF quants — why the distinction matters

The GGUF quant notes below (Unsloth, Bartowski, Ridge) come from an
independent analysis of the llama.cpp/GGUF ecosystem, **not** the HF
Transformers ecosystem used by this pipeline (`download_teacher.py`,
`bench_teacher.py`). The `Qwen/Qwen3.8-27B-FP8` checkpoint used here is an
**official vendor** quant, presumably better calibrated than an unprotected
third-party quant — but the general lesson still transfers:

> A broken quantized model looks exactly like a working one from the
> outside (tokens coming out, GPU happy, dashboard green). Nothing flags it
> automatically.

**Concrete recommendation**: before running `precompute_teacher_targets.py`
at scale, run `bench_teacher.py` in FP8 **and** in bf16 (`--dtype
bfloat16`, needs ~55 GB — only feasible on a few short prompts if the GPU
allows it) and compare the resulting Top-K logits — a divergence sanity
check, not a full audit.

## GGUF quant reference (llama.cpp ecosystem — for reference, outside the HF/FP8 scope)

A good quant of this model protects the output layer, attention value
projections, and down-projections at 5 bits or better, and lets the
embeddings and DeltaNet layers absorb more compression (not the other way
around) — same file size, more model left standing. Different teams get
there differently: **Ridge** (11.7 GB, refuses to compress the DeltaNet
state at all, pushes compression onto the middle feed-forward layers
instead), **Unsloth** ("dynamic"), **Bartowski** (importance matrix
calibrated on a 63% tool-calling corpus).

Independent benchmarks cited (Quesma, ~$3000 of rented GPU time, on GPQA
Diamond, IFBench, TerminalBench — 89 real terminal tasks in a container):
4-bit matches the full bf16 model on TerminalBench, instruction-following
doesn't budge down to 2-bit, but 1-bit collapses (score around random,
sometimes below). VRAM reference points (GGUF files, not FP8): 24 GB →
Unsloth UD-Q4_K_XL (17.92 GB); 32 GB → Q6/Q8; 16 GB → Bartowski IQ4_XS;
≤ 12 GB → use a smaller model rather than an aggressive quant of this one.
**Always pin the downloaded revision**: Unsloth's files were replaced on
2026-08-19, making benchmarks published against the earlier version stale.

KV-cache-specific pitfall (llama.cpp): compressing the KV cache to Q4
(instead of Q8/full precision) dropped output similarity to 8.3% against
the full-precision model (vs. 81.6% at Q8), on a different model (7B
coder) — not reproduced on Qwen3.8-27B but same flag, so avoid by default
if this pipeline ever moves to llama.cpp/vLLM to serve this Teacher.

## Default reasoning effort ("xhigh") — relevant to `bench_teacher.py`

This model thinks at **xhigh by default** (extended reasoning before
answering). Examples cited (video): 22,276 "thinking" tokens for a single
SVG-generation prompt; another job going from 21 min to 137 s just by
turning reasoning off.

**Real API (confirmed by the official Unsloth docs,
unsloth.ai/docs/models/qwen3.8)**: this is **not** a plain
`enable_thinking=True/False` boolean (that parameter exists for the
smaller dense Qwen3 models, not this one). Qwen3.8-27B exposes a
**`reasoning_effort`** with four levels, passed as a chat-template kwarg:
- `xhigh` (default) — deep reasoning
- `medium` — balances accuracy/speed
- `low` — optimized for speed
- `none` — no reasoning

Related setting: **"Preserve Thinking"**, which keeps the reasoning trace
from previous turns in a multi-turn conversation (uses more tokens,
potentially improves accuracy) — tied to the chat-template quirk mentioned
under "Out of scope" below (nested thinking blocks in multi-turn).

Implemented in `bench_teacher.py` via `--reasoning_effort
{model_default,xhigh,medium,low,none}`, forwarded to
`apply_chat_template(..., reasoning_effort=...)` (silent fallback if the
tokenizer doesn't accept it, e.g. the local gpt2 smoke-test). With
`--max_new_tokens 64` (current default), the default run
(`--reasoning_effort model_default`, i.e. xhigh) is almost certainly cut
off mid-reasoning — a warning is printed in that case. Use
`--reasoning_effort none` to measure post-thinking throughput.

**Correction (verified on real hardware, 2026-09-04, `abacus21`/A100)**: the
Unsloth docs' 4-level claim above is **not what this checkpoint's actual chat
template accepts**. `--reasoning_effort none` raises a hard
`jinja2.exceptions.TemplateError: Unexpected reasoning effort none. Supported
types are xhigh (default), medium, and low.` — it is **not** a silent
fallback, it's a crash. Only `xhigh`/`medium`/`low` are valid on this
snapshot. Use `--reasoning_effort low` as the closest available proxy for
"fastest, least reasoning" rather than `none`.

`precompute_teacher_targets.py` does a plain forward pass (no
`generate()`, no chat template), so it is **not** affected by this
reasoning mode.

### Recommended generation parameters (Unsloth docs)

Not yet applied in `bench_teacher.py` (which deliberately uses
`do_sample=False` for a reproducible, comparable throughput measurement) —
use these if this Teacher is ever used to actually generate text (data
augmentation, target generation) rather than just being benchmarked:

| | Temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking mode (`reasoning_effort` ≠ `none`) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct mode (`reasoning_effort=none`) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

Native max context: 262,144 tokens (extendable to 1M via YaRN).

### Quantization alternative: NVFP4

Besides the FP8 already used here, Unsloth also offers an **NVFP4** quant
(`unsloth/Qwen3.8-27B-NVFP4`), advertised as ~1.5x faster than bf16 with
92-97% top-1 accuracy — served via `vllm serve unsloth/Qwen3.8-27B-NVFP4`.
Not tested here, but worth keeping in mind as an alternative if FP8 turns
out to be a throughput bottleneck rather than a VRAM one.

### Quantization support added to the scripts

`bench_teacher.py` and `precompute_teacher_targets.py` now accept
`--quantization {none,bnb-4bit,bnb-8bit}` (via `bitsandbytes`
`BitsAndBytesConfig`, `nf4` quant type, compute dtype = `--dtype`), so an
Unsloth bf16 repo (e.g. `unsloth/Qwen3.8-27B`) or the vendor bf16 repo can
be loaded on smaller GPUs without a separate GGUF/llama.cpp stack. Needs
`pip install bitsandbytes`. This is separate from the vendor FP8 checkpoint
(`Qwen/Qwen3.8-27B-FP8`) already used as the default Teacher — use it only
to experiment with a different VRAM/quality trade-off, not as the default
path.

### VRAM reference by quantization level (Unsloth docs, GGUF ecosystem)

| Bits | VRAM required |
|---|---|
| 1-bit | 7-8 GB |
| 2-bit | 9-11 GB |
| 3-bit | 12-14 GB |
| 4-bit | 16-19 GB |
| 6-bit | 23-26 GB |
| 8-bit | 31 GB |
| BF16 | 56 GB |

Consistent with the ~30.9 GB measured for the official FP8 build (between
the 6-bit and 8-bit GGUF reference points above) and the ~55.6 GB of the
unquantized bf16 checkpoint ("Model identity" section).

## Out of scope for this pipeline (mentioned in the source video, no action needed)

- Built-in speculative decoding (draft MTP head): llama.cpp-specific, no
  effect on Apple Silicon, not applicable to `transformers`.
- Vision projector (separate file for image capabilities): not needed,
  this pipeline uses the model as text-only
  (`download_teacher.py`/`bench_teacher.py` already handle the text
  fallback via `AutoModelForCausalLM`).

## Reference numbers (video) — for comparison against our own runs

Raw figures cited in the video, kept as-is with their exact
hardware/format context. **Caveat**: most come from the GGUF/llama.cpp
ecosystem (not HF Transformers/FP8 like our pipeline) and from consumer
GPUs (3090/4090, Mac) rather than Grid'5000 (A100/H100/L40S) — not a valid
direct comparison in absolute terms, but useful as orders of magnitude and
ratios to spot a broken config (e.g. a `tokens_per_sec` near zero with
reasoning disabled, or a near-zero speedup from `--reasoning_effort none`
when the video shows a ~9x factor).

| Measurement | Value | Exact context |
|---|---|---|
| Full bf16 size | 55 GB | Unquantized `Qwen/Qwen3.8-27B` checkpoint (consistent with the ~55.6 GB measured via the Hub API, "Model identity" section) |
| Ridge quant | 11.7 GB | GGUF, refuses to compress the DeltaNet state |
| KL divergence, top-token agreement, 4-bit | 96% | Atomic Chat's (quant maker) internal benchmark, GGUF |
| KL divergence, top-token agreement, 8-bit | 98.9% | Same, +~2.5 GB over 4-bit for this marginal gain |
| TerminalBench (89 real terminal tasks) | 4-bit = full bf16 | Independent Quesma benchmark (~$3000 of rented GPU), GGUF, KV cache at full precision |
| Instruction-following (IFBench) | stable down to 2-bit | Same Quesma run |
| Reasoning benchmark (GPQA Diamond), 1-bit | ~random, sometimes worse | Same Quesma run; the smaller 1-bit build scored below random |
| Output similarity vs. full precision, KV cache at Q8 | 81.6% | Different model (7B coder, not Qwen3.8-27B), greedy decoding, fixed prompts, GGUF |
| Output similarity vs. full precision, KV cache at Q4 | 8.3% | Same setup — drastic drop for ~112 MB saved at short context: not worth it |
| "Thinking" tokens (xhigh, single SVG prompt) | 22,276 tokens | Real generation, default reasoning mode |
| Total time, same task, reasoning off | 21 min → 137 s | ~9x ratio — a reference point to check that `--reasoning_effort none` has an effect of the right order of magnitude on this pipeline |
| Throughput with speculative decoding (built-in draft MTP) | 31-41 tok/s | RTX 3090 (24 GB), GGUF UD-Q4_K_XL, self-reported (not a controlled benchmark) |
| Throughput with the same flag on Apple Silicon | 5.8 → 5.8 tok/s (no gain) | Mac M4, llama.cpp spec decoding unsupported on this platform -- not applicable to Grid'5000 |

Use these as sanity-check reference points for `bench_teacher.py`'s output
(`tokens_per_sec` in the JSON) rather than as targets to hit: our GPUs
(A100/H100/L40S) and our stack (HF Transformers, official FP8, no
speculative decoding or GGUF quant) give a completely different order of
magnitude in absolute terms, but an abnormally low `tokens_per_sec`, or a
`--reasoning_effort none` speedup factor very different from ~9x on a
comparable task (a short-answer prompt like SVG generation), are good
signs of a configuration error (attn_implementation silently falling back
to `eager`, `--num_gpus` misconfigured, a dtype forcing an unwanted
dequantize, etc.) rather than a real hardware limit.

## GPU site choice (checked 2026-09-03, availability changes constantly -- re-check before reserving)

`Qwen3.8-27B-FP8` (~31 GB) just needs a GPU that can hold it; native FP8
tensor-core speed additionally needs Hopper/Ada (H100/H200/L40S).

| Cluster (site) | GPU | Native FP8 | Status when checked |
|---|---|---|---|
| `hydra` (Lyon) | H200, 96 GB | ✅ | **free** (2/4 nodes) |
| `sirius` (Lyon) | A100×8, 40 GB | loadable only | **free** |
| `grouille` (Nancy) | A100×2, 40 GB | loadable only | **free** (1/2 nodes) |
| `abacus27` (Rennes) | H100×4, 94 GB | ✅ | `busy_besteffort` -- only besteffort jobs on it, a normal reservation preempts them, so effectively available |
| `abacus21` (Rennes) | A100, 40 GB | loadable only | `busy_besteffort`, same as above |
| `chuc` (Lille, 8 nodes) | A100×4, 40 GB | loadable only | mixed: 3/8 nodes `busy_besteffort` (available), rest genuinely busy |
| `abacus26` (Rennes) | L40S, 45 GB | ✅ | genuinely busy (real job running) |

Trade-off: staying at **Rennes** (`abacus27`) avoids transferring the ~31 GB
snapshot across sites after downloading it on `paradoxe-7` (also Rennes).
Going to **Lyon** (`hydra`) gets a currently-idle H200 but requires an
inter-site transfer of the downloaded snapshot (`Group_Storage`/`rsync`),
which the `grid5000` skill flags as a potential bottleneck for large volumes.
For a one-off ~31 GB transfer this is likely fine either way; prefer Rennes
if in doubt, to keep everything on one site's storage server.

Availability check command (run from any site's frontend, no netrc needed):
```bash
curl -s https://api.grid5000.fr/stable/sites/rennes/status.json \
  | jq -r '.nodes | to_entries[] | select(.key | test("^abacus(27|26|21)")) | "\(.key)\t\(.value.soft)"'
```
See `.claude/skills/grid5000/references/doc-map.md` for the multi-site
version and the `busy_besteffort` vs genuinely-busy distinction.

## See also

[`qwen3.8-benchmark-plan.md`](./qwen3.8-benchmark-plan.md) — the concrete experiment matrix
and GPU-specific instructions to actually validate this Teacher's setup on
Grid'5000, built from the findings above.

## Sources

- YouTube video "I Tested Every Qwen3.8-27B Quant: Here's the Best One For
  Your GPU" (auto-generated transcript fetched via `yt-dlp`, 2026-09-03).
- Official Unsloth documentation, unsloth.ai/docs/models/qwen3.8 (fetched
  2026-09-03) — source of the real `reasoning_effort` API (corrects the
  initial `enable_thinking` boolean assumption), the recommended generation
  parameters, the per-bit VRAM table, and the NVFP4 quant.
- Cross-checked against the Hub/GPU facts already present in this repo.
