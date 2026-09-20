# Distillation — onboarding, MFU/batch-size sweep, Teacher-target precompute

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-03/04 — Distillation onboarding: data prep, Teacher benchmark, first SFT + KD runs

Full run on Grid'5000 (cluster/job details live in `dev_notes/grid5000_usage.log.md`, not here).

### Data prep (reasoning / general / retrieval)

Ran all three `prepare_*_data.py` scripts against the real Hub datasets (previously only smoke-tested locally):
- `reasoning` (OpenR1-Math-220k): 38,057/93,733 kept (55,676 skipped — no verified `<think>` trace, or too long at `max_length=4096`), ~124 ex/s.
- `general` (WikiText-103 + TinyStories): 40,000/40,000 kept, 0 skipped, ~14s — TinyStories especially is very forgiving on length.
- `retrieval` (HotpotQA distractor): 90,446/90,447 kept, essentially no loss.

Added unfiltered `raw.jsonl` capture to all three scripts (writes every streamed example with a `kept: true/false` flag, not just the ones that pass filtering) — previously, skipped examples were discarded with no trace, which matters most for `reasoning` given its high skip rate. Re-ran `reasoning` with this enabled: same filtering outcome, `raw.jsonl` now available (93,733 rows) for future filter-criteria tuning.

### Teacher: Qwen3.8-27B-FP8

Chosen Teacher: `Qwen/Qwen3.8-27B-FP8`, a Qwen3.5 vision-language checkpoint (`Qwen3_5ForConditionalGeneration`) used purely as a text model. 30.89 GB on disk. Architecture detail that matters for anything touching hidden states: 64 layers total, only 16 "normal" attention, the other 48 are gated DeltaNet (linear attention with recurrent state) — see `learn/distill/qwen3.8-27b-notes.md` for the full writeup (quant landscape, reasoning_effort API, GPU-generation notes).

Bugs hit and fixed getting `bench_teacher.py`/`precompute_teacher_targets.py` to actually run this checkpoint:
- `--reasoning_effort none` is documented (Unsloth docs) as a valid 4th level alongside xhigh/medium/low, but this checkpoint's real chat template **rejects it with a hard `jinja2.exceptions.TemplateError`**, not a silent fallback as our own code comment assumed. Only xhigh/medium/low actually work; use `low` as the fastest available proxy.
- `model.config.num_hidden_layers` doesn't exist on `Qwen3_5Config` (a VLM wrapper) — the real value lives at `model.config.text_config.num_hidden_layers`.
- Missing `pillow`/`torchvision` block `AutoProcessor.from_pretrained` even though we never pass image input — the VLM image-processor class is still instantiated as part of loading.

Throughput investigation (Ampere, A100 40GB): first suspected missing `causal_conv1d`/`flash-linear-attention` optimized kernels (the 48 DeltaNet layers fall back to slow reference PyTorch ops without them) — compiled both from source (needed installing a CUDA toolkit via conda-forge/nvidia channels first, since the node has no `nvcc` by default) and confirmed they load and the "falling back" warnings disappear. **Throughput didn't change** (still ~0.3-0.6 tok/s). Real cause: `bench_teacher.py` forces `dtype=bfloat16` on load regardless of the checkpoint's native FP8 format, dequantizing to ~55.6 GB — too big for a 40 GB A100, so `transformers` silently offloads part of the model to CPU, and that CPU↔GPU traffic per generated token is what actually dominates. Not yet fixed in the script (should load without forcing bf16, or run on a GPU with native FP8 support instead). Switching to an L40S (Ada, compute capability 8.9, the lowest tier with native FP8 tensor cores) sidesteps the issue entirely — loaded fast, no offload, no dequantization warning.

Top-K=32 storage formula (`K*6+2` bytes/token) has now been validated exactly against real measurements three separate times (194.0 bytes/token measured every time) across different runs/GPUs.

### First SFT run (10M student, no KD)

`train_sft.py` had a real bug: it never moved the model or batch tensors to a CUDA device, so it would have silently trained on CPU even on a GPU reservation. Fixed. First real run on GPU: 9.44M-param random-init GPT-2-arch student (gpt2 tokenizer), 200 steps on the real `reasoning` data in 3.9s on an A100, loss 10.85→5.11. Confirms the full pipeline (real data → tokenizer → from-scratch model → GPU training loop) works end-to-end. This is plain next-token cross-entropy, no Teacher involved yet.

### First KD run (logit-level distillation)

Implemented in `train_sft.py`: `--teacher_targets`/`--kd_alpha` CLI args and a `topk_kd_loss()` function computing KL(teacher‖student) over the (K+1)-way categorical formed by the Teacher's Top-K token indices plus one merged "everything else" bucket, reconstructed via the residual log-sum-exp exactly as stored by `precompute_teacher_targets.py`. This is the only distribution shape exactly recoverable from Top-K-only storage — treating the untracked tail of the vocabulary as one lumped outcome rather than ignoring it or assuming it's zero.

Also fixed a latent bug: the student's model config always used `--base_config`'s own vocab size, silently ignoring `--tokenizer` — meant nothing crashed if you used a mismatched tokenizer, it would just produce token ids out of the embedding table's range. Now sets `config.vocab_size = len(tokenizer)` unconditionally.

Validated the loss function before spending any GPU time on it:
1. Local CPU smoke-test: tiny synthetic Top-K `.npz`, 1.7M-param gpt2-arch model, ran without shape errors, both CE and KD terms decreasing over 10 steps.
2. Numeric correctness check: built "teacher" targets directly from a student's own logits (so they should be identical distributions) and measured `KL ≈ 0` (`-1.6e-7`, i.e. numerical noise) — confirms the KL math itself is correct, not just that the code runs.

Real run: precomputed Top-K=32 targets for 200 `reasoning` examples using the actual Teacher on the L40S (had to drop `--max_length` from the default 4096 to 512 to avoid an OOM — the transient `(seq_len, vocab≈152k)` logits/mask tensors on top of the model's own ~40 GB footprint don't fit in 44 GB otherwise). Then trained a student with `--teacher_targets` pointed at that precomputed file, `--tokenizer` pointed at the Teacher's own snapshot (needed so Top-K indices refer to the same vocabulary as the student) — this pushed the student to **41.09M params** instead of the earlier 10M, because the Teacher's vocabulary's embedding table dominates parameter count at this width; the original "10M" target was implicitly calibrated for gpt2's much smaller (50k) vocab. Correction (2026-09-04): the Teacher's own tokenizer (`Qwen3.8-27B-FP8`, a Qwen3.5 VLM) has **248,077** tokens, not the ~151,936 figure cited in `README.md` for plain Qwen3 — that figure is correct for the *data-prep* tokenizer (`Qwen/Qwen3-0.6B`, used by `prepare_*_data.py`), a genuinely different, smaller tokenizer than the Teacher's own. The two are not interchangeable; KD training must use the Teacher's own tokenizer for index alignment (as it does), while data-prep length filtering used the smaller one — a minor, currently-unquantified discrepancy in effective token counts between the two stages. 200 steps in 37.9s: CE 11.38→~1.5-1.8, KD/KL 0.41→~0.29, best combined loss 0.673. Both loss components move sensibly, first genuine end-to-end KD result. Full config/metrics/log: `logs/EXP-003-kd-topk32-baseline/` (adopting the `logs/EXP-XXX/` convention already used elsewhere in this repo — going forward, this narrative log points to an EXP-XXX per run instead of inlining metrics).

Open follow-ups: decide whether to keep the ~41M student size (Teacher-vocab-aligned) or find a way to shrink it back toward the original 10M budget (smaller `n_embd`, or a separate small vocab with an explicit index-remapping layer against the Teacher's Top-K ids); scale the precompute step past the 200-example/512-token validation slice to the full 38,057-example `reasoning` set (needs the OOM addressed for the full 4096-token budget, likely via a bigger-VRAM GPU or a leaner residual computation).

### muP (hyperparameter transfer across width)

Decision (2026-09-04): given the project's stated target of scaling from ~10-41M up toward 500M-3B, tuning LR/init fresh at every tier is wasteful and, under standard parametrization, systematically misleading (optimal LR shifts with width, so values tuned small don't transfer). Implemented the two components of muP (Yang et al., *Tensor Programs V*) with the largest effect on transfer per the original paper: (1) hidden matmul weights (attention/MLP projections) get init std scaled by `1/sqrt(width_mult)` and Adam LR scaled by `1/width_mult`, both relative to a `--mup_base_width` reference; (2) the LM head is untied from the input embedding (a muP requirement — tied weights can't satisfy both the embedding's and the readout's scaling rules at once), zero-initialized, and its output logits divided by `width_mult` before the loss. Embeddings, LayerNorm, and biases keep the framework's default init and the base (unscaled) LR, per the muP table.

**Explicitly not implemented** ("muP-lite", not full muP): attention's `1/sqrt(d_head)` logit scaling is not patched to muP's `1/d_head` convention — would require reaching into the specific attention implementation's internals in a version-fragile way. Revisit if empirical transfer doesn't hold well across the widths actually used.

Validation done: local CPU smoke-tests (`--mup` alone, and combined with `--teacher_targets`/KD) ran without shape errors, losses decreasing sensibly. Sanity check on the zero-init readout: initial loss ≈ 10.82 ≈ ln(vocab_size≈50257), exactly matching the theoretical value for uniform logits at init — confirms the readout scaling wiring is correct.

**Real GPU width-transfer test (2026-09-04, `abacus21`/A100, real `reasoning` data)**: 4 learning rates (0.003/0.01/0.03/0.1) × 3 configs (narrow+muP at width 40, wide+muP at width 160, wide-without-muP at width 160), plain CE, 60 steps each. Result: a real if modest confirming signal — at the most aggressive LR (0.1), the no-muP wide model's loss drifted clearly worse (6.035) than the narrow-muP baseline (5.809), while the wide-muP model tracked the baseline almost exactly (5.811); 3 of the 4 LRs favored muP tracking the baseline more closely, one (0.01) was a coin flip. Not a dramatic explosion/instability demo, but a directionally consistent, codebase-specific result — an upgrade from the earlier inconclusive CPU toy test.

**Tied-head compromise (2026-09-04)**: canonical muP requires untying the LM head from the input embedding (they need different init/scaling rules, impossible to satisfy both on one shared tensor — see `learning_journal.md`'s "weight tying" entry for the full explanation). With this project's large Teacher-aligned vocabulary (248,077 tokens), untying doubles an already-huge table, which directly worked against a separate, related decision: **model size tiers now refer to core size (transformer blocks only), not total size**, since the vocab-sized head has nothing to do with the model capacity actually being scaled and dominates small cores badly (`README.md`'s "Student size" section has the full reasoning and the corrected 40M/150M tier numbers). Resolution: `train_sft.py --mup` now keeps the head tied by default (`--mup_untie_head` opts into canonical muP instead), still applies muP's core init/LR scaling and the logit/`width_mult` rescaling at the loss, just skips the zero-init/untie step — a documented, deliberate deviation from the paper, not the full recipe.

**150M-core tier run** (`logs/EXP-004-kd-150Mcore-mup/`): `n_layer=12, n_embd=1024, n_head=16` → 152.2M core + 254.0M tied head = 406.2M total (not the ~280M initially mis-estimated by reusing an earlier width's head size by mistake — corrected before writing anything down permanently). Reused the LR (0.003) found optimal for the muP base width (40) in the transfer test above, unchanged, across a 25.6× width jump — converged cleanly, best combined loss 0.288 (CE 12.19→0.60, KD 0.40→0.28), notably better than the 40M-core run's 0.673 on the same 200-example validation slice, with no signs of instability from the un-retuned LR.

Treat muP (the tied-head variant actually in use) as implemented, smoke-tested, and now backed by one real-data GPU transfer experiment showing the expected directional effect — not a fully rigorous, high-confidence validation (would want a bigger width ratio and a clearer LR-instability demonstration), but no longer "theory only."

**500M-core tier run** (`logs/EXP-005-kd-500Mcore-mup/`, job 4091486 on `abacus26` L40S): `n_layer=25, n_embd=1280, n_head=16` → 493.2M core + 317.5M tied head = 810.8M total. Same base LR (0.003) reused unchanged from the width-40 tuning run, now across an 80× width jump from the tuning base (32× from the 150M-core tier's own `width_mult`). Trained cleanly, no divergence: 200 steps in 77.3s, best combined loss 0.382 (CE 12.25→1.12, KD 0.41→0.28). Slightly worse than the 150M-core tier's 0.288 on the same 200-example validation slice (expected — capacity isn't the bottleneck at this data size, and this is still the tiny validation slice, not a real training run), but the LR transferred without retuning and without instability across three tiers now (40M → 150M → 500M-core), which is the actual thing muP was adopted to guarantee.

**LR/kd_alpha/weight_decay sweep, and a real limit of the muP-transfer claim (2026-09-04, `abacus30` V100×1, `logs/EXP-006-hp-sweep` group on W&B/MLflow)**: prompted by the observation that `lr=0.003`/`kd_alpha=0.5` had never actually been shown to be *good*, only shown to *transfer* — ran a full grid at the 40M-core tier (LR ∈ {1e-4..1e-1}, `kd_alpha` ∈ {0.25,0.5,0.75}, `weight_decay` ∈ {0,0.01,0.1}), all on the 200-example validation slice. Found a clear optimum at 40M-core: `lr=0.01` (best_loss 0.195, vs. 0.344 at the previously-used 0.003 — a 1.8× improvement), `kd_alpha=0.25` slightly better still (0.146, vs. 0.195 at alpha=0.5), `weight_decay` had negligible effect (0.192-0.199 across 0/0.01/0.1).

**Then re-tested this "improved" config at 150M-core to confirm transfer — it did not transfer.** `lr=0.01` (with the old `kd_alpha=0.5`) gave best_loss 1.276 at 150M-core, dramatically worse than the 0.288 already achieved with the original `lr=0.003`. A finer probe (`lr` ∈ {0.003, 0.005, 0.007, 0.01} at 150M-core, `kd_alpha=0.5` held fixed) showed a monotonically *increasing* loss with LR at this tier (0.288 → 0.306 → 0.374 → 1.276) — the exact opposite ranking from the 40M-core tier, where higher LR (up to 0.01) monotonically *decreased* loss. `kd_alpha=0.25` also transferred badly (0.423 at 150M vs. 0.288 at alpha=0.5, with `lr=0.003` held fixed to isolate the effect).

**Likely cause, not yet fixed**: the tier progression (40M→150M→500M-core) scales `n_layer` (4→12→25) *and* `n_embd` (160→1024→1280) simultaneously. muP's width-invariance guarantee applies to `n_embd` (width) only — it says nothing about depth, and deeper networks are well known to need lower LR for stability independent of muP (residual-stream accumulation across more layers). So the "transfer test" done across tiers was never a clean width-only test; depth was a confound the whole time. This explains why the original `lr=0.003`/`kd_alpha=0.5` — never verified as optimal at any single tier, just carried forward unchanged — turned out to be a better multi-tier compromise than the value that *was* verified optimal, but only at the shallowest, narrowest tier alone.

**Practical upshot (superseded below)**: no retraining needed — `EXP-003/004/005` already used the better (if accidentally so) hyperparameters for this depth+width co-scaling trajectory. The sweep's value was diagnostic, not a config upgrade: it turned an untested assumption into a confirmed (if not fully understood) one, and surfaced a real methodological gap — true muP guarantees would require holding depth fixed (isolating width) or adding a depth-specific correction (e.g. residual-branch `1/sqrt(n_layer)` scaling, "Depth-muP"-style) — neither implemented yet. Revisit before trusting LR transfer to much deeper tiers (1B/3B-core) if those also increase depth alongside width.

### Depth-muP-lite: implementing and testing the fix (2026-09-04, same session)

Implemented the fix flagged above: `apply_depth_mup_scaling()` in `train_sft.py` (new `--depth_mup`/`--mup_base_depth` flags) scales each transformer block's residual branch OUTPUT (attention and MLP, before the residual add) by `1/sqrt(n_layer / mup_base_depth)`, via `register_forward_hook` on each block's `.attn`/`.mlp` submodules rather than monkeypatching `GPT2Block.forward` (verified against the installed transformers version that `GPT2Attention` returns `(attn_output, present)` and `GPT2MLP` returns a plain tensor, so hooks can rescale the branch cleanly without touching the internal residual-add). "Lite" in the same sense as the existing width-muP-lite: only the forward-pass branch magnitude is corrected, not the branch's own init variance or a matching LR term — the single largest-effect piece, not the full canonical recipe.

Re-ran the exact case that broke transfer above (`lr=0.01`, `kd_alpha=0.5`, the config optimal at 40M-core) at both bigger tiers, with `--depth_mup --mup_base_depth 4` (4 = the 40M-core tier's own `n_layer`, so `depth_mult=1` there and the correction is a no-op at the tuning base — confirmed: 40M-core with `--depth_mup` gave the byte-identical 0.194570 as without it):

| tier (n_layer) | `lr=0.01`, no depth_mup | `lr=0.01`, with depth_mup | (reference: old `lr=0.003`, no correction) |
|---|---|---|---|
| 40M (4) | 0.195 | 0.195 (neutral, as expected) | 0.344 |
| 150M (12, depth_mult=3) | 1.276 | **0.469** (2.7× better) | 0.288 |
| 500M (25, depth_mult=6.25) | 2.356 | **0.320** (7.4× better, and *beats* the 0.382 old baseline) | 0.382 |

Depth-muP-lite substantially restores transfer at both deeper tiers, and at the largest depth ratio tested (500M, 6.25×) the "optimized-at-40M" LR combined with the depth correction actually **beats** the old untuned baseline — the first config found in this whole investigation that's simultaneously good at the small tuning tier AND competitive-or-better at the largest tier trained so far. At 150M (3× depth ratio) it closes most but not all of the gap to the baseline (0.469 vs. 0.288) — plausibly noise from the tiny 200-example single-seed validation slice, or a real residual gap that the "-lite" version (no init/LR-side depth correction) doesn't fully close; not yet distinguished.

**Practical upshot (current)**: `--mup --depth_mup --mup_base_depth 4 --lr 0.01` is now a credible candidate default for the model-size-tier roadmap — tune once at the smallest/shallowest tier as originally intended, transfer to both width and depth increases together. Given the remaining (if reduced) gap at 150M, this isn't yet a slam-dunk "always use this" result off a single 200-example slice; worth confirming on a larger validation slice (or the real training set) before committing it as the new default for the next tiers (1B/3B-core), but it's a real, working fix for the depth confound identified above, not just a diagnosis this time.

### ⚠️ Critical correctness bug found (2026-09-05): the FP8 Teacher's weights were never actually dequantized

While validating the FP8-vs-bf16 evaluation methodology (comparing Top-K
outputs from `Qwen/Qwen3.8-27B-FP8` against `Qwen/Qwen3.8-27B` bf16 on the
same 40-example slice), found **0% top-1 agreement** — nowhere near the
expected ~96-99% from the reference quantization literature. Root cause,
confirmed directly by inspecting the loaded model: `transformers`
(5.17.0.dev0) never applies the FP8 `weight_scale_inv` dequantization scale
for this checkpoint (`quant_method: fp8`, `fmt: e4m3`) — every scale tensor
is loaded then discarded as `UNEXPECTED`, `model.is_quantized` is `None`,
and every linear module is a plain `nn.Linear` holding the raw FP8 bytes
reinterpreted as bf16. Full writeup and evidence in
`learn/distill/qwen3.8-27b-notes.md`'s new "CRITICAL" section.

**This means every KD run above (`EXP-003/004/005`, the LR/kd_alpha/weight_decay
sweep, the Depth-muP-lite results in this same section) trained against
Top-K Teacher targets precomputed from this broken FP8 load path** — the
loss numbers are real, but the "Teacher signal" being distilled was
numerically incoherent, not real Qwen3.8-27B knowledge. The muP/Depth-muP
transfer conclusions (about LR/optimizer behavior across width and depth)
are probably still valid as relative comparisons — the targets were
garbage but *consistently* garbage across all three tiers, and the question
being tested was whether a given LR/depth-correction transfers, not how
good the resulting student is. But no number here should be read as
evidence of real distillation quality, and the combined-loss magnitudes
(0.19-2.36 across these tables) don't mean what they were assumed to mean.

**Fixed and verified same day.** Root cause: `load_model_and_tokenizer`
passed `quantization_config=None` explicitly, suppressing transformers'
auto-detection of the checkpoint's native FP8 scheme. Fix: omit the kwarg
unless bnb quantization is requested. Re-ran the comparison after the fix:
**98.92% top-1 agreement** vs bf16 (matches the ~98.9% 8-bit literature
reference) — confirms the fix and validates the comparison methodology.
Full writeup in `learn/distill/qwen3.8-27b-notes.md`'s "RESOLVED" section.
`EXP-003/004/005` above still predate the fix and used the broken load path
— still worth deciding whether to rerun them against real Teacher targets
before trusting any Teacher-signal-quality conclusion from them.

## 2026-09-13 — Distillation 500M-core MFU/batch-size sweep, and a missing-bf16 discovery (`abacus26` L40S, job 4104870)

Requested by thinker-e9 to replace the extrapolated MFU estimates in `learn/distill/README.md` with a real batch-size sweep at the 500M-core tier (810.8M total, `n_layer=25 n_embd=1280 n_head=16`). Found something more consequential than a batch-size curve: **`train_sft.py` had no mixed-precision at all** (`AutoModelForCausalLM.from_config` → fp32 by default, zero `autocast`/`GradScaler` in the file) — every MFU number in this project's history to date was an fp32-achieved-throughput compared against a **bf16** peak-FLOPS spec, not a like-for-like ratio.

Added `--bf16` to `train_sft.py` (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around the forward+loss computation; opt-in flag, no behavior change when unset). Results:
- fp32: batch=4 → 3,239 tok/s, batch=6 → 3,425 tok/s, **batch=8 OOMs** (44.39GB L40S already ~42GB used at batch=6).
- bf16: batch=6 → **5,748 tok/s — a real 1.68× speedup over fp32 at the identical batch size** (Tensor Core effect, confirms the missing-autocast finding was real and fixable). **batch=7 also OOMs under bf16.**
- **The batch ceiling (6) is identical in fp32 and bf16** — bf16 sped up compute but did not raise the memory ceiling, because the bottleneck is the Teacher-aligned tied vocab head (248,077 tokens): `topk_kd_loss`'s `logsumexp` over the full `(batch, block_size, vocab)` logits tensor, plus the Teacher's own fp32 top-K/residual tensors, dominate memory regardless of the model's own autocast dtype.

**Practical upshot**: `--bf16 --batch_size 6` (5,748 tok/s real) is now the best measured single-GPU config at this tier, giving ~392h (16.3 days) for `D=8.11B` tokens on one L40S — still past Grid'5000's ~1-week single-reservation limit, so checkpoint/resume across besteffort reservations (already implemented) remains necessary regardless. Full numbers and a chunked-loss idea (to remove the vocab-driven memory ceiling independently of precision, à la Liger-Kernel/"Cut Your Losses" — not yet implemented, next lever if more throughput is needed) are in `learn/distill/README.md`'s updated batch-size/precision sweep section.

## 13 Sep 2026 -- Teacher-target precompute sharding: A40/A100 (Ampere) are a bad fit for the FP8 Teacher checkpoint

Per the user's "as fast as possible" directive, sharded the 8000-example Teacher-target precompute (`precompute_teacher_targets.py`, K=32, max_length=1024) across 3 independent GPU jobs instead of running it sequentially on one node. Measured per-node throughput surfaced a real (not incidental) hardware-fit issue:

| Node (GPU) | Architecture | ex/s | Notes |
|---|---|---|---|
| abacus26 (L40S) | Ada Lovelace | 2.84-2.86 | native FP8 tensor cores |
| abacus27 (H100 NVL) | Hopper | 3.2-3.5 | native FP8 tensor cores |
| abacus4 (A40) | Ampere | 0.24 | **no native FP8 tensor cores** |

abacus4's shard was ~12x slower than the other two despite similar GPU memory headroom (58% util, only 126W draw on a ~300W TDP card -- clearly not compute-bound in the normal sense). Root cause: the Qwen3.8-27B-FP8 Teacher checkpoint is natively FP8-quantized; Ada (L40S) and Hopper (H100) have hardware FP8 Tensor Core support, Ampere (A40, and presumably A100) does not, so the `kernels` package's fine-grained FP8 path falls back to a much slower dequant/compute path on Ampere. This is a distinct failure mode from the earlier-documented "forcing `--dtype bfloat16` on a <56GB-VRAM GPU triggers CPU offload" collapse (both were previously conflated as "some GPUs are just slow for this") -- here `--dtype auto` was used correctly, and the model fit in VRAM without offload; the slowdown is purely an architecture/FP8-kernel-support mismatch.

### Full train/val curve, KD-run 500M-core (job 4105629, completed cleanly)

Completed all 13340 steps (`training_seconds=7669.2`, ~2h08, `best_loss=0.1227`), this time with the full periodic val curve preserved (`python -u` fix). Key points (step: val_ce / val_kd):

| step | val_ce | val_kd |
|---|---|---|
| 1 | 12.107 | 0.400 |
| 500 | 0.3145 | 0.671 |
| 1000 | 0.1909 | 0.706 |
| 2000 | 0.1315 | 0.732 |
| 3000 | 0.1134 | 0.750 |
| 5000 | 0.1004 | 0.778 |
| 8000 | 0.1062 | 0.786 |
| 10000 | 0.1038 | 0.802 |
| 13000 | 0.0974 (min) | 0.815 (max) |

**val_kd diverges almost immediately** (0.40->0.67 by step 500 alone, ~75% of its total eventual rise happens by step 2000-3000) and keeps climbing slowly and almost monotonically for the entire 13340-step run, never plateauing. **val_ce shows no comparable divergence** -- it oscillates in a noisy 0.10-0.13 band from step ~2500 onward, with its best value at the very last measured point (step 13000). Conclusion for model-design's question: KD-term memorization starts near-instantly and never stops climbing at this data scale (8000 examples); CE-based language-modeling generalization is unaffected across the whole run. Their suggested follow-up (try a lower `kd_alpha`, e.g. 0.1-0.2, to see if de-weighting the KD term changes the overall val_loss picture) is a reasonable next step, not yet run.

### GPU-scale attn_supervised grid (6 runs: 3 baseline, 3 attn_supervised) -- self-match fixed, task accuracy not

model-design's attention-supervision fix (auxiliary CE loss on q_proj/k_proj, no new params) was tested at `d_model=128` (vs. their CPU-scale `d_model=32` test) across 3 seeds each:

| variant | seed | final_acc | mean_rank (chance=1.50) | top1_rate (chance=0.25) |
|---|---|---|---|---|
| baseline | 0 | 0.246 | 1.250 | 0.336 |
| baseline | 1 | 0.264 | 1.264 | 0.401 |
| baseline | 2 | 0.256 | 1.590 | 0.272 |
| attn_supervised | 0 | 0.247 | **0.000** | **1.000** |
| attn_supervised | 1 | 0.245 | **0.000** | **1.000** |
| attn_supervised | 2 | 0.257 | **0.000** | **1.000** |

**Striking disconnect**: the self-match diagnostic goes from noisy/near-chance (baseline) to *perfect* (mean_rank=0, top1=100%, all 3 seeds) under attention supervision -- the auxiliary loss completely fixes the mechanistic problem it targets. But `final_acc` on the actual n_hops=2 chain task is essentially unchanged (baseline avg ~0.255, attn_supervised avg ~0.250) -- no better than the ~25-32% plateau documented throughout this project. This GPU-scale result (larger d_model, longer budget than model-design's CPU smoke test) does not reproduce their reported 72.9% accuracy at CPU scale -- a real discrepancy to flag, not just noise, since the self-match fix landed perfectly across all 3 seeds while accuracy stayed flat. Possible reading: perfect self-match among an episode's *own* candidate facts is necessary but not sufficient for the downstream task -- something else in the SM->output path (per model-design's own earlier hypothesis) may be the actual bottleneck once retrieval itself is no longer the failure mode.

**Actionable conclusion**: never schedule the Teacher-FP8 precompute (or presumably any FP8-checkpoint inference) on Ampere-generation GPUs (A40, A100) at this cluster -- restrict to Ada/Hopper (L40S, H100) or newer. The abacus4 job was killed mid-shard (besteffort preemption actually beat us to it) and its ~2520 remaining examples were re-split across the two already-idle Ada/Hopper nodes instead, which finished in ~8 additional minutes.

Follow-up: launched the real KD training run (`learn/distill/train_sft.py`, 500M-core tier, `--bf16 --mup --kd_alpha 0.5`, merged 8000-example Top-K32 Teacher targets) on the H100 node (fastest available at Rennes for this workload), 10 epochs (13,340 steps) budgeted at ~2h based on measured throughput, checkpointing every 500 steps to survive besteffort preemption.

### Real KD run result (500M-core, 8000 real examples, H100)

Completed cleanly, no preemption: **13,340/13,340 steps, `best_loss=0.1267`, `training_seconds=3110.6` (~51.8 min)** -- almost 2.5x faster than the ~2h05 estimate extrapolated from L40S throughput (0.552 s/step there vs. ~0.233 s/step actually achieved on the H100 NVL, a bigger gap than the ~15-20% suggested by the earlier precompute ex/s comparison -- KD training's compute mix, unlike single-example precompute inference, apparently favors H100 more strongly, plausibly batching/kernel-fusion effects rather than raw FP8 throughput alone).

Loss trajectory: 6.41 (step 1) -> 1.21 (step 95) -> ~0.15-0.18 (plateauing from roughly step 9000 onward, oscillating in that band through step 13340). Combined CE and KD components both bottomed out in the same range (`ce` ~0.10-0.15, `kd` ~0.17-0.22 at the end).

**Caveat worth flagging to model-design**: with only 8000 training examples and 13,340 steps at batch_size=6 (~10 full epochs), a loss collapse from 6.4 to ~0.15 is consistent with memorization/overfitting on this small a sample, not necessarily a generalizable KD signal -- the run validates the training *pipeline* (real data, real Teacher targets, checkpoint/resume, bf16, muP) end-to-end at this scale, but the loss curve itself shouldn't be read as "KD works well at 500M-core" without a held-out eval or a larger example count to rule out memorization.

### Held-out val check confirms memorization on the KD term

Per model-design's suggestion, ran the trained checkpoint against a held-out val split (from `prepare_reasoning_data.py`'s own `val.jsonl`, never seen in training -- distinct from `train_sample8000.jsonl`) using the same CE+KD loss (`learn/distill/eval_val_loss.py`, a new small script that reconstructs the exact architecture from the checkpoint's saved args/muP multipliers and runs a no-grad pass). First pass used a 40-example val slice (`val_sample40.jsonl`) whose Top-K32 Teacher targets happened to already exist from an earlier bf16-vs-fp8 precompute-dtype sweep session, letting this check run **without any GPU at all** (the eval only needs the small 810M student + precomputed targets, not the 27B Teacher -- ran on a plain CPU besteffort-free job while the three Ada/Hopper GPUs were all tied up by other users' jobs, see `grid5000_usage.log.md`).

Result: **val_loss=0.4585** vs. **train best_loss=0.1267** (~3.6x gap). Breaking down the two components separately is informative: `val_ce=0.0985` is actually in the same range as train's CE component (~0.10-0.15) -- plain next-token prediction generalizes fine -- but `val_kd=0.8185` is roughly 4-8x every train-time KD value logged (~0.10-0.22 range). **Conclusion: the loss collapse is memorization specifically of the fine-grained Teacher-logit alignment (the KD term), not of the underlying language-modeling task.** This matches the earlier caveat's prediction and settles the train/val question model-design asked for -- more examples (not just more steps) are needed before this run's loss curve says anything about real KD quality at this scale.

Caveat on this specific check: n=40 is a small val slice (chosen only because its Teacher targets already existed from an unrelated earlier sweep, avoiding a GPU-contended precompute just to get a first read); the qualitative CE-vs-KD split is unlikely to flip with more examples, but a tighter quantitative val_loss estimate would use a larger held-out slice (a 1000-example `val_sample1000.jsonl` is already prepared and staged for this, precompute pending GPU availability).

### Correction: "avoid Ampere for Teacher-FP8 precompute" was too broad -- the real constraint is VRAM, not architecture generation

Session-13's earlier `experiment.log.md` entry ("Teacher-target precompute sharding: A40/A100 are a bad fit for the FP8 Teacher checkpoint") concluded from the abacus4 (A40, 46GB) result alone that Ampere-generation GPUs should be avoided entirely for this workload. Investigating a Nantes site standby reservation (see `grid5000_usage.log.md`) turned up pre-existing logs from an earlier session's FP8-vs-bf16 comparison work on an **A100 80GB** (`ecotaxe` cluster) that contradict the blanket claim.

`transformers` itself explains the real mechanism on load: *"FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100) ... We will default to dequantizing the model to bf16"* -- A100 is compute capability 8.0, so it always dequantizes FP8->bf16 on load, exactly like A40. The dequantized model needs ~55.6GB VRAM (vs. ~30.9GB native FP8). **A40 (46GB) doesn't have enough VRAM for that, so it silently falls back to CPU offload -- a ~50-100x collapse, which is what the earlier 0.24 ex/s number actually measured.** A100 80GB has plenty of headroom for the same 55.6GB dequantized model, so no offload happens: `precompute_fp8_fixed.log` from that Nantes session shows the checkpoint loading in 12.6s and reaching a **steady-state throughput of ~3.1-3.6 ex/s** -- essentially on par with L40S (2.84-2.86 ex/s) and close to H100 (3.2-3.5 ex/s), not 12x slower.

**Corrected rule**: the deciding factor for this Teacher checkpoint's precompute speed is **available GPU VRAM relative to the ~56GB bf16-dequantized footprint**, not "Ampere vs. Hopper/Ada" as a category. A100-80GB (and presumably any other >=64GB-class Ampere card) is a fine precompute target; A40 (46GB) and any other <56GB card outside the native-FP8 Ada/Hopper set are not. Told model-design about this correction since the earlier (too-broad) version had already been passed along.

### 2026-09-20 -- KD wired into train_prompt_response.py (prompt/thinking/answer streams), char-to-token alignment problem solved and verified

`learn/indexed_attention/train_prompt_response.py` was CE-only until now (see its own docstring's earlier note). Landed `--teacher_targets`/`--val_teacher_targets`/`--kd_alpha`/`--teacher_max_length`, reusing `topk_kd_loss()` (`learn/distill/train_sft.py`) exactly like `train_real_text.py` -- but the alignment problem flagged back on 2026-09-20 (train_real_text.py's fixed-window-position convention doesn't apply here) needed real design work, done in `data/prompt_response_dataset.py`:

**The problem**: `precompute_teacher_targets.py` tokenizes the jsonl's whole `text` field (the CHATML string). `ReasoningPromptDataset`/`RetrievalPromptDataset` instead tokenize each response span (`thinking`, `answer`) **standalone**, via `_tokenize_padded` on the extracted substring -- not as a slice of `text`'s own tokenization. Two distinct failure modes had to be handled, not assumed away:

1. **String-level**: `ReasoningPromptDataset`'s `answer` field is the dataset's canonical verified answer, NOT a re-parse of the generated text after `</think>` (a paraphrase, see that class's own docstring) -- it may simply not appear verbatim in `text` at all.
2. **Token-level (found only by testing on real tokenizer output, not by design review)**: even when the span string DOES appear verbatim in `text` (guaranteed for `RetrievalPromptDataset`'s `answer`, template-substituted), standalone tokenization of the substring can produce **different token ids** than the same characters tokenized in their surrounding context, because BPE merges a leading space differently depending on what precedes it. Reproduced directly with gpt2 on a synthetic example: `"4"` tokenized alone -> id 19, but the same `"4"` inside `"...is 4."` -> id 604 (`" 4"` merged as one token). A positional-only alignment (assume span token i == text token tok_start+i) would silently feed the wrong Teacher row here.

**Fix**: `PromptResponseTeacherTargets.slice_span()` (`data/prompt_response_dataset.py`) re-tokenizes the whole `text` with `return_offsets_mapping=True` (same tokenizer/truncation/max_length as the precompute run -- deterministic, so no change needed to `precompute_teacher_targets.py` itself), locates the span's token range via character offsets, and **verifies exact token-id equality** between that slice and the span's own standalone tokenization before using it. Any mismatch -> that example's KD mask is all-False (pure CE for it, not a crash, not a silently wrong target). `doc_id` alignment also required a real fix: both dataset classes previously indexed `self.examples` by post-filter position, but `precompute_teacher_targets.py` iterates the UNFILTERED jsonl (some rows are skipped by e.g. `if not answer: continue`) -- `doc_id` is now the original pre-filter row index, stored per example.

**Verified with a synthetic smoke test** (`kd_smoke_test.py`, 4-example reasoning jsonl + a random-logit synthetic Teacher .npz built with the same tokenizer/text, gpt2 for offline reproducibility): `thinking` span aligned 4/4 examples (long span starting right after `"<think>\n"`, robust to the boundary effect); `answer` span in this reasoning case fell back cleanly 0/4 (the exact "4"/" 4" case above) -- confirms the fallback triggers correctly rather than silently misaligning. A follow-up direct check on a synthetic `RetrievalPromptDataset` example (verbatim-substituted answer, no paraphrase) DID align cleanly on every available token position, confirming the fallback is specifically about the tokenizer-boundary effect, not a general failure of the mechanism. Ran the actual CLI end-to-end (`kd_alpha=0` vs `kd_alpha=0.7`, same seed/data): both complete without error, KD run's total loss differs from CE-only as expected from the weighted combination, `kd_answer`/`kd_thinking` only appear in logs when `--teacher_targets` is set. Full `tests/` suite: 76/76 passed after this change (no regression).

**Not yet measured on real data**: what fraction of real OpenR1-Math-220k `answer` spans hit the fallback in practice (this smoke test used a deliberately adversarial short-numeric-answer case) -- worth checking once a real precompute exists, since a high fallback rate would mean the `answer` stream's KD signal on reasoning data is mostly absent even though the mechanism is correct. `thinking` and retrieval's `answer` are expected to be far more robust based on the mechanism above (long spans / verbatim substitution), but this is a prediction, not yet a measurement.

**Next step to unblock a real KD run on this pipeline**: precompute Teacher Top-K targets for `openr1_math`/`hotpotqa` with a tokenizer-matched Teacher, e.g.:
```
python3 learn/distill/precompute_teacher_targets.py \
  --input_file data/distill/openr1_math/train.jsonl \
  --model_dir LiquidAI/LFM2-1.2B --top_k 32 --max_length 4096 \
  --out_file data/distill/openr1_math/train_topk32.npz
python3 learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data data/distill/openr1_math/train.jsonl \
  --tokenizer lfm2 --teacher_targets data/distill/openr1_math/train_topk32.npz \
  --teacher_max_length 4096 --kd_alpha 0.5 [...]
```
(LFM2-1.2B v1, not the "Thinking"/2.5 variant -- see `core/model_families.py`'s `lfm2` comment for why the tokenizer must match exactly.)


## 2026-09-20 — KD alignment diagnostic: 0/2000 answer spans aligned on openr1_math (total CE-only fallback)

Diagnostic demandé par `model-design` (mesure du taux réel de fallback CE-only sur le stream `answer`, `ReasoningPromptDataset` + Teacher targets LFM2-1.2B) exécuté sur `graffiti-3` (Nancy) : **0/2000 exemples ont un span `answer` aligné** -- fallback total, aucun signal KD sur ce stream tel que généré actuellement.

Confirme la limitation explicitement anticipée par `model-design` avant ce diagnostic ("`answer` canonique/pas forcément verbatim dans le texte généré"). Le signal KD sur `reasoning` portera donc entièrement sur le stream `thinking` (le trace `<think>...</think>` lui-même, qui est verbatim par construction) -- pas bloquant pour le KD global, mais confirme qu'il faut soit ignorer/pondérer à zéro le KD sur `answer` pour ce dataset, soit revoir `prepare_reasoning_data.py` pour que l'`answer` canonique corresponde à un span verbatim du texte généré (si l'un des deux streams doit avoir du KD).

Commande exécutée : script fourni par `model-design` (`ReasoningPromptDataset` chargé avec `teacher_targets=data/distill/openr1_math/val_topk32_lfm2-1.2b.npz`), sur les 2000 exemples du split val.

## 2026-09-20 — Vérification tokenizer Qwen: alias `qwen` (Qwen3-0.6B) et Qwen3.8-27B-FP8 sont INCOMPATIBLES

Vérifié directement avant tout precompute (même méthode que LFM2/LFM2.5) : `Qwen/Qwen3-0.6B` (alias `qwen` dans `core/model_families.py`) a `vocab_size=151643, len=151669`, tandis que le Teacher déjà téléchargé `Qwen3.8-27B-FP8` (`storage3.rennes.grid5000.fr/thinker-distill/`) a `vocab_size=248044, len=248077` -- vocabulaires de taille différente, IDs différents sur une phrase test identique. **Tokenizers totalement incompatibles, pas de désalignement subtil -- une confusion évidente si on tentait de les traiter comme identiques.** Ne pas réutiliser l'alias `qwen` pour ce Teacher -- nécessite un alias distinct dans `core/model_families.py` (même traitement que `lfm2` vs `lfm2-thinking`), à ajouter par `model-design` avant tout precompute Qwen3.8-27B.

## 2026-09-20 — LFM2-1.2B precompute train scale (18k exemples), consigne "ne jamais laisser un GPU modeste inactif à attendre un gros Teacher"

Consigne explicite de l'utilisateur : quand un GPU est trop petit pour OLMo-7B/Qwen3.8-27B, l'utiliser pour avancer l'entraînement du student (Thinker) plutôt que le laisser inactif. Appliqué directement : `abacus11-1` (libéré par la fin de `useff_sweep`) redirigé vers le precompute LFM2-1.2B **train complet** (18000 exemples chacun, `openr1_math`/`hotpotqa`), sans attendre les jobs OLMo (A100, Waiting) / Qwen (H100, Waiting). Confirmé tournant (RTX A5000 25Go, chargement réel). Une fois prêt, lancement d'un vrai run KD (`train_prompt_response.py --teacher_targets ... --kd_alpha 0.5`) sur GPU modeste également, sans attendre les gros Teachers -- chantier multi-tokenizer (OLMo/Qwen) reste distinct et non-bloquant.

## 2026-09-20 — Bug réel : npz corrompu en tuant le process trop tôt après écriture

En nettoyant les process precompute LFM2 "terminés" (GPU à 0% util, fichier .npz déjà présent avec une taille plausible), j'ai tué le process reasoning avant qu'il ait fini d'écrire/fermer le fichier -- `zipfile.BadZipFile: File is not a zip file` au chargement par `train_prompt_response.py`. **Leçon : ne jamais tuer un process de precompute juste parce que le fichier de sortie existe déjà avec une taille non-nulle -- attendre la ligne de résumé explicite ("Wrote ... : N tokens across M examples") avant de le considérer terminé et sûr à interrompre.** Régénéré proprement, run KD relancé une fois le fichier confirmé complet.

**Run KD retrieval (lancé, confirmé actif)** : `loss` réel en baisse (5.79 au step 1, ce_answer 11.59→8.53 au step 20) -- mais **`kd_answer=0.0000`** dès le début, même symptôme que le diagnostic d'alignement sur reasoning (0/2000 answer spans alignés) -- suggère que le fallback CE-only total touche aussi le stream `answer` de retrieval, pas seulement reasoning. À vérifier avec le même diagnostic d'alignement que model-design avait demandé pour reasoning, appliqué à `RetrievalPromptDataset`.

## 2026-09-20 — KD retrieval kd_answer=0 EXPLIQUÉ: bug réel dans la comparaison d'alignement, pas un problème "pas verbatim"

Investigation demandée par `model-design` (prédiction : retrieval devrait aligner bien mieux que reasoning puisque `answer` y est substitué verbatim). Vérifié : `--max_length` du precompute (4096) correspond bien à `--teacher_max_length` par défaut (4096) -- pas de mismatch de troncature.

**Diagnostic quantitatif (`RetrievalPromptDataset`, train, 18000 exemples) : 0/18000 alignés.** Inspection manuelle de 3 échecs : dans les 3 cas, `ex["answer"]` **est bien trouvé verbatim** dans `ex["text"]` (`idx != -1`) -- donc ce n'est PAS le problème "canonique/paraphrase" de reasoning. C'est un vrai bug dans `slice_span`'s comparaison d'égalité de tokens, à deux causes qui se cumulent (vérifiées directement) :

1. **`_tokenize_padded(tokenizer, ex["answer"], ...)` tokenise la réponse standalone, ce qui ajoute un token BOS en tête** (`tokenizer("São Miguel")["input_ids"]` → `[1, 560, 2388, 22661]`, le `1` initial est BOS) -- alors que le slice extrait du texte complet ne contient jamais de BOS en position médiane. À lui seul, ça suffit à garantir `ids_slice[0] != span_ids[0]` sur 100% des exemples.
2. **Fusion BPE d'espace précédent** : même en ignorant le BOS, `"São Miguel"` seul tokenise en 3 tokens (`[560, 2388, 22661]`) alors qu'en contexte (précédé d'un espace/`ASSISTANT_MARKER`) il tokenise en 2 (`[17370, 22661]`) -- vérifié directement avec `tokenizer(" " + answer)["input_ids"]` qui donne `[1, 17370, 22661]`, correspondant exactement (hors BOS) au slice en contexte.

**Ces deux causes sont déterministes, pas des cas limites (short answers "yes"/"no" comme suspecté initialement) -- elles s'appliquent à TOUT exemple, expliquant le 0/18000 exact plutôt qu'un taux partiel.** Fix nécessaire côté `data/prompt_response_dataset.py` : `ans_ids` (utilisé pour la comparaison dans `slice_span`) devrait être dérivé (a) sans le token BOS, et (b) avec un espace précédent (ou, plus robuste, en re-slicant directement `ids_slice` depuis la tokenisation en contexte déjà calculée par `_locate_token_span`, au lieu de comparer à une tokenisation standalone séparée). Ne pas patcher moi-même -- code de `ff2attn`/`model-design`, à leur discrétion sur l'approche de fix. Run KD retrieval laissé actif tel quel (dégrade proprement en CE pur, pas de crash) en attendant.

## 2026-09-20 — Fix confirmé: kd_answer non-nul sur retrieval après resynchronisation

Fix `a71f729` synchronisé, run KD retrieval tué et relancé (checkpoint précédent abandonné, jamais eu de vrai signal KD sur `answer` comme prévenu). **Résultat immédiat : `kd_answer=11.14` au step 1 (non-nul), `kd_answer=8.04` au step 20 -- signal KD réel désormais actif**, contrairement au `kd_answer=0.0000` systématique d'avant le fix. Diagnostic d'alignement quantitatif relancé pour mesurer le taux exact (résultat à suivre).

## 2026-09-20 — Diagnostic d'alignement retrieval après fix: 18000/18000 (100%)

Confirme le fix `a71f729` de bout en bout : `RetrievalPromptDataset`, train, 18000 exemples -- **18000/18000 answer spans alignés (100%)**, contre 0/18000 avant. Exactement la prédiction de `model-design` (retrieval devait aligner presque parfaitement puisque `answer` y est verbatim, une fois la comparaison faite correctement). Run KD retrieval bénéficie maintenant d'un vrai signal KD sur 100% des exemples, pas de fallback CE résiduel à surveiller sur ce dataset.

## 2026-09-20 — KD run retrieval complet (8000/8000 pas): fort surapprentissage

`train_prompt_response.py --dataset_type retrieval --kd_alpha 0.5`, alignement KD 100% (fix confirmé). **Train `final_loss=2.07`, mais VAL `answer=7.54` (kd_answer=6.39) au step final** -- écart train/val massif, signe de surapprentissage marqué à ce budget (8000 pas, `d_model=1024`, pas de KD sur `thinking` ici puisque retrieval n'a qu'un stream `answer`). Le signal KD réel (confirmé actif) n'empêche pas l'écart -- à surveiller si un budget plus court ou une régularisation serait nécessaire avant de tirer des conclusions sur l'utilité du KD lui-même pour cette tâche.
