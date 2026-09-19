# S3 curriculum ffn2attn — A1/A2, S0, S0.5, attribution/probe/routeur

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-19 — A1/A2 (Phase 12 pre-check): a measurement artifact caught, real signal is quasi-orthogonality

`learn/indexed_attention/diagnose_olmo_ffn_memory_geometry.py`, `allenai/OLMo-2-0425-1B` pretrained weights, CPU, no training, ~2 min wall-clock (`paradoxe-27`, job 4121204). Goal: does unifying the FFN-key space (`W_gate`) with the attention-key space (`W_K`) cost a real distillation, or is it nearly free -- decides S1's branch (i) vs (ii) and informs S3's feasibility, before any conversion code is written.

**Measurement artifact caught before it produced a wrong conclusion**: the "energy of `W_gate` projected onto `span(W_K)`" metric read **exactly 1.0000 at every layer**, and the top-10 principal-angle cosines between the two subspaces are **exactly 1.0** too. This looks like "total overlap" but isn't -- `AutoConfig` confirms OLMo-2-1B uses full MHA (`num_attention_heads=16 == num_key_value_heads=16`, no GQA), so `W_K` has shape `(2048, 2048)` and is full column rank: `span(W_K)` is trivially the **entire** 2048-dimensional ambient space. Any vector at all -- `W_gate`'s rows included, or pure noise -- projects onto it with 100% energy and 0-degree principal angles. **This specific pair of metrics is uninformative whenever the reference subspace (`W_K` here) has rank >= `d_model`**, which is exactly OLMo's case; reporting the 1.0 values as "unification is nearly free" would have been a real, wrong conclusion in the opposite direction from what the data actually supports.

**The metric that survives this trap**: pairwise cosine similarity between individual `W_gate` key vectors and individual `W_K` key vectors (computed on a 512x512 random subsample, doesn't depend on either matrix's rank). Mean `|cos|` = **0.017-0.021 across all 16 layers**, p90 `|cos|` = 0.044 (layer 0). The expected mean `|cos|` between two independently random unit vectors in `R^2048` is `sqrt(2/(pi*2048))` approx **0.0176** -- OLMo's actual `W_gate`/`W_K` key vectors sit right at that random-vector floor, no measurable directional alignment beyond chance.

**Read (A1)**: `W_gate` (FFN "keys") and `W_K` (attention keys) are **quasi-orthogonal**, not overlapping -- the substantive finding the flawed energy/angle metric obscured. Per the decision table's second branch: **unifying the K/V space across the three memories is not nearly-free; it costs a real distillation to budget**, and S1 should keep "conversion first, anneal into the shared space" (start from S0's exact equivalence, not from a from-scratch shared-space init) rather than treating unification as elegant-and-cheap.

**A2 (cross-layer FFN redundancy)**: stacking all 16 layers' `W_gate` keys (131072 vectors total, `d_model=2048`) gives effective rank **1699/2048 (83%) at 90% energy, 2000/2048 (98%) at 99% energy** -- close to the full ambient dimension, i.e. **very little global redundancy relative to `d_model`** despite the huge nominal key count (131072). Adjacent-layer mean `|cos|` (0.0207) and distant-layer mean `|cos|` (0.0178) are both near the same random-vector floor as A1 and barely distinguishable from each other -- **no clear low/high stratification signal at this coarse per-vector-pair-cosine level** (doesn't rule out subspace-level stratification a finer analysis might catch, just not visible here).

**Read (A2)**: a shared "universal layer" memory (S3) is **not** cheaply compressible from redundancy -- the keys already nearly span the available `d_model`-dimensional space, so unioning them (per the plan's own "union, never average" rule) would need close to the full ambient rank, not a small shared subspace. Combined with A1's quasi-orthogonality finding, **both free pre-checks point the same direction: the elegant/cheap versions of unification (S1 branch (ii), and S3's shared layer) are not free lunches at this model's scale** -- the harder, distillation-requiring paths are the ones actually supported by the geometry. Full per-layer numbers in `runs/olmo_ffn_geometry/result.json` (Rennes home).

**[CORRECTION, same day, before design work started from this reading]**: `model-design` caught that **both A1 and A2's conclusions above outrun what their metrics actually measure**, independent of the rank artifact already caught:
- A1: mean pairwise cosine between individual `W_gate`/`W_K` row vectors doesn't distinguish "incompatible key spaces" from "same space, different rotation" -- two arbitrary orthonormal bases of the *same* `R^d` also average to the random-vector cosine floor. And since `W_K` is already established as full rank (its span is literally all of `R^2048`), the FFN keys live in that same ambient space by construction -- subspace membership was never the right question. The real question is **score scale/geometry under a real query**: do `q.k_ffn` and `q.k_attn` land at comparable magnitudes for an actual residual-stream query, or does one source dominate a unified softmax purely on scale (spec Sec.3.2's "mass vs. peak" risk) regardless of relevance?
- A2: rank of 131072 vectors in `R^2048` is bounded at 2048 *by construction* -- "not fully redundant" was guaranteed before any data was measured, uninformative about S3. The real question is per-key **substitutability**: does a key in layer L have a near-duplicate in another layer? A few thousand near-duplicates among 131k keys would be decisive for S3 and completely invisible in a mean-cosine or rank statistic -- needs a nearest-neighbor **tail** statistic instead.

Both re-measured directly on real activations/nearest-neighbor tails (`diagnose_olmo_ffn_memory_geometry_v2.py`, CPU, ~2 min) rather than left as an open retraction -- see the follow-up entry immediately below for the corrected numbers. The "quasi-orthogonal" / "not compressible" readings above are **retracted as stated**; whatever the v2 entry below says supersedes this one.

## 2026-09-19 — A1/A2 v2: corrected measures -- score-scale mismatch confirmed real, S3 near-duplication genuinely absent

`diagnose_olmo_ffn_memory_geometry_v2.py`, `allenai/OLMo-2-0425-1B`, real activations from `data/distill/general_realtext/train.jsonl` (512-token sample), CPU, `paradoxe-27`, ~2 min.

**A1 (score-scale geometry, real queries)** -- per layer (0, 4, 8, 11, 15), `q = q_proj(hidden)`, `k_attn = k_proj(hidden)` (512 real per-token keys), `k_ffn = W_gate` (8192 static rows), scores scaled by `1/sqrt(head_dim)` on both sides for a like-for-like comparison:

| layer | \|\|k_attn\|\| | \|\|k_ffn\|\| | score_attn (mean/p99/max) | score_ffn (mean/p99/max) | unified-softmax mass on FFN |
|---|---|---|---|---|---|
| 0 | 15.8 | 2.36 | 0.26 / 1.07 / 2.65 | -0.01 / 0.18 / 0.65 | **92.4%** |
| 4 | 16.6 | 1.92 | 0.14 / 0.83 / 9.88 | -0.00 / 0.16 / 0.67 | **93.2%** |
| 8 | 21.9 | 1.83 | 0.39 / 1.42 / 84.7 | 0.00 / 0.21 / 0.87 | 75.8% |
| 11 | 27.4 | 1.74 | 0.58 / 2.46 / 31.2 | 0.00 / 0.25 / 1.32 | 82.0% |
| 15 | 70.3 | 1.49 | 6.50 / 27.8 / 83.8 | -0.01 / 0.53 / 1.53 | **0.29%** |

**Read**: this is the real, decisive answer A1 v1 couldn't give. `k_attn` norms and scores **grow sharply with depth** (norm 15.8->70.3, score mean 0.26->6.50, max up to 84.7 by layer 15) while `k_ffn` scores stay essentially flat and near-zero at **every** depth (mean ~0, max never above 1.5). Concatenated under one softmax, this produces a wildly depth-inconsistent mixing ratio that has nothing to do with relevance: **FFN keys capture 92-93% of the softmax mass in early layers** (winning purely on count -- 8192 keys at near-zero score each still out-accumulate 512 keys at slightly-less-near-zero score) and then **collapse to 0.29% by layer 15** (attention's few huge-magnitude scores become winner-take-all). **This is exactly the "mass vs. peak" risk spec Sec.3.2 flagged, now confirmed as a real, large, depth-dependent effect on an actual pretrained model -- not a hypothetical.** Directly supports keeping **branch (i)** (separate reads with source-appropriate kernels, e.g. `relu`/`sigmoid` for the KB, softmax for the sequence, per the plan's already-stated default) rather than (ii) (naive unified softmax), or at minimum a per-source scale calibration before any unified softmax is attempted.

**A2 (nearest-neighbor substitutability)** -- adjacent-layer pairs, distant pairs, and intra-layer self-control (self-match excluded):

| pair type | mean(max_cos) | frac > 0.5 | frac > 0.7 | frac > 0.9 |
|---|---|---|---|---|
| adjacent (avg of 15 pairs) | 0.12-0.19 | 0.09-1.2% | ~0-0.06% | ~0% |
| distant (0-15, 0-8, 7-15, 3-12) | 0.084-0.091 | **0.0%** | 0.0% | 0.0% |
| intra-layer self-control (0,8,15) | 0.20-0.21 | 0.5-4.0% | 0-0.8% | 0-0.26% |

**Read**: this is a properly-posed substitutability tail statistic, and it's clean -- **near-duplicate FFN keys are genuinely rare everywhere**, including within a single layer's own 8192 keys (the intra-layer control, which should show the *highest* self-similarity of any comparison, still has essentially 0% of keys with a near-duplicate at `cos > 0.9`). Adjacent layers are mildly more self-similar than distant ones (0.12-0.19 vs 0.084-0.091 mean max-cosine) -- a small, consistent signal in the direction Geva's stratification predicts, but nowhere near "near-duplicates exist to exploit." **This reading survives the correction and can be kept**: a shared/universal-layer memory (S3) cannot be built by deduplication or nearest-neighbor merging -- any real compression there would need to be learned (distillation/projection), not found for free in the raw key geometry.

Full numbers: `runs/olmo_ffn_geometry/result_v2.json`.

## 2026-09-19/20 (nuit) — S3 curriculum étape 1 + S0: S0 exact (PASS parfait), étape 1's attribution signal absent (no-op query)

Executed on `paradoxe-27` (CPU, no GPU) on behalf of `model-design` (code written/validated by them, this session ran it on the cluster per their resource constraint), `diagnose_layer_source_attribution.py --layers 0,8,15` and `convert_ffn_to_kv.py --layers 0,4,8,11,15` against real `allenai/OLMo-2-0425-1B` weights.

**S0 (exact SwiGLU FFN -> gated-attention conversion): PASS, bit-for-bit, all 5 layers tested** -- `max_abs_err=0.0`, `max_rel_err=0.0` at layers {0,4,8,11,15}, no exceptions. Confirms the gated bilinear form (`silu(x.gate_i) * (x.up_i)`) reconstructs OLMo-2's real SwiGLU MLP exactly, not approximately. Harness correct per the plan's own decision gate.

**Étape 1 (does a real query naturally retrieve its own layer's FFN region in a union memory, no explicit tag): no signal, either query mode.** `mass_on_own_region` sits at ~0.330-0.341 for `identity` (Q=x) and ~0.326-0.335 for `q_proj`, both statistically indistinguishable from the shuffled-label chance control (~0.333-0.334) and the uniform floor (1/3). No layer, no query mode shows separation above noise. Full numbers in `runs/olmo_ffn_geometry/layer_source_attribution.json` and `s0_conversion_check.json` (Rennes home) -- relayed to `model-design` for the read/decision on next steps (per their own stated decision table: weak/absent signal -> do not chain into étape 2/3 as-is, revisit the labeling with them before investing training compute).

## 2026-09-19/20 (nuit) — S3 curriculum, layer-depth linear probe: signal is present in x_l, just not via a raw dot product against gate_proj

Executed on `paradoxe-2` (CPU, Rennes -- `paradoxe-27`'s job expired mid-session, moved without loss) on behalf of `model-design`: `diagnose_layer_linear_probe.py --layers 0,8,15`, real OLMo-2-1B activations, 10844 tokens/layer pooled from 64 text shards, single `nn.Linear` probe, 70/30 train/test split.

**Result: `test_acc=0.9997` vs `chance_floor=0.3333`** (train_acc=1.0000, 22773 train / 9759 test tokens) -- clears the script's own `chance+0.15` bar by a huge margin. **Reading (b) confirmed: layer-depth information IS present in `x_l`, essentially perfectly linearly separable -- it just isn't accessible via a raw, untrained dot product against `gate_proj` rows (étape 1's negative result was about that specific untrained-geometry test, not about whether the information exists at all).** A learned per-layer/depth signal is viable -- `model-design`'s own framing, relayed for their read/decision on next steps (regime 4's content-addressed tag stays alive per this result, per their own decision table).

Full result in `runs/olmo_ffn_geometry/layer_linear_probe.json` (Rennes home).

## 2026-09-19/20 (nuit) — S3 linear probe, norm-artifact control: directional signal confirmed, not a scale artifact

Re-run per `model-design`'s extension (norm-artifact control, motivated by A1's earlier `||k_attn||` depth-growth finding): `diagnose_layer_linear_probe.py --layers 0,8,15` now trains the probe on both raw `x_l` and L2-normalized `x_l` (unit norm, direction only). Executed on `paradoxe-2`, same 10844 tokens/layer as the previous run.

**`raw`: test_acc=0.9998. `l2_normalized`: test_acc=0.9995.** Both clear chance (0.3333) by the same huge margin -- the signal survives removing magnitude entirely. **Verdict: directional structure confirmed, not a norm/scale artifact** -- a learned, dot-product-compatible per-layer tag (regime 4) is genuinely viable, not just an accident of `||x_l||` growing with depth. Full JSON in `runs/olmo_ffn_geometry/layer_linear_probe.json` (Rennes home, overwritten).

## 2026-09-19/20 (nuit) — S0.5 complete: full-layer composition (real attn + converted FFN) exact on all 5 layers tested

Executed on `paradoxe-5` on behalf of `model-design`: `check_layer_composition_exact.py --layers 0,4,8,11,15` -- verifies S0's exact FFN conversion, embedded in OLMo-2's real Post-Norm residual structure with the real (unconverted) self-attention, reconstructs the full decoder layer output bit-for-bit, not just the isolated FFN.

**PASS, `max_abs_err=0.0` / `max_rel_err=0.0` on all 5 layers, no exceptions.** Full-layer composition introduces no silent bug -- confirms the checkpoint needed before Phase 12's étape 2 (composing a converted FFN-KB read with a sequence-side read inside one Thinker-style layer) can be trusted. Full JSON in `runs/olmo_ffn_geometry/s0_5_layer_composition_check.json` (Rennes home).

## 2026-09-19/20 (nuit) — S3, 16-layer extension of attribution + linear probe: same pattern holds at full stratification

Executed on `paradoxe-5` (P1, no dedicated reservation): both étape-1 diagnostics re-run across all 16 OLMo-2-1B layers (not just the 0/8/15 sample).

**Attribution (raw dot product, `diagnose_layer_source_attribution.py`)**: `mass_on_own_region` stays within ~0.061-0.064 of the `uniform_floor=0.0625` at every layer, both query modes -- essentially at chance everywhere, matching the 3-layer result. Slight departures at layers 14-15 under `identity` mode (0.0637/0.0643 vs 0.0625 floor) are the only points worth a second look, still tiny relative to the probe's signal below.

**Linear probe (`diagnose_layer_linear_probe.py`, 16-way classification)**: `raw` test_acc=0.9918, `l2_normalized` test_acc=0.9589, both against `chance_floor=0.0625` (121453 train / 52051 test tokens, 10844/layer). **Confirms the 3-layer finding at full stratification**: layer identity is strongly, directionally decodable from `x_l` across all 16 layers, not just a coarse low/mid/high split -- some expected softening at 16-way vs 3-way (l2_normalized drops from 0.9995 to 0.9589, still enormous vs chance) is consistent with finer-grained classes being harder, not a qualitative change in the finding.

Full JSON: `runs/olmo_ffn_geometry/layer_source_attribution_16layers.json`, `layer_linear_probe_16layers.json` (Rennes home).

