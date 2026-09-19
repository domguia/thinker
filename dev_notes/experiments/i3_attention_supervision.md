# I3 — supervision d'attention (étape 1 LR sweep, étape 2 comparaison des bras)

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-19/20 (nuit) — I3 étape 1 complete: attention supervision resolves hard-hops chaining, but bimodal -- success rate drops sharply with LR

`i3_attnsup_lrsweep`, `train_kb_chain_attn_supervised.py --attn_supervised --supervise node`, `n_hops=3, n_distractors=5` (hardened generator), `depth=2, block_size=4, d_model=256, n_step=12`, `lr` in `{1e-4, 2e-4, 3e-4, 4e-4}` x 3 seeds, `abacus11` (2x A5000), ~15 min/cell, 12/12 done.

| lr | seed0 | seed1 | seed2 | successes |
|---|---|---|---|---|
| 1e-4 | 99.84% | **17.99%** | 99.18% | 2/3 |
| 2e-4 | **23.16%** | 99.22% | 98.32% | 2/3 |
| 3e-4 | 29.30% | 30.86% | 98.77% | 1/3 |
| 4e-4 | 26.74% | 24.43% | 28.22% | 0/3 |

**Read**: every cell lands cleanly in one of two bands -- resolved (~98-100% acc) or collapsed (~18-31%, near the failed/no-supervision range) -- no intermediate outcomes, a textbook bimodal signature already seen elsewhere in this project (`pool_n_head`, `kdim128_decoupled`). **The success RATE, not just the absolute performance level, degrades monotonically with `lr`**: 2/3 at `1e-4` and `2e-4`, 1/3 at `3e-4`, 0/3 at `4e-4`. Qualitatively, this answers the open question I3 was launched for: **attention supervision does resolve hard-hops chaining (`n_hops=3`) when it lands in the right basin** -- it is not architecturally incapable at this scale -- but **no LR tested here is reliable** (best is 2/3 seeds, i.e. a 1/3 failure rate even at the safest point tested).

**Per the standing rule (bimodality found -> needs >=5 seeds to trust a cell, not 3)**: this result on its own cannot yet support "supervision reliably beats no-supervision at `n_hops=3`" for étape 2's comparison -- 1/3 to 2/3 seeds succeeding is exactly the regime that rule exists to catch. **À arbitrer**: whether to (a) extend the LR sweep further down (`3e-5`, `5e-5` -- the failure rate trend suggests lower might be safer, matching this project's repeated "narrow, low, non-monotone LR window" pattern) before committing étape 2's LR, or (b) accept the current best (`1e-4`/`2e-4`, both 2/3) and run étape 2 with enough seeds (>=5, ideally more given the observed 1/3 failure rate) to average over the bimodality honestly. Proceeding with **(a) first** since it's cheap and GPU is available -- extended-low-LR cells queued on `abacus11`.

**[Confirmed direction + stopping rule, `model-design`]**: continuing the low-LR extension is the right call -- a monotonically-decreasing success rate as `lr` increases is the same signature `pool_n_head`'s fine sweep showed before it eventually found a narrow stable window, so the window is plausibly below `1e-4`. **Stopping rule, to bound the search**: if no `lr` in the extended sweep reaches `>=4/5` seeds succeeding, stop looking and report the **success rate itself** as the finding ("supervision resolves hardened `n_hops=3` in k/n seeds, cleanly bimodal, no LR window tested is fully reliable") rather than continuing to hunt for a perfect window -- that is a real, publishable result on its own, an indefinite hunt is not, and it costs GPU needed elsewhere.

**Also: don't lose the actual question I3 was launched to answer.** The bimodality measured so far is in the **supervised arm alone** -- it says nothing yet about redundant-vs-necessary without the **unsupervised arm at the same LRs**. Étape 2 must run **both arms** (`--attn_supervised` on and off) at the same `lr` values and seed count, and compare **success RATES** (fraction of seeds resolving the task), not mean accuracies -- a mean across a bimodal distribution is not a meaningful summary statistic here.

**Extension complete (26/26), stopping rule triggered -- reporting the success rate, not chasing a perfect window further**:

| lr | 2e-5 | 5e-5 | **1e-4** | **2e-4** | 3e-4 | 4e-4 |
|---|---|---|---|---|---|---|
| n seeds | 5 | 5 | 5 | 5 | 3 | 3 |
| success rate | **0/5** | 0/5 | **2/5 (40%)** | **2/5 (40%)** | 1/3 | 0/3 |

**Correction to the "lower is safer" hypothesis this extension was launched to test**: it's falsified -- `2e-5` and `5e-5` (the newly-added, lower values) do *worse* than `1e-4`/`2e-4`, not better (0/5 success at both, versus 40% at the window's center). The success rate actually peaks at `1e-4`-`2e-4` and degrades on **both** sides -- a genuinely bracketed window this time, not an open-ended search, but its peak reliability is only 40%. **No `lr` reached the `>=4/5` bar** -- per the stopping rule above, halting the LR hunt here rather than probing further (e.g. `1.5e-4`) and reporting this as the finding: **hardened `n_hops=3` attention supervision resolves the task in at most 2/5 seeds at its best LR window (`1e-4`-`2e-4`), cleanly bimodal (resolved ~98-100% or collapsed ~18-31%, nothing between), with no LR tested giving reliable success.** This is the number to carry into étape 2, not a single "best" accuracy.

## 2026-09-19/20 (nuit) — I3 étape 2 complete (70/70): supervision looks harmful, but a real script discrepancy against item[5]'s baseline needs checking before trusting this

`i3_step2_bothArms` (Rennes -- `abacus11-1/18-1/25-2/29-1` relief job, plus `paradoxe-27` CPU for the 3 orphaned/reclaimed cells), `train_kb_chain_attn_supervised.py`, `attn_supervised` on/off, `n_hops` in `{3,4}`, `lr` in `{1e-4,2e-4}` (the étape 1 window), `n_step=12`, 14-20 seeds/cell depending on `n_hops`/`lr` combination (grid grew across several repowering passes tonight, seed counts not perfectly even but all comfortably >=14).

**Success rate (`final_acc > 0.5`, well clear of the resolved/collapsed bimodal bands seen throughout this thread) by `n_hops` x `attn_supervised`**:

| n_hops | unsupervised (`False`) | supervised (`True`) |
|---|---|---|
| 3 | **13/14 (93%)** | 3/16 (19%) |
| 4 | 2/20 (10%) | 0/20 (0%) |

**Read, with a major caveat attached**: taken at face value, this is a stark reversal of étape 1's framing -- supervision looks actively *harmful* here, not just unreliable, at both `n_hops`. **But the `n_hops=4` unsupervised number (2/20, 10%) directly contradicts `item[5]`'s result from earlier tonight at the exact same hyperparameters** (`n_hops=4, n_step=12, lr=1e-4`, `train_kb_chain.py`, no supervision path at all): item[5] got **99.9% on all 3 seeds** at that identical config. Same `n_hops`, same `n_step`, same `lr`, same hardened generator -- 99.9%/3-for-3 in one script, 10%/2-of-20 in the other. **This gap is too large to be seed variance** -- it points to a real difference between `train_kb_chain.py`'s training path and `train_kb_chain_attn_supervised.py`'s `attn_supervised=False` path (default hyperparameters not actually identical, a different code path even when the flag is off, or a bug in this script's baseline arm), not a genuine architectural effect of "having the supervision machinery present but disabled."

**Not reporting "supervision hurts" as a finding until this is resolved.** **À arbitrer / next step**: diff `train_kb_chain_attn_supervised.py`'s `attn_supervised=False` path against `train_kb_chain.py` for any silent default mismatch (e.g. optimizer, init, an extra loss term still active at weight 0 vs literally absent, a different default for an unlisted flag) before trusting either the `n_hops=3` (93% vs 19%) or `n_hops=4` (10% vs 0%) comparison as answering I3's original "is supervision redundant or necessary" question. The `n_hops=3` split (93% vs 19%) is at least directionally consistent with étape 1's own finding that supervision at `n_hops=3` is unreliable (peaked at 40% success in étape 1's own sweep) -- so the supervised-arm numbers here are plausible on their own. It's specifically the **unsupervised arm's implausibly low `n_hops=4` number** that doesn't match the already-trusted `item[5]` baseline and should block any headline claim until explained.

Full per-cell results in `runs/i3_step2_bothArms/state/*.json` (Rennes home).

## 2026-09-20 — Diagnostic complete: default-hyperparameter mismatch (most likely cause) + an independent, unrelated fidelity bug fixed

Full `diff` of `train_kb_chain.py` against `train_kb_chain_attn_supervised.py` (`model-design`, CPU, local, zero cluster cost -- per the plan's own "diagnostic bloquant, zéro compute" classification).

**Most likely root cause of the 10%-vs-99.9% gap: the two scripts have materially different DEFAULTS for architecture/training hyperparameters that i3_step2_bothArms's grid never explicitly overrode.** The grid swept only `n_hops`, `lr`, `n_step` (per the entry above) -- everything else fell back to each script's own `argparse` default:

| flag | `train_kb_chain.py` default | `train_kb_chain_attn_supervised.py` default |
|---|---|---|
| `--d_model` | 256 | **128** |
| `--depth` | 3 | **2** |
| `--n_head` | 4 | **1** |
| `--batch_size` | 64 | 256 |
| `--max_steps` | 200000 | 20000 |

If these weren't matched explicitly in the grid config, the "unsupervised" arm of `train_kb_chain_attn_supervised.py` trained a **single-head, one-fewer-hierarchy-level, half-width** model relative to `item[5]`'s `train_kb_chain.py` baseline -- `n_head=1` alone is a severe capacity cut for a 4-hop composition task, plausibly enough on its own to explain a collapse from 99.9% to ~10%, with nothing to do with supervision at all. **Not yet confirmed against the actual `i3_step2_bothArms/grid.jsonl` config on the cluster** (this diagnosis was done from the scripts alone, not from re-reading the dispatched grid file) -- confirming that read is the one remaining step before treating this as settled, but the defaults themselves are unambiguous and the hypothesis is well-supported.

**Independent finding, unrelated to the above, fixed regardless (commit `d789164`)**: `forward_with_optional_supervision()` (the manual unroll of `Thinker.forward()`'s loop, needed to insert the auxiliary loss mid-loop) claimed to implement "the same equations as `core/indexed_thinker_model.py`" but silently omitted two of `Thinker.forward()`'s branches -- the `detach_sm_keys` stop-gradient on new SM keys, and the `sm_cap` trim to the last N SM entries. Both flags default to off/None and weren't part of I3's swept axes, so this did **not** cause the specific 10%-vs-99.9% gap -- but it's a real, dormant correctness gap that would have silently made `--detach_sm_keys`/`--sm_cap` no-ops through this script. Fixed to match `Thinker.forward()` exactly; verified with a smoke run (`--detach_sm_keys --sm_cap 2`, CPU, 5 steps, runs end-to-end without error) and the full test suite (76/76 green, this file isn't imported by any existing test).

**Corrected re-run for I3 étape 3, once the grid-config read confirms the mismatch** -- explicitly match every architecture/training flag to `item[5]`'s `train_kb_chain.py` config rather than relying on defaults:
```
tools/exp/gridgen.py --out runs/i3_etape3/grid.jsonl \
  --script learn/indexed_attention/train_kb_chain_attn_supervised.py \
  --fixed n_hops=4 depth=3 block_size=4 d_model=256 n_head=4 batch_size=64 n_step=12 max_steps=20000 max_time_minutes=30 \
  --sweep attn_supervised=true,false lr=1e-4,2e-4 seed=0,1,2,3,4
```
(`max_steps=20000` kept at the supervised script's own budget rather than `train_kb_chain.py`'s 200000 -- item[5]'s 99.9% was reached well within 20000 steps per its own log entry, so this isn't expected to reintroduce the gap; flag if étape 3's unsupervised arm still disagrees with item[5] after this fix, since that would point to a genuine code-path difference instead.)


## 2026-09-20 — i3_etape3 launch bug found and fixed: n_distractors missing from --fixed

First launch of `i3_etape3` (the n_head-isolation grid) crashed all 20 cells immediately (rc=1, ~2-3s each): `AssertionError: max_facts=6 (-> 24 leaves) is not a multiple of block_size=4 ** depth=2 = 16`. Cause: `n_distractors` was not in the gridgen `--fixed` list, so the script's own default kicked in, giving `max_facts=n_hops+n_distractors=4+2=6`, not divisible per the hierarchy's leaf-count constraint. item[5]/item6's own grids always set `n_distractors=4` explicitly for `n_hops=4` -- this was silently omitted here. Fixed by adding `n_distractors=4` to `--fixed`, regenerated the grid, relaunched -- confirmed running (not crashing) on the second attempt. No compute wasted (crashed in seconds, caught via log check before drawing any conclusion from it), but ~15 min of wall-clock lost to the failed-launch-not-yet-checked window -- reinforces the standing "verify a launch actually produced real progress before moving on" discipline.
