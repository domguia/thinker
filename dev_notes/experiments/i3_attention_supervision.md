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

