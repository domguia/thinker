# N_step / LR law (item[5] plateau, item[8] extension n_hops 5-6)

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-19/20 (nuit) — item [5] complete (45/45): N_step plateau confirmed cleanly, LR law is a ceiling not a smooth 1/N_step decay

`item5_nstep_lrlaw` (Nancy, `graffiti-3`, 4x RTX 2080 Ti), `train_kb_chain.py`, hardened `n_hops=4, n_distractors=4, d_model=256`, `n_step` in `{2,4,6,8,12}` x `lr` in `{1e-4,3e-4,1e-3}` x 3 seeds, trained directly at each `n_step` (not an eval-time generalization probe like the retracted I7 -- no training-margin confound here, each cell is genuinely trained at its own `n_step`).

**Accuracy vs `n_step`, at each cell's best `lr`**:

| n_step | 2 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| best acc | 89.7% | **99.9%** | 99.8% | 99.9% | 99.8% |

**Read (N_step plateau)**: clean and exactly as the central thesis predicts. `n_step=2` is below the 4 real hops needed and correctly can't fully solve the task (89.7%, still far above chance but not saturated). **From `n_step=4` onward (matching the hop count) accuracy saturates near 100% and stays there flat through `n_step=12` (3x the needed count) -- no degradation from extra iterations.** This is the training-time version of the plateau I7 tried and failed to show cleanly (I7's checkpoint had a huge training margin baked in, confounding the read); this result has no such confound since every cell here is actually trained at its stated `n_step`. **This is the cleanest, most direct confirmation tonight that the loop's iteration count is not a fragile, fixed quantity -- the model trained with more steps than strictly needed doesn't get worse, matching the "iteration count can be free" thesis.**

**Full table, all lr**:

| n_step | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---|---|---|---|
| 2 | 84.6% | **89.7%** | 83.5% |
| 4 | **99.9%** | 99.8% | 14.4% |
| 6 | **99.8%** | 99.7% | 14.8% |
| 8 | **99.9%** | 99.7% | 14.8% |
| 12 | **99.8%** | 99.7% | 12.4% |

**Read (LR "law")**: **not** a smooth `lr ~ 1/n_step` decay as hypothesized after the Piste A finding -- the best `lr` stays at `1e-4` (with `3e-4` a close second) from `n_step=4` all the way to `n_step=12`, it doesn't keep shrinking as `n_step` grows further past the hop count. What the data actually shows is a **ceiling effect**: `lr=1e-3` is fine-ish at `n_step=2` (83.5%, only slightly worse than the other two) but **collapses catastrophically once `n_step>=4`** (12-15% at every `n_step` from 4 to 12, near the `1/vocab_size`-ish floor) -- consistent with Piste A's finding that looped configs need a substantially lower LR than flat ones, but the mechanism here reads more like "there's a stability boundary between `3e-4` and `1e-3` that appears once the loop is deep enough to matter" than a continuously-decreasing optimum. **Correcting the earlier prediction**: report this as a threshold, not a `1/n_step` power law -- the data doesn't support the smoother hypothesis as stated.

## 2026-09-19/20 (nuit) — item [8] complete (18/18): LR ceiling replicates and sharpens at n_hops=5/6

`item8_nhops56` (Nancy `graffiti-3`, 4x RTX 2080 Ti), `train_kb_chain.py`, hardened generator, `n_hops` in `{5,6}` (with `n_distractors={3,2}` resp., `max_facts=8` both), `d_model=256, depth=2, block_size=4, n_step=12`, `lr` in `{1e-4,3e-4,1e-3}` x 3 seeds, `max_time_minutes=20`.

| n_hops | lr=1e-4 | lr=3e-4 | lr=1e-3 |
|---|---|---|---|
| 5 | **99.7% / 99.9% / 99.7%** (3/3) | 25.1% / **99.1%** / 24.7% (1/3) | 2.4%* / 10.6% / 14.6% (0/3) |
| 6 | 33.8% / **98.9%** / **99.4%** (2/3) | 33.1% / **99.1%** / 32.5% (1/3) | 28.3% / 28.7% / 10.1%* (0/3) |

(*below chance=12.5%, `margin` negative -- a real collapse, not just "didn't finish".)

**Read: directly replicates and sharpens item[5]'s LR-ceiling finding at harder `n_hops`.** `lr=1e-4` is the clear safe point (3/3 at `n_hops=5`, 2/3 at `n_hops=6`, and the successes are near-saturated, 98.9-99.9%). `lr=1e-3` **fails completely at both `n_hops`** -- 0/3 each, several cells landing at or below chance level (not just "slow", genuinely collapsed) -- consistent with item[5]'s finding that `lr=1e-3` becomes catastrophic once the loop is deep enough to matter, now confirmed to get worse (not better) as `n_hops` increases. `lr=3e-4` is the same bimodal middle ground seen throughout this project (1/3 success at both `n_hops`, no partial-credit intermediate outcomes) -- consistent with the `pool_n_head`/I3-style bimodality signature, not a new phenomenon.

**No new architectural signal here** (this was a confirmation/extension run, not a new mechanism question) -- the value is in cementing "lr=1e-4 is the correct default for this scale/depth regime, `1e-3` is not a viable option past `n_hops>=4-5`" as settled going forward, backed now by 4 different `n_hops` values (2,3,4 elsewhere + 5,6 here) all showing the same qualitative pattern.

Full grid + raw curves in `runs/item8_nhops56/state/*.json` (Nancy home).


## 2026-09-19/20 (nuit) — item[6] étape 4 more-seeds complete (15/15): n_hops 2/3/4 all clean at 5 additional seeds

`item6_etape4_moreseeds` (hardened generator, `n_hops` in {2,3,4}, seeds 3-7, `lr=3e-4`), Rennes, all 15 done.

| n_hops | seeds 3-7 |
|---|---|
| 2 | 99.98-100% (margin 0.875 all) |
| 3 | 99.98-100% (margin 0.875 all) |
| 4 | 99.7-99.9% (margin 0.872-0.875) |

**Read: extends the original 3-seed Étape 4 result to 8 total seeds per `n_hops`, no exceptions, no bimodality.** Confirms the hardened generator's multi-hop chaining result is not a small-n artifact -- real, robust across seeds at all three `n_hops` values. Full grid in `runs/item6_etape4_moreseeds/state/*.json` (Rennes home).
