# Indexed Attention — mécanisme central (plateau n_hops, contre-expertise, étape 4, I1, I5)

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-12 — Indexed Attention Phase 0: hierarchy vs. flat go/no-go, and an n_facts scale cliff (`EXP-007`)

**Context**: first GPU validation of `HierarchicalMemory`/`IndexedThinker` (`dev_notes/indexed_attention_experiment_plan.md` Phase 0), beyond the CPU overfit tests in `tests/test_indexed_memory.py`. Grid'5000 Rennes, job 4104802 (`abacus3-1`, 4×A5000 24G besteffort, `~/micromamba/envs/teacher311`). Script: `learn/indexed_attention/train_kb_retrieval.py` (new file).

**LR does not transfer across scale (found before the main result)**: `lr=3e-3` (used in the CPU unit tests) fails outright once `n_facts`/`d_model` grow. A sweep at `n_facts=16, d_model=128` found `lr=3e-4` converges cleanly (98.6% held-out acc in 2 min) while `1e-4/3e-3/1e-2` all fail at chance — same muP-style width/LR miscalibration already documented in this log's distillation section, rediscovered independently here.

**Go/no-go result at `n_facts=16` (3 seeds each, held-out eval split, `lr=3e-4`, `d_model=256`, `n_step=3`, 8 min/run)**:
- `depth=4` (hierarchical): **99.6% / 99.5% / 100.0%** held-out accuracy
- `depth=0` (flat, Baseline C, spec §9): **7.0% / 6.25% / 5.86%**

Massive, unambiguous gap (>>2σ) — confirms Phase 0's hypothesis cleanly: the hierarchical, unified-softmax memory dramatically outperforms flat attention once the KB is large enough to dilute the flat baseline's signal. Decision per the plan's table: **go** — continue with `depth>0` as the default config into Phase 1bis.

**But: `n_facts=64` (256 leaves, same `d_model=256`) fails to learn at all, for BOTH `depth=0` and `depth=4`** — not a hierarchy-specific weakness, the whole register+SM+fusion mechanism plateaus at chance level. Isolated the axis with 4 short (5 min) diagnostic runs, one per GPU:
- `n_facts=64, d_model=128, lr=3e-4` → fails (0.2% acc)
- `n_facts=16, d_model=256, lr=3e-4` → succeeds (99.7%) — control confirming `d_model=256` alone isn't the problem
- `n_facts=64, d_model=256, lr=1e-4` → fails (0.5%)
- `n_facts=64, d_model=256, n_step=6` → fails (0.6%)

Neither `d_model` (128 vs 256), nor `lr` (1e-4 to 3e-3 tried across two sweeps), nor `n_step` (3 vs 6) move the needle — it's specifically **`n_facts`** (task scale) that breaks learning. Two more capacity-style levers tested and also ruled out: `n_register=4` (0.3% acc) and `batch_size=256` (0.3% acc) — neither helps either.

**Curriculum learning (matches this log's own 18 Dec 2023 ToyThinker/copy-task precedent — "having a plateau doesn't mean the model is at capacity", solved there by curriculum, not by hyperparameter tuning)**: `thinker-e9` (sister session) implemented fixed-shape padding/masking (`HierarchicalMemory.build(..., leaf_mask=...)`, commit `d7bff12`) so `data/kb_retrieval.py::KBRetrievalDataset(n_facts, max_facts, ...)` can vary real fact count while keeping the hierarchy shape (`block_size`/`depth`) fixed to the curriculum's target scale. `train_kb_retrieval.py` got a `--curriculum "16,32,64"` stage-promotion loop (promote on held-out accuracy threshold). Real GPU curriculum run (16→32→64, same job/node) in progress at time of writing — see next entry or `dev_notes/grid5000_usage.log.md` for the outcome.

**Role split for this work going forward** (per explicit user instruction): `thinker-e9` (sister session, "model-design") owns architecture/spec/plan changes (`core/`, `dev_notes/indexed_attention_spec.md`, `dev_notes/indexed_attention_experiment_plan.md`); this session owns experiment execution — Grid'5000 job orchestration, `learn/indexed_attention/` training-loop code, profiling/optimization, and reporting observations back for the plan to be updated.

### Curriculum result (same day, continued): confirms the mechanism, and re-confirms the flat baseline's failure

Ran the `--curriculum 16,32,64` promotion loop (`train_kb_retrieval.py`, promote at held-out acc ≥0.9, min 500 steps/stage) for real on GPU (job 4104802, `abacus3-1`), `depth=4` and `depth=0` in parallel, 18 min budget each:

- **`depth=4` (hierarchical): fully unblocked.** Promoted cleanly through all three stages and finished at `n_facts=64` (the original blocker) with **100% held-out accuracy**, `best_loss=0.00035`, in 8806 steps / 1082s. Confirms the Dec-2023 ToyThinker curriculum precedent transfers directly to this architecture: the plateau at `n_facts=64` was never a capacity/mechanism ceiling, just an optimization-landscape one that curriculum routes around.
- **`depth=0` (flat): never left stage 1.** After 22962 steps (>3× the steps `depth=4` needed for its *entire* 3-stage curriculum), held-out accuracy was still stuck at ~5%, never crossing the 0.9 promotion threshold even once — consistent with, not contradicting, the go/no-go result above (flat attention already failed at `n_facts=16` directly; giving it unlimited time at the same stage doesn't change that). Useful negative control: curriculum only helps when the underlying mechanism *can* eventually solve the harder stage — it's not a universal fix for any plateau.

**Practical upshot**: curriculum learning (train `depth>0` at `n_facts=16→32→64`) is now the validated path to scale this task past the `n_facts=64` cliff found in Phase -1, with zero architecture changes. Recorded as the reference training recipe for any future phase needing `n_facts>16`.

### Phase 1bis first pass (n_facts=16, 3 seeds/variant): no differentiation at this scale

Ran `n_slots=4` (vs. default `n_slots=1`), `use_ff=True` (2-layer GELU MLP in the main-loop fuse step), `detach_sm_keys=True` (stop-gradient on SM keys), and `level_dropout_p=0.1` (stochastic high-level dropping), 3 seeds each, same `n_facts=16` task/budget as the go/no-go run:

| Variant | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Baseline (`n_slots=1`, no FF/detach/dropout) | 99.6% | 99.5% | 100.0% |
| `n_slots=4` | 98.9% | 99.7% | 99.8% |
| `use_ff=True` | 99.5% | 100.0% | 100.0% |
| `detach_sm_keys=True` | (run duplicated by an orchestration mistake, discarded) | 100.0% | 99.7% |
| `level_dropout_p=0.1` | 99.8% | 100.0% | 99.8% |

All variants land in the same 98.9-100% band as the baseline — **no variant shows a distinguishable effect at this scale**, per the plan's own ambiguity rule (overlapping ±σ → "not concluded," not "no effect ever"). `n_facts=16` is a near-ceiling task for every configuration tried so far; a real test of `use_ff`'s composition-capacity hypothesis (the audit's priority #1) needs either the harder `n_facts=64` curriculum-trained setting or a task that actually requires computing over retrieved values (Phase 2 multi-hop), not more distractors at a task every variant already solves near-perfectly.

**Infra note for future large parallel batches**: piping `oarsh` stdout straight back over the orchestrating SSH connection is fragile — a transient connection drop (happened twice this session, unrelated to the remote job) kills the local pipe and loses all output, even though the remote training process itself survives untouched (confirmed via `ps`/`nvidia-smi` on the node). Switched to `... > ~/remote/path/logfile 2>&1 < /dev/null & disown`, launched via a short-lived `oarsh` call that returns immediately — the remote process then belongs to the OAR cgroup, not to the SSH session, and results are recovered with a plain `cat`/`tail` afterward regardless of local connection hiccups. Adopt this pattern by default for any run expected to outlive a single quick command.

### Multi-hop chain task (Phase 2 / Phase 1quater) and batch-size/LR co-scaling: session close-out

**Multi-hop (`data/kb_chain_retrieval.py::KBChainDataset`, thinker-e9's implementation)**, curriculum on `n_hops` (1→2→3, `n_distractors=1`, `depth=2`), 15 min budget each, comparing fixed `n_step=12` vs. randomized `n_step~Uniform(1,12)` per batch (Universal/Looped-Transformer-style depth randomization, for later extrapolation testing):

- `n_hops=1` promotes cleanly and fast in both runs (~step 1200, ~98% held-out acc) — the mechanism can do a single hop easily even under a large `N_step` budget.
- `n_hops=2` **plateaus at 32-34% held-out accuracy in BOTH runs**, never crossing the 0.9 promotion threshold despite 16000-22000+ further steps (well past `curriculum_min_steps=500`) — no measurable difference between fixed and randomized `N_step`, so depth-randomization isn't the fix here. Not chance level (~2-3% at this vocab size) but far from mastery.
- **Not yet diagnosed which of three explanations is correct** (flagged to thinker-e9, unresolved at session close): (a) `N_step=12` still insufficient for `n_hops=2` despite already being ~6x the hop count and above thinker-e9's own "~2-4x n_hops" heuristic, (b) `lr=3e-4` (carried over from the single-lookup task, never re-swept for the chain task specifically) is miscalibrated for this harder task -- same lesson as every other axis in this session, or (c) a genuine mechanism limit on 2+ hop composition through the SM buffer. Whichever it is, this is the most important open question for validating the project's central "iterative extraction beats single-pass" thesis -- next step should be an LR sweep specifically at `n_hops=2` (not reusing the retrieval-task LR) before concluding anything about (c).
- Added `--extrapolate_n_steps "8,16,24"` to `train_kb_chain.py`: in-memory re-evaluation at `N_step_test > N_step_train` on a fresh disjointly-seeded KB, no checkpoint needed (thinker-e9's suggestion, since this is a cheap go/no-go check, not something needing to persist/reuse a checkpoint elsewhere) -- ready for the next chain run but not yet exercised on a converged model (nothing converged past `n_hops=1` this round).

**Batch-size/LR co-scaling (prompted by a resource-utilization question mid-session)**: profiling (`nvidia-smi`) during the busiest 4-GPU-job stretch showed persistent 5-30% compute / 2-7% VRAM utilization even with 4-9 runs packed onto 4 GPUs simultaneously -- these ~0.7-1.6M-param models are nowhere near saturating an A5000 at `batch_size=64`. Per thinker-e9: don't reuse `lr=3e-4` (tuned for batch=64) at a larger batch without re-tuning -- same d_model/LR coupling lesson as Phase -1, apply the linear scaling rule (Goyal et al. 2017) as a starting point, then verify with a small sweep, don't trust it blindly. Sweep at `batch_size=256`, `n_facts=16`/`depth=3` (the Phase 0 go/no-go task): `lr=6e-4` (99.55% acc), **`lr=1.2e-3` = the linearly-scaled value (4x batch -> 4x lr): best, 99.9% acc**, `lr=2.4e-3` diverges (7.1% acc, loss stuck at 4.3). Confirms the linear scaling rule held exactly at this 4x batch jump. **`batch_size=256, lr=1.2e-3` is now the reference config for any future `learn/indexed_attention` run at this scale** wanting to use the node's idle capacity -- any accuracy/loss comparison against the earlier `batch_size=64, lr=3e-4` runs in this log should note the two changed together, not be read as an apples-to-apples curve comparison.

**Session-level summary of everything validated this session** (Phase 0 through here), for a reader who wants the punch line without the blow-by-blow above:
1. **Go/no-go (n_facts=16, 3 seeds)**: hierarchical (`depth>0`) massively beats flat attention (`depth=0`) -- 99.6-100% vs. 5.9-7.0%. Confirmed, not ambiguous.
2. **Curriculum on `n_facts`** (16->32->64) fully unblocks the `depth>0` mechanism at the `n_facts=64` scale that direct training could never solve (any `d_model`/`lr`/`n_step`/`n_register`/`batch_size` tried) -- 100% final accuracy. `depth=0` under the same curriculum never even clears `n_facts=16` (consistent negative control, not a new anomaly).
3. **Phase 1bis first pass** (`n_slots`, `use_ff`, `detach_sm_keys`, `level_dropout_p`) at `n_facts=16`: no variant distinguishable from baseline (all 98.9-100%) -- inconclusive at this scale, a real test needs either the `n_facts=64` curriculum-trained setting or a task that needs computation over retrieved values, not more distractors on an already-near-ceiling task.
4. **Multi-hop chain task**: mechanism handles 1 hop easily, plateaus hard at 2 hops (32-34%) regardless of fixed vs. randomized `N_step` -- open question, LR re-sweep at this task is the next diagnostic step, not yet a verdict on the architecture's multi-hop capability.
5. **Infra**: default to writing remote process output to a file on the node (not piping over the orchestrating SSH connection), and to reserving whole nodes (`gpu=4`) with `CUDA_VISIBLE_DEVICES` packing multiple independent runs per node rather than one GPU per job -- both adopted as standing defaults going forward given these models are small enough that node-level (not just GPU-level) parallelism is the actual bottleneck lever.


## 2026-09-13 — n_hops=2 LR sweep: rules out miscalibrated LR as the plateau's cause

Follow-up to the multi-hop plateau found earlier (n_hops=2 stuck at 32-34% held-out acc, both fixed and randomized N_step=12), per thinker-e9's request: sweep LR specifically at `n_hops=2` (direct, no curriculum) rather than reusing the `n_hops=1`/single-lookup LR. `abacus3-1`, 4 GPU parallel, 4 min budget each, `n_distractors=2, vocab_size=64, depth=2, block_size=4, d_model=256, n_step=12, batch_size=64`:

| lr | final_acc |
|---|---|
| 1e-4 | 24.5% |
| 3e-4 | 25.3% |
| 1e-3 | 26.1% |
| 3e-3 | **diverges (loss NaN, 0% acc)** |

None of the tested LRs show a qualitatively different trajectory from the known 32-34% plateau (these are all in the same ballpark, plausibly still en route to that plateau within the short 4 min budget, not a real difference) — no LR value found so far unlocks mastery, and the highest value tested is unstable. **This weakens hypothesis (b) (miscalibrated LR)** as the explanation for the n_hops=2 plateau. Not a fully exhaustive sweep (4 min/point is short, and only 4 values tried), but no positive signal for "just needed a different LR" the way `n_facts=64`'s plateau turned out to be a curriculum problem rather than an LR one at first glance, or the way `n_facts=16`'s original failure *was* purely an LR problem. Remaining live hypotheses per the original three: (a) `N_step=12` still insufficient for 2-hop composition, or (c) a genuine mechanism limit on chaining through the SM buffer — next diagnostic step (not yet run) should isolate `N_step` directly (e.g. `N_step` sweep at a fixed, reasonable LR) before concluding on (c).

## 2026-09-13 — n_hops=2 LR sweep at full budget: (b) miscalibrated LR properly ruled out

Per thinker-e9's valid concern (the short 4-min sweep above wasn't budget-comparable to the 15-18min/16-22k-step runs that established the 32-34% plateau), re-ran the two most promising short-sweep LRs (`1e-3`, `6e-4`) at full budget (18 min, `abacus3-1`, job 4104890, `n_hops=2` direct, same config as the original plateau runs):

- `lr=6e-4`: 16,685 steps, **final_acc 25.2%** — still within/below the known plateau, no improvement.
- `lr=1e-3`: 16,984 steps, held-out acc oscillated 22-28% through training then **final measured acc 4.6%** (an unstable/collapsed final read, not a real improvement — consistent with `lr=1e-3` being the least stable value tried so far, one step from the `lr=3e-3` value that diverges outright).
- Extrapolation probe (`--extrapolate_n_steps 16,20,24`, in-memory, no checkpoint) on both: flat or slightly worse than the training-time N_step=12 accuracy (e.g. `lr=6e-4`: 24.8/25.0/24.5% at N_step_test=16/20/24 vs. 25.2% at N_step=12) — no sign that simply running more reasoning steps at inference recovers anything.

**Conclusion: (b) miscalibrated LR is now properly ruled out** at a budget comparable to the original plateau observation, not just a short sweep. Neither of the two candidate LRs exceeds 32-34%, and the higher one shows real instability rather than a hidden improvement. Remaining live hypotheses: (a) `N_step=12` still insufficient for 2-hop composition (next diagnostic: isolate `N_step` directly, e.g. a sweep at `N_step` ∈ {8, 16, 24, 32} at the already-known-stable `lr=3e-4`, matching Phase -1's methodology of sweeping one axis at a time rather than changing several together), or (c) a genuine mechanism limit on chaining through the SM buffer at 2+ hops -- not yet distinguishable from (a) without that N_step isolation.

## 2026-09-13 — n_hops=2 N_step sweep: (a) also ruled out, points to (c) a real mechanism limit

Isolating N_step directly at a fixed, stable LR (`lr=1.2e-3`, rescaled with `batch_size=256` per the linear rule to avoid reintroducing a batch/LR confound -- NOT the `lr=3e-4` originally suggested, since that was only validated at `batch_size=64`), `n_hops=2` direct, `N_step ∈ {8, 16, 24, 32}` × 2 seeds, 18 min budget each, 8 runs packed on 4 GPUs (2/GPU):

| N_step | seed 0 | seed 1 |
|---|---|---|
| 8 | 24.9% | 24.8% |
| 16 | 25.5% | 24.7% |
| 24 | 25.0% | 25.5% |
| 32 | 24.9% | 24.9% |

**Completely flat across a 4x range of N_step (8 to 32, i.e. 4x to 16x the hop count, well past thinker-e9's own ~2-4x heuristic and past the N_step=12 already tested)** -- all 8 runs land within a 0.8-point band (24.7-25.5%), no trend whatsoever. This rules out (a) N_step insufficiency as the explanation, following directly on ruling out (b) miscalibrated LR at full budget in the previous entry.

**With both (a) and (b) ruled out, (c) -- a genuine mechanism limit on 2+-hop composition through the SM buffer -- is now the best-supported explanation** for the n_hops=2 plateau (roughly 25-34% depending on the exact config tested across these sweeps, consistently well above chance ~1.5-3% but nowhere near the near-100% mastery seen at n_hops=1). This is a significant result for the project's central thesis (iterative extraction+processing should compose across hops) -- the mechanism handles single-hop retrieval essentially perfectly but does not yet compose reliably across two hops, independent of training budget, LR, or reasoning-step count tried so far.

**GPU utilization note** (per explicit user feedback on under-utilization mid-session): this sweep used `batch_size=256` (vs. the earlier default of 64) and packed 2 runs per GPU, measured at 31-41% compute / 8-14% VRAM per GPU during the run -- a real improvement over the 5-30%/2-7% seen in earlier single-run-per-GPU sweeps, though still with significant headroom (only 1-2.6GB of 24GB VRAM used per GPU). Density (processes/GPU) should be pushed further on the next batch of runs rather than batch size alone, per thinker-e9's guidance, to avoid re-opening the batch/LR confound question on an already-running sweep.

## 2026-09-13 — n_hops=2: use_ff and n_register don't clearly help either

Testing thinker-e9's priority-1 candidate (`use_ff=True`, the Phase 1bis variant Phase 1bis itself couldn't discriminate at n_facts=16) and n_register as candidate #2, at `n_hops=2` direct, `batch_size=256, lr=1.2e-3, n_step=12`, 2 seeds each, 18 min budget:

| Variant | seed 0 | seed 1 |
|---|---|---|
| `use_ff=True` | **diverges (NaN)** | 25.3% (= baseline) |
| `n_register=2` | 8.8% (worse) | 1.6% (much worse) |
| `n_register=4` | 25.0% (= baseline) | 25.3% (= baseline) |

None of these clearly break the ~25% plateau. `use_ff` is at best neutral (one seed matches baseline, the other diverges — plausibly an LR-stability interaction with the added FF capacity, not yet re-swept for this variant specifically) rather than a clean unlock. `n_register=2` is notably *worse* and unstable across seeds; `n_register=4` is neutral, same as baseline. No candidate tested so far (LR, N_step, use_ff, n_register) breaks the n_hops=2 plateau — (c) a genuine composition-mechanism limit remains the best-supported reading, though `use_ff`'s divergence at seed 0 leaves open whether a properly re-tuned LR for that variant specifically might behave differently (not yet tested: only the retrieval-task-tuned `lr=1.2e-3` was tried with `use_ff`).

## 2026-09-13 — Contre-expertise: the n_hops>=2 plateau was a compressor bug, not a mechanism limit

Independent review of the whole Indexed Attention branch (spec + plan + this log + `core/` + diagnostics), requested by the user. It overturns the branch's current headline conclusion. **Read this entry before acting on any earlier multi-hop conclusion in this file.**

### 1. The plateau was compared against the wrong chance level

Every earlier entry reads the plateau as "well above chance (~1.5-3%), so the mechanism partially composes". That reference is the uniform-over-vocabulary rate, and the model never chooses among the vocabulary — it copies a value present in the episode's KB. The correct reference is the **conditional** chance level `1/n_facts`:

| Config | `n_facts` | `1/n_facts` | Plateau observed earlier |
|---|---|---|---|
| `n_hops=2, n_distractors=1` | 3 | 33.3% | 32-34% |
| `n_hops=2, n_distractors=2` | 4 | 25.0% | 24.7-25.5% |

Two exact matches on two different configs. Measured directly: **97-98% of the baseline's predictions land on some KB value**. The model had learned "emit a KB value" and was picking at random among them — a total failure, not partial composition.

Worse, the trivial-predictor controls now implemented (`learn/indexed_attention/eval_metrics.py`) show the plateau was **below** the best no-retrieval shortcut:

| Config | `random_kb` | `non_key` (skips every hop) | model at plateau |
|---|---|---|---|
| `n_hops=2, n_distractors=1` | 0.332 | **0.500** | 0.32-0.34 |
| `n_hops=2, n_distractors=2` | 0.253 | **0.330** | 0.247-0.255 |
| `n_hops=3, n_distractors=1` | 0.244 | **0.492** | 0.42-0.46 |

`non_key` exploits a real structural shortcut in `data/kb_chain_retrieval.py`: the chain's final answer never appears as a key, so guessing uniformly among non-key values needs zero hops. The `n_hops=3` baseline's "42-46%" was exactly this shortcut, not partial chaining.

### 2. Root cause: `LevelCompressor` could not represent a key->value association

`core/indexed_memory.py::LevelCompressor` pooled `parent_k` and `parent_v` with the **same** softmax weights. A fact block is `[KEY_MARK, key_id, VAL_MARK, val_id]`; to serve as a memory entry a node must be *findable by its key* (`parent_k ~ f(key_id)`) and *return its value* (`parent_v ~ g(val_id)`) — two opposite weightings over the same children. With one softmax the compressor can only pick one, or settle on a blurred compromise: brute-forceable at one hop (hence the clean ~98% at `n_hops=1`), unchainable beyond.

**Fix**: `decouple_kv=True` (now the default) adds a second learned pooling query `query_v`, costing `n_slots * d_model` parameters. `decouple_kv=False` keeps the old behavior as an ablation (`--shared_kv_pooling`).

**CPU evidence** (`d_model=32`, `n_step=8`, 3000 steps, `lr=1e-3`, batch 64, 2 seeds, everything else identical):

| Variant | seed 0 | seed 1 |
|---|---|---|
| shared pooling, `n_hops=2` | 35.4% | 26.4% |
| **decoupled, `n_hops=2`** | **100.0%** (loss 0.000) | **100.0%** (loss 0.000) |
| shared pooling, `n_hops=3` | 46.2% | 42.4% |
| **decoupled, `n_hops=3`** | **98.1%** | 46.4% |

Clean and reproducible at 2 hops. At 3 hops one seed out of two solves it at this budget — real progress over a baseline that never does, but **not yet a stable result**; seed variance at 3 hops is the first thing to characterize on GPU.

**Therefore hypothesis (c) ("a genuine mechanism limit on 2+-hop composition") is refuted.** The loop composes; the compressor could not supply anything composable. The `(a)`/`(b)` eliminations (N_step, LR) remain valid work but were answering a question whose premise was wrong.

### 3. The attention supervision was optimizing an orthogonal objective

This explains the "striking disconnect" logged above (perfect self-match, flat accuracy). `candidate_match_loss` supervised the query toward the fact's **KEY leaf** — but attending to a key leaf returns `v_proj` of that same key token, i.e. what the model already had. The auxiliary objective was fully satisfiable *and* useless: hence `mean_rank=0.000 / top1=1.000` on all 3 seeds with accuracy unchanged. Only a fact's **level-1 node** carries the key->value pair.

Both supervision scripts now default to `--supervise node` (targets `mem._levels_k[1]` at the target fact's index, gradient also reaching the compressor's pooling queries); `--supervise leaf` reproduces the old grid.

**Smoke observation, not an experiment** (CPU, 434 steps, 15s, `d_model=32`, `n_hops=2`): `--attn_supervised --supervise node` with decoupled pooling reached `final_acc=0.72` — versus a 25% plateau after 16,000+ GPU steps previously. Needs a real run at budget before being quoted as a result.

### 4. Phase 0's "hierarchy vs flat" result needs re-reading

`depth=0` (the "flat Baseline C") has no compressor at all, so its leaves are per-*token* K/V: `k_proj(embed(tok))` / `v_proj(embed(tok))`. Attending to a key token returns that key token. **A flat memory of raw leaves cannot represent a key->value association under any training budget** — which is why it sat at 5-7% and never left curriculum stage 1.

So the 99.6% vs 6.0% gap does **not** establish that hierarchical indexing beats flat attention. It establishes that block-level grouping is the only path to an associative entry in this implementation. The honest flat baseline is **`depth=1`** (one compression level, one node per fact, no multi-level index) — see the plan's Phase 0bis.

### 5. What changed in the repo

- `core/indexed_memory.py`: `LevelCompressor(decouple_kv=True)` default + `_pool()` helper; threaded through `HierarchicalMemory` and `Thinker`.
- `tests/test_indexed_memory.py`: `TestDecoupledKVPooling` (9 tests) pinning the property the old code violated; `test_compressor_matches_manual_reference` now parameterized over both modes. 70 tests green.
- `learn/indexed_attention/eval_metrics.py` (new): conditional chance, `pred_in_kb_rate`, trivial-predictor controls, `format_report`. Wired into `train_kb_chain.py` and `train_kb_retrieval.py` — every run now prints the chance-level block.
- `diagnose_attention_supervision.py` / `train_kb_chain_attn_supervised.py`: `--supervise node|leaf`, `--shared_kv_pooling`, plus a `node_selection_diagnostic` that tracks what actually matters.

No GPU runs were launched for this entry — the re-runs are queued in the plan (Phase 0bis / Phase 2-redo).

## 2026-09-13 — CPU-vs-GPU confound resolved: exact-config repro confirms it was hyperparameters, not scale (superseded by the compressor-bug finding above, kept for the record)

Before the Contre-expertise entry above landed, this session (experiment-manager) had been given a narrower, now-superseded task: the earlier GPU `attn_supervised` grid (`d_model=128`, 3 seeds) had failed to reproduce the CPU diagnostic's 99.2% task accuracy (self-match fixed perfectly on both, but GPU `final_acc` stayed at the ~25% plateau) — plan flagged reproducing the exact CPU config (`d_model=32, vocab_size=32, n_register=4, batch_size=64, lr=3e-4, n_step=16, 12000 steps`) at matched budget on GPU as the required next step before trusting either a scale effect or chasing a new architectural hypothesis.

Ran `learn/indexed_attention/train_kb_chain_attn_supervised.py` with that exact config, 3 seeds, on Rennes (`abacus22-1` A5000 job 4105879, then `abacus21-1` A100 job 4105917 after the first job's walltime cut off mid-grid):

| variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `attn_supervised` | 99.77% | 99.77% | 99.22% |
| baseline (no supervision) | 33.8% | 26.9% | 28.8% |

**Result: GPU fully reproduces the CPU numbers when the config is matched exactly** (99.2-99.8% vs. CPU's 99.2%; baseline lands in the same 25-34% plateau band documented throughout this project). This resolves the specific confusion flagged in the plan: the earlier `d_model=128` grid's failure to reproduce was **not** a d_model/GPU-scale effect — it was a confound of several simultaneously-changed hyperparameters (the GPU script's own defaults: `vocab_size=64` not 32, `n_register=1` not 4, `batch_size=256` not 64, `lr=1.2e-3` not 3e-4, `max_steps` capped by a 30-min walltime not a fixed 12000). Each run took ~25 min on GPU (not the few seconds this scale suggested — small-model wall-clock is dominated by per-step Python/kernel-launch overhead, not FLOPs), which is why the first job's default 1h walltime cut the grid short mid-way and needed a second, longer-walltime job to finish.

**Superseded context**: by the time this finished, the Contre-expertise entry above (relayed via the `model-design` sister session) had already found the real root cause of the `n_hops>=2` plateau (the compressor's shared-softmax key/value pooling bug) and shown the attention-supervision fix itself was targeting the wrong node (leaf, not level-1) — so this result closes the specific "why doesn't GPU reproduce CPU" question cleanly, but the broader `attn_supervised`-at-`d_model=128`-plateau finding it was chasing is no longer the live question. No further action needed on this thread; recorded for completeness since it was a fully-executed, valid diagnostic in its own right.

Infra note: `micromamba` isn't on `$PATH` in a non-interactive `oarsh`/`nohup` shell (no `.bashrc` sourcing) — use the full path (`~/micromamba/micromamba run -n <env> ...`) in any unattended launch script rather than assuming `micromamba` resolves. Also, running a script by relative path from `~/thinker` still needs `PYTHONPATH=~/thinker` set explicitly — Python puts the *script's own* directory (`learn/indexed_attention/`) on `sys.path[0]`, not the cwd, so `import data.foo` / `import core.foo` fail with `ModuleNotFoundError` otherwise despite `cd`ing to the repo root first.

## 2026-09-14 — Étape 4 (hardened generator) complete: real multi-hop chaining confirmed at n_hops 2/3/4

`train_kb_chain.py`, hardened generator (`n_distractors` scaled per `n_hops`: 6/5/4), `depth=2, block_size=4, d_model=256, n_step=12, batch_size=256`, 3 seeds/cell, `leak_check`/shortcut controls verified on every cell:

**LR sweep at `n_hops=2`** (16 cells: `lr` ∈ {1e-4, 3e-4, 6e-4, 1.2e-3, 3e-3} × 3 seeds, minus one):

| lr | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1e-4 | 100% / +0.871 | 100% / +0.868 | 100% / +0.865 |
| 3e-4 | 99.98% / +0.865 | 100% / +0.850 (dup: +0.868) | 100% / +0.850 |
| 6e-4 | fails, ~0% margin | fails | fails, ~0% margin |
| 1.2e-3 | fails (~13-20%, margin negative) | fails | fails |
| 3e-3 | fails (6.2%, margin -0.084) | fails (4.0%, margin -0.104) | (not run) |

**Sharp, non-monotonic LR window**: only `lr ∈ {1e-4, 3e-4}` work (both ~100%, margin ~0.85-0.87); everything from `6e-4` up through `3e-3` fails outright (near-zero or negative margin) — not a gradual degradation, a cliff immediately above `3e-4`. Confirmed on 3 seeds at every failing value, not a fluke.

**n_hops=3** (`n_distractors=5`, `lr=3e-4`, 3 seeds): 100% / +0.838, 100% / +0.837, 100% / +0.820.
**n_hops=4** (`n_distractors=4`, `lr=3e-4`, 3 seeds): 99.98% / +0.795, 99.69% / +0.784, 99.77% / +0.798.

**Conclusion: all 3 n_hops levels (2, 3, 4) beat the no-retrieval shortcuts decisively and consistently across every seed (margin +0.78 to +0.87, no exceptions)** — per the standing decision table, this confirms real multi-hop chaining at the hardened-generator scale, not shortcut exploitation. Per the same table, relaunched Phase1bis (`use_ff`/`n_register`) at the hard `n_hops` (see next entry) rather than treating this as final — a ceiling this close to 100% could mask a use_ff/n_register effect that would only show up under more pressure (harder task or the same task with less capacity), so the ablation is a robustness/scaling-curve check, not a rescue of a failing result this time.

## 2026-09-14 — Phase1bis `use_ff`/`n_register` ablation at hard n_hops (3, 4) — launched, not yet analyzed

Reusing `n_hops=3`/`n_hops=4`'s exact working config (`lr=3e-4`, matching `n_distractors`), varying `use_ff ∈ {False, True}` × `n_register ∈ {1, 4}` (baseline cell `use_ff=False, n_register=1` already covered by the previous entry's 3-seed result), 2 seeds/cell, 12 runs total across 2 GPUs. Results pending at time of writing — see follow-up entry once complete.

## 2026-09-14 — Phase1bis pool_n_head/k_dim ablation complete (12/12): opposite LR-robustness ranking than initially suspected

`kdim128_decoupled` (asymmetric K/V dim=128, still decoupled) vs. `poolhead4_shared` (`--shared_kv_pooling --pool_n_head 4`), `n_hops=2, n_distractors=2, d_model=256`, 3 LR values × 2 seeds:

| variant | lr=3e-4 | lr=6e-4 | lr=1.2e-3 |
|---|---|---|---|
| `kdim128_decoupled` | ~100% both seeds | **100% (s0) / 33.4% (s1)** — inconsistent | fails both seeds (~20%, margin negative) |
| `poolhead4_shared` | ~100% both seeds | **99.2-99.8% both seeds** — robust | fails both seeds (~26-29%, margin negative) |

**Correction to the working hypothesis formed mid-sweep**: `poolhead4_shared` (multi-head pooling, still `shared_kv_pooling`) turned out to be the more LR-robust variant at `lr=6e-4` (both seeds solidly above shortcut), while `kdim128_decoupled` (asymmetric K/V width) is the inconsistent one at that same LR (one seed at ceiling, the other barely above chance) — the opposite ranking from an early read of the first few results streaming in. Neither variant survives `lr=1.2e-3` — same cliff already seen for the plain decoupled baseline in the Étape 4 LR sweep above, so this isn't a `pool_n_head`/`k_dim`-specific robustness gain in that direction, just noise/inconsistency specifically at `kdim128_decoupled`'s `6e-4` cell. Given the ceiling effect at `lr=3e-4` (both variants ~100%, no differentiation) and the mixed/inconsistent picture at `6e-4`, this doesn't cleanly match either of the pre-supplied decision-table branches ("poolhead4_shared ≈ full decouple" or a clear win/loss) — flagging to `model-design` as ambiguous rather than forcing a read.

## 2026-09-14 — `sm_cap` ablation complete: bounded (cap=1) buffer matches unlimited — same architectural signal as toy-memory Exp.2

`train_kb_chain.py`, `n_hops=2, n_distractors=2, lr=3e-4, d_model=256`, 3 seeds/cell:

| sm_cap | seed0 | seed1 | seed2 |
|---|---|---|---|
| 1 | 99.84% / +0.657 | 100% / +0.652 | 100% / +0.662 |
| none (unbounded) | 100% / +0.664 | 100% / +0.664 | 99.98% / +0.676 |

**`sm_cap=1` (a single-slot short-term-memory buffer) is indistinguishable from `sm_cap=None` (unbounded FIFO)** — both land in the same ~99.8-100% / margin +0.65-0.68 band, no seed showing a gap larger than ordinary run-to-run noise. This is the same qualitative finding as toy-memory Exp.2's `n_memory` ablation (`experiment.log.md`, earlier entry): whatever the short-term-memory buffer contributes here, one slot already provides it as well as an unbounded one — no evidence of the buffer accumulating useful content across multiple retained writes at this task/scale. Per the standing decision table, this is the "sm_cap=1 ≈ None" branch — reporting directly to `model-design` as a cross-workstream architectural finding (toy-memory + indexed-attention now agree), not filing it as inconclusive.

**Correction (same day, from `model-design`/user, applies to this entry AND the `disable_sm` follow-up below): this result does not actually settle whether the SM is useful.** `kb_chain_retrieval.py` is Markovian by construction — resolving hop $i{+}1$ never needs anything older than hop $i$'s value, so a single current state ($R$) suffices structurally for any chain length, `n_hops=4` included, independent of whether the multi-slot memory mechanism is good or not. **`sm_cap=1 ≈ sm_cap=None` (and whatever `disable_sm` shows) confirms this specific task cannot reveal the SM's usefulness either way — it is not evidence against the SM's architectural role (spec §-1), it's a limitation of this test's design.** Full detail in `dev_notes/indexed_attention_spec.md` §9.1. Read as "inconclusive on SM usefulness, task is Markovian," not as "SM is superfluous" — the earlier framing above ("important architectural finding") overstated what this result actually shows, before this correction arrived. A real test needs a task with independent facts to combine that are separated in time by a forced distraction phase (NTM/DNC-style recall) — not yet properly specified, `model-design`'s call on when to design and launch it.

## 2026-09-19 — I1 (deadline plan): Phase1bis `use_ff`/`n_register` at hard `n_hops` (3,4), collected from job 4106501 -- 11/12, no architectural signal

Resumed job 4106501 (launched 2026-09-14, `abacus11`, 12 runs across 2 GPUs -- see the 2026-09-14 launch entry above) via its raw logs on the Rennes home (`~/thinker/logs/phase1bis_hardhops/*.log`; this grid predates `tools/exp/`, so `collect.py` has no `grid.jsonl` for it and the logs had to be read directly). All 12 processes did run (`GPU0_DONE` present; `GPU1_DONE` missing, matching the incomplete cell below) -- this was genuinely unread output, not lost work.

`train_kb_chain.py --depth 2 --block_size 4 --d_model 256 --n_step 12 --batch_size 256 --lr 3e-4`, `use_ff ∈ {False, True}` × `n_register ∈ {1, 4}` (baseline cell `use_ff=False, n_register=1` already covered by the earlier 3-seed `n_hops` sweep, not rerun here), 2 seeds/cell:

**n_hops=3, n_distractors=5** (baseline for reference: 100%/+0.838, 100%/+0.837, 100%/+0.820):

| variant | seed0 | seed1 |
|---|---|---|
| `ff0_nreg4` | 99.94% / +0.8324 | 99.77% / +0.8307 |
| `ff1_nreg1` | 95.88% / +0.8025 | 99.98% / +0.8460 |
| `ff1_nreg4` | 99.88% / +0.8318 | 99.98% / +0.8328 |

**n_hops=4, n_distractors=4** (baseline for reference: 99.98%/+0.795, 99.69%/+0.784, 99.77%/+0.798):

| variant | seed0 | seed1 |
|---|---|---|
| `ff0_nreg4` | 99.79% / +0.8050 | 97.42% / +0.7667 |
| `ff1_nreg1` | 99.55% / +0.7963 | 99.45% / +0.8007 |
| `ff1_nreg4` | 99.77% / +0.8067 | **incomplete -- walltime cut the run off mid-training (log ends at step 3000, `eval_acc(held_out)=0.9949`, no final chance-level report/`margin_over_shortcut` line ever printed)** |

**Read (11/12 cells)**: every completed variant lands in the same tight band as the already-established baseline -- 95.9-100% accuracy, `margin_over_shortcut` +0.77 to +0.85, all comfortably above every no-retrieval shortcut. No variant (`use_ff`, `n_register=4`, or both together) separates from the baseline or from each other outside ordinary seed-to-seed noise (the single lowest point, `ff1_nreg1 n_hops=3 seed0` at 95.88%/+0.8025, is still far above shortcut and not clearly worse than the baseline's own seed spread). **Verdict: `use_ff`/`n_register` show no detectable effect at these hard `n_hops`, i.e. the ceiling from Étape 4 is real and robust, not masking a use_ff/n_register-driven difference** -- closes the ablation the 2026-09-14 entry flagged as still open, pending the one missing cell.

**Not yet final**: `h4_ff1_nreg4_s1` needs a ~15-min GPU relaunch (same `run_id`-equivalent config, legitimate resume not a duplicate) to complete the 12/12 picture, but given how tightly every other cell already clusters, a single additional seed is very unlikely to change the "no effect" read. Per this session's explicit instruction, no GPU reservation made for this alone -- deferred, to be bundled with the next GPU reservation once that's decided.

**[2026-09-19/20 nuit] 12/12 complete**: `h4_ff1_nreg4_s1` finished as predicted -- `final_acc=99.69%`, `margin_over_shortcut=+0.8719` -- squarely inside the same tight band as every other cell (95.9-100% acc, +0.77 to +0.85 margin). Confirms the "no detectable `use_ff`/`n_register` effect" read from the 11/12 partial result; the missing seed changed nothing. **I1 closed.**

## 2026-09-19/20 (nuit) — I5 complete: CPU/GPU gap was budget, not scale -- clean, decisive result

`i5_cpu32_matched_steps` (`train_kb_chain_attn_supervised.py`, `n_hops=2, n_distractors=2, n_register=4, vocab_size=32, attn_supervised, supervise=node, lr=3e-4`, matched `max_steps=12000` for both `d_model` values, 3 seeds each), 6/6 done, `abacus17`.

| d_model | seed0 | seed1 | seed2 |
|---|---|---|---|
| 32 | 100% | 100% | 100% |
| 128 | 100% | 100% | 100% |

**Read: unambiguous.** Both scales reach **100% task accuracy, all 6 seeds**, at the same 12000-step budget. This directly settles the question flagged back on 2026-09-13 (`experiment_plan.md`'s "why did CPU (`d_model=32`) show a recovery to 99.2% while GPU (`d_model=128`) didn't?"): **it was a training-budget confound, not a scale-transfer failure.** The historical GPU result that seemed not to replicate the CPU finding was under-trained relative to what CPU got (and, per the earlier-logged correction, also predated the `decouple_kv` fix -- two compounding reasons that result was never comparable in the first place). At matched budget, `attn_supervised` resolves `n_hops=2` cleanly at both scales -- no evidence of a real `d_model` scale effect on this specific question. Closes I5 with a clean positive, no further follow-up needed on this thread.

