# Real-text baselines & Piste A (LFM2/OLMo/Qwen, A/B/C, LR sweeps)

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-14 — Phase 3 real-text: LR sweep result + a real blocker for "continue plus longtemps"

`train_real_text.py`, `depth=1`, `data/distill/general_realtext/train.jsonl` (2700 docs), 2 seeds/LR:

| lr | seed0 final_loss | seed1 final_loss |
|---|---|---|
| 1e-4 | 6.99 | 6.80 |
| 3e-4 | 6.49 | 6.36 |
| 1e-3 | 6.05 | 5.89 |
| 3e-3 | 5.89 | 5.58 |
| 1e-2 | 6.81 | 6.10 |
| 3e-2 | 16.10 | 16.91 |

Clean, monotonic loss decrease from `1e-4` to `3e-3`, then degradation at `1e-2` (worse than `3e-3` but still training) and outright divergence at `3e-2`. **`lr=3e-3` is the optimum, cleanly bracketed on both sides** — not just the best of the originally-tested range.

**Blocker found while trying to act on "loss décroît proprement → continue plus longtemps"**: `train_real_text.py`'s `LockstepLaneBatcher` does a **single, non-repeating pass** over the 2700-document dataset — it exhausts the data and the generator stops after ~1537-1538 steps (~37-40s wall-clock) regardless of `--max_time_minutes`. Confirmed directly: a run launched with `--max_time_minutes 18` (vs. the sweep's 12) still stopped at `num_steps=1537`, `elapsed=0.65m` — i.e. it was a duplicate of the `lr=3e-3 seed=0` sweep cell, not a longer run. **"Continue plus longtemps" is not currently possible with this script/dataset as-is** — needs either (a) a multi-epoch/repeat mechanism added to `LockstepLaneBatcher` (currently `refill()` returns `False` and stops once `next_doc_ptr` exhausts `doc_order`, no wraparound), or (b) more real-text data than 2700 documents. Flagging to `model-design` rather than silently working around it — this directly blocks Phase 3's next planned step (Baselines A/B/C also still not implemented per the script's own docstring, separately).

## 2026-09-16 — Baselines A/B/C real-text grid (LFM2/OLMo/Qwen), a real `tokenizer.vocab_size` bug found, 13/15 done

Relayed by `model-design` on behalf of domguia: Baselines A (`--n_step 1`) / B (`--disable_kb`) / C (main, `depth=1`) x {lfm2, olmo} x {`d_model=128`, `d_model=1024` spec §13 sizing}, plus `qwen`/`d_model=1024` (checked first, not a duplicate). 15 cells via `tools/exp/`, see `grid5000_usage.log.md` for the launch/infra details.

**Bug found**: both `qwen` cells that actually build/use the KB embedding path (`n_step=1` and the main/`depth=1` run) crashed identically at the very first forward pass -- `IndexError: index out of range in self` in `self.embed(kb_tokens)`. Root cause: `train_real_text.py` sized the embedding table (and the output head) from `tokenizer.vocab_size`, but that attribute reports the tokenizer's **base** vocab, excluding added/special tokens -- confirmed directly (`Qwen/Qwen3-0.6B`: `vocab_size=151643` vs `len(tokenizer)=151669`, a 26-token gap; `LiquidAI/LFM2-350M` and `allenai/OLMo-2-0425-1B` both have zero gap, so their 12 completed cells are unaffected). The `disable_kb` cell for `qwen` happened to succeed anyway -- it never exercises the `self.embed(kb_tokens)` path that hit the id above `vocab_size-1`. **Fix**: use `len(tok)` instead of `tok.vocab_size` everywhere the embedding/head is sized (`learn/indexed_attention/train_real_text.py`). Verified with a 5-step smoke run on `qwen`/`d_model=1024` post-fix (`params=321.32M`, no crash). Not yet committed. Stale `state`/`claims` files for the two crashed cells deleted and re-queued through the same worker pool (still using the fixed code now synced to the node) rather than a fresh grid -- run_id is a hash of (script, config), unaffected by an internal code fix, so this is a legitimate resume, not a duplicate.

**Preliminary read on the 13 completed cells (`final_loss`, lower is better; all still far from converged -- 1000-5500 steps out of the 200000-step ceiling, this is an early-training snapshot, not an asymptotic comparison)**:

| tokenizer | d_model | A (`n_step=1`) | B (`disable_kb`) | C (main, `depth=1`) |
|---|---|---|---|---|
| lfm2 | 128 | **4.948** (5493 steps) | 6.383 (5386 steps) | 5.525 (4233 steps) |
| olmo | 128 | **5.405** (3393 steps) | 6.113 (3314 steps) | 6.153 (2987 steps) |
| lfm2 | 1024 | **4.828** (1915 steps) | 6.966 (1936 steps) | 7.678 (1549 steps) |
| olmo | 1024 | **5.902** (1303 steps) | 9.803 (1344 steps) | 8.447 (1081 steps) |

**Baseline A (flat retrieval, no loop) has the lowest loss in all 4 cells -- beating the main model (C) at this budget, on both tokenizers, at both `d_model`.** Read with real caution before treating this as "iteration doesn't help": each baseline gets a different number of gradient steps for the same wall-clock budget, because `n_step` directly multiplies per-step compute (A's `n_step=1` vs C's default `n_step=6` -- confirmed in the step counts above, A consistently reaches ~1.5-2x more steps than C in the same 30/45 min). This comparison confounds "does the loop help" with "same wall time, fewer steps" -- not a clean ablation as currently run. B (`disable_kb`, loop but no memory) is worst everywhere as expected (no memory access should hurt), which is at least a sane sanity check on the harness itself. **Not drawing an architecture conclusion from this table** -- flagging the wall-time-vs-step-count confound to `model-design`; a fair comparison needs either matched step count (not wall time) or accounting for the per-step cost difference explicitly.

**Update: 15/15 cells done** (the 2 re-queued `qwen` cells finished cleanly, well before the reserving job's 6h walltime ran out and reclaimed the node). Final row for `qwen`/`d_model=1024`, same shape as the rest of the table above:

| tokenizer | d_model | A (`n_step=1`) | B (`disable_kb`) | C (main, `depth=1`) |
|---|---|---|---|---|
| qwen | 1024 | **4.897** (2686 steps) | 11.726 (1544 steps) | 6.101 (2267 steps) |

Same pattern as lfm2/olmo: A lowest loss, B highest, C in between -- consistent with the wall-time/step-count confound already flagged (A's `n_step=1` reaches ~1.2-1.7x more steps than C here too). Grid fully collected; no new read beyond what's already flagged to `model-design` (step-count confound needs resolving before any "does the loop help" claim).

## 2026-09-19 — Piste A: step-matched real-text baselines rerun -- A still beats C, gap widens at larger d_model (early-training snapshot, not asymptotic)

Goal: the 2026-09-16 Baselines A/B/C grid (`experiment.log.md` above) confounded "does the loop help" with "same wall time, fewer gradient steps" -- Baseline A (`n_step=1`) reached 1.2-2x more steps than C (`n_step=6`, main config) in the same budget. Reran **step-matched** instead: `--max_steps 1000` fixed identically for every cell (chosen conservative -- the slowest historical cell, C at `d_model=1024`, reached only 1081-1549 steps in 30-45 min wall-clock on 2026-09-16, so 1000 is comfortably reachable by every variant), `--max_time_minutes 60` generous/non-binding. 36 cells (A/B/C x {lfm2, olmo} x {`d_model`=128, 1024} x 3 seeds), via `tools/exp/`, `abacus18-1` (3x RTX 6000), all 36 done in ~16 minutes wall-clock (much faster than the original wall-time-matched grid, expected -- 1000 steps is a low, fixed bar every cell clears quickly rather than running until a time budget expires).

**Mean `final_loss` at step 1000 (lower is better), 6 cells averaged per (baseline, d_model) cell -- both tokenizers, 3 seeds pooled**:

| d_model | A (`n_step=1`) | C (main, `n_step=6`) | B (`disable_kb`) |
|---|---|---|---|
| 128 | **6.49** | 6.88 | 7.44 |
| 1024 | **6.04** | 8.43 | 9.66 |

**[CORRECTION, same day, before this was ever read as final -- LR was never revalidated for this grid]**: every cell in this grid used `train_real_text.py`'s bare default `--lr 3e-4` -- confirmed by checking the grid config directly, `lr` was never set in `gridgen`'s `--fixed`/`--sweep`. That is the exact "changed axis, reused an old LR" pattern this project's own standing rule exists to catch (already burned twice: `decouple_kv` at `lr=1.2e-3`, `pool_n_head` bimodality at `lr=6e-4`). **Pulled the actual loss curves from the `RunLogger` state history (`progress()` every 20 steps) before trusting the table above, per that same rule -- and it matters**:

- `d_model=1024`, C (main, `n_step=6`), seed 0, lfm2 (`train_real_text-5c73e1061b`): loss is **not** descending cleanly -- `11.6 -> 12.3 -> 12.0 -> 16.6 -> 11.6 -> 9.0 -> 10.1 -> 10.5 -> 23.1 -> 12.5` (steps 0-900, every 100). Wild oscillation with a spike to 23 at step 800 -- textbook LR-too-high instability, not a converging run.
- `d_model=1024`, A (`n_step=1`), seed 0, lfm2 (`train_real_text-f6a6b848f6`): smooth, roughly monotonic descent, `11.4 -> 7.8 -> 7.1 -> 6.8 -> 6.5 -> 6.2 -> 6.3 -> 6.2 -> 6.1 -> 6.0`.
- `d_model=128`, same comparison: **both** A and C descend smoothly (C: `11.5 -> 8.9 -> 8.2 -> 7.8 -> 7.4 -> 7.3 -> 7.3 -> 7.2 -> 6.9 -> 6.99`), no instability on either side.

**So the widening A-C gap at `d_model=1024` is very likely an artifact of `lr=3e-4` being too high specifically for C's config at that scale (`n_step=6`, more compounded gradient paths per step than A's `n_step=1`), not a genuine "the loop scales worse" finding.** `d_model=1024` is a real axis change (bigger model) on top of `n_step` already differing between A and C -- exactly the situation the LR-revalidation rule targets, and this grid skipped it entirely.

**Retracting the "real negative result" framing above -- this is NOT yet a validated result.** What actually still stands: at `d_model=128`, both curves are stable and A beats C at matched steps by a modest, currently-uncorrupted margin (0.38) -- worth keeping as a tentative, small-scale-only observation. The `d_model=1024` comparison (and its "gap widens" claim) is retracted pending an LR sweep for **A and C separately at `d_model=1024`** (their optimization landscapes differ -- `n_step=6` vs `n_step=1` is not the same mechanism, per the standing rule's own guidance) before any conclusion, positive or negative, is drawn at that scale.

**Both wall-time-matched (2026-09-16) and this step-matched attempt remain open questions for the `d_model=1024` / main-config comparison specifically** -- not yet a defensible "A beats C" claim at that scale until LR is recalibrated there. The `d_model=128` step-matched reading is the only piece of this entry currently safe to cite.

## 2026-09-19/20 (nuit) — Piste A LR sweep at d_model=1024: A and C do NOT share an optimal LR, and step-matched != optimization-progress-matched

`pistea_lrsweep_d1024` (A: `n_step=1`, C: main/`n_step=6`, both `d_model=1024`, `tokenizer=lfm2`, `max_steps=1000` matched, 3 seeds/cell), launched to resolve whether the retracted "A beats C, gap widens to 2.39" reading (above) survives a proper LR sweep instead of the shared, unvalidated `lr=3e-4`.

**Wave 1 (`lr` in `{1e-5, 3e-5, 1e-4, 3e-4}`) mean `final_loss`**:

| lr | A (`n_step=1`) | C (`n_step=6`, main) |
|---|---|---|
| 1e-5 | 7.014 | 7.761 |
| 3e-5 | 6.587 | **7.672 -- clean bracketed minimum for C** (worse on both sides: 1e-5 and 1e-4) |
| 1e-4 | 6.181 | 7.881 |
| 3e-4 | **6.050 -- best A so far, still at the range's top edge** | 8.691, unstable (one seed 9.289) |

**C's optimum is real and bracketed at `lr=3e-5`** -- not a boundary artifact. **A's optimum is not yet bracketed** (still improving at the range's top edge); wave 2 (`lr` in `{1e-3, 3e-3}`) is resolving it.

**Central finding, independent of A's still-open optimum**: C's best LR (~3e-5) is **roughly 10x lower** than A's (>=3e-4) -- the looped mechanism (`n_step=6`) needs a substantially lower, narrower LR window than the flat baseline (`n_step=1`) at this scale. This on its own explains why the original shared-`lr=3e-4` comparison was invalid (C was unstable at that LR, A was fine).

**[Methodological correction from `model-design`, catches a second confound the step-matching alone didn't remove]**: pairing `max_steps` removes the wall-clock/step-count confound (the original 2026-09-16 issue), but **introduces a different one once the two architectures' optimal LRs differ by an order of magnitude: at matched step count but each at its own optimal LR, the two architectures do not travel the same distance through parameter space.** C at `lr=3e-5` for 1000 steps moves far less than A at `lr=3e-4` for 1000 steps -- so a 1000-step comparison, however carefully LR-tuned, is not actually an apples-to-apples "does the loop help" measurement; it confounds "architecture quality" with "optimization progress made in the same step budget." **This is a limit of the comparison, not a result -- the 1000-step gap (currently 1.62 at each architecture's best-known LR) must not be reported as a negative finding on its own.** The only way to actually settle "does the loop help" is a longer, per-architecture-optimal-LR run where the loss curves either converge, cross, or stay apart -- see the extended-budget follow-up (next entry, queued once A's LR is bracketed).

Not yet concluding anything from this entry (charte d'autonomie: pas de conclusion sur résultat partiel) -- wave 2 still running.

**À arbitrer**: none yet from this thread -- flagged here as a placeholder since the eventual extended-budget read (curves crossing late vs. staying apart) may itself land in "ambiguous, needs a human call" territory depending on what the curves actually do.

**Wave 2 complete (`lr` in `{1e-3, 3e-3}` added) -- both optima now cleanly bracketed**:

| lr | A (`n_step=1`) mean | C (`n_step=6`) mean |
|---|---|---|
| 1e-5 | 7.014 | 7.761 |
| 3e-5 | 6.587 | **7.672 -- C's optimum, bracketed** |
| 1e-4 | 6.181 | 7.881 |
| 3e-4 | **6.050 -- A's optimum, bracketed** | 8.691 |
| 1e-3 | 6.495 (worse than 3e-4) | 91.6 (exploding) |
| 3e-3 | 10.47 (diverging) | NaN (fully diverged) |

**A's optimum is `lr=3e-4`** (worse on both sides: 1e-4 and 1e-3), **C's optimum is `lr=3e-5`** (worse on both sides: 1e-5 and 1e-4) -- confirmed 10x apart, both now solidly bracketed rather than open questions. At each architecture's own best LR: A=6.050, C=7.672, gap=1.622. **Per the methodological correction above, this gap is NOT reported as a "does the loop help" verdict** -- it's the necessary input to the extended-budget follow-up (next entry), which is the actual test.

Extended-budget follow-up (item [2], each architecture at its own bracketed optimum: A `lr=3e-4`, C `lr=3e-5`, `d_model` in {128, 1024}, 8000 steps, 2 seeds, full curve logged every 20 steps) launched on `abacus18` once the sweep freed it. Results pending, see next entry.


## 2026-09-19/20 (nuit) — Piste A extended-budget complete (8/8): A beats C at both scales, gap widens

`pistea_extended_budget` (`train_real_text.py`, `d_model` in {128,1024}, `n_step` in {1 (A), 6 (C, main)}, 2 seeds each), Rennes, all 8 done.

| d_model | A (n_step=1) mean | C (n_step=6) mean |
|---|---|---|
| 128 | **4.522** (4.568, 4.477) | 6.423 (6.455, 6.391) |
| 1024 | **3.205** (3.179, 3.231) | 5.699 (5.645, 5.754) |

**Read: A (flat, no loop) beats C (main, looped) at both scales, and the gap widens at larger `d_model`** (128: 1.90 loss points; 1024: 2.49 points) -- same direction and pattern as the earlier step-matched result, now at extended budget rather than the earlier 1000-step matched grid. Consistent with the LR-sweep finding above (A and C do not share an optimal LR) -- this comparison still uses the shared `lr=3e-4`, so **the gap here should be read as "at a shared, not-necessarily-optimal-for-either-arm LR," not yet the fairest possible comparison** (the untested revalidated-LR-per-arm comparison remains the open item). Full grid in `runs/pistea_extended_budget/state/*.json` (Rennes home).

## 2026-09-20 — User reframing: A>C is expected at this stage (C is novel, needs more optimization research than classical A), don't stop -- keep exploring C in parallel; new evaluation toolkit built for the actual project goal

User's point: it's not surprising A beats C right now -- A is architecturally compatible with well-understood classical training practice, C (the loop) is novel and plausibly needs more hyperparameter/schedule research before being judged. Don't conclude, explore C harder. Dispatched (see this file's later entries once results land): `pistea_ext2` (progressive budget escalation 12k/16k/20k steps, not a single jump), `pistea_c_nstep_sweep` (n_step in {2,4,6,8,10,12} at d_model 128/1024), `pistea_c_lr_warmup_sweep` (LR fine grid x warmup_steps in {0,200,500}). New code: `--lr_warmup_steps`/`--lr_warmup_init` added to `train_real_text.py` (same convention as `learn/toy_memory/train_toy_memory.py`), verified via unit-checked ramp logic + a live smoke run (commit `f48192f`).

**Restated project priority (user, explicit)**: architecture-variant exploration (n_slots, etc.) is useful for information/cluster utilization but secondary -- the priority is a working, evaluable Thinker on real text that can "think for long" and be compared to real LLMs on general tasks, while also behaving consistently with the architecture's own hypotheses (memory manipulation, step extrapolation).

**Evaluation toolkit built for this goal (none of it existed before today)**:
- `--val_data` (commit `f1c06aa`): genuine held-out split for `train_real_text.py` -- this project had NO held-out mechanism for real text before this (only training loss was ever reported).
- `--save_checkpoint_path`/`--extrapolate_n_steps` (commit `657cd39`): checkpoint saving + in-memory n_step extrapolation probe, same convention as `train_kb_chain.py`'s flags of the same name.
- `eval_checkpoint.py` (commit `e958c8b`): loads a saved checkpoint, runs (a) a **memory ablation** (toggles `model.disable_kb` between two held-out eval passes -- if disabling the KB/long-range read barely hurts, the model isn't manipulating memory the way spec Sec.-1 hypothesizes, reported as a real finding rather than assumed) and (b) the step-extrapolation probe on held-out data. Refuses to run without `--val_data`.
- `eval_llm_baseline.py` (commit `6b1f8d7`): evaluates a reference LLM (default LFM2-350M, already this project's own tokenizer alias -- no vocab mismatch) on the exact same held-out windows, for a directly comparable loss/ppl number against Thinker's own.

All four verified end-to-end via smoke runs (tiny CPU models/corpora) before being reported here; 76/76 existing tests green throughout.

**Next milestone once the C-exploration sweeps above produce a result**: pick the best C config, run a proper (not smoke-scale) training with `--val_data`/`--save_checkpoint_path`, then run `eval_checkpoint.py` + `eval_llm_baseline.py` on the resulting checkpoint -- this is the first candidate for "a working Thinker we can evaluate," the project's stated priority ahead of further architecture-variant exploration.

## 2026-09-20 — Garde-fou méthodologique sur pistea_c_nstep_sweep: LR fixe à travers n_step, même piège que item[5]/item[8]

**À appliquer avant toute lecture définitive de la tendance monotone observée sur `pistea_c_nstep_sweep`** (loss croissante avec `n_step` à `d_model=1024`, `n_step=2` optimal, `lr=3e-5` fixe partout) : `item[5]`/`item[8]` (`dev_notes/experiments/nstep_lr_law.md`) ont déjà montré sur le synthétique que le LR n'est pas une loi lisse en `1/n_step` mais un effet de seuil -- une fenêtre stable à petit `n_step` peut s'effondrer brutalement à `n_step` plus grand. Le sweep actuel tient `lr` fixe sur toute la plage `n_step`, exactement la configuration qui a produit une fausse lecture ailleurs dans ce projet avant correction.

**La tendance monotone actuelle peut donc signifier soit (a) "lr=3e-5 est déjà au-delà du seuil de stabilité pour n_step>=4" (confond LR), soit (b) "la boucle profonde est intrinsèquement pire sur texte réel" (résultat réel) -- indistinguable sans un sweep LR par `n_step`.** Ne pas conclure sur (b) tant que `pistea_c_nstep_lr_joint` (sweep croisé `n_step` x `lr`, lancé juste après complétion de ce sweep-ci) n'a pas confirmé que même au meilleur LR par `n_step`, la perte continue de croître.

## 2026-09-20 — Plan : branchement KD dans `train_real_text.py` + choix de dimensionnement pour le run final

**Contexte, discussion utilisateur** : les sweeps `pistea_c_*` entraînent déjà le vrai `Thinker` sur texte réel (correction d'une lecture inexacte : il y a bien du Thinker en entraînement en ce moment), mais (a) uniquement en cross-entropy pure, jamais en KD (vérifié : `teacher_targets`/`topk_kd_loss`/`kd_alpha` n'existent que dans `learn/distill/`, aucun script important `Thinker` ne les référence), et (b) ce sont des balayages d'hyperparamètres, pas un run délibérément dimensionné pour l'évaluation finale. Décision : brancher le KD maintenant, dimensionnement du run final à `d_model=1024`, `depth=1` (garder le MVP déjà validé, isoler KD comme seule nouveauté), `n_step` fixé une fois le sweep croisé `n_step`x`lr` conclu, `n_register=8` (pas de preuve contraire), `n_ctx` à trancher plus tard.

**Format des cibles Teacher, réutilisé tel quel** (`learn/distill/precompute_teacher_targets.py`, déjà écrit/testé) : un `.npz` par fichier JSONL source, arrays plats `(total_tokens, ...)` -- `indices`/`values` (Top-K, forme `(N,K)`), `residual` (`(N,)`, log-sum-exp du reste), `offsets` (`(n_docs+1,)`, `offsets[i]:offsets[i+1]` = lignes du document `i`), `k` = taille du Top-K. `topk_kd_loss()` (`learn/distill/train_sft.py`) déjà écrite/testée, réutilisée telle quelle (import, pas de réimplémentation) : `KL(teacher || student)` sur la catégorielle `(K+1)`-way, prend `student_logits (B,T,V)`, `teacher_indices/values (B,T,K)`, `teacher_residual/mask (B,T)`.

**Le point non trivial, à documenter avant de coder pour ne pas se tromper silencieusement -- convention d'alignement position/logit** :
- `precompute_teacher_targets.py` stocke `logits = out.logits[0]` bruts (convention causale standard HF) : la ligne `q` du document est la distribution du Teacher pour prédire le token `q+1`, en ayant vu les tokens `0..q`.
- `RealTextWindowDataset._build_window` construit, pour une fenêtre `(doc_id, p)` : `labels[t] = ids[p+t]` et `target_input[t] = ids[p+t-1]` (avec repli sur `pad_id` si `p=0`, mais ce cas n'arrive jamais en pratique -- `_add_windows` exige `real_context >= min_real_context=1`, donc `p>=1` toujours pour une fenêtre incluse). `Thinker`, à l'étape `t`, reçoit `target_input[t]` comme requête et prédit `labels[t]`.
- **Donc la ligne Teacher à utiliser pour la position `t` d'une fenêtre commençant en `p` est `q = p + t - 1`** (le Teacher a vu les tokens `0..p+t-1`, prédit `p+t = labels[t]` -- exactement ce que Thinker fait avec `target_input[t]=ids[p+t-1]`). Les lignes nécessaires pour une fenêtre entière sont donc `range(p-1, p+t_tgt-1)`, à extraire de `teacher["indices"][offsets[doc_id]+p-1 : offsets[doc_id]+p+t_tgt-1]` (et pareil pour `values`/`residual`).
- **Contrainte à vérifier/documenter au moment du precompute** : le fichier `.npz` doit avoir été calculé avec un `--max_length` couvrant au moins `p + t_tgt - 1` pour toute fenêtre utilisée -- sinon certaines fenêtres tombent hors de la plage précalculée (`offsets[doc_id+1] - offsets[doc_id] < p + t_tgt - 1`). À gérer par un masque par fenêtre (`teacher_mask`, déjà prévu par `topk_kd_loss`) plutôt qu'un crash -- toute fenêtre partiellement hors plage doit avoir ses positions hors plage masquées, pas toute la fenêtre écartée.

**Changements de code prévus** :
1. `data/real_text_windows.py` : ajouter `"window_pos": p` au dict retourné par `_build_window` (actuellement absent, nécessaire pour indexer les cibles Teacher).
2. `learn/indexed_attention/train_real_text.py` : nouveaux arguments `--teacher_targets`/`--val_teacher_targets`/`--kd_alpha` (même convention que `train_sft.py`) ; chargement du `.npz` (une fois, aligné sur l'ordre des documents de `RealTextWindowDataset`, en supposant même fichier source/même ordre -- hypothèse à vérifier, pas juste supposée) ; dans la boucle, si `teacher_targets` fourni, construire `teacher_indices/values/residual/mask (B, t_tgt, ...)` par fenêtre via `doc_id`+`window_pos`, appeler `topk_kd_loss()` (importée, pas réécrite), combiner avec la CE existante via `kd_alpha` -- même formule que `train_sft.py` (`loss = (1-kd_alpha)*ce + kd_alpha*kd`).
3. `collate_lane_batch` : propager `window_pos` (actuellement ignoré comme `doc_id`/`is_first_window`, à traiter pareil -- métadonnée, pas une entrée du modèle).

Implémentation en cours.

## 2026-09-19/20 (nuit) — pistea_ext2 complete (24/24): A bat C reste vrai à budget encore plus élevé, gap ne se referme pas

`pistea_ext2` (Nancy `graffiti-3`, relancé correctement via `oarsh` après le fix frontend), `n_step` in {1 (A), 6 (C)}, `d_model` in {128, 1024}, `max_steps` in {12000, 16000, 20000}, `lr=3e-4` fixe, 2 seeds.

**Moyennes par (n_step, d_model), toutes valeurs de max_steps/seeds confondues** :

| | d_model=128 | d_model=1024 |
|---|---|---|
| A (n_step=1) | **~3.99** | **~2.77** |
| C (n_step=6) | ~4.12 | ~3.69 |

**Read: A bat C à budget encore plus élevé (12k-20k pas) qu'auparavant (8000 pas), le gap ne se referme pas, et s'élargit même à `d_model=1024`** (0.92 pt d'écart vs 0.13 pt à `d_model=128`) -- cohérent avec `pistea_extended_budget` (8000 pas) et l'escalade de budget déjà loguée. **Rappel du garde-fou déjà posé** (entrée précédente) : ce sweep tient `lr=3e-4` fixe, potentiellement pas optimal pour C -- le sweep croisé `n_step`x`lr` (lancé séparément) est ce qui distinguera "C a juste besoin d'un LR différent" de "la boucle est intrinsèquement pire à budget égal". Ne pas lire ce résultat isolément comme définitif sur la boucle.

Full grid : `runs/pistea_ext2/state/*.json` (Nancy home).

## 2026-09-20 — pistea_c_nstep_sweep complete (24/24): confirms monotone n_step degradation at lr=3e-5 fixed, joint sweep launched to test if it's a LR confound

`lr=3e-5` fixe, `max_steps=8000`, 2 seeds/cellule.

| n_step | d_model=128 mean | d_model=1024 mean |
|---|---|---|
| 2 | 6.352 | **4.709** |
| 4 | 6.435 | 5.223 |
| 6 | 6.424 | 5.631 |
| 8 | 6.434 | 6.184 |
| 10 | 6.468 | 6.624 |
| 12 | 6.528 | **7.199** |

**`d_model=128` reste quasi plat (6.35-6.53, dérive légère mais faible). `d_model=1024` montre une dégradation monotone franche, `n_step=2` (4.71) à `n_step=12` (7.20), +2.49 points de loss.** Confirme la tendance déjà vue sur les 18 premières cellules. **Garde-fou toujours actif** (entrée précédente) : `lr=3e-5` est fixe partout, donc cette dégradation peut être soit un vrai effet architectural, soit `lr=3e-5` au-delà du seuil de stabilité dès `n_step>=4` à cette échelle -- indistinguable sans le sweep croisé. `pistea_c_nstep_lr_joint` lancé immédiatement après (même jour) pour trancher.

Full grid : `runs/pistea_c_nstep_sweep/state/*.json` (Rennes home).

## 2026-09-20 — pistea_c_lr_warmup_sweep complete (30/30): LR optimal pour C plus haut que testé, warmup n'aide pas

`d_model=1024, n_step=6 (C), max_steps=8000`, `lr` in {1e-5,2e-5,3e-5,5e-5,7e-5} x `lr_warmup_steps` in {0,200,500}, 2 seeds.

**Moyenne par lr (warmup=0)** : 1e-5→6.40, 2e-5→5.92, 3e-5→5.79, 5e-5→5.41, **7e-5→4.82**.

**Read : tendance monotone croissante jusqu'au bout de la plage testée (7e-5, la valeur la plus haute) -- aucun signe de plafond ou de cliff dans cette fenêtre.** Ça contredit l'hypothèse de travail qui motivait ce sweep (chercher un LR *plus bas* pour C) -- au contraire, le LR optimal pour C semble être *au-delà* de 7e-5, pas en-deçà. **Implication directe pour `pistea_c_nstep_sweep`/`pistea_c_nstep_lr_joint`** : ces sweeps testent `lr` jusqu'à 1e-4 (`nstep_lr_joint`) -- la lecture de ces deux sweeps doit maintenant se faire à la lumière de "peut-être encore trop bas", pas seulement "peut-être trop haut" comme le garde-fou initial le supposait implicitement.

**Warmup : n'aide pas, tend à nuire légèrement.** À `lr=7e-5` (le point le plus net) : `warmup=0` → 4.82, `warmup=200` → 4.94, `warmup=500` → 5.05 -- dégradation monotone avec plus de warmup. Même tendance plus faible aux LR plus bas. **Conclusion : pas de bénéfice du warmup dans cette plage de LR/pas pour ce réglage de C** -- contrairement à l'intuition "pratique standard pour les architectures récurrentes/bouclées" qui motivait ce test.

Full grid : `runs/pistea_c_lr_warmup_sweep/state/*.json` (Rennes home).

## 2026-09-20 — pistea_c_useff_sweep complete (8/8): use_ff=True bat clairement use_ff=False sur texte réel

`d_model=1024, n_step=6, lr=3e-5, ff_hidden_mult=4, max_steps=8000`, 4 seeds.

| | moyenne final_loss |
|---|---|
| `use_ff=True` | **4.71** (4.61-4.78) |
| `use_ff=False` | 5.64 (5.38-5.80) |

**Read : signal net et cohérent sur les 4 seeds (aucun chevauchement des plages) -- `use_ff=True` bat `use_ff=False` de ~1 point de loss.** Selon la lecture déjà posée par `model-design` avant le lancement de ce sweep : **un gain net de `use_ff=True` indique une capacité de composition/calcul manquante dans le stream (qui devrait rester un lecteur léger selon spec §11bis/§-1), pas que la boucle Thinker elle-même est une mauvaise idée.** Ce résultat va dans le sens de la contre-hypothèse -- le stream fait un travail de calcul non trivial au-delà de la simple lecture, à documenter et discuter plutôt qu'à écarter. Pertinent aussi pour la lecture du résultat "A bat C" de Piste A : une partie de l'écart pourrait venir d'une capacité insuffisante côté `fuse`/stream plutôt que d'une limite intrinsèque de la boucle.

Full grid : `runs/pistea_c_useff_sweep/state/*.json` (Rennes home).
