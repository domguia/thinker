# Item[10] — audit des conclusions sous-alimentées

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-19/20 (nuit) — item [10]: audit complete -- conclusions resting on <5 seeds or un-revalidated LR

Swept `experiment.log.md` for numeric-result entries, checked real seed count behind each headline conclusion and whether the `lr` used was revalidated on that exact config or inherited from a different one. Most entries in this log already self-flag their own fragility (explicit "retracted"/"à arbitrer" markers) -- those are not re-flagged here, just listed for completeness. Genuinely new findings below.

**(c) New fragility, not previously flagged:**

1. **I1's "no detectable `use_ff`/`n_register` effect" (2026-09-19, closed 12/12) rests on 2 seeds/cell, not the project's own >=5-seed bar.** The stated justification (all 6 variant cells cluster tightly, 95.9-100%/+0.77-0.85, no bimodality) is legitimate under the project's own rule ("`>=5` seeds only needed once bimodality is found") -- but that rule was designed to catch bimodal *positive* results, not to license a *null* result at low n. A genuinely modest `use_ff`/`n_register` effect (e.g. a few points, not a cliff) would not reliably surface as bimodality at 2 seeds/cell -- it would just look like noise. **Recommendation**: fine to keep as-is for an internal "no obvious architectural lever here" read, but caveat explicitly in the paper as "no large effect detected at 2 seeds/cell" rather than "no effect," if this ablation is cited there.

2. **`kdim128_decoupled` vs `poolhead4_shared` (2026-09-14, Phase1bis pool_n_head/k_dim) used 2 seeds/cell and already shows a bimodal split at `lr=6e-4`** (`kdim128_decoupled`: 100% one seed, 33.4% the other) -- already flagged "ambiguous" to `model-design` in the log itself, so this is category (b) (already known), listed here only to confirm the audit caught it and it's not forgotten -- no new action.

3. **I7 real-scale margin=4 (trained `N_step=8`) is explicitly marked "mixed... not yet clean" already** -- category (b), no new flag needed, but worth surfacing here as the one `I7` margin bucket that should NOT be cited as settled in the paper without more seeds (1/3 clear degradation, 2/3 flat) -- the other three margin buckets (1, 2, 12) are each internally consistent (3/3 or 2/3-with-explained-failure) and safer to cite as-is.

**LR-revalidation check**: no case found where a conclusion silently reused an LR from a *materially different* config without either (a) an explicit sweep on the new config, or (b) an explicit flag that this is exactly the open question (e.g. Piste A's `d_model=1024` sweep was launched *specifically because* the shared `lr=3e-4` was suspect, and the result is `à arbitrer` pending the extended-budget read -- correct handling, not a gap). `item8_nhops56`, `item5_nstep_lrlaw`, `i5_cpu32_matched_steps` all ran their own sweep or reused an LR already validated on the *same* `n_hops`/scale, not a borrowed one.

**Not re-flagged (already correctly marked in-line, listed for completeness only, category (b)):** I3 étape 1 bimodality (led directly to étape 2's repowering, itself now running), the `cumsum` `seq_len=32` mixed result (toy-model memory thread, explicitly left open), I7's original small-scale margin confound (fully retracted).

No other new <5-seed or unrevalidated-LR fragility found beyond the two items above -- the project's discipline of self-flagging (retracted/à arbitrer markers) is catching the large majority of cases already; this audit's marginal value was items 1-2 above, both about ablations reporting a *null* result at seed counts calibrated for detecting *positive* bimodality, not absence of effect.

