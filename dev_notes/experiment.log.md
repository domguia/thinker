# Journal d'experiences -- index

Ce fichier est un **index chronologique court**. Le detail complet de chaque
fil (lancement, resultats, corrections, retractations) vit dans
`dev_notes/experiments/<fil>.md`, groupe par sujet plutot que par date --
consulter le fichier lie pour le contexte complet d'une ligne donnee.

Migration effectuee le 2026-09-20 (le fichier plat depassait 1290 lignes,
melangeant chronologie et sujets -- ex. I3 etape 1 et etape 2 etaient
separees de plus de 100 lignes). Contenu original integralement preserve,
zero perte -- voir les fichiers `dev_notes/experiments/*.md`.

## Fichiers par fil

- [`archive_2023_prototyping.md`](experiments/archive_2023_prototyping.md) -- prototypage initial (2023)
- [`distillation.md`](experiments/distillation.md) -- onboarding, MFU sweep, Teacher precompute
- [`indexed_attention_core_mechanism.md`](experiments/indexed_attention_core_mechanism.md) -- plateau n_hops, contre-expertise, etape 4, I1, I5
- [`toy_memory.md`](experiments/toy_memory.md) -- memoire a moyen terme (read_step, n_memory)
- [`real_text_baselines.md`](experiments/real_text_baselines.md) -- Piste A, baselines A/B/C
- [`b1_associative_recall.md`](experiments/b1_associative_recall.md) -- n_facts curriculum, B1
- [`i4_i6_i7_diagnostics.md`](experiments/i4_i6_i7_diagnostics.md) -- I4/I6/I7 (attribution, phase probe, N_step generalization)
- [`i3_attention_supervision.md`](experiments/i3_attention_supervision.md) -- I3 etapes 1 et 2
- [`s3_ffn_attention_curriculum.md`](experiments/s3_ffn_attention_curriculum.md) -- A1/A2, S0/S0.5, curriculum ffn2attn
- [`nstep_lr_law.md`](experiments/nstep_lr_law.md) -- item[5]/item[8], loi N_step/LR
- [`item10_audit.md`](experiments/item10_audit.md) -- audit des conclusions sous-alimentees

## Index chronologique

- **2023-12-05/18** -- [Prototypage initial (copy task, debug inplace/permute bugs)](experiments/archive_2023_prototyping.md)
- **varying position embedding (2023)** -- [Curriculum copy task, position embedding](experiments/archive_2023_prototyping.md)
- **2026-09-03/04** -- [Distillation onboarding: data prep, Teacher benchmark, premiers SFT+KD](experiments/distillation.md)
- **2026-09-12** -- [Phase 0 go/no-go hierarchie vs flat, n_facts scale cliff (EXP-007)](experiments/indexed_attention_core_mechanism.md)
- **2026-09-13** -- [500M-core MFU/batch-size sweep, missing-bf16 discovery](experiments/distillation.md)
- **2026-09-13** -- [n_hops=2 LR/N_step sweeps -- (a)/(b) ecartes, (c) mecanisme reel](experiments/indexed_attention_core_mechanism.md)
- **2026-09-13** -- [Teacher-target precompute sharding, A40/A100 mauvais fit FP8](experiments/distillation.md)
- **2026-09-13** -- [Contre-expertise: plateau n_hops>=2 = bug de compresseur, pas limite de mecanisme](experiments/indexed_attention_core_mechanism.md)
- **2026-09-13** -- [Toy model: memoire moyen-terme jamais testee, read_step sweep, retraction label-leakage](experiments/toy_memory.md)
- **2026-09-14** -- [Etape 4 (generateur durci): chainage multi-hop confirme n_hops 2/3/4; Phase1bis/sm_cap](experiments/indexed_attention_core_mechanism.md)
- **2026-09-14** -- [Redo leak-free: copy porte l'info, cumsum sous-capacite; n_memory ablation](experiments/toy_memory.md)
- **2026-09-14** -- [Phase 3 real-text LR sweep + blocage LockstepLaneBatcher single-pass](experiments/real_text_baselines.md)
- **2026-09-16** -- [Bug n_facts_curriculum trouve et corrige (bloque au stage 1)](experiments/b1_associative_recall.md)
- **2026-09-16** -- [Baselines A/B/C (LFM2/OLMo/Qwen), bug tokenizer.vocab_size, 15/15](experiments/real_text_baselines.md)
- **2026-09-19** -- [I1: Phase1bis use_ff/n_register a n_hops durs -- aucun signal architectural (12/12)](experiments/indexed_attention_core_mechanism.md)
- **2026-09-19** -- [I4: mesure d'attribution douce -- signal reel, caveat checkpoint](experiments/i4_i6_i7_diagnostics.md)
- **2026-09-19** -- [Piste A: baselines real-text step-matched -- A bat C, ecart s'elargit](experiments/real_text_baselines.md)
- **2026-09-19** -- [A1/A2 (pre-check Phase 12): artefact de mesure trouve, signal reel = quasi-orthogonalite](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19** -- [A1/A2 v2: mesures corrigees -- score-scale mismatch confirme, quasi-duplication absente](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19** -- [Piste A LR sweep d_model=1024: A et C n'ont pas le meme LR optimal](experiments/real_text_baselines.md)
- **2026-09-19/20 (nuit)** -- [I7: [RETRACTE] confond de marge d'entrainement identifie](experiments/i4_i6_i7_diagnostics.md)
- **2026-09-19/20 (nuit)** -- [I6: probe de phase lineaire -- fonctionne, explique surtout par derive d'echelle](experiments/i4_i6_i7_diagnostics.md)
- **2026-09-20** -- [KD cablee dans train_prompt_response.py -- alignement char/token verifie, fallback CE-only sur mismatch tokenizer](experiments/distillation.md)
- **2026-09-19/20 (nuit)** -- [I3 etape 1: supervision resout le chainage dur mais bimodal, plafond 40%](experiments/i3_attention_supervision.md)
- **2026-09-19/20 (nuit)** -- [I5: ecart CPU/GPU n_hops=2 = budget d'entrainement, pas d'echelle](experiments/indexed_attention_core_mechanism.md)
- **2026-09-19/20 (nuit)** -- [item[5] (45/45): plateau N_step confirme, LR = seuil pas loi 1/N_step](experiments/nstep_lr_law.md)
- **2026-09-19/20 (nuit)** -- [I7 real-scale rerun (corrige): marge d'entrainement controle l'extrapolation](experiments/i4_i6_i7_diagnostics.md)
- **2026-09-19/20 (nuit)** -- [item[8] (18/18): plafond de LR replique et affine a n_hops=5/6](experiments/nstep_lr_law.md)
- **2026-09-19/20 (nuit)** -- [item[10]: audit des conclusions <5 seeds ou LR non revalide](experiments/item10_audit.md)
- **2026-09-19/20 (nuit)** -- [I3 etape 2 (70/70): supervision semble nuisible, mais ecart avec item[5] a eclaircir](experiments/i3_attention_supervision.md)
- **2026-09-19/20 (nuit)** -- [S3 etape 1 + S0: S0 PASS parfait, signal d'attribution brut absent](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19/20 (nuit)** -- [S3 probe lineaire: signal present dans x_l (test_acc=0.9997)](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19/20 (nuit)** -- [S3 probe lineaire, controle norme/direction: signal directionnel confirme (l2norm=0.9995)](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19/20 (nuit)** -- [S0.5: composition couche complete exacte (5/5 couches)](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19/20 (nuit)** -- [S3 extension 16 couches: attribution a chance, probe confirme signal directionnel](experiments/s3_ffn_attention_curriculum.md)
- **2026-09-19/20 (nuit)** -- [B1 (100/100): echappement du stage 2 rare (9-12%), pas typique](experiments/b1_associative_recall.md)
- **2026-09-19/20 (nuit)** -- [Piste A extended-budget (8/8): A bat C aux deux echelles, ecart s'elargit](experiments/real_text_baselines.md)
- **2026-09-19/20 (nuit)** -- [I2 (15/15): fenetre LR kdim128_decoupled propre et fiable, clos](experiments/indexed_attention_core_mechanism.md)
- **2026-09-19/20 (nuit)** -- [item[6] etape 4 more-seeds (15/15): n_hops 2/3/4 confirmes a 8 seeds](experiments/nstep_lr_law.md)
- **2026-09-20** -- [I3: diagnostic du confond 10%-vs-99.9% -- mismatch de defauts d'archi, pas la supervision (+ fix detach_sm_keys/sm_cap independant)](experiments/i3_attention_supervision.md)
- **2026-09-20** -- [i3_etape3: bug de lancement (n_distractors manquant) trouve et corrige](experiments/i3_attention_supervision.md)
- **2026-09-20** -- [Nouveau toolkit d'evaluation real-text: val split, checkpoint+extrapolation, ablation memoire, baseline LLM (LFM2-350M)](experiments/real_text_baselines.md)

- **2026-09-20** -- [Garde-fou methodologique sur pistea_c_nstep_sweep: LR fixe = meme piege que item5/item8](experiments/real_text_baselines.md)

- **2026-09-19/20 (nuit)** -- [S3 router entraine (3-couches + distant): mecanisme de routage valide](experiments/s3_ffn_attention_curriculum.md)

- **2026-09-20** -- [Pipeline prompt/thinking/answer: bug API target_input dict vs tenseur unique](experiments/prompt_response_pipeline.md)

- **2026-09-19/20 (nuit)** -- [pistea_ext2 (24/24): A bat C a budget eleve, gap ne se referme pas](experiments/real_text_baselines.md)

- **2026-09-20** -- [pistea_c_nstep_sweep complet (24/24): degradation monotone confirmee a d_model=1024, sweep croise lance](experiments/real_text_baselines.md)

- **2026-09-20** -- [i3_etape3 (20/20): n_head=1 confirme cause unique, supervision semble nuire une fois corrige](experiments/i3_attention_supervision.md)
- **2026-09-20** -- [n_slots sweep (9/9): M=1 echoue completement, M>=2 necessaire](experiments/i3_attention_supervision.md)

- **2026-09-20** -- [KD alignment: 0/2000 answer spans alignes sur openr1_math -- fallback CE-only total](experiments/distillation.md)

- **2026-09-20** -- [LFM2-1.2B precompute train scale (18k), jamais laisser un GPU modeste inactif](experiments/distillation.md)

- **2026-09-20** -- [§8ter KB ingestion MVP: smoke test reel, aucun crash, cout x7 confirme](experiments/kb_ingestion_8ter.md)

- **2026-09-20** -- [lr_warmup_sweep (30/30): LR optimal pour C plus haut que teste, warmup n'aide pas](experiments/real_text_baselines.md)
- **2026-09-20** -- [useff_sweep (8/8): use_ff=True bat clairement use_ff=False, capacite manquante possible](experiments/real_text_baselines.md)

## Résumé de nuit 2026-09-19 -> 2026-09-20 (autonomie confirmée par l'utilisateur ~15h35)

**En cours de rédaction, mis à jour au fil de la nuit -- lire la version la plus récente en tête de ce fichier au réveil.**

Réservations posées avant expiration :
- GPU : job 4121144 (7 GPU, abacus11/17/18, jusqu'à ~20:54) -- relève posée job **4121241** (7 GPU, cluster élargi abacus3/10/11/17/18/19/20/21/22/25/29, besteffort, walltime 12h, soumis ~15:38, Waiting).
- CPU : job 4121204 (paradoxe-27, jusqu'à ~21:21) -- relève posée job **4121245** (paradoxe, host=2, walltime 14h, queue normale, soumis ~15:39, Waiting).

Fils lancés ce soir (voir entrées datées ci-dessous pour le détail) : I1 (clos, 12/12), I4 (fait), Piste A step-matched (fait, **partiellement rétracté** -- LR non revalidé à d_model=1024) puis extended-budget (fait, A bat C, écart s'élargit), A1/A2 (fait, v1 rétracté pour artefact de mesure, v2 corrigé), I3 (étapes 1+2 collectées, écart avec item[5] à éclaircir), **I2 (clos, 15/15)**, I5 (clos), B1 (clos, 100/100). Balayage LR Piste A à d_model=1024 (item [1] de la file de nuit) lancé ~15:38.

**[MISE À JOUR ~22h]** : correction -- I2 était marqué "en cours" ci-dessus par erreur, en réalité clos (15/15) depuis un moment, découvert non-journalisé lors d'une vérification live. Toujours vérifier `runs/<nom>/state/` directement plutôt que de se fier à ce statut résumé, qui peut être en retard sur l'état réel du cluster.

**Points "à arbitrer" (à trancher par un humain, pas décidés seuls cette nuit)** -- liste vide pour l'instant, sera remplie au fil des résultats ambigus.

---

