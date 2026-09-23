# État courant -- experiment-agent

(Fichier d'état, pas un journal -- écrase à chaque mise à jour majeure. Le
détail chronologique/résultats complets restent dans
`dev_notes/experiments/prompt_response_pipeline.md`.)

Dernière mise à jour : 2026-09-23 ~19:45, session locale sur le point d'être fermée
par l'utilisateur -- les 4 jobs GPU (Nancy graffiti-4, job group 6938131-6938134)
continuent de tourner indépendamment sur Grid'5000, seul le suivi local (Monitor/
ScheduleWakeup) sera perdu. À la reprise : relire ce fichier + relancer un check
des logs (commande ci-dessous), pas besoin de relancer les jobs.

## En cours -- X1 (H2, extrapolation algorithmique OOD)
- Discipline "économie de tokens" active (consigne permanente supervisor-agent) :
  rapports courts, batchés, escalade uniquement selon X1_DISPATCH.md §6.
- **T1 (addition) : G1 VALIDÉE** (99% EM in-dist, 20000 steps, ~16min sur H100/GPU
  Nancy graffiti-3, job 6938128). Bug trouvé+fixé : `attention_mask` manquant dans
  `greedy_generate()` (learn/x1/train_dense.py) cassait toute génération malgré
  loss d'entraînement saine (commit 2aaea79). OOD=6.5% -- normal, c'est la question
  H2 elle-même, pas un échec de gate.
- **G2/T1 (M3 looped-dense) EN COURS** sur graffiti-4, 4 jobs parallèles (besteffort,
  job group 6938131-6938134), tous sains à la dernière lecture (2026-09-23 ~19:45,
  aucune erreur, loss en baisse) :
  - `x1_g2_m3_addition_v2.log` : step=11000/20000, loss~0.12-0.19, n_step curriculum OK
  - `x1_m4_addition_seed1.log` (réplicat seed1 M4/T1) : step=13380/20000, loss~0.04
  - `x1_m4_addition_seed2.log` (réplicat seed2 M4/T1) : step=12960/20000, loss~0.01-0.02
  - `x1_m3_addition_seed1_v2.log` (M3/T1 seed1) : step=10760/20000, loss~0.11-0.15,
    dernier EM lu à step 3000 = 0.1100 (n_step_test=8), normal si tôt dans training
  - Bug déjà rencontré + fixé sur ces 4 runs : `use_cache=False` manquant dans
    `learn/x1/train_looped_dense.py` (commit 5f3cc40) -- ne pas rediagnostiquer si
    ça réapparaît, juste vérifier que le fix est bien dans le fichier committé.
  - Commande de vérification à la reprise :
    `ssh nancy.g5k 'for f in x1_g2_m3_addition_v2 x1_m4_addition_seed1 x1_m4_addition_seed2 x1_m3_addition_seed1_v2; do echo "=== $f ==="; OAR_JOB_ID=6938131 oarsh graffiti-4 "tail -5 ~/thinker/logs/$f.log 2>/dev/null"; done'`
  - Une fois `FINAL in-distribution EM=` >=95% sur `x1_g2_m3_addition_v2.log` :
    G2/T1 validée, committer résultat (journal + ce fichier), comme pour G1/T1.
- **T3 (prefix_sum) : G1 + G2 VALIDÉES par data-agent** (EM=1.0 in-dist les deux).
  data-agent construit maintenant le harnais Thinker (M1) pour le vocab synthétique
  X1 -- plan : disable_kb=True (Baseline B, pas de KB externe pour ces tâches),
  kb_tokens=prompt complet, génération autorégressive avec position_ids place-value.
  Je ne duplique pas ce travail (accordé par message).
- G3 (M1 Thinker) pas encore commencé sur aucune tâche -- bloqué sur le harnais
  Thinker en cours de construction par data-agent.
- Grille complète (5 tâches x 4 modèles x 3 seeds = 60 runs) pas commencée au-delà
  des gates.

## Terminé et rapporté avant X1 (résumé, voir journal pour détails)
- E5 (diagnostic mécanistique) : R converge en direction mais diverge en norme
  (sans borne), corrobore l'ablation outer_norm (E13, agent2) comme fix candidat.
- Baseline C, phase17-22/E3 (récurrence poids-partagés seule ne reproduit PAS le
  collapse Thinker), benchmark efficience (5.8x moins de params, 8.5x plus rapide),
  E1 (gap extrapolation +8.82 fixe vs +0.23 aléatoire, 6/6 seeds).
- E8 (nmax=4/16 filler priority) EN PAUSE, moins prioritaire que X1.

## Notes pour la prochaine reprise (moi-même ou un autre agent)
- Toujours vérifier `git log --oneline -20` et la fin de
  `dev_notes/experiments/prompt_response_pipeline.md` en complément de ce fichier.
- Lire `thesis/research/RESEARCH_CHARTER.md` + `thesis/research/X1_DISPATCH.md`
  en premier si reprise sur X1 -- table de décision §6 pour enchaîner sans
  repasser par supervisor sauf conditions d'escalade explicites.
- Split de travail avec data-agent : je fais T1, data-agent fait T3 (+ harnais
  Thinker générique, réutilisable pour T1 une fois prêt) -- se coordonner avant
  de prendre une tâche/modèle pour éviter collision.
- GPU actif : Nancy graffiti-3, job 6938128 (besteffort, peut être préempté --
  checkpoints via --save_dir survivent, mais train_dense.py n'a pas encore de
  `--init_from_checkpoint`, à ajouter si une préemption survient en plein run long).
