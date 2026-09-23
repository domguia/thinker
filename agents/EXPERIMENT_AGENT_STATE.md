# État courant -- experiment-agent

(Fichier d'état, pas un journal -- écrase à chaque mise à jour majeure. Le
détail chronologique/résultats complets restent dans
`dev_notes/experiments/prompt_response_pipeline.md`.)

Dernière mise à jour : 2026-09-23 ~19:14, en cours de session (deadline X1 25/09 12h00).

## En cours -- X1 (H2, extrapolation algorithmique OOD)
- Discipline "économie de tokens" active (consigne permanente supervisor-agent) :
  rapports courts, batchés, escalade uniquement selon X1_DISPATCH.md §6.
- **T1 (addition) : G1 VALIDÉE** (99% EM in-dist, 20000 steps, ~16min sur H100/GPU
  Nancy graffiti-3, job 6938128). Bug trouvé+fixé : `attention_mask` manquant dans
  `greedy_generate()` (learn/x1/train_dense.py) cassait toute génération malgré
  loss d'entraînement saine (commit 2aaea79). OOD=6.5% -- normal, c'est la question
  H2 elle-même, pas un échec de gate.
  - Prochaine étape : G2/T1 (M3 looped-dense) -- réutiliser
    `learn/x1/train_looped_dense.py` (générique, déjà validé par data-agent sur T3).
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
