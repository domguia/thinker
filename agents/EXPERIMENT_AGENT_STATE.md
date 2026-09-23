# État courant -- experiment-agent

(Fichier d'état, pas un journal -- écrase à chaque mise à jour majeure. Le
détail chronologique/résultats complets restent dans
`dev_notes/experiments/prompt_response_pipeline.md`.)

Dernière mise à jour : 2026-09-23 ~16:45, en cours de session (deadline papier 26/09).

## En cours
- **E5** (WRITING_PLAN §4, P1) : diagnostic mécanistique par itération (norme/rang effectif
  de R, similarité cosinus, logit-lens via sm_k/sm_v) -- script écrit
  (`learn/indexed_attention/diagnose_iteration_dynamics.py`), en cours de transfert/lancement
  sur Rennes, checkpoint cible `retrieval_kd_fullcov_best.pt`.

## Terminé et rapporté cette session (résumé, voir journal pour détails)
- Baseline C (dense transformer non-récurrent, wikitext) : calibration 21.2% vs Thinker ~0.3-0.5%.
- Phase17-21 : CE-vs-KD sur 4 datasets (wikitext/retrieval/math/tinystories), collapse universel confirmé.
- Phase20 : retrieval KD couverture top-K full -- collapse persiste (couverture pas la cause).
- Phase22/E3 : scaling (d_model 256->512) ET récurrence poids-partagés seule (transformer dense en boucle)
  -- NI L'UN NI L'AUTRE ne reproduit le collapse Thinker. Cause reste spécifique à l'architecture Thinker.
- Benchmark efficience : Thinker 5.8x moins de params, 8.5x plus rapide que Qwen3.5-0.8B.
- E1 (6/6 seeds, fixed vs random n_step) : gap extrapolation +8.82 (fixe) vs +0.23 (aléatoire), robuste,
  aucun chevauchement sur 3 seeds.

## En attente de décision supervisor-agent
- Rien pour l'instant (E5 en cours d'exécution autonome, "enchaîne" déjà donné).

## GPUs actifs connus (peut être obsolète -- vérifier oarstat avant de supposer)
- abacus27-1 (Rennes, H100) : libre après random_seed2, candidat pour E5.
- graffiti-1/graffiti-11 (Nancy) : libres après complétion des runs E1.

## Notes pour la prochaine reprise (moi-même ou un autre agent)
- Toujours vérifier `git log --oneline -20` et la fin de
  `dev_notes/experiments/prompt_response_pipeline.md` en complément de ce fichier --
  ce fichier donne le "quoi", le journal donne le "pourquoi/comment/chiffres".
- Convention chemins : Nancy home != Rennes home != storage3 partagé -- toujours vérifier avant de lancer
  un script sur un nouveau site/node (voir leçons `.claude/skills/grid5000/SKILL.md`).
