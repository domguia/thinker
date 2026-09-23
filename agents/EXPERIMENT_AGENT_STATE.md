# État courant -- experiment-agent

(Fichier d'état, pas un journal -- écrase à chaque mise à jour majeure. Le
détail chronologique/résultats complets restent dans
`dev_notes/experiments/prompt_response_pipeline.md`.)

## REPRISE RAPIDE (préparé pour interruption transfert de compte, 21:20)
1. Lire ce fichier en entier (état complet ci-dessous), puis vérifier
   `git log --oneline -10` pour confirmer aucun commit perdu.
2. Vérifier le run en cours : `ssh nancy.g5k 'OAR_JOB_ID=6938205 oarsh
   graffiti-3 "tail -8 ~/thinker/logs/x1_m3_addition_seed2_retry.log"'`
   (M3/T1 seed2, ETA ~21:47, devrait être FINAL ou proche à la reprise).
3. Si FINAL >=95% : grille T1 M1-M4 (3 seeds chacun) COMPLÈTE. Consolider
   dans `dev_notes/experiments/X1/results.csv` (même format que data-agent
   pour T3), MAJ `results_inventory.md` + `OBJECTIVES_LOG.md`, commit,
   informer data-agent + supervisor-agent que T1 est clos.
4. Ensuite : T4 (p-hop induction, data-agent) en attente d'arbitrage GPU --
   voir si T1+T3 sont bien tous deux clos avant d'investir dessus. T5/T2
   pas commencés (prochains dans l'ordre du dispatch après T4).
5. Les GPU jobs Grid'5000 continuent de tourner indépendamment d'une
   interruption de session Claude -- rien à relancer sauf si un job a
   effectivement crashé (vérifier `oarstat -j <id> -f`).

Dernière mise à jour : 2026-09-23 ~20:05. Les monitors précédents sont tombés
(session fermée/rouverte entre-temps) -- vérifié directement, les 4 jobs GPU
tournent toujours sains (Nancy graffiti-4, job group 6938131-6938134), aucune
perte. Contexte plus court désormais -- je log plus fréquemment dans ce fichier
à chaque check, pas seulement en fin de gate.

Résultats FINAL à 20:15 :
- M4/T1 seed1 : 99.00% in-dist EM, OOD 6.00% -- **G1 reconfirmé** sur seed
  supplémentaire (déjà VALIDÉE avec seed0 à 99%).
- M4/T1 seed2 : 99.00% in-dist EM, OOD 6.50% -- **G1 reconfirmé** (3e seed).
- G2/T1 (M3 addition) seed0 (n_step_test=8, n_step_train_max=8) : **ÉCHEC DE
  SEUIL** 17.50% in-dist EM (pas un crash -- loss basse ~0.085 mais EM basse,
  différent du bug attention_mask de G1 qui donnait EM=0.0000 strict).
- M3/T1 seed1 (même config) : 70.00% in-dist EM -- variance seed→seed forte,
  suggère sous-capacité/sous-entraînement plutôt qu'un bug structurel.
- Hypothèse : n_step_test=8 insuffisant pour la propagation de retenue sur
  addition multi-chiffres (1-20 chiffres) -- tâche différente de T3
  (prefix_sum/parité) où data-agent a eu EM=1.0 avec la même mécanique.
- **Relancé** : x1_g2_m3_addition_v3.log, job en cours sur graffiti-4,
  n_step_train_max=16 (au lieu de 8), max_steps=30000 (au lieu de 20000),
  seed=0, save_dir runs/x1_g2_m3_addition_v3. Démarré 20:16, ~13min pour
  20k steps précédemment donc ETA ~20min pour 30k. Si toujours <95% après
  ce relancement, ce sera la 2e tentative de gate -- envisager escalade
  selon X1_DISPATCH.md §6 ou accepter comme résultat négatif si loss
  plafonne clairement (pattern G3/T3 de data-agent : EM=0 stable = résultat
  valide, pas un bug).
- data-agent : G3/T3 (Thinker/M1, prefix_sum) terminé, EM=0.0000 in-dist et
  OOD -- cause identifiée (mean-pooling du register détruit l'ordre des bits,
  incompatible avec la dépendance à l'ordre du prefix-sum/parité). Résultat
  négatif valide (classe H8), pas un blocage, committé (f23cf8d). Harnais
  `learn/x1/train_thinker.py` générique, réutilisable pour G3/T1 une fois
  G2/T1 clos. data-agent passe à la grille complète T3 sur graffiti-11.

## Proactivité GPU (20:20, directive supervisor-agent + infra-agent)
3 des 4 GPU graffiti-4 étaient devenus idle (jobs M4 seed1/seed2 + M3 seed1
terminés, GPU 0% util confirmé nvidia-smi) pendant que G2/T1 v3 tournait sur
le 4e -- relancés immédiatement, sans attendre G2/T1 :
- `x1_g3_m1_addition.log` (job 6938132) : G3/T1, M1 Thinker baseline B,
  n_step_train_max=16, n_step_test_sweep 1..32, seed=0, 30k steps.
- `x1_x2a_m2_addition.log` (job 6938133) : X2(a)/T1, M2 Thinker+outer_norm
  (remède déjà implémenté dans core/indexed_thinker_model.py, flag
  --outer_norm), même config, seed=0. Justifié par la table de décision
  X1_DISPATCH §5 (data-agent a déjà observé le défaut H8 sur G3/T3, donc
  tester le remède outer_norm en parallèle sur T1 est autorisé sans attendre).
- `x1_g3_m1_addition_seed1.log` (job 6938134) : 2e seed de G3/T1 M1, même
  config, seed=1 -- tâche la plus incertaine/bloquante, priorité aux seeds.
- X2(b)/(c)/(d) (recall input, lecture multi-latents, tête d'arrêt) : PAS
  encore implémentés dans le code (vérifié, seul outer_norm=X2a existe) --
  nécessitent du dev, pas juste un lancement. À faire si G3/T1+X2a confirment
  le besoin d'un remède plus poussé.
- Les 4 GPU graffiti-4 sont maintenant tous actifs (aucun idle).

## G2/T1 VALIDÉE (20:44) -- n_step_train_max=16
FINAL in-distribution EM=0.9950 (n_step_test=16), OOD=0.50% (attendu, c'est
H2). Détail complet dans le journal (prompt_response_pipeline.md).

## G3/T1 -- défaut H8 confirmé sur addition aussi, X2(a) inefficace (20:44)
M1 Thinker (seed0+seed1) : loss plafonne ~2.2, EM=0.0000 -- même défaut que
G3/T3 de data-agent (mean-pooling casse l'ordre place-value). X2(a)
outer_norm testé en parallèle : ne corrige pas (EM≈0.5%). Détail dans le
journal. X2(b)/(c)/(d) pas implémentés -- besoin de dev avant de tester.

## Prochaines actions (20:44)
- G1+G2/T1 validées, G3/T1 négatif (résultat valide, classe H8) -- T1 peut
  passer à la grille complète (M1-M4 x n_step_test sweep x 3 seeds) sur les
  GPU libres, en parallèle de T4 (prochaine tâche selon l'ordre §2).
- Les 2 runs G3/T1 (M1 seed0+seed1) et X2a/T1 continuent jusqu'à leur fin
  naturelle (30k steps) -- pas besoin de les tuer, ils fournissent le sweep
  n_step_test complet (1..32) en fin de run pour la grille finale.
- GPU graffiti-4 (job 6938131, ex-G2/T1 v3) réutilisé : `x1_m3_addition_seed2_v3.log`,
  M3/T1 seed2 (n_step_train_max=16, n_step_test=16 fixe), complète les 3 seeds
  de la grille pour M3/T1.

## G3/T1 + X2(a)/T1 terminés (21:00) -- défaut H8 confirmé sur T1
M1 seed0+seed1 + X2a seed0 : sweep complet (n_step_test 1..32), EM=0.0000
partout, in-dist ET OOD -- identique à G3/T3 de data-agent (24/24 cellules à
0 chez eux aussi). data-agent a aussi testé X2(b) (--enable_kb, recall input)
: négatif également (loss plafonne ~ln(2), même signature). 2 remèdes
isolés testés, tous deux négatifs -- avis donné à data-agent : tenter X2(c)
une fois avec budget capé, sinon documenter comme limite ouverte et prioriser
la fin de grille vu la deadline (2j restants).

3 GPU graffiti-4 libérés relancés immédiatement (job 6938132/33/34) pour
compléter les seeds manquantes de la grille T1 : `x1_g3_m1_addition_seed2.log`
(M1 seed2), `x1_x2a_m2_addition_seed1.log` / `_seed2.log` (M2 seed1+2).
Les 4 GPU graffiti-4 actifs (M3 seed2, M1 seed2, M2 seed1, M2 seed2).

## INCIDENT (22:09) -- panne nœud graffiti-4, 4 jobs en Error simultanément
Les 4 jobs (6938131-6938134) sont tombés en état `Error` en même temps
(`SWITCH_INTO_ERROR_STATE`, pas une préemption normale -- vraisemblablement
panne matérielle/nœud). Signalé à infra-agent.

**Sauvé (données sur NFS avant la panne)** :
- M1/T1 seed2 : sweep complet jusqu'à n_step_test=32, EM=0.0000 partout --
  **3e seed confirmant le défaut H8** (avec seed0/seed1 déjà clos).
- M2/T1 (outer_norm) seed1 : sweep complet, EM≈0.005 in-dist (négligeable),
  0.0000 OOD -- confirme X2(a) inefficace.
- M2/T1 seed2 : sweep complet, EM=0.0000 partout.
- **Grille M1+M2/T1 (3 seeds chacun) est donc COMPLÈTE et négative** --
  résultat prêt à consolider dans results.csv/results_inventory.md.

**Perdu (à relancer)** : M3/T1 seed2 (looped-dense), coupé à step 25220/30000
(loss ~0.05-0.09, sur la bonne trajectoire vu G2/T1 v3). Nouveau GPU réservé
(job 6938205, besteffort, en attente de démarrage) pour relancer.

infra-agent confirme (21:16) : graffiti-4 juste "Absent (standby)"
(comment=OK, maintenance=NO), pas de panne matérielle réelle signalée --
probable reboot/coupure réseau transitoire, pas à éviter à l'avenir.

Job 6938205 démarré sur graffiti-3 (pas graffiti-4) -- relancé M3/T1 seed2 :
`x1_m3_addition_seed2_retry.log`, save_dir `runs/x1_g2_m3_addition_seed2_retry`,
même config (n_step_train_max=16, n_step_test=16, 30k steps). ETA ~25-30min.
Une fois FINAL >=95% (quasi certain), grille T1 M1/M2/M3/M4 (3 seeds chacun)
sera complète -- consolider dans dev_notes/experiments/X1/results.csv comme
data-agent l'a fait pour T3, mettre à jour results_inventory.md/OBJECTIVES_LOG.md.

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
