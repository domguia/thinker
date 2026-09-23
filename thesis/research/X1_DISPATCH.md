# X1 — Dispatch pour experiment-manager (autonome ~48 h)

> Lire d'abord `RESEARCH_CHARTER.md` (§1, §3 H1/H2/H8, §4). Hypothèse testée : **H2** (et débogage du modèle).
> **CE-only validé par l'utilisateur le 2026-09-23** pour ces tâches synthétiques (pas de Teacher de même famille pour un vocabulaire de chiffres). Écrire `CE-only (validé 2026-09-23, X1)` dans le README de chaque run. Cette exception ne vaut **que** pour X1.

## 1. Question
Plus d'itérations au test (n_step_test > n_step_train) améliorent-elles l'exact match sur des instances **plus grandes que celles d'entraînement** ? Et Thinker se comporte-t-il comme un looped dense à params égaux ?

## 2. Grille (tout lancer en parallèle, remplir tous les GPU libres, 2e cluster autorisé)
| Tâche | Train | Test OOD | Réf. |
|---|---|---|---|
| T1 addition (embeddings Abacus / indices de position de chiffre) | 1–20 chiffres | 21–100 | McLeish 2024, 2405.17399 |
| T2 multiplication | 1–5 × 1–5 | jusqu'à 10 × 10 | McLeish 2024 |
| T3 prefix sums (parité binaire cumulée) | longueur 32 | 64–512 | Schwarzschild 2021 |
| T4 p-hop induction | p ≤ 8 | p 9–32 | Saunshi 2025 |
| T5 labyrinthes | 9×9 | 13×13 à 33×33 | Bansal 2022 |

Modèles, **à params égaux (±10 %)**, petits (≈5–20M), même tokenizer, même optimiseur :
- M1 Thinker (`core/indexed_thinker_model.py`)
- M2 Thinker + `--outer_norm`
- M3 looped dense (poids partagés, setup E3)
- M4 dense non bouclé (profondeur = n_step_train moyen)

Entraînement récurrent : n_step ~ U(1,8) (recette E8/E13). Test : n_step ∈ {1,2,4,8,12,16,24,32}. **3 seeds.** Total : 5 × 4 × 3 = 60 runs.

**Ordre** : T1 et T3 d'abord (rapides et les plus cités), puis T4, puis T5 et T2.

## 3. Portes de débogage (bloquantes, à passer AVANT la grille complète, 1 seed)
- G1 : M4 dense atteint ≥ 95 % d'EM **en distribution** sur T1 et T3. Sinon c'est un bug dans les données ou l'éval → corriger avant tout.
- G2 : M3 looped dense ≥ 95 % EM en distribution. Sinon, revalider le LR (voir la mémoire : budget → LR → config héritée, avant de conclure).
- G3 : M1 Thinker en distribution.
  - < 50 % alors que M3 ≥ 95 % → c'est le même défaut que H8. **Continuer quand même la grille** (c'est un résultat), et lancer en parallèle les variantes X2 sur T1 : (b) recall de l'input, (c) lecture depuis tous les latents.
- Vérifier l'absence de fuite : générateurs train et test disjoints, avec seeds séparées ; l'EM se calcule sur la réponse complète.

## 4. Mesures et sorties
- EM par (tâche, modèle, seed, taille, n_step_test). Aussi la CE, et la norme du latent par itération (lien avec le résultat E5 : divergence de direction).
- Un CSV par run, plus un agrégat `dev_notes/experiments/X1/results.csv`.
- Figure clé : EM vs taille, une courbe par n_step_test, un panneau par modèle.
- Mettre à jour `thesis/research/results_inventory.md` et `agents/OBJECTIVES_LOG.md` à chaque tâche terminée.

## 5. Table de décision (enchaîner sans attendre de retour)
| Observation | Action |
|---|---|
| M1 : l'EM OOD croît avec n_step_test (≥ +20 pts de 8 → 32 sur au moins une taille OOD) | **Résultat principal A.** Ajouter 2 seeds sur cette tâche. Étendre la plage OOD. |
| M1 ≈ M3, et les deux extrapolent | Résultat A, présenté comme « l'architecture préserve la propriété ». Priorité à X3 (efficience). |
| M3 extrapole, M1 non | Rapporter ; renforce le résultat C. Tester M2 et les variantes X2 sur la même tâche. |
| M2 corrige M1 | Remède trouvé : répliquer sur les 5 tâches, puis sur le setup LM de l'ablation (X2). |
| Aucun modèle n'extrapole sur une tâche | Vérifier l'encodage de position (Abacus pour T1/T2). Un seul essai. Sinon noter l'échec et passer à la tâche suivante. |
| Un run diverge (NaN) | Relancer une fois avec LR/2. Si ça recommence, noter et passer. |
| GPU libres | Seeds supplémentaires sur la tâche la plus prometteuse, puis X3, X5, X6 (`RESEARCH_DIRECTION.md` §3). |

## 6. Quand escalader vers l'utilisateur (sinon, ne pas interrompre)
- G1 échoue deux fois.
- Un changement d'architecture de M1 semble nécessaire, au-delà des flags existants.
- Le budget dépasse 2× l'estimation, ou il faudrait utiliser abacus27 pour autre chose que des seeds.
- **Échéance** : résultats agrégés au plus tard le **25/09 à 12:00 heure de Paris**, même partiels.
