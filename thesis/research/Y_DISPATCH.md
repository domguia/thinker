# Y — Dispatch « Séparation calcul / connaissance » (2e experiment-manager, en parallèle de X1)

> Lire d'abord `RESEARCH_CHARTER.md` (§1, §2bis, §4). Indépendant de X1 (`X1_DISPATCH.md`) : autre cluster, autres GPU, **aucun fichier partagé** hors `results_inventory.md`, qui est en ajout seul.
> Objectif : des **pilotes** qui montrent que la séparation est réelle et exploitable. Pas un résultat final. Viser des premiers chiffres en ~12 h, et tout finir en ≤ 48 h.
> **Échéance** : agrégat partiel au plus tard le **25/09 à 12:00 heure de Paris**.

## 0. Ce qu'on veut pouvoir écrire
« La connaissance vit dans la mémoire, le calcul vit dans le cœur. » Pour le justifier, il faut trois types de preuve :
1. **Double dissociation** : retirer la mémoire casse les tâches de connaissance mais pas le calcul ; réduire n_step casse le calcul mais pas le rappel.
2. **Échange de mémoire** : remplacer la mémoire par une nouvelle base de connaissances, **sans réentraîner le cœur**, et observer que le modèle utilise les nouvelles connaissances.
3. **Connaissance procédurale en mémoire** : une *règle* stockée en mémoire (et pas seulement un fait) est exécutée par le cœur, y compris une règle jamais vue.

Positionnement face à la littérature, à citer :
- RETRO et Memorizing Transformers échangent déjà une base de données, mais pour des faits, sans cœur récurrent.
- Memory Layers (Berges 2024) mettent la mémoire dans les poids.

Notre nouveauté : (a) la dissociation mesurée dans les deux sens, (b) la mémoire procédurale, (c) l'accès itératif, synchronisé sur les itérations du thinking.

## 1. Règle d'entraînement
- **Y0 et Y4 (LM réel)** : KD, comme le flagship (`--teacher_targets`, `--kd_alpha 0.5`, Teacher de même famille via le skill `model-families`, `--hidden_layers` incluant `last`).
- **Y1 à Y3 (synthétiques)** : CE-only est une **déviation à faire valider par l'utilisateur**, comme pour X1. En attendant, préparer les données et passer les portes de débogage, mais ne lancer la grille qu'une fois la validation reçue.

## 2. Expériences, par ordre de priorité

### Y0 — Ablation de la mémoire sur les checkpoints existants (analyse seule, à lancer tout de suite, ~2 h)
- Sur les checkpoints LM existants de Thinker, comparer à l'évaluation trois conditions : (i) mémoire normale, (ii) mémoire mise à zéro ou permutée entre documents, (iii) mémoire d'un autre domaine.
- Mesurer la ΔCE **par type de token** : entités nommées et nombres (connaissance), mots-outils et ponctuation (syntaxe). Utiliser un tagger NER/POS léger, spaCy par exemple.
- *Si* ΔCE(entités) ≫ ΔCE(mots-outils) → première évidence de séparation, directement sur le LM réel.
- *Si* ΔCE ≈ 0 partout → le modèle **n'utilise pas** sa mémoire. C'est un résultat important, lié au retrieval EM=0 %. Signaler immédiatement, et prioriser la porte G1 ci-dessous.

### Y1 — Échange de base de connaissances (synthétique, le cœur de la preuve)
- **Données** : biographies synthétiques, du type bioS de Allen-Zhu & Li, *Physics of LMs 3.1*. Une personne a environ 6 attributs. Les faits sont écrits dans la **mémoire externe** (entrées de l'index), pas dans l'input. Les questions portent sur une personne et un attribut.
- **Protocole** :
  - Entraîner sur la base A, soit N personnes.
  - Évaluer sur la base B : des personnes **nouvelles**, écrites en mémoire sans aucun gradient.
  - Faire varier N_B, la taille de la mémoire au test : 1×, 4× et 16× N_A, pour simuler des distracteurs.
- **Baselines** :
  - (a) un dense de même taille, avec les faits dans le contexte : c'est la borne haute, de type RAG ;
  - (b) un dense entraîné sur A, sans mémoire : il doit échouer sur B.
- *Si* EM(B) ≥ 70 % de EM(A) → la séparation est démontrée : la connaissance est remplaçable sans réentraînement.
- Mesure bonus : en posant des questions à deux sauts (« le patron de X est né où ? »), l'EM doit croître avec n_step. Ce lien avec H2 et H3 est le cœur du propos.

### Y2 — Double dissociation (reprend les modèles de Y1, et de X1 si possible)
- Suite mixte : des tâches de connaissance (Y1) et des tâches de calcul (addition ou prefix sums avec le même générateur que X1 ; se coordonner via le code, pas via les runs).
- Matrice 2×2 : {mémoire intacte / supprimée} × {n_step plein / n_step = 1}.
- Résultat attendu :

| | mémoire supprimée | n_step = 1 |
|---|---|---|
| tâches de connaissance | s'effondrent | tiennent |
| tâches de calcul | tiennent | s'effondrent |

- Figure clé du papier : une heatmap 2×2 par tâche.
- *Si* le calcul chute aussi quand on retire la mémoire → la mémoire sert de « scratchpad ». C'est intéressant, le noter, sans le cacher.

### Y3 — Mémoire procédurale (le plus original)
- **Tâche** : une table de fonction f : {0..K-1} → {0..K-1} est écrite en mémoire. L'input est (x, k), et la sortie attendue est f^k(x), c'est-à-dire appliquer f k fois (pointer chasing).
- **Protocole** : entraîner avec des tables f aléatoires et k ≤ 8. Tester sur des **tables jamais vues** (échangées en mémoire) et k de 9 à 32, avec n_step_test croissant.
- *Si* l'EM est élevée sur des tables nouvelles et croît avec n_step → **la règle est en mémoire, l'exécution est dans le cœur**. C'est la démonstration la plus directe de la vision (§2bis : « le savoir-faire est externe »).
- Variante, si le temps le permet : les opérations elles-mêmes en mémoire, par exemple une table d'addition modulo m, avec m échangé au test.

### Y4 — Distillation vers la séparation (combine la question KD, H6)
- Sur le LM réel, en KD depuis le Teacher de même famille, comparer trois variantes, une variable à la fois :
  - (a) mémoire apprise librement ;
  - (b) valeurs de la mémoire **initialisées avec les hidden states `last` du Teacher**, par cluster de documents, en réutilisant `precompute_doc_clusters.py` ;
  - (c) repr-KD sur les lectures mémoire.
- Mesures : la CE, et la ΔCE de Y0 (entités vs mots-outils) pour chaque variante.
- *Si* (b) ou (c) augmente la dépendance à la mémoire (ΔCE entités plus grande) sans dégrader la CE → **la distillation peut « déposer » la connaissance du Teacher dans la mémoire**. C'est l'argument « on hérite du savoir des LLM existants sans pré-entraîner ».

## 3. Portes de débogage (bloquantes, 1 seed, avant les grilles)
- **G1** : sur Y1, avec une base A minuscule (100 personnes) et les faits placés dans l'**input**, le modèle atteint ≥ 95 % d'EM. Sinon, bug de données ou d'éval.
- **G2** : même test avec les faits placés en **mémoire**, sur une base A de 100 personnes.
  - Si l'EM est < 50 % alors que G1 est ≥ 95 % → le chemin d'accès à la mémoire est cassé (cf. la mémoire projet sur le bug de pooling K/V).
  - Dans ce cas, **arrêter Y1 à Y3**, diagnostiquer le chemin mémoire (attention sur l'index, gradients vers les clés, top-k), faire un rapport et escalader. Y0 et Y4 continuent.
- **G3** : contrôle de fuite. Personnes et tables disjointes entre train et test, et vérification que le cœur ne peut pas mémoriser B (vu qu'il n'a reçu aucun gradient).

## 4. Table de décision (enchaîner sans attendre de retour)
| Observation | Action |
|---|---|
| G2 réussit | Grilles Y1 → Y3 → Y2, 3 seeds, sur tous les GPU libres |
| Y1 réussit | Augmenter N_B jusqu'à ce que ça casse (courbe EM vs distracteurs) ; ajouter les questions à 2 sauts |
| Y1 échoue mais G2 réussit | Échec de généralisation, pas d'accès. Essayer ×10 de diversité en train (plus de bases A), une seule fois |
| Y3 réussit | Priorité absolue : 3 seeds, figure EM vs k × n_step, variante avec les opérations en mémoire |
| Y4 (b) ou (c) augmente la dépendance à la mémoire | Répliquer sur un 2e seed ; lancer X5 (localité) sur ce checkpoint |
| Un run diverge | LR/2, une seule fois, sinon noter et passer |

## 5. Escalader vers l'utilisateur uniquement si
- G2 échoue (le chemin mémoire est cassé).
- La validation CE-only de Y1 à Y3 est nécessaire.
- Une modification d'architecture au-delà des flags existants semble requise.
- Y0 montre ΔCE ≈ 0.

## 6. Sorties
- `dev_notes/experiments/Y/{Y0..Y4}/results.csv`, et un README par expérience (hypothèse, commande, seed, régime CE ou KD).
- Figures : barres Y0 par type de token ; EM vs distracteurs (Y1) ; heatmap 2×2 (Y2) ; EM vs k × n_step (Y3).
- Ajouts à `thesis/research/results_inventory.md` et `agents/OBJECTIVES_LOG.md`.
