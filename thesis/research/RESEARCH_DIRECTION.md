# RESEARCH DIRECTION — synthèse du 2026-09-23 (remplace le cadrage §1 de WRITING_PLAN.md)

Sources : `RESEARCH_CHARTER.md`, `results_inventory.md`, `lit_recurrence.md`, `lit_memory_inference.md`, `iclr2027_rules.md`, `vision_vs_specs.md`.

## 1. Ce que les faits disent

| Constat | Conséquence |
|---|---|
| Entraînement à n_step aléatoire → extrapolation déjà publiée (Geiping 2025, Kohli 2026 arXiv:2604.07822, Kuo 2026 arXiv:2606.29983) | C1 **n'est pas une contribution** : c'est une réplication dans notre architecture |
| E3 : le transformer dense bouclé (poids partagés) atteint 20.5 % d'accuracy ; Thinker est à 0.3–0.5 % | L'effondrement est **propre à Thinker** (lecture par cross-attention depuis le latent / goulot du latent), pas à la récurrence en soi → un défaut de conception localisable |
| Localité et préchargement déjà exploités dans les MoE (MoE-Infinity, Pre-gated MoE, Fiddler) ; spécialisation par domaine des experts contestée ("Myth of Expert Specialization" 2026) | H4/H5 ne sont défendables que **mesurés sur Thinker lui-même** ; la nouveauté est un accès front-loaded par itération, à comparer à la variation token par token des MoE |
| Aucun travail ne combine une mémoire KV auto-générée par token spécial, une KB inter-instances avec distracteurs et un entraînement de bout en bout (proches : Cartridges, ICAE, xRAG, Memorizing Transformers) | **H7 est le vrai trou dans la littérature** — mais pas testable proprement avant le 25 |

## 2. Cadrage recommandé pour ICLR (deadline 25/09 23:59 AoE)

**Type** : « architecture + analyse », honnête. Pas un papier de vision pure (rejeté sans résultat).

> Titre de travail : « Separating Computation from Knowledge: A Latent Recurrent Architecture with Indexed Memory — Extrapolation, Efficiency, and Where the Latent Readout Breaks »

1. **Motivation / vision** (≈1 p.) : I1–I3 de la charte (décomposition en étapes, compute mal condensé, connaissance externe organisée). Positionné vs looped/recurrent-depth (calcul) et memory layers/retrieval (connaissance) : personne ne combine les deux avec un accès mémoire planifiable.
2. **Résultat A** : réplication de l'extrapolation en profondeur dans une architecture à latent cross-attention (+ H2 algorithmique si X1 réussit — **c'est ce qui rend le papier convaincant**).
3. **Résultat B** : efficience d'inférence, **comparaison équitable** (X3).
4. **Résultat C** : l'effondrement de calibration est **localisé** : absent du looped dense, présent dans Thinker, 6 causes écartées ; + remède si X2 marche.
5. **Résultat D (si X5/X6 prêts)** : premières mesures de localité mémoire à travers les itérations → l'argument infra.
6. Limitations + agenda : H4, H6, H7, montée en échelle.

## 3. Expériences — pour experiment-manager (chacune : hypothèse → décision)

### Avant la deadline (≤ 48 h, lancer en parallèle)
- **X1 — Extrapolation algorithmique (H2)** : addition de longueur croissante (protocole Abacus, McLeish 2024 arXiv:2405.17399 : entraîner ≤20 chiffres, tester 30–100) et p-hop induction (Saunshi 2025). Modèles : Thinker vs looped dense (E3) vs dense, même params, n_step aléatoire, **3 seeds**. Mesure : exact match vs longueur × n_step_test. *Si* l'accuracy de Thinker croît avec n_step sur les longueurs hors distribution → résultat principal du papier. *Si* Thinker < looped dense → le rapporter, et le résultat C prend le dessus.
  - Note : tâche synthétique sans Teacher ; entraînement CE-only faute de Teacher pertinent (déviation à la règle KD **justifiée** : pas de Teacher de même famille sur ce vocabulaire de chiffres — l'utilisateur doit valider).
- **X2 — Remèdes à l'effondrement (H8)**, sur le setup existant de l'ablation, une variable à la fois : (a) outer normalization du latent entre itérations (Labovich 2026) ; (b) ré-injection de l'input / « recall » (Bansal 2022) ; (c) lecture de sortie depuis tous les latents ou depuis un résiduel non récurrent (cf. "Readout Blind Spot" arXiv:2606.24898 — à lire, voir X7) ; (d) tête d'arrêt stochastique (Kuo 2026). *Si* un remède ramène l'accuracy vers ~20 % → le résultat C devient « diagnostic + fix ».
- **X3 — Bench d'efficience équitable** : Thinker vs looped dense vs dense à params égaux, + Qwen3.5-0.8B pour référence ; FLOPs/tok, latence, tok/s, mémoire pic, **en fonction de la longueur de contexte** (512→8k) et de n_step ; KV-cache activé côté dense.
- **X4 — Seeds** : 3 seeds pour l'extrapolation n_step (existant) et Baseline C.
- **X5 — Localité d'accès mémoire (H5), analyse sur checkpoints existants** : instrumenter les accès à l'index par itération ; mesurer la fraction de mémoire touchée, le Jaccard entre l'itération 1 et l'itération N, et les octets chargés après l'itération 1, sur des batches mono-domaine vs multi-domaines. Coût : analyse seule, pas d'entraînement.
- **X6 — Organisation de l'index (H4), analyse sur checkpoints existants** : pureté/NMI des entrées ou clusters de l'index vs labels de domaine des données.
- **X7 — Lecture approfondie** (sous-agent + skill `read-arxiv-paper`) : 2606.24898 Readout Blind Spot, 2606.29983, 2604.15259, 2606.18206 Fixed-Point Reasoners, 2604.12946 Parcae → une note « quel remède correspond à notre effondrement ».

### Après la deadline (programme de recherche)
- **X8 — H7 mémoire auto-générée**, refaite proprement : baseline Cartridges/ICAE, nombre de distracteurs croissant.
- **X9 — H6 boosters** : flux de sortie auxiliaires greffés puis retirés, deep supervision, puis RL.
- **X10 — Index guidé** par clusters d'embeddings pré-entraînés vs index libre (suite de X6).
- **X11 — Échelle** 200M → 1B (abacus27), seconde famille (OLMo).
- **X12 — Comparaison MoE** de la localité (Mixtral/OLMoE instrumentés comme X5).

## 4. Questions ouvertes pour l'utilisateur
1. Validation du cadrage §2.
2. X1 en CE-only (tâche synthétique sans Teacher) : OK ?
3. Relectures réciproques ICLR : un auteur éligible (publication antérieure dans une venue listée) est requis, sinon rejet d'office — à vérifier sur OpenReview.
