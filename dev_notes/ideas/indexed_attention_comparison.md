# Comparaison détaillée — Thinker "Indexed Attention" vs mécanismes de sparsité existants

Objectif de ce document : avant toute implémentation, comparer point par point les mécanismes d'attention creuse/indexée existants (QSA, DSA, NSA, GDN — discutés en tout début de `raw/Branch-•-Indexed-Attention.md`, lignes 1-500) avec la variante "Indexed Attention" imaginée pour Thinker (synthétisée dans `dev_notes/ideas/branch_indexed_attention_synthesis.md`). Aucun choix d'implémentation n'est fait ici — c'est une base de décision.

**Références retrouvées et vérifiées (2026-09-13, demande explicite de l'utilisateur — ces trois mécanismes ont été discutés dans `raw/Branch-•-Indexed-Attention.md` sans citation formelle jusqu'ici ; ajoutées aussi dans `thesis/paper/references.typ` [34]-[36])** :
- **NSA** : Yuan, J., Gao, H., Dai, D., et al. (2025). *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*. arXiv:2502.11089. (DeepSeek.)
- **DSA** : DeepSeek-AI (2025). *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models* — introduit DeepSeek Sparse Attention et son "Lightning Indexer". arXiv:2512.02556.
- **QSA** : Qiu, Z., Wang, Z., Li, X., et al. (2026). *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability*. arXiv:2608.30320. (Qwen Team, Alibaba — introduit Qwen Sparse Attention dans Qwen3.8-Flash-Next, un hybride avec Gated DeltaNet, une couche QSA pour 3 couches GDN — cohérent avec ce qui était déjà documenté ici à partir des notes de conversation.)

**Clarification terminologique** : "Indexed Attention" est le nom de la variante Thinker elle-même (registre latent + SM + KB hiérarchique indexée). "Branch" dans le nom du fichier source n'est qu'un intitulé de conversation/branche de discussion — ce n'est pas un mécanisme ou un concept architectural séparé. Il n'y a donc pas deux idées concurrentes, une seule : Thinker enrichi d'une mémoire externe indexée.

## 0. Un point de contexte essentiel : la littérature et Thinker ne résolvent pas le même problème

Les trois mécanismes de la littérature (QSA, DSA, NSA) sparsifient l'attention **sur la séquence en cours** d'un transformer standard (context window jusqu'à 1M tokens, mais éphémère — recalculé à chaque nouvelle conversation/document). Rien n'y est persistant entre deux inférences indépendantes.

Le besoin Thinker est différent : une **base de connaissance externe, statique, préremplie, persistante entre toutes les inférences** (potentiellement des millions à des milliards de tokens), consultée par un registre latent récurrent qui, lui, maintient en plus une mémoire court-terme (SM) propre à l'inférence en cours. Les mécanismes de la littérature sont donc des **briques à emprunter**, pas des architectures à copier telles quelles — le problème de fond (indexer un KV cache massif et permanent) n'est traité nativement par aucun des trois.

## 1. Panorama des mécanismes existants (état de l'art discuté dans la conversation)

### 1.1 QSA (Qwen Sparse Attention)
- **Granularité** : micro-blocs fixes de 4 tokens.
- **Indexation** : clé de bloc = moyenne non paramétrique des clés individuelles ($\bar K_b = \frac1B\sum K_{b,i}$), score = $Q\bar K_b^T$, Top-k blocs retenus pour attention fine.
- **Entraînement de l'indexeur** : **aucune perte auxiliaire, aucun STE** — l'indexeur est généralement gelé pendant le SFT ; seules les clés des blocs effectivement sélectionnés reçoivent du gradient (via l'attention fine standard sur les tokens retenus).
- **Conséquence** : blocs jamais sélectionnés = "dead blocks", pas de mise à jour, risque de boucle "les riches s'enrichissent". Un drift de $\bar K_b$ est possible même sans paramètre propre (car $W_K$ continue d'évoluer via l'attention fine, donc la moyenne se déplace) si l'indexeur n'est pas gelé.
- **Rôle architectural** : alterné avec des couches récurrentes (Gated DeltaNet) — 1 couche QSA pour 3 couches GDN. QSA n'est pas censé porter seul la mémoire du modèle.
- **Coût mémoire entraînement** : $O(N\cdot K)$ direct, pas de calcul dense.

### 1.2 DSA (DeepSeek Sparse Attention)
- **Granularité** : indexeur au niveau token (pas de blocs fixes).
- **Indexation** : perte auxiliaire de distillation $\mathcal L_{align} = D_{KL}(P_{dense} \| P_{indexer})$ pendant une **phase d'alignement** courte (le calcul dense complet est temporairement nécessaire, donc coûteux en VRAM $O(N^2)$ pendant cette phase seulement).
- **Entraînement de l'indexeur** : gradient direct via la perte KL — l'indexeur apprend activement à imiter l'attention dense, sans dépendre uniquement des blocs sélectionnés.
- **Coût mémoire entraînement** : $O(N^2)$ pendant l'alignement, puis $O(N\cdot K)$ une fois l'indexeur stabilisé/gelé.
- **Limite** : la phase d'alignement dense est en tension directe avec l'objectif même de la sparsité (elle nécessite temporairement ce qu'on cherche à éviter) — acceptable seulement si elle reste courte/ponctuelle.

### 1.3 NSA (DeepSeek Native Sparse Attention)
- **Granularité** : 3 branches parallèles — locale (fenêtre glissante), compressée (blocs de 16-32 tokens compressés par une projection **apprenable**), sélectionnée (Top-k sur les blocs compressés, puis attention fine sur les tokens bruts des blocs retenus).
- **Fusion** : $O = g_{loc}\odot O_{loc} + g_{cmp}\odot O_{cmp} + g_{sel}\odot O_{sel}$, portes $g$ apprises (sigmoïde/softmax sur une couche linéaire).
- **Entraînement de l'indexeur** : **pas de perte auxiliaire nécessaire** — la branche compressée $O_{cmp}$ participe directement à la sortie et donc à la perte principale ; son gradient met à jour la compression sans distillation externe.
- **Rôle intuitif de $\tilde V_b$** (au-delà de faire circuler le gradient) : un résumé sémantique du bloc ("vue panoramique"), qui sert de **filet de sécurité** si l'indexeur rate un bloc pertinent au Top-k — ce bloc contribue quand même via la branche compressée.
- **Coût mémoire entraînement/inférence** : 3 branches calculées à chaque étape (mais la branche compressée reste légère, $N/C$ vecteurs).

### 1.4 Gated DeltaNet (GDN) — récurrence complémentaire, pas un mécanisme d'indexation
- Récurrent **le long de la séquence** (façon Mamba), pas en profondeur (façon Looped Transformer/Thinker). État cumulatif de taille fixe mis à jour via une règle Delta (correction d'erreur) + gating adaptatif.
- Utilisé en alternance avec QSA (pas un concurrent de l'indexation, un complément pour la continuité globale à coût $O(N)$).
- Pertinent pour Thinker seulement par analogie : le registre latent `Z_t`/`L_n` de Thinker joue déjà un rôle de "mémoire compressée continue", mais via récurrence **en profondeur** (poids partagés entre itérations), pas en séquence — donc pas directement transposable, à noter comme piste distincte si un jour on veut fusionner les deux formes de récurrence.

### Tableau de synthèse — état de l'art

| Mécanisme | Granularité | Comment l'indexeur apprend | Gradient sur blocs non sélectionnés | Coût mémoire entraînement | Risque principal |
|---|---|---|---|---|---|
| QSA | micro-blocs (4 tok) | rien (souvent gelé) | aucun | $O(N \cdot K)$ | dead blocks, drift silencieux |
| DSA | token-level | perte KL vs dense (phase d'alignement) | indirect via KL | $O(N^2)$ pendant alignement | coût de la phase dense, désynchronisation post-alignement |
| NSA | blocs (16-32 tok) | natif, via $O_{cmp}$ dans la perte principale | indirect via branche compressée (filet de sécurité) | $O(N\cdot(W+N/C+k))$, 3 branches | complexité d'implémentation (3 branches + gating) |

## 2. Les axes de la variante "Indexed Attention" de Thinker, mis en regard de ces mécanismes

Cette section relie chaque variante identifiée dans `branch_indexed_attention_synthesis.md` (§4) au mécanisme de la littérature dont elle s'inspire ou dont elle diverge, avec le compromis induit.

| # | Axe de variation (Thinker) | Analogue dans la littérature | Divergence propre à Thinker |
|---|---|---|---|
| 1 | Agrégation `M=1` vs `M>1` par nœud | QSA/NSA compressent en **un seul** vecteur par bloc ($M=1$ implicite) | `M>1` (façon Slot Attention) est une extension **absente de QSA/DSA/NSA** — aucun des trois ne teste plusieurs slots sémantiques par bloc |
| 2 | Fonction d'agrégation (pooling / Perceiver / récurrent à poids partagés) | QSA = moyenne non paramétrique ; NSA = projection/pooling **apprenable** mais poids propres par étage | Thinker propose un **partage de poids inter-étages illimité** (le même compresseur à tous les niveaux de l'arbre) — ni QSA ni NSA ne partagent les poids de compression entre niveaux (NSA n'a qu'un seul niveau de compression, pas une hiérarchie) |
| 3 | Softmax unifié multi-niveaux vs branches à gating séparées | NSA = branches séparées + gating appris (3 branches, **pas de hiérarchie de niveaux**) | Le softmax unifié est une proposition propre à l'utilisateur, jamais testée dans la littérature citée ici — risque mathématique identifié (biais d'échelle) nécessitant RMSNorm par niveau, non requis dans NSA car chaque branche a son propre gate normalisé séparément |
| 4 | Dropout stochastique des niveaux hauts à l'entraînement | Aucun analogue direct dans QSA/DSA/NSA (qui n'ont pas de hiérarchie multi-niveaux) — plus proche d'un *stochastic depth*/*layer dropout* générique | Spécifique à la hiérarchie Thinker, à valider empiriquement |
| 5 | Résolution implicite (softmax arbitre) vs routeur explicite | QSA/DSA/NSA utilisent tous un score explicite + Top-k dur (routeur explicite, discret) | Thinker envisage une **résolution implicite continue** (le softmax unifié absorbe le choix de niveau) — plus proche d'une relaxation Soft Top-K que d'un Top-K dur, evite le blocage de gradient de QSA mais hérite du risque de dilution documenté (§ "biais d'échelle") |
| 6 | Stop-gradient sur les clés vs bout-en-bout | Directement comparable à QSA (indexeur gelé = stop-gradient total) vs NSA (bout-en-bout natif, aucun gel) | Le choix "plus de stop-gradient" rapprocherait Thinker de NSA (tout différentiable) plutôt que QSA (gelé) — cohérent avec le choix déjà fait d'un softmax unifié (qui, comme $O_{cmp}$ dans NSA, fait circuler le gradient nativement, sans Top-K dur ni perte auxiliaire de type DSA) |
| 7 | Requêtes découplées $Q_{KB}$ vs $Q_{SM}$ | Analogue lointain : NSA utilise la **même** query pour les 3 branches (pas de découplage) | Suggestion Gemini non testée dans la littérature citée ; découpler augmenterait le coût (projections séparées) sans précédent direct montrant le gain |
| 8 | Largeur adaptative / No-Op (seuil, bypass, sentinel, sparsemax, température) | Aucun des 3 mécanismes cités n'implémente de "No-Op" — QSA/DSA/NSA sélectionnent toujours un budget fixe de blocs, jamais zéro | Idée propre à Thinker, explicitement mise hors-scope MVP par l'utilisateur — cohérent avec la littérature qui ne traite pas ce cas non plus |
| 9 | Statut de l'input (cross-attention dédiée vs SM) | Sans objet dans la littérature (QSA/DSA/NSA traitent une seule séquence, pas de distinction input/mémoire) | Spécifique à la structure à 3 niveaux de Thinker (input / SM / KB) |
| 10 | Stratégie d'entraînement de la KB (œuf/poule) | **Absent de la littérature citée** — QSA/DSA/NSA indexent toujours la séquence courante du batch, jamais une base externe pré-existante et statique | Nœud réellement propre à Thinker, sans précédent direct dans les 3 mécanismes ; le choix retenu ("espace unifié input/KB avec biais de priorité") s'inspire vaguement de la façon dont NSA traite input et query dans un espace commun, mais sans équivalent pour une KB *externe et persistante* |
| 11 | Curriculum largeur "large → étroit" | Rappelle la logique du seuil de bloc $B$ dans QSA (bloc trop large = dilution, trop petit = surcoût) mais appliqué dynamiquement dans le temps plutôt que fixé une fois pour toutes | Idée propre, non formalisée, aucun précédent direct |

## 3. Comparaison critique appliquée au cas d'usage Thinker

| Critère | QSA (gelé) | DSA (KL align) | NSA (natif, branches) | Indexed Attention (softmax unifié, sans stop-grad) |
|---|---|---|---|---|
| Coût entraînement (mémoire) | Faible ($O(N\cdot K)$) | Élevé pendant alignement ($O(N^2)$) | Moyen (3 branches, mais légères) | Moyen — un seul softmax mais sur un ensemble concaténé feuilles+niveaux, taille à mesurer selon profondeur de l'arbre |
| Gradient sur l'indexeur | Aucun (gelé) → risque de dead blocks | Indirect via KL, coûteux à maintenir à jour | Direct et natif via $O_{cmp}$ | Direct et natif (comme NSA) si le choix "plus de stop-gradient" est confirmé — **c'est le point où Indexed Attention se rapproche le plus de NSA et s'éloigne de QSA** |
| Hiérarchie multi-niveaux | Non (un seul niveau de bloc) | Non | Non (un seul niveau de compression) | **Oui** — c'est la contribution structurelle propre à Thinker, absente des 3 références |
| KB externe persistante entre inférences | Non | Non | Non | **Oui** — aucun équivalent dans la littérature citée, donc aucune recette prête à l'emploi pour le problème de l'œuf et de la poule (axe 10) |
| Risque principal identifié | Dead blocks / drift silencieux | Coût + désynchronisation post-alignement | Complexité d'implémentation (3 branches + gating) | Biais d'échelle du softmax unifié (nécessite RMSNorm par niveau, non encore vérifié empiriquement) + le nœud non résolu de l'entraînement de la KB |

## 4. Ce qui ressort de la comparaison

1. **Indexed Attention n'est pas un concurrent de QSA/DSA/NSA — c'est une extension hiérarchique et persistante d'un problème que ces trois mécanismes ne traitent pas** (ils sparsifient une séquence courante, pas une base de connaissance externe statique). Aucun des trois n'offre de solution prête pour le nœud le plus critique identifié (axe 10, stratégie d'entraînement de la KB) : c'est un problème réellement nouveau, pas un simple choix parmi des recettes existantes.
2. Sur le plan du **flux de gradient**, le choix déjà esquissé par l'utilisateur (softmax unifié + abandon du stop-gradient) positionne Indexed Attention plus près de **NSA** (apprentissage natif bout-en-bout, sans gel ni perte auxiliaire) que de **QSA** (gelé) ou **DSA** (KL coûteux) — c'est cohérent en interne, et évite à la fois le risque de dead blocks de QSA et le surcoût de la phase d'alignement de DSA.
3. La **hiérarchie multi-niveaux à poids partagés** est la vraie nouveauté structurelle : ni QSA, ni DSA, ni NSA n'empilent plusieurs étages de compression récursive. C'est aussi la source du principal risque non encore validé (biais d'échelle dans le softmax unifié entre niveaux de compression différents).
4. Le mécanisme de **No-Op / largeur adaptative** (axe 8) n'a pas d'équivalent dans la littérature citée — cohérent avec la décision déjà prise de le reporter hors du MVP.

## 5. Ce que ce document ne tranche pas (volontairement)

Ce document compare ; il ne décide pas. Restent ouverts, à trancher séparément :
- La stratégie d'entraînement de la KB pendant la phase d'entraînement (espace unifié input/KB retenu comme piste à explorer en premier, mais pas encore formalisée mathématiquement).
- La profondeur de hiérarchie et la fonction d'agrégation à utiliser pour un premier MVP.
- Le statut exact de l'input (axe 9).
- La formalisation du RMSNorm par niveau pour neutraliser le biais du softmax unifié.
