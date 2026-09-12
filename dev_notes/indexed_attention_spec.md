# Spec formelle — Thinker "Indexed Attention"

Document vivant, mis à jour au fur et à mesure de l'implémentation, pour permettre un cross-checking systématique entre la spec mathématique et le code. Ne pas faire confiance à l'implémentation existante (`core/toy_model.py`, `core/layers.py`) comme référence de vérité — certaines parties (ex. l'usage de `F.scaled_dot_product_attention`/patterns "flex attention") sont des raccourcis d'implémentation, pas des choix mathématiques validés. Ce document part des notes manuscrites originelles, pas du code.

Chaque équation/décision est taguée :
- **[NOTES]** — transcription fidèle des notes manuscrites de l'utilisateur (source de vérité première).
- **[CONFIRMÉ]** — décidé explicitement par l'utilisateur dans la conversation ou cette session.
- **[DÉFAUT]** — choix par défaut que j'ai posé pour avancer, révisable, pas encore validé par l'utilisateur.
- **[OUVERT]** — question non tranchée, à clarifier.

## -1. Motivation architecturale fondamentale **[CONFIRMÉ]**

Ceci prime sur toute lecture des sections qui suivent — à relire avant de juger si un composant "manque" un FF, une capacité, ou une transformation apprise.

Dans un transformer standard, les couches **feed-forward (FF)** ne sont pas juste un module de calcul générique : la recherche en interprétabilité mécaniste (ex. Geva et al., *"Transformer Feed-Forward Layers Are Key-Value Memories"*) montre qu'elles fonctionnent largement comme des **mémoires clé-valeur associatives** qui stockent la connaissance factuelle du modèle directement dans leurs poids (chaque neurone de la couche FF agissant comme une "clé" associée à une distribution de sortie "valeur"). C'est le mécanisme principal par lequel un LLM classique "sait" des faits.

**L'idée fondamentale de l'architecture Thinker (Indexed Attention) est de déplacer cette connaissance hors des poids FF et de la placer explicitement dans les paires clé-valeur (latentes) de la base de connaissance externe (KB/LM)**, plutôt que de la laisser diffuse et implicite dans les matrices de poids d'un FF. La KB est *littéralement* structurée en KV — ce n'est pas un hasard, c'est le point : elle joue le rôle que les couches FF jouent dans un transformer classique, mais sous une forme explicite, indexée, non bornée en taille, et interrogeable par attention plutôt que figée dans des poids de taille fixe.

Le rôle du registre latent récurrent (le "core") n'est donc **pas** de mémoriser des faits dans ses propres poids (que ce soit dans un FF ou ailleurs) — c'est d'apprendre, à travers les itérations, à :
1. **Extraire** les connaissances pertinentes de la mémoire long terme (KB) et les **placer** dans la mémoire court terme (SM) ;
2. **Travailler** sur ce qui est maintenant disponible dans la SM — exécuter le "programme"/raisonnement nécessaire sur ces connaissances extraites — pour produire la sortie demandée.

Conséquence directe pour la lecture du reste de ce document : la présence ou l'absence d'un FF dans tel ou tel composant (compresseur hiérarchique, boucle principale, output streams) n'est **pas** à juger à l'aune de "est-ce que ça ressemble à un transformer standard ?", mais à l'aune de "est-ce que ce composant a besoin de stocker de la connaissance factuelle dans ses poids, ou seulement d'exécuter une transformation/un calcul général ?". Un FF dans le core ou les streams sert au second rôle (calcul), jamais au premier (stockage de faits) — les faits doivent rester dans les feuilles de la KB (des embeddings dérivés des données, pas des poids appris), pas se diffuser dans les poids du core au fil de l'entraînement. Ceci reste à vérifier empiriquement (rien ne garantit qu'un FF dans le core ne se mette pas, de facto, à mémoriser des faits si on ne contraint pas sa taille/capacité) — mais c'est l'intention architecturale qui doit guider les choix de conception, pas une imitation par défaut des blocs transformer standards.

**[CONFIRMÉ]** Bénéfice d'entraînement attendu, au-delà de l'argument de principe ci-dessus : supprimer les FF ne fait pas que réduire le nombre de poids (donc simplifier l'optimisation par la taille) — ça **force le modèle à simplifier sa représentation des $K$, $V$** eux-mêmes. Sans FF pour absorber/compenser une représentation complexe ou mal formée, les projections $K$/$V$ (et le pooling du compresseur) doivent produire des vecteurs directement exploitables par un simple produit scalaire + moyenne pondérée — il n'y a nulle part où "cacher" de la complexité résiduelle non-linéaire. Cette contrainte de simplicité représentationnelle est elle-même un facteur de simplification de l'entraînement (moins de degrés de liberté à apprendre, une seule famille de transformation — linéaire + attention — à optimiser plutôt que deux entremêlées), indépendamment de la réduction du nombre de paramètres. À vérifier empiriquement aussi (une contrainte de capacité peut simplifier *ou* rendre plus difficile l'optimisation selon la tâche), mais c'est l'hypothèse de travail qui motive ce choix.

## 0. Notation de base

- $d$ : dimension cachée commune (on assume $q=n=e=m=d$ par souci de simplicité, **[NOTES]**, §"Par souci de simplicité").
- $L_n = [v_1, \dots, v_n]$, $v_i \in \mathbb{R}^d$ : le **registre latent** (appelé "register/latent" dans les notes). **[NOTES]**
  - ⚠️ Collision de notation à lever : dans `thesis/paper/architecture.typ` (architecture actuelle), le nombre de vecteurs latents est noté $L$ et l'état latent $Z_t \in \mathbb{R}^{L\times d}$. Dans les notes manuscrites, $L_n$ désigne *le registre lui-même* (pas sa taille). **Dans ce document, je renomme le registre $R_t \in \mathbb{R}^{n\times d}$** (état du registre à l'itération $t$, $n$ = nombre de vecteurs) pour éviter la confusion avec $L$ = niveau de hiérarchie (voir §2). **[DÉFAUT — notation seulement, pas un choix architectural]**
- $\text{Attn}(Q,K,V)$ : mécanisme d'attention standard. **[NOTES]**
- $Q_e$ (ou $Q$) : query externe, de dimension $e$, utilisée pour interroger la mémoire long terme / KB. **[NOTES]**
- $K_q^l, V_m^l$ : clés (dim $q$) et valeurs (dim $m$) de la mémoire long terme (KB/LM). **[NOTES]**
- $K_q^s, V_n^s$ : clés/valeurs de la mémoire court terme (SM), mêmes dimensions que la mémoire long terme par hypothèse de simplicité. **[NOTES]**

## 1. Trois composants (vision d'ensemble, non contestée par l'utilisateur)

1. **Registre latent récurrent $R_t$** — mis à jour itérativement par une couche à **poids partagés** (Looped Transformer). **[NOTES + CONFIRMÉ]**
2. **Mémoire court terme (SM)** $(K^s, V^s)$ — mise à jour incrémentalement à chaque itération par `APPEND`. **[NOTES]**
3. **Mémoire long terme / base de connaissance (KB/LM)** $(K^l, V^l)$ — massive, statique, préremplie, **indexée hiérarchiquement** ("indexed mem like DSA"). **[NOTES]**

## 2. Flux principal par itération $t$ (transcription du schéma)

Pour chaque itération $t$ :

1. **Génération de la query externe** : $Q_{e,t} = f_Q(R_{t-1})$ — projection du registre latent. **[NOTES]**
2. **Interrogation de la KB (indexée hiérarchiquement)** : récupération de valeurs depuis $(K^l, V^l)$ via $Q_{e,t}$. **[NOTES]**
3. **Concaténation/projection** des valeurs récupérées avec $R_{t-1}$. **[NOTES]**
4. **Interrogation de la SM** : $(K^s, V^s)$. **[NOTES]**
5. **Écriture mémoire** : production de nouveaux $(K_{\text{new}}, V_{\text{new}})$, empilés (`APPEND`) dans SM. **[NOTES]**
6. **Skip connection additive** : $R_t = R_{t-1} + \Delta(\cdot)$, où $\Delta$ résulte de la fusion KB+SM ci-dessus (l'algèbre exacte de fusion n'est pas encore formalisée dans les notes — **[OUVERT]**, voir §6.1). **[NOTES]**

Note manuscrite : *« Il peut arriver qu'on force le KB lookup ou SM lookup, ou qu'on ne compute pas (ou ne garde pas de trace mémoire non-SM) »* — anticipe un mécanisme de contrôle/bypass, formalisé plus bas comme candidat pour le No-Op (hors scope MVP). **[NOTES]**

### 2.1 Encadré "Asynchronous Skip Connection / Delayed Retrieval"

Mention manuscrite : *« asynchronous skip connection allows delayed KB retrieval »*, avec un schéma de "multiple detached streams" et connexions résiduelles/skips asynchrones de $Q_e$ à travers plusieurs étapes. **[NOTES]**

**[OUVERT]** — jamais creusé dans la conversation ni cette session. Signification et utilité à clarifier avec l'utilisateur avant toute tentative de formalisation ou d'implémentation. Je ne propose pas de défaut ici : le risque de mal interpréter est trop grand pour un mécanisme jamais explicité.

Seul indice trouvé ailleurs dans le fichier source (ligne 1261, glose de Gemini dans un inventaire, **non confirmée par vous**) : *« Delayed retrieval pipelines allowing computation to proceed while slow memory reads complete »* — suggère un découplage pipeline entre le calcul du registre latent et la latence de lecture d'une KB massive stockée sur mémoire lente (VRAM insuffisante → offload disque/CPU). Plausible mais pas une confirmation de votre intention réelle.

## 3. Short-Term Memory — options de couplage KB/SM (Page 2 des notes)

Trois options manuscrites pour combiner l'index long-terme avec la structuration de SM, décrites mais dont les équations exactes n'ont pas été retranscrites en détail lisible dans l'export texte (schémas visuels non capturés) :

- **Option 1** : concaténation de $K$, $V$ & $Q$ — note en marge : *« ici on réutilise l'info en particulier l'index long terme qui peut aider à structurer l'index SM, et $Q_e$ peut aider à orienter la prochaine recherche »*. **[NOTES]**
- **Option 2** : mentionnée, contenu visuel non retranscrit. **[OUVERT — schéma manquant]**
- **Option 3** : un mécanisme qui "scanne $K_M$ et trie les valeurs" pour focaliser l'attention, ou utilise $Q_e^{SM}$ pour une attention classique. **[NOTES]**

**[OUVERT]** — aucune de ces trois options n'est tranchée. Nécessaire de clarifier avec l'utilisateur laquelle (ou quelle combinaison) retenir, idéalement en revenant aux schémas visuels originaux si disponibles.

## 4. Gradient et entraînement (encadré notes, Page 2)

Citation directe : *« Bien qu'il serait bien d'éviter de donner des rôles multiples à $Q_e$ (afin de le rendre moins précis dans sa recherche de $K_e^{lr}$), le même souci peut être aussi pour $K_e^{lr}$. Un stop-gradient peut aider ? Ou peut-être mieux éviter de les réutiliser et recalculer tout à partir du nouveau contexte. »* **[NOTES]**

Réponse initiale de l'utilisateur (clarification directe, plus tard dans la conversation) : *« Stop gradient était juste sur les clés car j'ai pas encore trouvé comment tous les indexer »* — donc stop-gradient **initialement limité aux clés $K$** entrant dans le buffer SM, pas sur les valeurs, ni sur la KB long-terme en tant que telle. **[CONFIRMÉ, état initial]**

### 4.1 [OUVERT — question à vous poser directement]

Plus tard : *« Plus de stop gradient, on pourra implémenter les variantes M>1 et M=1 et tester »* (ligne 1057 du fichier source). Cette phrase est **ambiguë** et je ne veux pas trancher à votre place :
- Lecture A : "il n'y a plus de stop-gradient" (abandon du stop-gradient décrit en §4, entraînement bout-en-bout).
- Lecture B : "plus de stop-gradient" au sens de "davantage de stop-gradient" (renforcement/généralisation au-delà des seules clés).

**Question directe : laquelle des deux lectures est correcte ?** Le reste de ce document utilisera votre réponse — en attendant, je marque toute équation qui dépend de ce choix comme **[BLOQUÉ PAR 4.1]**.

## 5. Indexation hiérarchique de la KB (extension propre à l'utilisateur, hors transcription des notes manuscrites)

Cette partie vient de l'enrichissement ultérieur du concept (ligne 672+), pas des notes manuscrites elles-mêmes — je la garde distincte pour ne pas mélanger les deux sources.

### 5.1 Construction de l'arbre

Soit $\ell \in \{0, 1, \dots, D\}$ l'indice de niveau ($\ell=0$ = feuilles = tokens/faits bruts, $\ell=D$ = racine). Soit $C$ la taille de bloc (nombre d'enfants par nœud parent). **[DÉFAUT — $C$ et $D$ sont des hyperparamètres non fixés par l'utilisateur]**

Pour chaque nœud parent $p$ au niveau $\ell+1$, regroupant les nœuds enfants $\{c_1,\dots,c_C\}$ au niveau $\ell$ :
$$
\tilde K_p^{(\ell+1)}, \tilde V_p^{(\ell+1)} = \text{Compress}_\theta\big(K_{c_1}^{(\ell)}, \dots, K_{c_C}^{(\ell)}, V_{c_1}^{(\ell)}, \dots, V_{c_C}^{(\ell)}\big)
$$

où $\text{Compress}_\theta$ est **partagé entre tous les niveaux** ($\theta$ identique quel que soit $\ell$) — **[NOTES]**, "Pour un multi-étage illimité on peut share weight du compressor peu importe l'étage".

**[OUVERT]** Choix de $\text{Compress}_\theta$ — non tranché par l'utilisateur (voir `indexed_attention_comparison.md` §2, axe 2) :
- pooling non-paramétrique (moyenne + RMSNorm),
- projection linéaire/Conv1D à taille de bloc fixe,
- **cross-attention Perceiver-style à 1 query apprise ($M=1$)** — direction vers laquelle l'utilisateur penche mais sans choix final,
- compresseur récurrent à poids partagés (DeltaNet/GRU-like).

Granularité de sortie : $M=1$ (idée initiale de l'utilisateur) vs $M>1$ (slots multiples façon Slot Attention, variante confirmée à tester en parallèle — **[CONFIRMÉ]**, "on pourra implémenter les variantes M>1 et M=1 et tester").

### 5.2 Softmax unifié multi-niveaux

Proposition propre de l'utilisateur (pas de la littérature) : au lieu de fusionner les niveaux par gating séparé (façon NSA), concaténer dans **un seul softmax** les clés de toutes les feuilles et de tous les niveaux compressés :
$$
K_{\text{unifié}} = \big[K^{(0)}; \tilde K^{(1)}; \dots; \tilde K^{(D)}\big], \quad V_{\text{unifié}} = \big[V^{(0)}; \tilde V^{(1)}; \dots; \tilde V^{(D)}\big]
$$
$$
O = \text{Softmax}\!\left(\frac{Q_e\, K_{\text{unifié}}^\top}{\sqrt d}\right) V_{\text{unifié}}
$$
**[CONFIRMÉ — proposition de l'utilisateur, ligne 672+]**

**Correction de biais d'échelle [DÉFAUT, non encore vérifié empiriquement]** : appliquer une RMSNorm **propre à chaque niveau** sur les clés avant le produit scalaire, pour éviter que les clés compressées (norme différente des clés brutes) ne biaisent systématiquement le softmax :
$$
\hat K^{(\ell)} = \text{RMSNorm}_\ell(K^{(\ell)}) \quad \text{(paramètres } \gamma_\ell \text{ propres à chaque niveau, non partagés)}
$$
Ce point est un risque mathématique signalé par Gemini (pas testé) — traité ici comme un défaut d'implémentation à valider par test unitaire (voir plan de tests), pas comme un fait acquis.

### 5.3 Résolution implicite vs explicite

Le softmax unifié arbitre *implicitement* le niveau de résolution (feuille précise vs résumé de haut niveau) via les scores d'attention — pas de routeur dédié séparé. Formulation précise (ligne ~1189, non contestée par l'utilisateur après proposition) :
> « En plaçant les nœuds de différents niveaux dans le même Softmax normalisé (RMSNorm), le modèle n'a pas besoin de prédire une décision de niveau binaire. Si un concept haut niveau $\tilde K_{\text{haut}}$ a une forte affinité avec $Q$, son score $s_i$ dépasse le seuil, tandis que les sous-arbres non pertinents restent en dessous sans être explorés. »

L'utilisateur a lui-même clarifié (ligne 1181) qu'il distingue deux notions de "seuil" à ne pas confondre : un **seuil de largeur** (nombre de top éléments retenus par l'index, ce qu'il visait initialement) vs un **seuil de profondeur/résolution** (quel niveau de la hiérarchie explorer) — et suggère que le second peut rester **implicite** via l'indexation ($Q$ naturellement plus proche d'un $\tilde V$ de haut niveau ou d'un $V$ de bas niveau selon le cas) plutôt qu'un choix de niveau explicite. **[CONFIRMÉ — distinction largeur/profondeur clarifiée par l'utilisateur ; résolution de profondeur implicite favorisée]**

## 6. Registre latent — mise à jour complète

### 6.1 Fusion et skip connection

**[OUVERT]** — l'algèbre exacte de $\Delta$ dans $R_t = R_{t-1} + \Delta(\text{KB}, \text{SM}, R_{t-1})$ n'est formalisée nulle part dans les notes ni la conversation lue jusqu'ici (mentionné comme prochaine étape possible par Gemini, jamais creusé par l'utilisateur). À formaliser une fois §3 (couplage SM) et §4.1 (stop-gradient) clarifiés.

### 6.2 Statut de l'input

**[CONFIRMÉ dans cette session]** : espace unifié input/KB avec biais de priorité (pas de cross-attention dédiée séparée façon Perceiver-IO, qui était la suggestion de Gemini non retenue). Formalisation proposée **[DÉFAUT]** :
$$
\hat E_{\text{tok}} = E_{\text{tok}} + b_{\text{source}}, \quad b_{\text{source}} \in \{b_{\text{input}}, b_{\text{KB}}\} \text{ (embedding à 2 entrées)}
$$
appliqué avant projection en $K^{(0)}, V^{(0)}$ — l'input et la KB partagent alors le même pipeline d'indexation hiérarchique (§5), distingués uniquement par ce biais additif.

**Risque identifié explicitement (ligne 1593-1599, Gemini, non contesté par l'utilisateur)** : *« Le piège de la dilution : si la KB contient des milliers de blocs, la somme des exponentielles du Softmax risque d'étouffer l'input »*. Deux solutions possibles au piège :
1. **Biais de priorité additif** (ce que formalise $b_{\text{source}}$ ci-dessus) — solution la plus simple, retenue par défaut pour le MVP.
2. Forcer l'input à résider dans une **branche locale dédiée** (façon branche locale de NSA, garantie d'une part fixe de bande passante d'attention) — alternative plus lourde, non retenue pour le MVP mais à garder en tête si (1) s'avère insuffisant empiriquement (à vérifier par un test dédié : mesurer si le score d'attention moyen sur l'input s'effondre quand la taille de la KB augmente).

## 7. Hors scope MVP (mentionné dans les notes/conversation mais volontairement reporté)

### 7.1 Largeur adaptative / No-Op — **[CONFIRMÉ hors scope MVP par l'utilisateur]**, formalisé ici pour référence future

Formulation la plus aboutie atteinte dans la conversation (lignes 1181-1244), unifiant No-Op et largeur variable :

Le registre produit une requête $Q$ et un seuil de coupure $\theta = W_\theta R_{t-1} + b_\theta$. Pour chaque candidat $i$ (nœud résumé ou feuille) : $s_i = \dfrac{Q K_i^\top}{\sqrt d}$.

- **Hard (inférence)** : $\mathcal{I}_{\text{active}} = \{i \mid s_i > \theta\}$. Si $\max_i(s_i) \le \theta$ : $\mathcal{I}_{\text{active}}=\emptyset$, $O_{\text{mem}} = \mathbf 0$ (No-Op, aucun transfert mémoire).
- **Soft (entraînement, différentiable)** : $w_i = \sigma\!\left(\dfrac{s_i-\theta}{\tau}\right)$, puis
$$\alpha_i = \frac{\exp(s_i)\,w_i}{\sum_j \exp(s_j)\,w_j + \epsilon}, \qquad O_{\text{mem}} = \sum_i \alpha_i V_i$$
  Si tous les $w_i \to 0$, $O_{\text{mem}}\to \mathbf 0$ de façon différentiable (pas de discontinuité train/inférence).
- **Pénalité de parcimonie** : $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tâche}} + \lambda \sum_i w_i$, avec un risque explicitement identifié et non résolu dans la conversation : un $\lambda$ trop élevé tôt dans l'entraînement peut faire collapser le modèle vers un No-Op permanent ($\theta$ poussé arbitrairement haut) — aucune stratégie de warm-up de $\lambda$/$\tau$ n'a été validée.

Reporté hors MVP parce que : (a) l'utilisateur l'a lui-même écarté pour une première baseline, (b) le risque de collapse ci-dessus est un vrai piège d'entraînement non résolu, (c) le MVP doit d'abord valider que la hiérarchie + softmax unifié fonctionnent sans cette complexité additionnelle.
- Requêtes découplées $Q_{KB}$ vs $Q_{SM}$ — suggestion de Gemini, jamais confirmée, non implémentée pour l'instant.
- Dropout stochastique des niveaux hauts à l'entraînement — idée propre de l'utilisateur, non implémentée pour l'instant.
- Asynchronous skip connection / delayed retrieval (§2.1) — non formalisable sans clarification.

## 8. Stratégie de construction de la KB pendant l'entraînement (nœud "poule et œuf")

Problème posé par l'utilisateur (ligne 1577) : la KB long-terme est censée être statique/préremplie, mais elle n'existe pas encore pendant l'entraînement — comment superviser le retrieval sans index figé a priori ?

Quatre stratégies évaluées (lignes 1577-1651), synthèse comparative de Gemini :

| # | Stratégie | Stabilité gradient | Coût calcul/batch | Risque d'effondrement |
|---|---|---|---|---|
| 1 | **Espace unifié input/KB, projections partagées** | Moyen | Faible | Dilution de l'input (voir §6.2) |
| 2 | Sub-KB générée à la volée par batch + distracteurs négatifs | Élevée | Élevé (prefill à chaque pas) | Faible |
| 3 | Découverte libre, sans dataset dédié | Faible | Faible | Élevé (No-Op systématique) |
| 4 | Hybride (curriculum en 3 phases : ancrage agrégateur → Sub-KB contrôlée → généralisation) | Maximale | Équilibré | Très faible |

**[CONFIRMÉ dans cette session]** : stratégie **1 (espace unifié)** retenue pour le MVP — malgré le risque de dilution identifié, cohérent avec la priorité "MVP minimal direct" déjà actée et avec le fait que la stratégie 4 (recommandée par Gemini comme "généralement la voie la plus stable") a un coût d'ingénierie plus élevé, jugé prématuré avant d'avoir un premier signal sur la stratégie 1.

## 9. Baselines & protocole d'évaluation (cadre de test, pas encore implémenté)

Trois baselines architecturales pour isoler la source de tout gain observé (lignes 1358-1455), à garder en tête pour la conception de `tests/test_indexed_memory.py` et de futurs bancs d'essai :

- **Baseline A (RAG plat / mono-passage, $T=1$)** : une seule cross-attention sur le Top-$k$ récupéré, **sans** boucle latente ni hiérarchie. Répond à : *le raisonnement itératif apporte-t-il un gain sur la récupération simple ?*
- **Baseline B (boucle pure, sans mémoire)** : même registre latent bouclé $T$ fois, mais accès mémoire désactivé. Répond à : *le gain vient-il de la boucle elle-même ou de la mémoire externe ?*
- **Baseline C (attention dense sur tout le contexte)** : borne supérieure de référence (si la longueur de séquence le permet à petite échelle).

Découpage attendu des splits d'évaluation (ligne 1443-1447, 1736-1739), pertinent pour la conception de `data/kb_retrieval.py` :
1. **Raisonnement pur** (tout est dans l'input, pas besoin de KB) — Baseline B doit égaler le modèle complet ici.
2. **Récupération pure** (un seul fait à extraire d'une large KB statique) — Baseline A doit égaler le modèle complet ici.
3. **Raisonnement multi-sauts sur mémoire** (fait X → débloque Y → débloque Z) — seul le modèle complet doit réussir ; c'est le test décisif.

**[OUVERT]** : l'utilisateur a lui-même mis en doute (ligne 1718) que la baseline MVP proposée par Gemini (Top-k fixe, KB injectée en contexte plutôt que préremplie/hiérarchique) respecte fidèlement l'objectif "LLM généraliste à faible mémoire paramétrique" énoncé plus haut — pas de résolution finale dans le texte lu. À garder en tête : le MVP de ce document (§5-§6) vise à tester la **mécanique** (hiérarchie + softmax unifié + gradient natif), pas encore le protocole d'évaluation complet ci-dessus.

## 10. Curriculum de largeur "large → étroit" (idée propre, non formalisée mathématiquement)

Idée de l'utilisateur (ligne 1824) : en début de traitement (itérations basses), la sélection/largeur d'attention devrait être **large** pour bien explorer input + mémoire ; puis se **resserrer progressivement** au fil des itérations pour converger vers une décision, avec la possibilité de laisser le modèle apprendre ce planning lui-même plus tard (post-training). **[NOTES/CONFIRMÉ comme idée, non implémentée]**

Point de vigilance soulevé par Gemini (ligne 1844, non contesté) : un resserrement trop strict interdit le *backtracking* — si une déduction tardive révèle un besoin imprévu d'information, le modèle doit garder la possibilité de rouvrir une fenêtre large ponctuellement. **[OUVERT]** — aucune formalisation mathématique proposée dans la conversation ; hors scope MVP (dépend du mécanisme de largeur adaptative §7.1, lui-même hors MVP).

## 11bis. Output Streams (source : `raw/Distill-reasonning-stream.md`, absent des sections précédentes)

Concept distinct de la hiérarchie KB (§5-6) mais qui touche directement à la question "les poids qui calculent la sortie sont-ils partagés ou indépendants ?" — à traiter dans ce document plutôt que de le laisser implicite.

### Principe **[NOTES]**

Le registre récurrent produit, à chaque itération $t$, un état $h^{(t)}$ qui est empilé dans la mémoire court terme (SM) : $K_{sm}, V_{sm} = \text{Proj}([h^{(1)}, \dots, h^{(T)}])$ — c'est exactement le buffer SM déjà utilisé par `IndexedThinker` (§11). Au lieu de lire l'état final du registre directement, **un ou plusieurs "Output Streams" légers (1-2 couches) interrogent ce buffer par leur propre cross-attention** :

$$
O_{\text{stream}_i} = \text{Attn}\big(Q_{\text{stream}_i}, K_{sm}, V_{sm}\big), \qquad \text{logits}_i = \text{Head}_i(O_{\text{stream}_i})
$$

**Point clé (répond directement à la question posée) : $Q_{\text{stream}_i}$ et $\text{Head}_i$ sont des poids propres à chaque stream $i$, non partagés entre streams, ni avec le core récurrent.** Citation directe : *« chaque Stream […] chacun train indépendamment »* ; *« vous pouvez geler le core et entraîner les output streams indépendamment »* (lignes 230, 307). Seul le buffer SM lui-même (une activation, pas un poids) est partagé en entrée de tous les streams.

### Motivation supplémentaire, à garder en tête mais **non confirmée** : stabilisation du Looped Transformer **[NOTES, hypothèse théorique — infirmée comme explication du cas observé cette session]**

Citation directe (lignes 213-217) : *« Les transformers à poids partagés récurrents souffrent d'instabilité dynamique : la magnitude des hidden states explose ou s'effondre facilement au fil des boucles. Contraindre les états latents du loop via une perte MSE ou Cosine par rapport aux couches successives d'un Teacher pré-entraîné agit comme une régularisation pour borner les représentations. »*

**Historique de cette session, pour traçabilité** : un échec d'entraînement observé sur GPU (`thinker-e9`, `depth=4`, 64 facts, loss bloquée au niveau du hasard) puis reproduit localement avait d'abord été attribué à ce phénomène d'instabilité — l'échec ne se résorbait pas avec un `d_model`/`n_register` plus grand, ni avec `lr=1e-3`. **Corrigé ensuite** : `thinker-e9` a identifié par un balayage de LR plus complet un **plateau de convergence étroit autour de `lr=3e-4`** (confirmé indépendamment en local : mêmes hyperparamètres, `lr=3e-4` fait passer l'accuracy de ~10% à 98-100% entre les steps 800 et 1000, alors que `1e-4`/`1e-3`/`3e-3`/`1e-2` échouent tous). **La cause réelle était une sensibilité au learning rate (fenêtre étroite), pas une instabilité structurelle du looped transformer** — l'hypothèse de supervision intermédiaire ci-dessous n'a donc pas été implémentée. Gardée dans ce document car la citation source reste théoriquement valide et pourrait redevenir pertinente à plus grande échelle (plus d'itérations, plus de profondeur), mais **ce n'est plus le diagnostic retenu pour le cas observé ici**.

**Piste correspondante, reportée sans être écartée** : si l'instabilité redevient un candidat plausible plus tard (ex. après un balayage LR complet qui échoue à trouver un plateau stable à plus grande échelle), le principe d'une supervision intermédiaire sans Teacher reste disponible : sur une tâche synthétique on connaît la vérité terrain de la récupération à chaque étape, donc on peut superviser directement l'attention plutôt que seulement le token final (version sans Teacher de l'Option B "White-Box Trajectory Distillation" déjà documentée ailleurs dans la conversation source). **[OUVERT]**, non prioritaire tant que le balayage LR n'a pas d'abord été fait correctement (voir plan, Phase -1).

### Streams envisagés **[NOTES]**

- **Stream Answer** (tokens) : supervisé par cross-entropy + Top-K KL du Teacher sur les logits finaux.
- **Stream Thinking (embedding)** : auto-génératif sur des embeddings (pas de vocabulaire), aligné sur une couche médiane du Teacher (~40-65% de profondeur) via une perte de similarité/MSE — *« pour laisser le thinker un peu libre sur sa représentation interne »* (ligne 529).
- **Stream Thinking (tokens)** : décodage explicite d'une chaîne de pensée `<think>...</think>`, introduit plus tard dans le curriculum.
- Le nombre d'itérations du core **n'a pas besoin de correspondre** au nombre de couches du Teacher — c'est le mécanisme d'attention propre à chaque stream qui apprend à pondérer les pas récurrents pertinents pour sa tâche, pas un alignement pas-à-pas forcé (ligne 304, 478-479).

### Curriculum d'extinction **[NOTES]**

Idée notée (lignes 574-587) : les streams latents (embedding) sont utiles tôt dans l'entraînement pour guider le core, puis **atténués progressivement** (`weight decay sur la perte latente -> 0`) jusqu'à ne garder en inférence finale que les flux textuels visibles (Stream Thinking en tokens + Stream Answer). **[OUVERT]** — non formalisé mathématiquement (juste "atténuation linéaire" mentionnée, pas de fonction de schedule précisée dans le texte lu).

### État d'implémentation

**Corrigé dans cette session.** `IndexedThinker` expose désormais `self.streams: nn.ModuleDict[str, OutputStream]` (`core/indexed_thinker_model.py`) : chaque `OutputStream` a sa propre requête apprise (`query_seed`) + **1 à 3 couches** de cross-attention empilées (`OutputStreamLayer`, pré-norm + résiduel) + sa propre tête, et interroge par cross-attention le **SM complet accumulé sur toute la boucle** (pas seulement l'état final `R`). Le nombre de couches est configurable par stream (`stream_n_layers={'answer': 1, 'thinking': 2}`, etc.). `forward()` retourne `(R, stream_outputs)` — un dict `{nom: sortie}`, un par stream enregistré. Testé (`tests/test_indexed_memory.py::TestOutputStreamsIndependence`) : paramètres disjoints entre deux streams, `backward()` sur un stream ne peuple aucun gradient sur un autre, gradient qui atteint bien chaque couche d'un stream à 3 couches. Seul un stream `answer` (logits vocabulaire) est branché pour l'instant sur `data/kb_retrieval.py` ; un stream `thinking` en embedding nécessiterait un vrai Teacher (hors scope tant qu'on reste sur la tâche synthétique).

**Aller-retour sur le FF de `OutputStreamLayer`** : chaque couche avait initialement, en plus de la cross-attention, un FF résiduel (motif transformer standard). **Retiré ensuite** (voir §-1) : un stream ne doit faire que lire/décoder ce qui est déjà dans la SM, pas mémoriser ses propres associations factuelles indépendamment de ce qui y est réellement présent. `OutputStreamLayer` ne fait plus que cross-attention + résiduel, sans FF.

## 11. Implémentation MVP (première version, testée localement sur CPU)

Fichiers : `core/indexed_memory.py` (`HierarchicalMemory`, `LevelCompressor`), `core/indexed_thinker_model.py` (`IndexedThinker`), `data/kb_retrieval.py` (tâche synthétique), `tests/test_indexed_memory.py` (10 tests, tous verts).

Choix concrets faits pendant l'implémentation, non explicitement fixés par la spec ci-dessus — **[DÉFAUT, implémentation]** :

- **`Compress_θ` (§5.1)** : un unique jeu de $M$ requêtes apprises par la même instance à tous les niveaux ; les poids de pooling (un seul softmax sur les enfants) sont réutilisés à la fois pour résumer $K$ et pour agréger $V$ — une instanciation spécifique de "cross-attention Perceiver-style", plus simple qu'une paire de projections $K$/$V$ de sortie séparées. À raffiner si les tests empiriques suggèrent que $K$ et $V$ ont besoin d'être résumés différemment.
  - **Aller-retour sur le FF du compresseur** : une première correction avait ajouté un FF résiduel après le pooling (motif "un vrai bloc Perceiver a toujours attention *et* FF"). **Revenu en arrière ensuite** (voir §-1) : par souci de cohérence avec la motivation fondamentale de l'architecture (la connaissance doit vivre dans les KV de la KB, pas dans des poids de transformation), le compresseur ne fait plus que du pooling par attention pur, sans FF. `LevelCompressor.forward()` est de nouveau exactement la pondération softmax des enfants, sans transformation apprise supplémentaire.
  - **Gap trouvé par comparaison avec une implémentation de référence externe (NSA, `lucidrains/native-sparse-attention-pytorch`)** : le pooling par attention pure est une fonction d'ensemble — invariante par permutation de l'ordre des enfants dans un bloc. Deux blocs avec le même contenu dans un ordre différent se seraient compressés en un résultat identique. Ni la spec ni les tests écrits jusque-là n'auraient attrapé ce problème (aucun ne teste la sensibilité à l'ordre). NSA évite explicitement ce piège en ajoutant un **embedding de position intra-bloc** aux clés/valeurs des enfants avant la compression. **Corrigé** : `LevelCompressor` prend maintenant `block_size` en paramètre et ajoute `self.intrablock_pos` (une `nn.Embedding(block_size, d_model)`, poids partagés entre niveaux comme le reste du compresseur) à $K$ et $V$ des enfants avant le pooling — ce n'est **pas** une réintroduction de FF (un embedding additif ne porte aucune capacité de mémoire associative), donc cohérent avec §-1. Testé explicitement (`test_compressor_is_not_permutation_invariant`) : permuter l'ordre des enfants change désormais le résultat du pooling.
  - **Ce que cette comparaison a aussi confirmé, sans nécessiter de changement** : NSA n'a pas besoin d'une RMSNorm par niveau parce que chaque branche (locale/compressée/sélectionnée) a son propre softmax séparé, fusionné ensuite par gating en espace de sortie — il n'y a jamais de collision d'échelle dans un même softmax. Mon choix du "softmax unifié" (proposition propre à l'utilisateur, §5.2) crée justement cette collision, ce qui confirme a posteriori que la RMSNorm par niveau déjà ajoutée n'est pas un ajout arbitraire mais une conséquence directe et nécessaire d'avoir choisi l'unification plutôt que la séparation par branches façon NSA.
  - **Simplification non corrigée, documentée comme limite connue** : NSA compresse avec un recouvrement possible entre blocs (stride < taille de bloc) pour une couverture plus fine ; `HierarchicalMemory` n'utilise que des blocs disjoints sans recouvrement — plus simple et moins coûteux, mais peut manquer de l'information à la frontière entre deux blocs adjacents. Non corrigé pour l'instant (ajouterait de la complexité et du coût de calcul sans preuve que c'est nécessaire à cette échelle) — à revisiter si les expériences de la Phase 2+ du plan d'expérimentation montrent une perte d'information aux frontières de bloc.
- **Padding/masking** : implémenté (voir plan d'expérimentation, Phase -1 — nécessaire pour un curriculum sur `n_facts` qui garde `block_size`/`depth` fixes entre paliers). `HierarchicalMemory.build(..., leaf_mask=...)` accepte un masque booléen par feuille ; le masque est propagé à chaque niveau (`parent_mask = any(children_mask)`) et appliqué avant chaque softmax (pooling du compresseur ET attention unifiée finale), avec `nan_to_num` pour les blocs entièrement masqués (softmax sur tout `-inf`). `N == block_size**depth` reste une contrainte de forme fixe — le padding sert à remplir cette forme avec moins de contenu réel, pas à la contourner. Testé : absence de NaN même avec des blocs entièrement masqués, et invariance du résultat au contenu du padding (le padding peut contenir n'importe quoi, masqué = jamais attendu) — y compris sur le chemin `n_head>1`, qui n'avait auparavant aucune couverture de test (gap relevé par l'audit critique de cette session).
- **`depth=0` dégénère en attention plate sans hiérarchie** — utilisé directement comme équivalent de la Baseline C (§9) dans les tests, sans code dédié supplémentaire.
- **SM (§3)** : implémentée comme un simple buffer plat qui grandit par `APPEND` (pas de choix entre Options 1/2/3, qui restent `[OUVERT]`) ; les nouveaux $(K,V)$ sont produits par une unique projection du registre après fusion, pas par une requête $Q_e$ séparée ré-interrogeant la KB.
- **Fusion $\Delta$ (§6.1, marqué `[OUVERT]` plus haut)** : $R_t = R_{t-1} + W_{\text{fuse}}\,\text{RMSNorm}([O_{kb}; O_{sm}; R_{t-1}])$ — une **simple projection linéaire** (pas de FF/MLP) sur la concaténation. Une première version utilisait un MLP à 2 couches (GELU) ; **retiré** après la clarification de §-1 : le core recombine ce qu'il a récupéré, il ne doit pas disposer d'une capacité de transformation non-linéaire supplémentaire qui pourrait se mettre à encoder des faits dans ses propres poids plutôt que de les lire depuis KB/SM. Pas justifié par les notes manuscrites (qui ne formalisent pas $\Delta$), c'est le choix le plus direct compatible avec §-1.
- **Amorçage du registre** : $R_0 = R_{\text{init}} + \bar E(\text{query})$ (embedding moyen de la requête ajouté au registre initial appris) — un choix arbitraire pour permettre au test de sur-apprentissage de fonctionner, pas une décision architecturale mûrie.
- **Tâche de test (`data/kb_retrieval.py`)** : couvre uniquement le split "récupération pure" du §9 (une KB de faits distracteurs, une clé-requête, une valeur-cible) ; ne couvre pas encore "raisonnement pur" ni "multi-sauts" — répond partiellement à la question ouverte #6 en commençant par le split le plus simple à vérifier mécaniquement.

Résultats des tests (CPU) : les 10 tests passent, y compris le test de sur-apprentissage (accuracy ≥ 90% sur un batch fixe de 8 exemples, hiérarchie profondeur 2) et son équivalent Baseline C (`depth=0`). Le test `TestRMSNormScaleBias` confirme empiriquement que la RMSNorm par niveau réduit un écart d'échelle brut de facteur >5 à un facteur <1.5 (risque identifié en §5.2, maintenant vérifié).

## Journal des questions ouvertes (à répondre quand vous voulez, je continue en parallèle)

1. **§4.1** : "Plus de stop gradient" = abandon ou renforcement du stop-gradient ?
2. **§3** : quelle option de couplage KB↔SM retenir (Option 1/2/3) — schémas visuels non capturés dans l'export texte, avez-vous les images originales ?
3. **§2.1** : que signifie concrètement l'asynchronous skip connection / delayed retrieval ?
4. **§5.1** : quelle fonction $\text{Compress}_\theta$ retenir pour un premier MVP (je penche pour cross-attention $M=1$ par cohérence avec vos préférences déjà exprimées, mais pas encore confirmé) ?
5. **§6.1** : l'algèbre de fusion $\Delta$ — à formaliser ensemble une fois §3/§4.1 clarifiés.
6. **§9** : reste-t-on sur le MVP "mécanique" (tester hiérarchie + softmax unifié, sans encore le protocole d'évaluation généraliste à 4 volets), ou faut-il déjà caler `data/kb_retrieval.py` sur les 3 splits (raisonnement pur / récupération pure / multi-sauts) dès cette itération ?
