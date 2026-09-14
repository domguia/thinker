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

### Comment interpréter un résultat "avec-FF gagne" (précision de l'utilisateur, à ne pas perdre de vue)

**[CONFIRMÉ]** Si la variante `use_ff=True` (Phase 1bis) performe mieux, ça ne réfute **pas** la thèse de §-1 et ça ne doit **pas** conduire à l'abandonner ou à remettre un FF partout par défaut. La distinction importante n'est pas "présence vs absence de non-linéarité" — c'est **où et à quelle échelle statistique** cette non-linéarité opère :

- Un FF classique de transformer est appliqué **identiquement à travers des milliers d'exemples d'entraînement différents** — c'est cette réutilisation massive à travers la distribution d'entraînement qui lui permet d'accumuler des associations statistiques stables (des "faits") dans ses poids, au sens de Geva et al.
- Un FF ajouté dans la boucle principale d'`Thinker` (`use_ff=True`) opère sur $[O_{kb}, O_{sm}, R]$ **déjà spécifiques à l'exemple courant** (ce qui a été récupéré *pour cette requête précise*) — sa fonction potentielle n'est pas de mémoriser des faits génériques mais de **composer/calculer** sur des valeurs déjà extraites (ex. l'exemple des notes manuscrites : soustraire deux dates récupérées). Ce n'est pas categoriquement la même chose qu'un FF qui stocke "Paris = capitale de la France" dans ses poids.

**Donc, si `use_ff=True` gagne significativement** : la bonne lecture n'est pas "la prémisse §-1 est fausse, remettons des FF partout", c'est "**il manque une capacité de calcul/composition sur le contenu déjà récupéré**, et il faut trouver comment l'ajouter sans que ça redevienne, de facto, un stockage de faits à travers les exemples d'entraînement" — ex. contraindre fortement la capacité de ce FF spécifique, vérifier par un diagnostic dédié qu'il ne mémorise pas de faits génériques (est-ce que ses poids répondent différemment selon le contenu de $O_{kb}$ pour DEUX exemples différents avec la même structure de tâche, ou converge-t-il vers une fonction fixe indépendante du contenu récupéré ?), ou explorer des alternatives structurelles (ex. un FF appliqué seulement après la lecture SM, jamais avant/pendant l'indexation KB elle-même).

**Principe général d'interprétation, à appliquer à toutes les phases du plan, pas seulement celle-ci** : un résultat qui contredit une hypothèse de départ est un signal diagnostique sur *comment* atteindre l'objectif, pas un référendum sur *si* on continue dans cette direction. On a une direction (séparer raisonnement et connaissance) ; la question posée par chaque expérience est "qu'est-ce qui doit changer dans l'implémentation pour s'en rapprocher", pas "doit-on abandonner l'idée".

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

Ce choix-là reste ouvert, mais une **sous-question a été tranchée** : quelle que soit la famille de $\text{Compress}_\theta$ retenue, la production de $\tilde K_p$ et celle de $\tilde V_p$ doivent être **découplées** (§5.1bis) — un compresseur qui dérive les deux d'une même pondération ne peut pas produire une entrée de mémoire associative. Ce critère s'applique aussi aux trois autres familles ci-dessus si l'une d'elles est testée un jour (ex. un pooling non-paramétrique moyenne+RMSNorm y échouerait par construction, puisque la moyenne est la même pour $K$ et $V$).

Granularité de sortie : $M=1$ (idée initiale de l'utilisateur) vs $M>1$ (slots multiples façon Slot Attention, variante confirmée à tester en parallèle — **[CONFIRMÉ]**, "on pourra implémenter les variantes M>1 et M=1 et tester").

#### 5.1bis Pooling découplé $K$/$V$ **[CONFIRMÉ 2026-09-13 — correction structurelle, contre-expertise]**

Jusqu'au 2026-09-13, `LevelCompressor` poolait $\tilde K_p$ et $\tilde V_p$ avec **le même** vecteur de poids d'attention. C'est une **impossibilité structurelle** pour une mémoire associative, et c'était la cause du plateau multi-sauts.

Un bloc-fait a la forme $[\texttt{KEY\_MARK}, \texttt{key}, \texttt{VAL\_MARK}, \texttt{val}]$. Pour qu'un nœud serve d'entrée de mémoire, il faut simultanément :
$$\tilde K_p \approx f(\texttt{key}) \quad \text{(pour être trouvé par la requête)}, \qquad \tilde V_p \approx g(\texttt{val}) \quad \text{(pour restituer la réponse)}$$
soit **deux pondérations opposées** sur les mêmes enfants. Avec un softmax unique, le compresseur ne peut satisfaire que l'une des deux, ou converger vers un compromis flou — exploitable en force brute à un saut (d'où les ~98 % à `n_hops=1`), inchaînable au-delà.

**Correction** : deux requêtes apprises indépendantes,
$$w^K = \operatorname{softmax}\!\Big(\tfrac{q_K \tilde K_c^\top}{\sqrt d}\Big),\quad w^V = \operatorname{softmax}\!\Big(\tfrac{q_V \tilde K_c^\top}{\sqrt d}\Big),\quad \tilde K_p = \textstyle\sum_c w^K_c \tilde K_c,\quad \tilde V_p = \textstyle\sum_c w^V_c \tilde V_c$$
Les deux scores restent calculés contre les **clés** des enfants (seule la cible du pooling diffère). Coût : $n_{\text{slots}} \times d_{\text{model}}$ paramètres. Ce n'est **pas** une réintroduction de FF au sens de §-1 : aucune capacité de mémoire associative n'est ajoutée aux poids, on lève une contrainte d'expressivité du routage.

`decouple_kv=True` est le défaut (`core/indexed_memory.py`) ; `decouple_kv=False` reste disponible comme ablation (`--shared_kv_pooling`).

**Vérification CPU** ($d_{\text{model}}=32$, $N_{\text{step}}=8$, 3000 pas, `lr=1e-3`, 2 seeds) : `n_hops=2` passe de 26-35 % à **100 % (loss 0,000)** sur les deux seeds ; `n_hops=3` atteint 98,1 % sur une seed (46,4 % sur l'autre — variance à caractériser sur GPU). Voir `dev_notes/experiment.log.md`, entrée « Contre-expertise », et plan, Phase 2-redo.

**Leçon de méthode à retenir** (au-delà de ce cas) : la question « ce composant peut-il, *en principe*, représenter la fonction qu'on lui demande ? » doit être posée **avant** tout balayage d'hyperparamètres. Ni les 10 tests unitaires existants (formes, masquage, NaN, non-invariance par permutation) ni le débat FF/sans-FF ne pouvaient attraper ce défaut, et quatre sessions GPU ont balayé LR, $N_{\text{step}}$, `use_ff`, `n_register` autour d'un mécanisme structurellement incapable de réussir.

**[IMPLÉMENTÉ 2026-09-14] Pooling multi-tête (question de l'utilisateur, 2026-09-13) — le pooling résoudrait-il le problème « par accident » ?** `LevelCompressor._pool()` n'avait **jamais** eu de paramètre `n_head` — un seul score par enfant, calculé sur les `d_model` dimensions entières d'un coup (à ne pas confondre avec `HierarchicalMemory.attend()`, qui lui *est* multi-tête, mais c'est la recherche finale en aval, pas la construction des nœuds). Découper le pooling en $H$ têtes donne à chaque tête sa propre distribution de poids sur les enfants du bloc, avec la possibilité qu'une tête se spécialise "position clé" et une autre "position valeur". **Évaluation, avant test empirique : atténuation statistique plausible, pas une correction propre** — à l'intérieur d'une même tête, le même poids sert toujours à moyenner la tranche-$K$ et la tranche-$V$, donc le conflit structurel persiste localement ; $\tilde K_p$ et $\tilde V_p$ finiraient chacun avec un mélange de tranches propres et contaminées selon le sens de spécialisation de chaque tête, sans garantie contrairement au découplage explicite. Coût quasi nul (pas de nouveau paramètre, juste un découpage de la requête existante).

**Implémentation** : `LevelCompressor(pool_n_head=1)`, propagé via `HierarchicalMemory`/`Thinker`. `pool_n_head=1` (défaut) reproduit exactement le comportement précédent. Testé : 46 tests existants toujours verts ; nouveau test manuel (`pool_n_head=4`, seul et combiné à `decouple_kv`/`k_dim`/`attend()` multi-tête) — formes correctes, gradient atteint tous les paramètres, pas de NaN avec masquage.

**[RÉSULTAT 2026-09-14] Prédiction ci-dessus INFIRMÉE par l'empirique, après un premier faux départ méthodologique.** Un résultat initial à `lr=6e-4` (2 seeds, 99,2-99,8%) avait semblé confirmer une "victoire nette" contraire à la prédiction — mais un suivi à 7 seeds a montré un motif bimodal (3/7 réussite, 4/7 échec franc), retiré comme non robuste et d'abord relu comme "fenêtre LR mal calée, comme partout ailleurs sur ce projet" (cf. plan, leçon LR). Un balayage LR dédié et fin (`4e-4` à `5,5e-4`, 5 seeds/cellule) a ensuite tranché : **`lr=4e-4` est robuste à 5/5 (99,6-99,9%)**, tandis que `4,5e-4` — un seul cran au-dessus — échoue déjà nettement ; le point `6e-4` testé initialement était simplement hors de cette fenêtre étroite, ce qui explique complètement le bimodal observé. **Conclusion, une fois le LR correctement calé : `pool_n_head=4` seul atteint bien une victoire nette (99,6-99,9%), comparable au découplage explicite, pas juste une atténuation statistique.** Le raisonnement structurel ci-dessus ("le conflit persiste localement à l'intérieur d'une tête") n'est donc pas confirmé empiriquement à cette échelle — soit la spécialisation par tête suffit en pratique à séparer proprement les tranches $K$/$V$, soit un autre mécanisme compense, mais dans les deux cas le résultat mesuré prime sur la prédiction *a priori*. Implication pratique : `pool_n_head` (zéro nouveau paramètre) pourrait être une alternative moins coûteuse que `decouple_kv` (coût $n_{\text{slots}} \times d_{\text{model}}$) pour ce problème précis — à garder en tête pour Phase 3/4, pas encore décidé comme remplacement.

**[RÉSULTAT 2026-09-14, texte réel] Ne transfère PAS à l'échelle/au corpus de Phase 3 — le remplacement envisagé ci-dessus est écarté.** Même config que la principale (`depth=1`, `lr=3e-3` revalidé par sweep rapide), seul `pool_n_head=4` change : `final_loss=1,5049` contre `1,3517` pour la principale (découplée) — nettement pire, contraire à l'hypothèse ("au moins ne pas dégrader"). Lecture la plus plausible : le pooling multi-tête répartit `d_model` entre les têtes, réduisant la capacité par tête disponible pour résumer chaque bloc — un coût négligeable sur la tâche synthétique (faits courts, vocabulaire réduit, contenu structuré) mais significatif sur du texte réel (vocabulaire riche, contenu à haute entropie par bloc), où `decouple_kv` (deux projections dédiées, chacune à pleine largeur $d_{\text{model}}$) préserve toute la capacité de représentation. **Conclusion révisée : `pool_n_head` reste une victoire confirmée sur la tâche synthétique de petite échelle, mais n'est PAS une alternative viable à `decouple_kv` sur du texte réel à cette échelle — `decouple_kv` reste le choix par défaut pour Phase 3/4.** Utile comme rappel méthodologique de plus : un résultat validé sur banc d'essai synthétique ne se transfère pas automatiquement à l'échelle réelle, à revérifier systématiquement plutôt qu'à supposer.

**[IMPLÉMENTÉ 2026-09-14] Asymétrie $\dim(K) < \dim(V)$ (§5.4)** — proposée le 2026-09-13 (précédent Product-Key Memory / Memory Layers at Scale), jamais codée jusqu'ici. `HierarchicalMemory(k_dim=None)` : `k_dim=None` (défaut) reproduit exactement l'ancien comportement symétrique (`k_proj`/`q_proj` sortent `d_model`, un seul `intrablock_pos` partagé). Quand `k_dim` est explicitement plus petit : `k_proj`/`q_proj` sortent `k_dim`, `v_proj` reste à `d_model`, une seconde table `intrablock_pos_v` apparaît (le biais de position ne peut plus être partagé entre K et V de largeurs différentes), et `attend()` en mode multi-tête calcule désormais deux tailles de tête séparées (`k_dim/n_head` pour Q/K, `d_model/n_head` pour V — `F.scaled_dot_product_attention` supporte nativement $E_{qk} \neq E_v$). **Bug trouvé et corrigé pendant l'implémentation** : `HierarchicalMemory.build()` réutilisait une seule variable de dimension (`dc`, dérivée de `cur_k`) pour reformer *et* les enfants-$K$ *et* les enfants-$V$ à chaque niveau de la hiérarchie — silencieusement faux dès que $\dim(K) \neq \dim(V)$ (`RuntimeError` de reshape immédiat, heureusement — pas un échec silencieux). Corrigé en trackant les deux dimensions séparément. Testé comme le pooling multi-tête ci-dessus, y compris combiné (`k_dim` + `pool_n_head` + `decouple_kv` + `attend()` multi-tête simultanément) et de bout en bout à travers `Thinker.forward()` avec rétropropagation complète.

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

### 5.2bis Coût de l'indexeur (`build`/`attend`) et pertinence de FlashAttention **[NOTES, clarification 2026-09-13]**

Question posée : le goulot d'étranglement mémoire déjà trouvé côté distillation (le tenseur logits `(B,T,248077)` qui sature la VRAM, cf. `learn/distill/README.md`) pourrait-il venir de l'indexeur ? **Non, pas actuellement** — `learn/distill/train_sft.py` valide le pipeline avec un transformer dense standard, **`Thinker`/`HierarchicalMemory` n'y sont pas encore branchés**. La question reste pertinente pour la suite, une fois branchés :

- **`build()`** : un seul passage, vectorisé par niveau (§5.1, pas de boucle Python par nœud) — coût proportionnel au nombre de feuilles $\times d_{\text{model}}$, effectué **une fois par KB** (pas à chaque itération $t$), analogue à un coût de *prefill* de contexte long dans un transformer standard.
- **`attend()`** : ré-exécuté à **chaque** itération $t$ (la requête change avec $R_t$) — c'est une vraie attention scaled-dot-product standard sur $K_{\text{unifié}}$ (feuilles + tous les niveaux compressés, §5.2). Sa complexité est celle d'une attention sur une séquence de longueur = nombre total de nœuds de la hiérarchie (feuilles $\times \frac{\text{block\_size}}{\text{block\_size}-1}$ environ, la somme géométrique des niveaux au-dessus des feuilles restant petite) — donc **FlashAttention s'applique directement ici**, et deviendra pertinent dès que la KB réelle (Phase 3+) et $N_{\text{step}}$ (Phase 1quater) sont assez grands pour que ce coût, répété $N_{\text{step}}$ fois par forward, devienne significatif. C'est un point d'attention distinct du problème de logits de vocabulaire (qui n'est pas une opération d'attention — softmax sur le vocabulaire, pas requête-clé — et que FlashAttention n'accélère donc pas ; la piste pour celui-là reste la loss KD chunkée, cf. `learn/distill/README.md`).
- **Mémoire d'activation de la hiérarchie** : garder $K,V$ de tous les niveaux en mémoire pendant `attend()` ajoute un facteur multiplicatif modeste au-dessus des feuilles seules (~$\frac{\text{block\_size}}{\text{block\_size}-1}$, ex. ~1,14× à block_size=8) — pas une explosion, la hiérarchie elle-même reste bon marché en mémoire ; c'est répéter `attend()` $N_{\text{step}}$ fois qui multiplie le **calcul**, pas la taille de la hiérarchie stockée.

### 5.3 Résolution implicite vs explicite

Le softmax unifié arbitre *implicitement* le niveau de résolution (feuille précise vs résumé de haut niveau) via les scores d'attention — pas de routeur dédié séparé. Formulation précise (ligne ~1189, non contestée par l'utilisateur après proposition) :
> « En plaçant les nœuds de différents niveaux dans le même Softmax normalisé (RMSNorm), le modèle n'a pas besoin de prédire une décision de niveau binaire. Si un concept haut niveau $\tilde K_{\text{haut}}$ a une forte affinité avec $Q$, son score $s_i$ dépasse le seuil, tandis que les sous-arbres non pertinents restent en dessous sans être explorés. »

L'utilisateur a lui-même clarifié (ligne 1181) qu'il distingue deux notions de "seuil" à ne pas confondre : un **seuil de largeur** (nombre de top éléments retenus par l'index, ce qu'il visait initialement) vs un **seuil de profondeur/résolution** (quel niveau de la hiérarchie explorer) — et suggère que le second peut rester **implicite** via l'indexation ($Q$ naturellement plus proche d'un $\tilde V$ de haut niveau ou d'un $V$ de bas niveau selon le cas) plutôt qu'un choix de niveau explicite. **[CONFIRMÉ — distinction largeur/profondeur clarifiée par l'utilisateur ; résolution de profondeur implicite favorisée]**

### 5.4 KB comme coût proche d'un KV-cache, et asymétrie $\dim(K) < \dim(V)$ **[NOTES, précision de l'utilisateur, changement pas encore implémenté]**

**Précision de l'utilisateur (2026-09-13)** : la KB (par épisode, §8, ou persistante, §8bis) se comporte, du point de vue de l'empreinte mémoire, plus comme un **KV-cache** (coût qui grandit avec ce qui doit être tenu en mémoire) que comme un FF (coût fixe, indépendant de ce qui est traité) — exactement pourquoi l'indexation hiérarchique (§5) est importante : elle réduit cette empreinte en compressant, et sélectionne les $V$ pertinents pour la tâche en cours plutôt que de tout garder à résolution pleine. Corollaire attendu : à nombre de faits égal, une KB explicite est probablement **moins efficiente en paramètres** qu'un FF équivalent (le FF encode implicitement une fonction compressée de la distribution des faits ; la KB stocke les faits eux-mêmes, plus explicite mais plus volumineux) — la FF pourrait donc servir de **référence de taille** pour dimensionner $V$ (et possiblement $K$) plutôt que l'inverse.

**Question posée : $\dim(K) < \dim(V)$, que dit la littérature ?** Confirmé, c'est un choix délibéré et documenté dans la littérature des mémoires apprises à grande échelle, pas une intuition isolée :
- **Product-Key Memory** (Lample et al. 2019) et sa mise à jour **"Memory Layers at Scale"** (He et al., Meta 2024, arXiv 2412.09764) : $\dim(V)$ vaut **par défaut $d_{\text{model}}$** (la valeur est mélangée directement dans le flux résiduel, elle doit porter toute l'information utile) ; $\dim(K)$ est **choisi plus petit** (couramment $d_{\text{model}}/2$ dans leurs expériences) — motivé non pas par "la clé compte moins" mais par la **tractabilité de l'adressage** : une clé de plus petite dimension rend la recherche du plus proche voisin (exacte, via clés produits) praticable sur un nombre de slots énorme. Leur article note aussi que $\dim(V)$ peut être réduit et compensé par une projection supplémentaire si on veut échanger dimension de valeur contre nombre de slots, à budget de paramètres égal.
- **MemSizer / "Linearizing Transformer with Key-Value Memory"** : va plus loin avec un mécanisme "unbalanced key-value" explicite — clés indépendantes de l'input (apprises, fixes), valeurs dépendantes de l'input — asymétrie de nature différente de la nôtre mais qui confirme le principe général : $K$ et $V$ n'ont aucune raison d'être traités symétriquement dans une mémoire, contrairement à l'attention standard où ils sont dimensionnellement liés par construction.

**État actuel du code, pas encore aligné avec cette proposition** : `HierarchicalMemory.k_proj` et `.v_proj` (`core/indexed_memory.py:124-125`) sont tous deux `nn.Linear(d_model, d_model)` — $\dim(K)=\dim(V)=d_{\text{model}}$, aucune asymétrie. Changement proposé (pas encore fait) : réduire la dimension de sortie de `k_proj` (ex. $d_{\text{model}}/2$), garder `v_proj` à $d_{\text{model}}$ pleine, ajuster le facteur d'échelle du produit scalaire ($\sqrt{\dim(K)}$, pas $\sqrt{d_{\text{model}}}$) dans `LevelCompressor` et dans l'attention finale de `attend()`. À tester comme variante (candidate pour un futur flag Phase 1bis, ou pour la KB persistante de la Phase 10 dès sa conception plutôt qu'en rétrofit).

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

**[RÉTRACTÉ le 2026-09-13 par l'utilisateur]** — la tentative d'unification ci-dessus (Sentinel = No-Op + largeur variable + step compute-only) était une erreur de logique : le Sentinel ne sert qu'au cas binaire "ne rien récupérer du tout" (No-Op). Il **n'a rien à voir** avec :
- la **largeur de sélection de l'index** (rôle : rendre le modèle plus précis en fonction de la tâche en cours — combien de largeur/résolution de la hiérarchie consulter, pas juste "consulter ou non") ;
- les **steps "compute-only"** (itérations de la boucle qui ne font aucun appel mémoire).

Une paire $(\mathbf k_\emptyset,\mathbf v_\emptyset)$ ajoutée au softmax ne donne qu'un signal tout-ou-rien (le Sentinel gagne ou pas) — elle ne fournit pas de contrôle gradué sur la largeur/précision de la recherche, ni de mécanisme dédié pour "sauter" un appel mémoire à une itération donnée.

**[OUVERT]** : l'utilisateur estime qu'on pourrait avoir besoin des deux (largeur variable + step compute-only) et soupçonne qu'une approche unifiée existe **entre ces deux-là spécifiquement** (pas via le Sentinel) — non résolu, à explorer. Le mécanisme à seuil $\theta$/pénalité $\lambda$ formalisé plus haut dans cette section reste, lui, un candidat plausible pour la largeur variable seule (il donne un contrôle gradué réel, contrairement au Sentinel) mais son lien avec un step "compute-only" n'est pas non plus établi, et son risque de collapse documenté reste un problème ouvert. Ne pas considérer cette question comme résolue tant qu'une proposition concrète n'a pas été validée.
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

## 8bis. KB persistante apprise (mémoire long-terme, distincte de l'espace unifié input/KB) **[NOTES, précision de l'utilisateur — extension à implémenter, pas encore présente]**

**Distinction posée par l'utilisateur** : la §8 (stratégie 1, espace unifié) ne couvre que la récupération **par épisode** — les faits de la KB sont donnés en tokens d'entrée à chaque exemple, comme un contexte fourni de l'extérieur (comportement proche d'un long contexte / KV-cache, cf. §KV-cache ci-dessous). Ce n'est **pas** la même chose qu'une KB long-terme que le modèle **construit et organise lui-même** au fil de l'entraînement, avec des embeddings **appris** (paramètres entraînés par descente de gradient sur tout le corpus, pas dérivés d'un exemple donné) — c'est cette seconde capacité qui manque actuellement pour tenir pleinement la thèse §-1 ("connaissance dans le KV, pas dans les poids") : sans elle, le mécanisme ne teste que la **récupération** d'information déjà donnée, jamais l'**accumulation** de connaissance à travers les exemples d'entraînement — exactement le phénomène statistique cross-exemple que §-1 attribue aux FF (déjà noté comme limite de `n_facts=16` en Phase 1bis, cf. plan).

**Précédent direct en littérature — Product-Key Memory** (Lample et al. 2019, *"Large Memory Layers with Product Keys"*, et sa mise à jour 2024 *"Memory Layers at Scale"*, Meta) : une couche de mémoire à clés produits est un ensemble de $(K, V)$ **entièrement appris** (paramètres, pas dérivés de l'input), organisé pour une recherche approximative du plus proche voisin efficace ($|K|=512^2$ slots dans l'article original, seuls $k=32$ retrouvés par requête). Résultat clé cité : un modèle à 12 couches + une couche mémoire dépasse un modèle sans mémoire à 24 couches de même largeur — la mémoire apprise est une façon *compétitive* d'ajouter de la capacité, pas seulement une alternative moins efficace au FF. C'est le modèle le plus proche de ce que demande l'utilisateur : un ensemble de slots KV **fixe, appris, toujours résident**, par opposition à la KB actuelle dont la taille (et donc l'empreinte mémoire) dépend du nombre de faits fournis par épisode.

**Proposition d'implémentation (à discuter, pas encore codée)** : ajouter un niveau de mémoire **supplémentaire**, distinct de la hiérarchie construite par épisode (`HierarchicalMemory`, §5) — un ensemble fixe de $(K,V)$ appris comme paramètres du modèle (`nn.Parameter`, pas dérivés de `self.embed`), interrogé par la même requête unifiée que les autres niveaux (softmax unifié, §5.2), entraîné bout-en-bout par le seul signal de la tâche (pas d'opération d'écriture différentiable explicite façon NTM/DNC — la littérature Product-Key montre que le simple gradient suffit à "organiser" ce que la mémoire stocke). Les deux modes de mémoire coexisteraient : la KB par épisode (§8, donnée, taille variable, coût proche d'un KV-cache) pour ce qui est fourni en contexte, et cette KB persistante (fixe, apprise, coût proche d'un FF en taille mais organisé en KV adressable) pour ce que le modèle doit retenir à travers les exemples.

**Corpus proposé pour l'entraîner (question de l'utilisateur)** : un corpus où des faits **récurrent réellement à travers les exemples** est nécessaire — sinon rien n'incite le modèle à écrire dans la mémoire persistante plutôt que dans la KB par épisode. Les tâches synthétiques actuelles (`kb_retrieval`, `kb_chain_retrieval`) sont délibérément conçues avec des faits **uniques par épisode** (resamplés à chaque batch) — elles ne peuvent pas servir à tester ce mécanisme, il faut un corpus à connaissance partagée. Deux candidats proposés par l'utilisateur, tous deux pertinents :
- **TinyStories** : déjà dans le pipeline distillation (`learn/distill/prepare_general_data.py`), vocabulaire/faits simples qui se répètent naturellement à travers les histoires (relations de personnages, objets du quotidien) — bon premier test, peu coûteux.
- **Données orientées raisonnement/math** (`open-r1/OpenR1-Math-220k`, déjà dans le pipeline) — les formules/identités récurrentes across exemples sont un test plus proche de la thèse (connaissance factuelle réutilisable, pas seulement du vocabulaire général).

**Impact sur le travail en cours** : n'invalide rien de ce qui est déjà testé (Phase 1bis/2 restent les bons bancs d'essai pour le mécanisme de **récupération** par épisode) — c'est un **nouveau composant/phase à ajouter**, pas une correction. Proposé comme nouvelle phase du plan (voir `indexed_attention_experiment_plan.md`), à mener après (ou en parallèle indépendant de) Phase 2, une fois `Thinker` correctement isolé du `train_sft.py` de pipeline-validation (qui utilise actuellement un transformer dense standard, pas encore l'architecture Indexed Attention — cf. §11).

**Précision de l'utilisateur (2026-09-13) — pas de `k_proj`/`v_proj` pour ce niveau** : contrairement à `HierarchicalMemory` (§5), où $K,V$ **doivent** être dérivés du contenu (`self.embed(kb_tokens)` puis projection) parce que le contenu change à chaque épisode, la KB persistante n'a pas ce besoin — $K$ et $V$ **sont** les paramètres appris directement (`nn.Parameter(n_slots, dim_k)` / `nn.Parameter(n_slots, dim_v)`, initialisés aléatoirement puis mis à jour uniquement par le gradient), sans aucune couche de projection appliquée dessus à chaque forward — les recalculer via une projection serait un travail redondant sans contenu variable à projeter. Seul $Q$ reste calculé (projeté depuis le registre latent et/ou la SM courante, comme pour les autres niveaux). Confirme et précise la proposition d'implémentation ci-dessus ; c'est exactement la structure de Product-Key Memory ($K,V$ sont des tables de paramètres, pas des projections).

**Précédents déjà présents dans les notes brutes de l'utilisateur, à relier ici** (`raw/Branch-•-Indexed-Attention.md`) :
- **Ligne ~543-544** : *Titans* (Google) et *Infini-Transformer* — mémoire neuronale **persistante et différentiable**, mise à jour par une règle d'apprentissage associatif (métrique de surprise), qui compresse l'historique **sans limite de taille de contexte** ; et les systèmes de mémoire étagée (MemGPT/Letta, MemWalker) qui distinguent mémoire de travail (VRAM/contexte) et mémoire persistante externe. **Différence importante à trancher** : Titans/Infini-Transformer *écrivent* dans leur mémoire en continu, y compris à l'inférence (mise à jour test-time via la règle associative) — alors que Product-Key Memory n'apprend que par le gradient d'entraînement standard, mémoire figée à l'inférence. La proposition ci-dessus (PKM-style) est la plus simple à implémenter en premier (pas de règle d'écriture différentiable dédiée à concevoir) — une extension test-time façon Titans reste une piste **[OUVERT]**, à ne considérer qu'après un premier résultat PKM-style.
- **Ligne 1290** : *Perceiver Cross-Attention* (learned query tokens interrogeant des blocs K/V sous-jacents) — déjà le principe du `LevelCompressor` existant (§5.1), rassurant sur la cohérence de cette famille de mécanismes avec ce qui est déjà implémenté.
- **Ligne 1313** : *Sentinel/No-Op token* — une paire $(\mathbf{k}_\emptyset, \mathbf{v}_\emptyset)$ **apprise**, absorbant l'attention quand aucune mémoire n'est nécessaire. C'est en fait un cas particulier, à 1 seul slot, exactement de ce qui est proposé ici (un $(K,V)$ appris, non dérivé de l'input) — déjà noté comme hors scope MVP (§7.1) mais conceptuellement le même mécanisme de base ; les deux pourraient être unifiés (le slot No-Op devient un slot de plus dans la KB persistante) plutôt qu'implémentés séparément.

## 9. Baselines & protocole d'évaluation (cadre de test, pas encore implémenté)

Trois baselines architecturales pour isoler la source de tout gain observé (lignes 1358-1455), à garder en tête pour la conception de `tests/test_indexed_memory.py` et de futurs bancs d'essai :

- **Baseline A (RAG plat / mono-passage, $T=1$)** : une seule cross-attention sur le Top-$k$ récupéré, **sans** boucle latente ni hiérarchie. Répond à : *le raisonnement itératif apporte-t-il un gain sur la récupération simple ?*
- **Baseline B (boucle pure, sans mémoire)** : même registre latent bouclé $T$ fois, mais accès mémoire désactivé. Répond à : *le gain vient-il de la boucle elle-même ou de la mémoire externe ?*
- **Baseline C (attention dense sur tout le contexte)** : borne supérieure de référence (si la longueur de séquence le permet à petite échelle).
  - **[CORRIGÉ 2026-09-13]** `depth=0` a été utilisé comme incarnation de cette baseline — **c'est un mauvais choix**. Sans compresseur, les feuilles sont des $K$/$V$ **par token** : $K_i = W_k E(\text{tok}_i)$, $V_i = W_v E(\text{tok}_i)$, donc attendre sur un token-clé retourne ce token-clé. Une mémoire plate de feuilles brutes ne peut représenter **aucune** association clé→valeur, à n'importe quel budget — les 5-7 % de la Phase 0 sont structurels. Le gap « 99,6 % vs 6,0 % » n'établit donc pas la supériorité de l'indexation hiérarchique, seulement que le groupement en blocs est le seul chemin vers une entrée associative dans cette implémentation.
  - La baseline honnête « mémoire **sans index** » est **`depth=1`** : un niveau de compression, un nœud par fait, attention dense sur les nœuds, aucune hiérarchie multi-niveaux. C'est elle qu'il faut opposer à `depth≥2` (plan, Phase 0bis). Corollaire pratique : l'intégration sur texte réel peut démarrer **sans index** (`depth=1`) — la hiérarchie répond au *coût* d'une mémoire massive, pas à la capacité d'association.

Découpage attendu des splits d'évaluation (ligne 1443-1447, 1736-1739), pertinent pour la conception de `data/kb_retrieval.py` :
1. **Raisonnement pur** (tout est dans l'input, pas besoin de KB) — Baseline B doit égaler le modèle complet ici.
2. **Récupération pure** (un seul fait à extraire d'une large KB statique) — Baseline A doit égaler le modèle complet ici.
3. **Raisonnement multi-sauts sur mémoire** (fait X → débloque Y → débloque Z) — seul le modèle complet doit réussir ; c'est le test décisif.

**Niveaux de hasard et prédicteurs triviaux — obligatoire depuis 2026-09-13 [CONFIRMÉ]** : une accuracy sur ces tâches n'est interprétable qu'avec sa référence correcte. Le modèle ne choisit jamais dans le vocabulaire : il émet une valeur présente dans la KB de l'épisode (mesuré : 97-98 %). La référence est donc le **hasard conditionnel** $1/n_{\text{facts}}$, jamais $1/|V|$. S'y ajoutent les raccourcis structurels de la tâche (sur une chaîne, la réponse finale n'apparaît jamais comme clé ⇒ un tirage parmi les valeurs non-clés atteint 33-50 % sans aucun saut). `learn/indexed_attention/eval_metrics.py` calcule tout ça et l'imprime à chaque run ; aucune accuracy ne doit être citée sans ce bloc. C'est l'absence de ce contrôle qui a fait lire un échec total (plateau $= 1/n_{\text{facts}}$, sous le meilleur raccourci) comme une « composition partielle » pendant plusieurs sessions.

**[OUVERT]** : l'utilisateur a lui-même mis en doute (ligne 1718) que la baseline MVP proposée par Gemini (Top-k fixe, KB injectée en contexte plutôt que préremplie/hiérarchique) respecte fidèlement l'objectif "LLM généraliste à faible mémoire paramétrique" énoncé plus haut — pas de résolution finale dans le texte lu. À garder en tête : le MVP de ce document (§5-§6) vise à tester la **mécanique** (hiérarchie + softmax unifié + gradient natif), pas encore le protocole d'évaluation complet ci-dessus.

### 9.1 [CORRECTIF MÉTHODOLOGIQUE IMPORTANT, 2026-09-14] — les tâches en chaîne ne peuvent pas discriminer l'utilité de la SM multi-emplacements

**Constat** (retour direct de l'utilisateur, qui a raison de ne pas laisser conclure trop vite à l'inutilité de la SM à partir des résultats `sm_cap`/`n_memory` de ce soir) : `data/kb_chain_retrieval.py` (fait $i$ → débloque fait $i{+}1$ → ...) est **markovienne par construction** — retrouver le fait $i{+}1$ ne demande jamais rien de plus ancien que la valeur retrouvée au fait $i$. Un seul état courant ($R$) suffit donc structurellement pour résoudre une chaîne, **peu importe la qualité réelle du mécanisme de mémoire multi-emplacements** — même à `n_hops=4`.

**Conséquence directe** : les deux résultats nuls de ce soir (`sm_cap=1`≈`sm_cap=None` sur `n_hops=2`, Indexed Attention ; `n_memory=1`≈`n_memory=10000` sur `copy`, toy model) **ne démontrent pas que la SM/le buffer multi-emplacements est inutile** — ils démontrent que la tâche utilisée pour les mesurer ne pouvait pas révéler cette utilité dans un sens ou dans l'autre. Ce n'est pas une régression sur le rôle architectural de la SM (spec §-1 : la SM existe précisément pour lever le goulot d'étranglement d'information que le seul registre $R$ imposerait) — c'est un angle mort de conception de tâche, à corriger avant de tirer quelque conclusion que ce soit sur ce composant.

**Tâche correcte à construire (pas encore implémentée)** : une tâche qui force à garder **plusieurs faits distincts, temporellement séparés, non réductibles l'un à l'autre** — ex. récupérer le fait $A$ tôt dans le raisonnement, récupérer un fait $B$ **indépendant** (pas lié à $A$ par une clé partagée) bien plus tard, puis combiner $A$ et $B$ à la sortie (concaténation, comparaison, opération si numérique). Un registre récurrent unique devrait avoir du mal à porter $A$ intact pendant que le modèle travaille sur autre chose entre les deux récupérations ; un buffer multi-emplacements n'a qu'à le garder de côté. C'est cette tâche, pas une chaîne, qui teste réellement la question posée par `sm_cap`/`disable_sm`.

**Ne pas généraliser prématurément à l'inverse non plus** : un résultat positif sur une chaîne (SM utile même là) serait en soi intéressant, mais son absence ne prouve rien — traiter tout résultat `sm_cap`/`disable_sm` obtenu sur `kb_chain_retrieval.py` comme un signal secondaire, pas une réponse définitive, tant que la tâche à faits indépendants n'existe pas.

**[OUVERT] Second piège identifié en concevant la tâche de remplacement, pas encore résolu — ne pas implémenter avant de le trancher.** Une tâche naïve "2 faits indépendants $A$/$B$, requête donnant les deux clés d'un coup, sortie = combinaison de leurs valeurs" ne force pas nécessairement de séparation temporelle : le softmax unifié d'`attend()` (§5.2) n'est pas restreint à une seule position — rien n'empêche, en un seul appel à `attend()`, de mettre ~50% du poids sur le nœud de $A$ et ~50% sur celui de $B$, produisant $O_{kb} \approx 0{,}5 V_A + 0{,}5 V_B$ en une seule étape, sans jamais avoir besoin de mémoire à travers le temps. Le problème n'est donc pas seulement "combien de faits à combiner" mais "est-ce que la tâche autorise une solution en un seul passage d'attention" — une contrainte supplémentaire, distincte de la markovianité de §9.1.

De plus, `HierarchicalMemory.build()` construit toute la KB **une seule fois avant la boucle** — rien dans l'architecture actuelle ne permet de révéler une information *au milieu* du raisonnement ; toute la KB est visible dès l'itération 0. Ça pourrait signifier qu'aucune tâche à KB statique et entièrement visible dès le départ ne peut forcer une vraie dépendance temporelle, indépendamment de sa conception — une limite structurelle du protocole de test actuel, pas seulement de `kb_chain_retrieval.py`.

**Piste la mieux établie dans la littérature pour ce problème précis** (mémoire associative testée par un délai forcé — NTM/DNC, tâches de rappel associatif type "store then recall after a distractor phase") : stocker des paires, imposer une phase de calcul **sans rapport** avec les faits stockés (occupant plusieurs itérations, pour forcer $R$ à évoluer sans que les faits y soient réutilisés), puis demander de rappeler un fait spécifique choisi après coup. Le délai/la distraction est ce qui écraserait un $R$ non tamponné mais pas une mémoire externe correcte. Pas encore spécifié précisément ni implémenté — prochaine tâche de conception, à ne pas bâcler étant donné le premier piège découvert ci-dessus.

## 10. Curriculum de largeur "large → étroit" (idée propre, non formalisée mathématiquement)

Idée de l'utilisateur (ligne 1824) : en début de traitement (itérations basses), la sélection/largeur d'attention devrait être **large** pour bien explorer input + mémoire ; puis se **resserrer progressivement** au fil des itérations pour converger vers une décision, avec la possibilité de laisser le modèle apprendre ce planning lui-même plus tard (post-training). **[NOTES/CONFIRMÉ comme idée, non implémentée]**

Point de vigilance soulevé par Gemini (ligne 1844, non contesté) : un resserrement trop strict interdit le *backtracking* — si une déduction tardive révèle un besoin imprévu d'information, le modèle doit garder la possibilité de rouvrir une fenêtre large ponctuellement. **[OUVERT]** — aucune formalisation mathématique proposée dans la conversation ; hors scope MVP (dépend du mécanisme de largeur adaptative §7.1, lui-même hors MVP).

## 11bis. Output Streams (source : `raw/Distill-reasonning-stream.md`, absent des sections précédentes)

Concept distinct de la hiérarchie KB (§5-6) mais qui touche directement à la question "les poids qui calculent la sortie sont-ils partagés ou indépendants ?" — à traiter dans ce document plutôt que de le laisser implicite.

### Principe **[NOTES]**

Le registre récurrent produit, à chaque itération $t$, un état $h^{(t)}$ qui est empilé dans la mémoire court terme (SM) : $K_{sm}, V_{sm} = \text{Proj}([h^{(1)}, \dots, h^{(T)}])$ — c'est exactement le buffer SM déjà utilisé par `Thinker` (§11). Au lieu de lire l'état final du registre directement, **un ou plusieurs "Output Streams" légers (1-2 couches) interrogent ce buffer par leur propre cross-attention** :

$$
O_{\text{stream}_i} = \text{Attn}\big(Q_{\text{stream}_i}, K_{sm}, V_{sm}\big), \qquad \text{logits}_i = \text{Head}_i(O_{\text{stream}_i})
$$

**Point clé (répond directement à la question posée) : $Q_{\text{stream}_i}$ et $\text{Head}_i$ sont des poids propres à chaque stream $i$, non partagés entre streams, ni avec le core récurrent.** Citation directe : *« chaque Stream […] chacun train indépendamment »* ; *« vous pouvez geler le core et entraîner les output streams indépendamment »* (lignes 230, 307). Seul le buffer SM lui-même (une activation, pas un poids) est partagé en entrée de tous les streams.

### Motivation supplémentaire, à garder en tête mais **non confirmée** : stabilisation du Looped Transformer **[NOTES, hypothèse théorique — infirmée comme explication du cas observé cette session]**

Citation directe (lignes 213-217) : *« Les transformers à poids partagés récurrents souffrent d'instabilité dynamique : la magnitude des hidden states explose ou s'effondre facilement au fil des boucles. Contraindre les états latents du loop via une perte MSE ou Cosine par rapport aux couches successives d'un Teacher pré-entraîné agit comme une régularisation pour borner les représentations. »*

**Historique de cette session, pour traçabilité** : un échec d'entraînement observé sur GPU (`thinker-e9`, `depth=4`, 64 facts, loss bloquée au niveau du hasard) puis reproduit localement avait d'abord été attribué à ce phénomène d'instabilité — l'échec ne se résorbait pas avec un `d_model`/`n_register` plus grand, ni avec `lr=1e-3`. **Corrigé ensuite** : `thinker-e9` a identifié par un balayage de LR plus complet un **plateau de convergence étroit autour de `lr=3e-4`** (confirmé indépendamment en local : mêmes hyperparamètres, `lr=3e-4` fait passer l'accuracy de ~10% à 98-100% entre les steps 800 et 1000, alors que `1e-4`/`1e-3`/`3e-3`/`1e-2` échouent tous). **La cause réelle était une sensibilité au learning rate (fenêtre étroite), pas une instabilité structurelle du looped transformer** — l'hypothèse de supervision intermédiaire ci-dessous n'a donc pas été implémentée. Gardée dans ce document car la citation source reste théoriquement valide et pourrait redevenir pertinente à plus grande échelle (plus d'itérations, plus de profondeur), mais **ce n'est plus le diagnostic retenu pour le cas observé ici**.

**[CORRIGÉ 2026-09-13 — la supervision d'attention visait la mauvaise cible]** : l'implémentation de cette piste (`diagnose_attention_supervision.py`, `train_kb_chain_attn_supervised.py`) poussait la requête vers la **feuille KEY** du fait cible. Or attendre sur une feuille-clé retourne $W_v E(\text{key})$ — la clé elle-même, jamais la valeur du fait. L'objectif auxiliaire était donc **parfaitement satisfaisable et orthogonal à la tâche**, ce qui explique exactement le résultat GPU resté inexpliqué (self-match : `mean_rank=0.000`, `top1=1.000` sur 3 seeds ; accuracy inchangée à ~25 %). Seul le **nœud de niveau 1** porte la paire clé→valeur. Les deux scripts ciblent désormais ce nœud par défaut (`--supervise node`) ; `--supervise leaf` reproduit l'ancienne grille. Utile seulement conjointement au pooling découplé (§5.1bis).

**Piste correspondante, reportée sans être écartée** : si l'instabilité redevient un candidat plausible plus tard (ex. après un balayage LR complet qui échoue à trouver un plateau stable à plus grande échelle), le principe d'une supervision intermédiaire sans Teacher reste disponible : sur une tâche synthétique on connaît la vérité terrain de la récupération à chaque étape, donc on peut superviser directement l'attention plutôt que seulement le token final (version sans Teacher de l'Option B "White-Box Trajectory Distillation" déjà documentée ailleurs dans la conversation source). **[OUVERT]**, non prioritaire tant que le balayage LR n'a pas d'abord été fait correctement (voir plan, Phase -1).

### Streams envisagés **[NOTES]**

- **Stream Answer** (tokens) : supervisé par cross-entropy + Top-K KL du Teacher sur les logits finaux.
- **Stream Thinking (embedding)** : auto-génératif sur des embeddings (pas de vocabulaire), aligné sur une couche médiane du Teacher (~40-65% de profondeur) via une perte de similarité/MSE — *« pour laisser le thinker un peu libre sur sa représentation interne »* (ligne 529).
- **Stream Thinking (tokens)** : décodage explicite d'une chaîne de pensée `<think>...</think>`, introduit plus tard dans le curriculum.
- Le nombre d'itérations du core **n'a pas besoin de correspondre** au nombre de couches du Teacher — c'est le mécanisme d'attention propre à chaque stream qui apprend à pondérer les pas récurrents pertinents pour sa tâche, pas un alignement pas-à-pas forcé (ligne 304, 478-479).

### Curriculum d'extinction **[NOTES]**

Idée notée (lignes 574-587) : les streams latents (embedding) sont utiles tôt dans l'entraînement pour guider le core, puis **atténués progressivement** (`weight decay sur la perte latente -> 0`) jusqu'à ne garder en inférence finale que les flux textuels visibles (Stream Thinking en tokens + Stream Answer). **[OUVERT]** — non formalisé mathématiquement (juste "atténuation linéaire" mentionnée, pas de fonction de schedule précisée dans le texte lu).

### État d'implémentation

**Corrigé dans cette session.** `Thinker` expose désormais `self.streams: nn.ModuleDict[str, OutputStream]` (`core/indexed_thinker_model.py`) : chaque `OutputStream` a sa propre requête apprise (`query_seed`) + **1 à 3 couches** de cross-attention empilées (`OutputStreamLayer`, pré-norm + résiduel) + sa propre tête, et interroge par cross-attention le **SM complet accumulé sur toute la boucle** (pas seulement l'état final `R`). Le nombre de couches est configurable par stream (`stream_n_layers={'answer': 1, 'thinking': 2}`, etc.). `forward()` retourne `(R, stream_outputs)` — un dict `{nom: sortie}`, un par stream enregistré. Testé (`tests/test_indexed_memory.py::TestOutputStreamsIndependence`) : paramètres disjoints entre deux streams, `backward()` sur un stream ne peuple aucun gradient sur un autre, gradient qui atteint bien chaque couche d'un stream à 3 couches. Seul un stream `answer` (logits vocabulaire) est branché pour l'instant sur `data/kb_retrieval.py`.

**Stream `thinking` en embedding : plus hors scope, à tester dès les prochaines expériences [CONFIRMÉ, décision de l'utilisateur].** Aucun changement d'architecture nécessaire : `Thinker(stream_dims={'answer': vocab_size, 'thinking': teacher_hidden_dim})` fonctionne déjà — `OutputStream`/`Head` n'ont aucune hypothèse spécifique au vocabulaire, `teacher_hidden_dim` est juste une autre valeur de `out_dim`. Ce qui manquait : l'extraction des couches intermédiaires d'un vrai Teacher, ajoutée (`core/compressor/model_wrapper.py::HFModelWrapper.get_hidden_states()`, réutilise l'infra HF déjà présente pour le projet compressor plutôt que d'en recréer une).

**Critère de choix de la couche [NOTES, `raw/Distill-reasonning-stream.md` lignes 311-320, 562-563]** : couches basses (0-25% de profondeur) = lexical/syntaxique pur, à éviter ; couches médianes (~40-65%) = abstraction sémantique/résolution d'entités, cible privilégiée pour `thinking` ; avant-dernière couche = compression de la décision finale, plutôt pertinente pour aligner `answer`. Risque explicite si mal choisi : *« si une couche est trop basse (syntaxe brute) ou trop haute (effondrement décisionnel vers la lm_head), le stream aura un loss erratique ou forcera des gradients contradictoires »*. **Protocole recommandé, pas un choix a priori** : sonder (cosinus/CKA) à quelle itération de boucle l'état du registre/SM développe la corrélation la plus forte et la plus stable/monotone avec chaque couche candidate, entraîné d'abord avec seulement la supervision `answer` — retenir la couche qui donne cette convergence la plus propre, plutôt que de fixer un pourcentage de profondeur sans vérification empirique.

**Sur le risque d'instabilité de cette perte d'alignement**, discuté en détail avec l'utilisateur cette session : ce n'est pas la distance de la cible à l'initialisation qui pose problème (la CE aussi part loin) mais (a) l'absence de normalisation naturelle d'une perte MSE brute (contrairement à la CE, bornée par le softmax) — atténuable avec une perte cosinus/normalisée plutôt que MSE brute — et (b) un conflit multi-objectif sur les mêmes poids partagés tant que le core n'a pas encore de structure organisée. Les deux sont atténuables (normalisation, montée en poids progressive de cette perte), pas des raisons d'écarter l'idée — à vérifier empiriquement, pas supposer.

**Aller-retour sur le FF de `OutputStreamLayer`** : chaque couche avait initialement, en plus de la cross-attention, un FF résiduel (motif transformer standard). **Retiré ensuite** (voir §-1) : un stream ne doit faire que lire/décoder ce qui est déjà dans la SM, pas mémoriser ses propres associations factuelles indépendamment de ce qui y est réellement présent. `OutputStreamLayer` ne fait plus que cross-attention + résiduel, sans FF.

## 11. Implémentation MVP (première version, testée localement sur CPU)

**Nom de la classe — [RENOMMÉ 2026-09-13]** : `IndexedThinker` → `Thinker`. Précision de l'utilisateur : c'est bien **le** Thinker du projet (celui que décrit la thèse §-1), pas une variante à côté — `core/toy_model.py::ToyThinker` était la version simplifiée utilisée pour déboguer les tâches jouets initiales (mémoire plate concaténée, couche transformer basique, aucune indexation), pas une version antérieure de ce modèle, et n'est pas renommée. `core/thinker_model.py::Th1nker` reste un fichier ancien distinct, inactif/non référencé — à ne pas confondre.

Fichiers : `core/indexed_memory.py` (`HierarchicalMemory`, `LevelCompressor`), `core/indexed_thinker_model.py` (`Thinker`), `data/kb_retrieval.py` (tâche synthétique), `tests/test_indexed_memory.py` (10 tests, tous verts).

Choix concrets faits pendant l'implémentation, non explicitement fixés par la spec ci-dessus — **[DÉFAUT, implémentation]** :

- **`Compress_θ` (§5.1)** : un jeu de $M$ requêtes apprises par la même instance à tous les niveaux, instanciation spécifique de "cross-attention Perceiver-style".
  - **[PÉRIMÉ depuis le 2026-09-13 — décrit ci-dessous l'état AVANT correction]** La version initiale réutilisait **les mêmes poids de pooling** (un seul softmax sur les enfants) à la fois pour résumer $K$ et pour agréger $V$ — présenté alors comme "plus simple qu'une paire de projections $K$/$V$ séparées", avec la réserve "à raffiner si les tests empiriques suggèrent que $K$ et $V$ ont besoin d'être résumés différemment". Les tests empiriques l'ont montré, et de façon plus forte que "raffinement" : ce partage rend une association clé→valeur **structurellement irreprésentable** (§5.1bis), et c'était la cause du plateau multi-sauts.
  - **État actuel** : deux requêtes apprises indépendantes $q_K$, $q_V$ (§5.1bis), `decouple_kv=True` par défaut. Le partage reste accessible comme ablation (`decouple_kv=False` / `--shared_kv_pooling`).
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

## 12. Flags de variantes pour la Phase 1bis du plan (ajoutés à la demande d'`experiment-manager`)

Flags ajoutés pour rendre testables des variantes sans dupliquer le modèle :

- **`Thinker(use_ff: bool = False, ff_hidden_mult: int = 4)`** : par défaut, aucun FF (§-1). `use_ff=True` remplace `fuse_proj` (projection linéaire) par un MLP à 2 couches GELU (`fuse_in`/`fuse_out`) **uniquement dans la boucle principale** — pas réintroduit dans le compresseur ni les output streams, cohérent avec la branche d'action de la Phase 1bis ("réintroduire un FF minimal dans la boucle principale seulement" si le sans-FF s'avère insuffisant pour la composition).
- **`Thinker(detach_sm_keys: bool = False)`** : teste la lecture A de §4.1 (stop-gradient sur les clés SM uniquement, valeurs non affectées) contre la lecture B (bout-en-bout, défaut actuel).
- **`HierarchicalMemory(level_dropout_p: float = 0.0)`**, propagé via `Thinker`: dropout stochastique des niveaux hauts pendant l'entraînement uniquement (`self.training`), jamais sur les feuilles (niveau 0, pour toujours garder un minimum d'ancrage réel). Probabilité croissante avec le niveau ($p_i = p \cdot i / \text{depth}$), implémenté en réutilisant le mécanisme de masquage déjà en place pour le padding (§11) plutôt qu'un chemin séparé — un niveau "droppé" pour un forward donné a simplement son masque mis à `False` en entier.

- **`LevelCompressor(decouple_kv: bool = True)`**, propagé via `HierarchicalMemory` et `Thinker` (CLI : `--shared_kv_pooling` pour l'ablation) — **[2026-09-13]**. Statut différent des trois flags ci-dessus : ce n'est **pas** une variante ouverte à arbitrer empiriquement, c'est une **correction structurelle** dont le défaut par défaut (`True`) est acquis (§5.1bis). Le flag n'existe que pour (a) reproduire les runs antérieurs au 2026-09-13, (b) servir de contrôle négatif dans la grille Phase 2-redo du plan (l'ablation doit rester collée à `conditional_chance`, le découplé doit s'en détacher nettement). Ne pas le traiter comme un hyperparamètre à balayer.
  - Coût : $n_{\text{slots}} \times d_{\text{model}}$ paramètres (une requête apprise de plus).
  - **Interaction à connaître** : la supervision d'attention au niveau nœud (`--supervise node`, §11bis) n'a d'effet utile qu'avec `decouple_kv=True` — sous pooling partagé, même un nœud parfaitement sélectionné ne peut pas restituer la valeur du fait.

Testés (`tests/test_indexed_memory.py::TestPhase1bisVariantFlags`, et `::TestDecoupledKVPooling` pour `decouple_kv` — 9 tests dont un qui énonce explicitement le défaut de l'ablation : sous pooling partagé, `parent_v` est forcément la même combinaison convexe que `parent_k`) : `use_ff` change bien la structure de poids et le gradient atteint `fuse_in`/`fuse_out` ; `detach_sm_keys` change effectivement le gradient reçu par `sm_write_proj` par rapport à la version sans détachement ; `level_dropout_p=1.0` droppe effectivement des niveaux en mode entraînement sans produire de NaN, et n'a aucun effet en mode évaluation (`level_dropout_p=0.0` est un no-op vérifié séparément).

## 13. Dimensionnement d'`Thinker` — synthèse (2026-09-13)

Question posée : au regard de toute la discussion précédente (axes paramètres/profondeur de calcul/mémoire KB, §5.4/§8bis, comparaison Qwen3.5-0.8B), comment dimensionner concrètement le modèle réel ? Récapitulatif en quatre axes **indépendants**, chacun avec sa propre logique de décision :

**1. `d_model` (partagé embedding + espace de travail du core, pas de projection séparée actuellement)** — pas un choix libre isolé : `self.embed` (§8bis) est la table liée au vocabulaire du Teacher (248 077 tokens), donc $d_{\text{model}} \times 248\,077$ est déjà fixé une fois $d_{\text{model}}$ choisi. Point de départ retenu pour les comparaisons "barre de qualité" (§ Reference target) : $d_{\text{model}}=1024$, identique à `Qwen3.5-0.8B` — embedding/tête ≈ 254,3M, directement comparable.

**2. Cœur récurrent (`HierarchicalMemory.{q,k,v}_proj` + `sm_q_proj` + `sm_write_proj` + `fuse_proj`/`fuse_in`+`fuse_out`)** — **calculé, pas un hyperparamètre libre**, une fois $d_{\text{model}}$ fixé. À $d_{\text{model}}=1024$, **sans FF** (défaut §-1) :
$$\underbrace{1{,}048{,}576}_{q\_proj} + \underbrace{1{,}048{,}576}_{sm\_q\_proj} + \underbrace{2{,}097{,}152}_{sm\_write\_proj} + \underbrace{3{,}145{,}728}_{fuse\_proj} + \underbrace{2{,}097{,}152}_{k\_proj+v\_proj \text{ (one-time/KB)}} \approx \mathbf{9{,}44M}$$
Comparé à ~23-27M pour **un seul layer** de `Qwen3.5-0.8B` (même $d_{\text{model}}$) — le cœur sans-FF est déjà **~2,5-3× plus petit qu'une seule couche Qwen**, avant même d'appliquer le raisonnement "cœur ≈ 1 layer, N_step fait la profondeur" de la discussion précédente. **Avec `use_ff=True`** (`ff_hidden_mult=4`) : $\approx 23{,}06M$ — tombe presque exactement sur la taille d'un layer Qwen (cohérence rassurante : réintroduire le FF standard reconstruit à peu près le budget d'un layer transformer classique, confirme que le "manque" retiré par §-1 est bien de cet ordre de grandeur).

**3. `N_step`** — **seul axe encore activement déterminé empiriquement**, pas déductible à l'avance : le sweep en cours (`N_step ∈ {8,16,24,32}`, Phase 2/1quater) doit trancher entre "insuffisant" et "limite de mécanisme" avant de fixer une valeur de référence. L'heuristique $N_{\text{step}} \gtrsim 2\text{-}4\times L_{\text{cible}}$ ne sert que de fourchette de départ pour le sweep, pas de valeur finale.

**4. KB persistante (Phase 10, §8bis, pas encore codée) — [CORRIGÉ 2026-09-13, budget initial sous-estimé]** : l'utilisateur a relevé à juste titre que référencer **un seul** layer de FF (~13,6M) pour dimensionner la mémoire persistante était incohérent avec le raisonnement "cœur ≈ 1 layer, $N_{\text{step}}$ fait la profondeur" — la mémoire persistante doit remplacer la capacité de stockage de faits **cumulée sur toute la profondeur** d'un modèle standard comparable (chaque couche FF contribue potentiellement des associations différentes, c'est additif sur la profondeur, pas un budget à compter une seule fois), donc le budget de référence correct est le **FF agrégé sur les 24 couches** de `Qwen3.5-0.8B` (même $d_{\text{model}}=1024$) : $24 \times 11{,}01\text{M} \approx \mathbf{264\text{M}}$ — pas $13{,}6$M. Avec $\dim(V)=1024$, $\dim(K)=512$ (§5.4) : $n_{\text{slots}} \approx 264\text{M}/(512+1024) \approx \mathbf{172\,000}$ slots — un ordre de grandeur qui rejoint d'ailleurs directement celui de Product-Key Memory (Lample et al. 2019, $|K|=512^2 \approx 262\,000$ slots), une cohérence rassurante plutôt qu'une coïncidence : les deux problèmes (stocker une capacité factuelle comparable à celle d'un modèle de profondeur donnée) sont d'échelle similaire.

**Total trainable estimé, config de départ ($d_{\text{model}}=1024$, sans FF, KB persistante corrigée)** : $\approx 254{,}3\text{M (embed)} + 9{,}44\text{M (cœur)} + 264\text{M (KB persistante)} + \text{quelques M (streams)} \approx \mathbf{\sim 530M}$ — **beaucoup plus proche** des ~0,8-0,9B de `Qwen3.5-0.8B` qu'estimé initialement, pas "nettement sous". **Reformulation de l'argument d'efficacité** (l'estimation précédente le sur-vendait) : l'économie de paramètres se situe spécifiquement au niveau du **cœur récurrent** (9,44M vs ~264M de FF cumulés sur 24 couches, ~28× plus petit), pas sur la capacité de stockage factuel totale — stocker une quantité comparable de connaissance générale, que ce soit en FF ou en KV explicite, coûte un budget de paramètres du même ordre de grandeur quelque part dans le modèle ; ce n'est pas gratuit de le déplacer, l'argument du projet est que le déplacer **rend le cœur de raisonnement plus petit et plus facilement itérable** ($N_{\text{step}}$ à la place de la profondeur), pas que le modèle entier devient beaucoup plus petit. **Reste conditionné** aux résultats du sweep `N_step`/`use_ff` en cours et à la Phase 10 (pas encore implémentée) — ce chiffre est une cible espérée, pas une conception validée.

**Ce qui reste volontairement hors de cette synthèse** : `n_register` (largeur du registre — doit suivre le parallélisme de raisonnement nécessaire, pas un budget de taille, §Phase -1), `block_size`/`depth` de la hiérarchie par épisode (déterminés par la taille de KB réelle visée, pas un choix indépendant), `batch_size` (axe d'ingénierie GPU, découplé de l'architecture, cf. plan).

### 13.1 Initialisation de l'embedding/tête depuis le Teacher — éviter d'apprendre les ~254M de zéro **[NOTES, proposition 2026-09-13, pas encore implémentée]**

Question posée : peut-on éviter (ou limiter) le travail d'apprentissage des paramètres d'embedding/tête (la plus grosse partie du budget, §13) en initialisant depuis les poids existants du Teacher, plutôt que d'un init aléatoire ?

**Config réelle du Teacher vérifiée** (`Qwen/Qwen3.8-27B`, `config.json`) : `hidden_size=5120`, `vocab_size=248\,320` (léger écart avec les 248 077 mentionnés précédemment dans ce document — à réconcilier avec le tokenizer réellement téléchargé par `learn/distill/download_teacher.py`, pas avec un nouveau fetch), **`tie_word_embeddings=false`** — le Teacher a donc deux matrices **séparées** : l'embedding d'entrée (`vocab×5120`) et une tête de sortie `lm_head` indépendante (`5120×vocab`), pas une seule table partagée.

**Bonne nouvelle architecturale** : dans `Thinker` (contrairement au transformer dense de `train_sft.py`, qui utilise une tête liée par défaut, cf. `learn/distill/README.md` "muP and the tied-head compromise"), l'entrée (`self.embed`) et la sortie (`OutputStream.head` du stream `answer`, un `nn.Linear(d_model, vocab_size)` indépendant) **sont déjà deux matrices séparées** — aucune tension d'untying à gérer, chacune peut être initialisée indépendamment depuis la partie correspondante du Teacher (embedding d'entrée → `self.embed`, `lm_head` → `OutputStream['answer'].head`).

**Proposition concrète** :
1. Extraire les deux matrices du Teacher (déjà téléchargé, `download_teacher.py`) — poids CPU, opération offline, pas de GPU nécessaire pour cette étape.
2. `d_model$ (1024 dans l'exemple ci-dessus) $< 5120$ (hidden size du Teacher) — une projection est nécessaire, pas une copie directe. Deux options : (a) projection aléatoire fixe (rapide, pas d'optimalité particulière) ; (b) **SVD/PCA** sur la matrice d'embedding du Teacher, ne garder que les $d_{\text{model}}$ composantes principales — capture le plus de variance possible de l'espace d'origine avec le budget dimensionnel choisi, coût CPU raisonnable pour une matrice $248\,320\times 5120$.
3. **Limiter le travail d'apprentissage demandé, pas forcément l'annuler complètement** : geler `self.embed` (ou lui donner un LR très réduit) est plausible — c'est une table de correspondance token→représentation, un rôle largement indépendant de l'architecture qui la consomme ensuite. Geler `OutputStream['answer'].head` est plus risqué : la tête doit apprendre à prédire depuis l'état interne **de notre propre modèle** (différent de celui du Teacher), donc probablement la laisser entraînable mais initialisée depuis le Teacher plutôt que d'un `nn.Linear` aléatoire — un bon point de départ reste un point de départ, même si elle continue d'apprendre.
4. **Non résolu, à trancher empiriquement** : geler vs. LR réduit vs. entraînable dès le départ pour `self.embed` — proposé comme une variante de plus à tester (dans l'esprit des flags Phase 1bis déjà en place), pas une décision à prendre a priori.

## 14. Intégration d'`Thinker` sur texte réel — vers un objectif LM par position **[PARTIELLEMENT IMPLÉMENTÉ 2026-09-13 — voir statut par sous-section]**

### 14.0 Le problème exact à résoudre

`Thinker.forward` (`core/indexed_thinker_model.py:136-194`) a une interface **une-KB-in / une-réponse-out** : un seul `query_tokens`, un seul vecteur de sortie par stream (`OutputStream.query_seed`, `(1, d_model)`, `core/indexed_thinker_model.py:80,86`). C'est adapté à `kb_chain_retrieval.py` (une question, une réponse) mais **pas** à un entraînement LM classique sur texte réel, qui demande une perte cross-entropy **à chaque position** d'un document potentiellement long — ce que ni la Phase 3 du plan (déjà existante : texte/vocabulaire réels mais toujours sous forme synthétique "une requête → une réponse", cf. `wiki_samples.json`) ni aucune phase actuelle ne couvrent. Ce qui suit est la conception de cette extension, pas encore codée, distincte de la Phase 3.

### 14.1 Fenêtrage glissant du document **[IMPLÉMENTÉ]**

`data/real_text_windows.py::RealTextWindowDataset` (17 tests CPU, `tests/test_real_text_windows.py`), voir §14.5 pour le détail.

Un document tokenisé est découpé en fenêtres consécutives, chacune composée de :
- un **contexte** de $N_{\text{ctx}} = \text{block\_size}^{\text{depth}}$ leaves (contrainte déjà imposée par `HierarchicalMemory.build`, `core/indexed_memory.py:156-160`), lui-même scindé en deux sous-populations distinguées par `source_ids` (§6.2, déjà l'existant) :
  - **input** (`source_id=0`) : les $T_{\text{local}}$ tokens immédiatement avant la cible (contexte récent, local) ;
  - **KB** (`source_id=1`) : les tokens plus anciens du **même document**, remplissant le reste des $N_{\text{ctx}} - T_{\text{local}}$ slots — donne un vrai test de rappel long-terme sur données réelles (pas une KB de faits fabriqués), sans attendre un mécanisme de retrieval externe (§14.7). En début de document, `leaf_mask` (déjà supporté, `core/indexed_memory.py:139-151`) pad les slots manquants.
- une **cible** de $T_{\text{tgt}}$ tokens, sur laquelle la perte est calculée.

**Stride proposé** : $\text{stride} = T_{\text{tgt}}$ (cibles non chevauchantes) — chaque token du document est prédit exactement une fois par epoch, convention standard d'entraînement LM par blocs.

### 14.2 Récurrence du registre $R$ entre fenêtres, SM remise à zéro **[MÉCANISME CÔTÉ MODÈLE IMPLÉMENTÉ, boucle d'entraînement pas encore écrite]**

`Thinker.forward(..., register_init_override=...)` (`core/indexed_thinker_model.py`) accepte désormais le $R$ reporté à la place de `self.register_init` — testé (`tests/test_indexed_memory.py::TestRealTextIntegrationWiring`, y compris une simulation bout-en-bout à deux fenêtres). Le **stop-gradient** reste, comme prévu ci-dessous, la responsabilité de l'appelant (`Thinker` accepte n'importe quel tenseur, détaché ou non) — aucune boucle d'entraînement réelle ne l'applique encore (pas de script pour ça, cf. §14.5/Phase 11).

Proposition : $R$ (registre final de la fenêtre $i$) devient la base du registre de la fenêtre $i+1$ :
$$R_{\text{init}}^{(i+1)} = \text{stopgrad}\big(R_{\text{final}}^{(i)}\big) + \bar q^{(i+1)}$$
où $\bar q^{(i+1)}$ est le même terme qu'aujourd'hui (moyenne des embeddings du contexte de la fenêtre, `core/indexed_thinker_model.py:160-161`), et `register_init` (paramètre appris) ne sert alors qu'à amorcer la toute première fenêtre d'un document.

- **Stop-gradient à la frontière de fenêtre par défaut** (à la Transformer-XL, segment-level recurrence) : borne le graphe de calcul/BPTT à une seule fenêtre au lieu de tout le document — flag d'ablation `detach_register_across_windows: bool = True`, à tester (le cas `False` est un simple TBPTT complet, plus coûteux mais potentiellement plus cohérent en gradient).
- **SM n'est PAS reportée entre fenêtres** — remise à zéro à chaque fenêtre, comportement déjà actuel (`sm_k`/`sm_v` initialisés vides en tête de `forward`, lignes 163-164). Cette séparation est volontaire et s'aligne avec la terminologie du projet : SM = trace de travail **vraiment court-terme** de cette fenêtre de raisonnement, $R$ = résumé compressé **long-terme** qui traverse tout le document, KB = détail brut local/plus-ancien. Aucun des trois ne fait double emploi avec un autre.

### 14.3 Généralisation de l'`OutputStream` à une sortie multi-position **[IMPLÉMENTÉ]**

Nécessaire pour obtenir une perte par position (le point bloquant identifié en 14.0). Remplace la requête unique apprise `query_seed` par une requête **par position de la cible**, construite par teacher forcing :
$$q_t = \text{embed}(\text{target\_token}_{t-1}) + \text{pos\_embed}(t), \quad t = 1 \ldots T_{\text{tgt}}$$
chaque $q_t$ attend **indépendamment** (pas de self-attention causale entre positions ajoutée) sur `sm_k`/`sm_v` accumulées pendant les `n_step` itérations de la fenêtre courante — exactement le mécanisme `OutputStreamLayer.forward` existant (`core/indexed_thinker_model.py:59-62`), simplement batché sur la dimension $T_{\text{tgt}}$ au lieu de $1$.

**Implémentation** : `OutputStream(sequence_mode=True, max_seq_len=...)` (`core/indexed_thinker_model.py`) ajoute `self.pos_embed` et prend `query_input` (les embeddings de token déjà calculés, $t-1$) en argument de `forward()` plutôt que de recalculer l'embedding lui-même — c'est `Thinker.forward` qui embed `target_input` (nom de code retenu pour $\text{target\_token}_{t-1}$, cohérent avec le champ `target_input` de `data/real_text_windows.py`, §14.5) une seule fois et le redistribue à tous les streams `sequence_mode=True`, plutôt que chaque stream ne réembed indépendamment. 7 tests dédiés (`TestSequenceModeOutputStream`) : forme, indépendance stricte entre positions (perturber `query_input` à une position ne doit rien changer aux autres — vérifié explicitement, confirme l'absence de self-attention), `pos_embed` différencie bien des embeddings de token identiques, gradient atteint `pos_embed`/`query_input`.

- Reste "léger, sans FF" au sens de §11bis : on ajoute une dimension de batch à la requête, pas de nouvelle couche/poids de type FF.
- **Entraînement** : CE standard sur les $T_{\text{tgt}}$ positions en parallèle (teacher forcing avec les vrais tokens cible décalés — identique en substance à un décodeur Transformer standard).
- **Inférence/génération** : nécessairement autorégressive, un token à la fois (puisque $q_t$ dépend de $\text{target\_token}_{t-1}$) — aucun écart par rapport à la pratique standard d'un LM causal.
- **Lecture directe de la KB en plus de SM** (`read_kb_directly`, optionnelle) : pas retenue par défaut pour le MVP — `o_kb` est déjà intégrée dans SM à chaque itération du cœur (§2), donc probablement redondante ; à garder comme variante d'ablation si le stream peine à récupérer un détail fin noyé dans la trace SM compressée.

### 14.4 Objectif de perte

Réutilise tel quel l'infrastructure de `learn/distill/train_sft.py` : CE standard (`F.cross_entropy` sur `(B, T_{\text{tgt}}, \text{vocab})` vs labels) comme premier mode, KD Top-K (`topk_kd_loss`, déjà validé EXP-003 à EXP-006) branchable à l'identique une fois des cibles Teacher précalculées **par position de fenêtre** (même pipeline `precompute_teacher_targets.py`, tranché par fenêtre plutôt que par exemple) — aucun changement de format de perte nécessaire, seul l'alignement fenêtre/position doit être géré côté chargeur de données.

### 14.5 Nouveau chargeur de données **[IMPLÉMENTÉ]**

`data/real_text_windows.py::RealTextWindowDataset` (distinct de `data/kb_chain_retrieval.py`, la tâche synthétique multi-sauts) : tokenise un document entier une fois, puis produit des fenêtres glissantes de forme fixe — un `dict` par fenêtre : `kb_tokens`, `kb_source_ids`, `kb_leaf_mask` (taille $N_{\text{ctx}} = \text{block\_size}^{\text{depth}}$), `target_input`/`labels` (taille $T_{\text{tgt}}$, décalage teacher-forcing déjà appliqué, §14.3), plus `doc_id`/`is_first_window` pour qu'une boucle d'entraînement détecte les frontières de document (§14.2). $T_{\text{local}}$/$T_{\text{tgt}}$/stride configurables ; une fenêtre à contexte réel nul est exclue par défaut (`min_real_context=1`). Source : `train.jsonl`/`val.jsonl` déjà produits par `learn/distill/prepare_*_data.py` (champ `text`), aucun nouveau format de données créé. 17 tests CPU (`tests/test_real_text_windows.py`), tous verts, y compris l'intégration avec `torch.utils.data.DataLoader` (le collate par défaut suffit — toutes les fenêtres ont la même forme, contrairement à `train_sft.py::JsonlTextDataset`).

### 14.6 Changements d'interface pour `Thinker.forward` **[IMPLÉMENTÉ]**

- Nouveaux arguments optionnels (noms de code alignés sur `data/real_text_windows.py`, pas exactement ceux esquissés initialement ci-dessus) : `target_input: (B, T_tgt) = None` (teacher forcing, active le mode multi-position de §14.3 — requis dès qu'au moins un stream a `sequence_mode=True`, vérifié par une assertion) et `register_init_override: (B, n_register, d_model) = None` (injecte le $R$ reporté de la fenêtre précédente, §14.2, remplaçant `self.register_init.unsqueeze(0).expand(...)` comme base avant l'ajout de $\bar q$).
- `R` final est déjà retourné (`return R, stream_outputs`) — rien à changer côté retour pour permettre le report d'une fenêtre à l'autre ; l'appelant (boucle d'entraînement, **pas encore écrite**) a la responsabilité de le transmettre et d'appliquer le stop-gradient de §14.2 (`Thinker` ne détache rien lui-même).
- Testé (`tests/test_indexed_memory.py::TestRealTextIntegrationWiring`) : le chemin par défaut (les deux arguments à `None`) reproduit exactement le comportement précédent (non-régression `kb_retrieval`/`kb_chain_retrieval`).

### 14.7 Explicitement hors de ce document de conception

- **Vraie récupération externe** : ici, la "KB" d'une fenêtre = tokens plus anciens du **même document**, pas un corpus externe interrogé par similarité. Un futur mécanisme de retrieval (BM25/embedding search) alimenterait la population `source_id=1` de façon plus riche, sans changer l'interface ci-dessus — seul le choix des leaves KB en amont (côté chargeur de données) changerait.
- **KB persistante apprise** (Phase 10, §8bis) : orthogonal à ce document, s'intégrerait comme un niveau supplémentaire dans `HierarchicalMemory.attend()` (ou un terme additionné à `o_kb`), indépendant du fenêtrage texte réel décrit ici.
- **Coût GPU réel du fenêtrage** (un `build()` par fenêtre, potentiellement coûteux si $N_{\text{ctx}}$ est grand, cf. §5.2bis) : non mesuré, nécessitera un test Grid5000 dédié une fois codé.

### 14.8 Résumé des décisions et statut d'implémentation

| Axe | Choix | Statut |
|---|---|---|
| Fenêtrage | glissant, cibles non chevauchantes, stride = $T_{\text{tgt}}$ | ✅ `data/real_text_windows.py` (§14.5) |
| Contexte unifié | récence locale = input (`source_id=0`), plus ancien du même document = KB (`source_id=1`) | ✅ idem |
| Registre $R$ entre fenêtres | reporté via `register_init_override`, stop-gradient à la frontière **laissé à l'appelant** | ✅ mécanisme modèle (§14.2) — ⬜ pas encore appliqué par une vraie boucle d'entraînement |
| SM entre fenêtres | non reportée — remise à zéro à chaque fenêtre (comportement déjà actuel) | ✅ inchangé, pas de code à ajouter |
| Sortie | généralisation multi-position d'`OutputStream` (`sequence_mode`), cross-attention indépendante par position, pas de self-attention causale ajoutée | ✅ (§14.3) |
| Perte | CE standard + KD optionnel, réutilise `topk_kd_loss` tel quel | ⬜ pas encore branché — nécessite l'alignement Teacher-par-fenêtre (§14.4) et une boucle d'entraînement |

**Ce qu'il reste pour un run réel bout-en-bout** (voir plan Phase 11) : une boucle d'entraînement qui (a) itère les fenêtres d'un même document dans l'ordre, (b) applique le stop-gradient de §14.2 avant de passer `R` en `register_init_override` à la fenêtre suivante, (c) aligne/précalcule les cibles Teacher par fenêtre (§14.4) plutôt que par exemple entier, (d) fixe des valeurs concrètes de $N_{\text{ctx}}$/$T_{\text{local}}$/$T_{\text{tgt}}$/`block_size`/`depth` — aucune de ces quatre n'est encore validée empiriquement pour un run à l'échelle.

Relation avec le plan : voir Phase 11 (`dev_notes/indexed_attention_experiment_plan.md`), qui référence cette section pour le protocole de test.
