# Synthèse — "Branch • Indexed Attention" (conversation Gemini)

Source : `raw/Branch-•-Indexed-Attention.md` (export Gemini, 1861 lignes). Ce document distingue explicitement ce qui vient des tours **« you asked »** (idées/décisions réelles de l'utilisateur, faisant foi) de ce qui vient des tours **« gemini response »** (propositions de l'IA, à considérer comme du matériau brut à valider, pas des décisions prises).

## 1. Comment la conversation a démarré (contexte, pas l'idée elle-même)

La conversation débute par une exploration pure de mécanismes d'attention creuse existants (2026) : **QSA** (Qwen Sparse Attention, micro-blocs de 4 tokens + indexeur gelé), **DSA** (DeepSeek Sparse Attention, indexeur entraîné par distillation KL contre une attention dense), **NSA** (DeepSeek Native Sparse Attention, 3 branches — locale/compressée/sélectionnée — fusionnées par gating appris). L'utilisateur pousse Gemini à clarifier, pour chacune : la circulation du gradient à travers le Top-K (discret, bloquant), le rôle du pooling/moyenne dans l'indexeur, le risque de *drift* de l'indexeur, le rôle de la branche compressée de NSA au-delà de la simple circulation du gradient. **Ceci est du matériau de compréhension théorique, pas encore l'idée du projet** — mais tous les concepts réapparaissent ensuite comme briques de l'idée propre à l'utilisateur.

**Références formelles retrouvées (2026-09-13)** — ces trois mécanismes n'étaient cités que par acronyme dans les notes de conversation d'origine ; voir `dev_notes/ideas/indexed_attention_comparison.md` (en tête de fichier) pour les citations complètes (NSA : Yuan et al. 2025, arXiv:2502.11089 ; DSA : DeepSeek-AI 2025, arXiv:2512.02556 ; QSA : Qiu et al. 2026, arXiv:2608.30320) — aussi ajoutées dans `thesis/paper/references.typ` [34]-[36].

Le tournant a lieu ligne 488 : l'utilisateur formule sa propre problématique (un très large KV cache statique préconstruit, de plusieurs millions à milliards de tokens, représentant une base de connaissances, avec une compression multi-étage à poids partagés et un besoin d'indexation pour éviter de charger tous les K en mémoire).

## 2. L'idée centrale de l'utilisateur (à partir des notes manuscrites transcrites, lignes 897-969)

L'utilisateur a fourni la transcription de notes manuscrites qui posent l'architecture cible. Elle combine trois composants :

1. **Un registre latent récurrent `L_n`** (`L_n = [v_1, ..., v_n]`), qui joue le rôle de "registre" — traité en **Looped Transformer** : la même couche est appelée itérativement avec des poids partagés (pas de profondeur fixe, exactement l'esprit de l'architecture Thinker actuelle).
2. **Une mémoire à court terme (SM)** : `(K^s, V^s)`, mise à jour de façon incrémentale à chaque itération — nouveaux K/V empilés puis ajoutés (`APPEND`) — conceptuellement proche du "cache mémoire" déjà présent dans `architecture.typ` actuel.
3. **Une base de connaissances / mémoire long terme (KB/LM)** : `(K_q^l, V_m^l)`, **massive, statique, préremplie**, indexée hiérarchiquement, avec la mention explicite dans les notes : *« indexed mem like deepseek sparse attention (DSA) »*.

Le flux : `L_n` génère une query externe `Q_e`, interroge KB/LM (indexée), récupère des valeurs, les concatène/projette avec `L_n`, interroge ensuite SM, produit de nouveaux K/V ajoutés à SM, puis une skip connection additive redonne `L_new = L_n + ...`. Une **« asynchronous skip connection »** est notée dans le schéma comme permettant un *retrieval différé* (delayed retrieval) avec plusieurs flux détachés en parallèle — mécanisme mentionné mais **jamais creusé dans le reste de la conversation** (point non résolu, voir §5).

Points déjà tranchés par l'utilisateur dans les notes elles-mêmes :
- Un **stop-gradient sur les clés `K`** entrant dans un buffer/index — mais uniquement sur les clés, faute d'avoir encore trouvé comment indexer/stop-gradienter tout le reste (« Stop gradient était juste sur les clé car j'ai pas encore trouvé comment tous les indexer »). **Ce choix est ensuite abandonné plus tard dans la conversation** (voir §4, point 6).
- Le statut de l'input (SM ? LM ? un flux spécial ?) est explicitement **non décidé** par l'utilisateur.
- Il précise que ceci est un **draft**, composant d'un projet plus large.

## 3. Enrichissement du concept (indexation hiérarchique multi-échelle) — proposition initialement de l'utilisateur

Après la transcription des notes, l'utilisateur pousse plus loin (ligne 672, en son nom propre, pas une suggestion de Gemini) une extension au mécanisme KB : un **arbre d'indexation multi-niveaux** où chaque nœud parent résume un bloc de nœuds enfants (comme NSA, mais appliqué récursivement sur plusieurs étages), avec quatre propositions personnelles :
- **Softmax unifié** : concaténer dans un même softmax les clés brutes (feuilles) et les clés compressées de tous les niveaux supérieurs, plutôt que les fusionner via des branches à gating séparé (façon NSA classique).
- **Dropout stochastique des niveaux supérieurs** pendant l'entraînement (plus un niveau est haut, plus il est masqué aléatoirement souvent) — pour forcer la robustesse multi-échelle.
- **L'indexeur peut cibler directement des nœuds internes** (pas seulement les feuilles) comme résultat final d'attention — permettant de s'arrêter à un résumé si suffisant.
- Doute explicite de l'utilisateur : les concepts de haut niveau sont-ils trop génériques pour être utiles ? (Gemini répond que oui mais qu'ils servent doublement de routage et de contexte global — **réponse de Gemini, pas validée explicitement par l'utilisateur**.)

Un point de vigilance mathématique soulevé par Gemini (à vérifier empiriquement) : un softmax unifié entre vecteurs compressés (normes/variances différentes) et vecteurs bruts peut biaiser systématiquement vers les feuilles ; solution proposée = RMSNorm par niveau avant le produit scalaire.

L'utilisateur confirme vouloir passer en mode **Perceiver / cross-attention** pour agréger les niveaux (plutôt que pooling non-paramétrique) — dans ce cadre, la question de position (RoPE sur un intervalle de tokens, ambiguë pour un nœud parent) devient secondaire puisque l'accès est par contenu (« Q veut utiliser ce système... la position ne pose pas trop problème »).

## 4. Variantes explicitement identifiées à tester

L'utilisateur a lui-même demandé un inventaire exhaustif (ligne 1246) ; Gemini l'a produit (lignes ~1252-1335), organisé par sous-système. En triant ce qui constitue de vraies variantes **à tester** (vs. simple vocabulaire ou explication) :

1. **Granularité d'agrégation `M=1` vs `M>1` (slots multiples)** : idée initiale de l'utilisateur = `M=1` (un vecteur unique par nœud parent, façon Perceiver classique). Les slots multiples (`M>1`, façon Slot Attention, un vecteur par "aspect sémantique" du bloc) sont une variante suggérée par Gemini que l'utilisateur a explicitement acceptée de tester en parallèle : *« Plus de stop gradient, on pourra implémenter les variantes M>1 et M=1 et tester »* (ligne 1057). **C'est une variante confirmée par l'utilisateur, pas seulement une suggestion de Gemini.**
2. **Fonction d'agrégation `L → L+1`** : pooling non-paramétrique (moyenne/max + RMSNorm), projection linéaire/Conv1D à taille de bloc fixe, cross-attention Perceiver (query apprise), compresseur récurrent à poids partagés (type DeltaNet/GRU). L'utilisateur penche vers Perceiver mais **aucun choix final tranché** dans le texte.
3. **Softmax unifié multi-niveaux vs fusion par branches à gating (NSA classique)** — proposition propre de l'utilisateur, à opposer à l'approche NSA d'origine.
4. **Stochastic level dropping** à l'entraînement (masquage aléatoire des niveaux hauts) — idée propre de l'utilisateur.
5. **Choix de résolution implicite** (le softmax normalisé arbitre naturellement le niveau) **vs explicite** (un routeur dédié prédit la profondeur à atteindre) — l'utilisateur soulève la question lui-même (ligne 1183) en semblant favoriser l'implicite, sans trancher formellement.
6. **Gradient sur les clés : stop-gradient (choix initial des notes manuscrites) vs entraînement de bout en bout sans stop-gradient** — **changement de position explicite de l'utilisateur** : après avoir posé le stop-gradient dans ses notes initiales, il écrit ensuite « Plus de stop gradient, on pourra implémenter les variantes M>1 et M=1 et tester » (ligne 1057), ce qui semble indiquer qu'il abandonne l'idée de stop-gradient au profit de tests directs des variantes M=1/M>1. **Ambigu** : il n'est pas certain si "plus de stop gradient" signifie "on ajoute encore plus de stop-gradient" ou "on n'en met plus" — à clarifier avec l'utilisateur (voir §5).
7. **Requêtes découplées `Q_KB` vs `Q_SM`** (projections séparées pour interroger KB et SM séparément) : **suggestion de Gemini**, jamais confirmée explicitement par l'utilisateur dans le texte lu.
8. **Mécanisme de largeur adaptative / No-Op** : plusieurs variantes distinctes proposées par Gemini — seuil dur `θ` prédit par `L_n` (coupure Top-K implicite par score), porte de bypass sigmoïde explicite, token sentinelle/null key absorbant l'attention, Sparsemax/α-entmax, température adaptative. L'utilisateur a validé la direction générale (seuil `θ` + No-Op unifié + relaxation continue `w_i = σ((s_i-θ)/τ)` + pénalité de parcimonie `λ Σw_i`) en demandant à l'approfondir mathématiquement (ligne 1183-1187), mais **sans choisir entre les variantes de contrôle** (bypass vs sentinel vs sparsemax vs température) — et lui-même signale que **le No-Op/l'efficience peut être ignoré pour une première baseline** (ligne 1666).
9. **Statut de l'input** : cross-attention dédiée (façon Perceiver-IO, suggestion de Gemini, jamais confirmée par l'utilisateur) vs traité comme SM classique — explicitement non tranché par l'utilisateur dans les notes ET dans la suite de la conversation.
10. **Stratégie pour la construction de la KB pendant l'entraînement** (problème de l'œuf et de la poule : la KB long-terme est censée être fixe/statique, mais elle n'existe pas encore pendant l'entraînement) — l'utilisateur propose lui-même 4 stratégies (ligne 1577) : espace unifié input/KB avec biais de priorité, génération d'une "Sub-KB" à la volée par forward pass avec distracteurs négatifs, laisser le modèle découvrir seul sans dataset dédié, approche hybride/curriculum. **Aucune n'est tranchée** — c'est un vrai nœud ouvert.
11. **Curriculum de largeur d'attention "large puis étroit"** (idée propre de l'utilisateur, ligne 1824, explicitement notée comme à garder) : au début du traitement, la sélection devrait être large pour bien explorer les inputs et la mémoire, puis se resserrer progressivement au fil des itérations — avec la possibilité de laisser le modèle apprendre ce planning lui-même plus tard, ou via post-training. Idée conceptuelle, **non implémentée, non formalisée mathématiquement**.

## 5. Divergences, hésitations et changements d'avis relevés

- **Stop-gradient** : présent dans les notes manuscrites initiales (spécifiquement sur les clés K), mais une phrase ultérieure de l'utilisateur (« Plus de stop gradient... ») suggère un changement de cap — **formulation ambiguë, à faire clarifier par l'utilisateur** plutôt que d'assumer une lecture.
- **No-Op / largeur adaptative** : traité en détail mathématique sur demande de l'utilisateur, puis l'utilisateur lui-même le déclare **hors scope pour une première baseline** (« le no op (efficiency) peux être ignoré dans un début », ligne 1666) — donc une caractéristique du concept complet mais volontairement reportée pour la mise en œuvre initiale.
- **Complexité de la baseline MVP proposée par Gemini** (Top-k fixe, sans seuil, KB injectée directement en contexte plutôt que préremplie/hiérarchique, distillation "black-box" pure) : l'utilisateur a explicitement remis en question si cette proposition respectait bien ses objectifs de LLM généraliste énoncés plus haut (« Tu as vraiment tenu compte de mes objectifs... », ligne 1718) — **signal que le MVP proposé par Gemini est jugé possiblement trop réducteur/mal aligné par l'utilisateur**, sans qu'une résolution finale soit donnée dans le texte lu.
- **Slots multiples (`M>1`)** : présenté par Gemini comme variante à part entière ; l'utilisateur clarifie que son idée initiale était `M=1` mais accepte ensuite de tester les deux — pas un rejet, une extension.
- **Distillation comme stratégie d'accélération** : l'utilisateur introduit lui-même l'idée de distiller un LLM existant plutôt que d'entraîner from-scratch (ligne 1346), ce qui est manifestement la genèse du chantier de distillation actuellement actif dans le repo (`Distill-getting-start.md`, `Distill-reasonning-stream.md`, expériences EXP-003 à EXP-006 dans `dev_notes/experiment.log.md`) — **confirmation que le chantier distillation actuel est un sous-produit direct de cette conversation architecture**, pas un sujet complètement séparé. La conversation se termine (dernière partie lue) sur la décision de l'utilisateur de d'abord se familiariser avec la distillation générique/reasoning sur des architectures standards **avant** de revenir tester sa propre architecture — ce qui explique l'état actuel du projet (distillation en cours, nouvelle architecture en attente).

## 6. Comparaison explicite avec l'architecture actuelle (`thesis/paper/architecture.typ`, `core/thinker_model.py`, `core/toy_model.py`)

**Ce qui est conservé :**
- Le registre latent récurrent `Z_t` / `L_n`, mis à jour itérativement par un bloc à poids partagés (Looped Transformer) — c'est exactement le même principe dans les deux versions.
- Le principe de cross-attention comme mécanisme central d'accès à la mémoire/l'input (façon Perceiver).
- Un cache mémoire qui grandit en cours d'inférence par append (le "memory cache" actuel ≈ la "Short-Term Memory (SM)" du nouveau concept).

**Ce qui change fondamentalement :**
- **Ajout d'une KB/LM externe massive et statique** (millions à milliards de tokens), absente de l'architecture actuelle. Le "cache d'entrée" actuel (`K_in, V_in`) n'est qu'un encodage de la séquence d'entrée courante — pas une base de connaissance externe persistante à travers les inférences.
- **Indexation hiérarchique en arbre** sur cette KB (compression multi-étage à poids partagés, plusieurs niveaux de résumé) — l'architecture actuelle n'a aucune structure hiérarchique ni indexation ; c'est une simple concaténation plate `[K_in, K_mem]`.
- **Softmax unifié multi-échelle avec normalisation par niveau** pour arbitrer nativement entre "résumé" et "détail fin" — mécanisme entièrement nouveau, absent de l'architecture actuelle.
- **Largeur d'attention adaptative + No-Op** (seuil dynamique, possibilité de sauter complètement l'accès mémoire à une itération donnée) — l'architecture actuelle attend toujours sur les mêmes caches, sans mécanisme de coupure/sparsité dynamique.
- **Changement de philosophie produit** : le nouveau concept vise explicitement un modèle à **très peu de mémoire paramétrique**, qui délègue les faits à une mémoire externe et concentre ses poids sur le raisonnement/la récupération — un positionnement plus radical que l'architecture actuelle, qui reste un encodeur-processeur d'une séquence d'entrée donnée sans notion de base de connaissance externe persistante.

## 7. Questions ouvertes à trancher avec l'utilisateur

1. La phrase « Plus de stop gradient » (ligne 1057) signifie-t-elle l'abandon du stop-gradient sur les clés, ou son renforcement/généralisation ? Impact direct sur l'implémentation.
2. Aucune décision finale sur le statut de l'input (SM, LM, ou flux dédié via cross-attention) — nécessaire avant toute implémentation.
3. Aucune décision sur la fonction d'agrégation `L → L+1` (Perceiver vs pooling vs récurrent) — a un impact fort sur le coût d'entraînement et la stabilité du gradient.
4. Le nœud le plus critique et non résolu : **comment entraîner la KB long-terme alors qu'elle est censée être statique/préremplie mais n'existe pas encore pendant l'entraînement** — 4 stratégies proposées par l'utilisateur, aucune choisie.
5. La baseline MVP proposée par Gemini (Top-k fixe, sans seuil, sans hiérarchie, KB en contexte) a été explicitement questionnée par l'utilisateur comme possiblement non fidèle à l'objectif "LLM généraliste" — reste à clarifier ce qui constitue un MVP acceptable qui teste vraiment l'idée centrale sans être aussi complexe que la vision complète.
6. Le mécanisme d'« asynchronous skip connection / delayed retrieval » mentionné dans les notes manuscrites (multiple detached streams) n'a jamais été creusé dans la conversation — signification et utilité à clarifier directement avec l'utilisateur.
