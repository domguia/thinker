# §8quater — Clustering token-level (sans pooling) de la densité documentaire

## 2026-09-20 — Révision `precompute_token_clusters.py` : remplace le pooling par document

`precompute_doc_clusters.py` (approche précédente) réduisait chaque document à UN SEUL vecteur (mean pooling via `sentence-transformers/all-MiniLM-L6-v2`, `max_seq_length=256`) avant clustering -- aplatit l'hétérogénéité interne d'un document et tronque tout au-delà de 256 tokens. Nouveau script (`learn/distill/precompute_token_clusters.py`, commit `f753f5c`) : chaque TOKEN de chaque document garde son propre embedding (`BAAI/bge-large-en-v1.5`, AutoModel brut sans pooling, `last_hidden_state`), le clustering (GMM récursif + HDBSCAN récursif, mêmes méthodes que `precompute_doc_clusters.py`) tourne directement sur le nuage de points token-level à travers tous les documents ET tous les datasets (retrieval/reasoning/general) à la fois, avec `token_doc_id` traçant explicitement l'appartenance. Dispersion intra-document (distance cosinus moyenne entre les propres tokens du document, jamais un vecteur poolé) calculée en lecture seule après coup.

Testé localement par `long-term-memory-builder` avant la demande (CPU, 8 docs/dataset synthétiques + extrait HotpotQA réel) -- tourne sans erreur, sorties cohérentes, mais pas testé à l'échelle réelle ni avec le modèle d'embedding cible.

## 2026-09-20 — Premier passage réel (200 docs/dataset) : HDBSCAN bloque, GMM fonctionne

Réservation dédiée H100 (job `4122747`, `abacus27-1`) réutilisée après le test §8ter scale-up. Dépendance manquante trouvée et corrigée en route : `precompute_doc_clusters.py` (dont `precompute_token_clusters.py` importe `reduce_dim`/`gmm_recursive_depth`/`hdbscan_recursive_depth`) n'avait jamais été synchronisé vers Rennes (fichier untracked local) -- copié avant le premier lancement réussi. `hdbscan` manquant dans `teacher311`, installé (`0.8.44`).

**200 docs/dataset (retrieval:1984 instances car HotpotQA compte chaque document du contexte séparément, reasoning:200, general:200 -- 302 112 tokens au total)** :
- Extraction embeddings GPU : **26.9s** pour 302K tokens (H100).
- Dispersion intra-doc : 1.3s.
- PCA (32 dims) : 3.6s.
- GMM récursif (par token) : **12.3s** -- fonctionne bien à cette échelle.
- HDBSCAN récursif (par token) : **bloqué >27min CPU sans terminer ni logger de progression intermédiaire** -- coût combinatoire de la récursion HDBSCAN qui explose avec N à ce volume de points. Tué sur demande de `long-term-memory-builder` plutôt que de laisser tourner à l'aveugle.

## 2026-09-20 — Relance réduite (50 docs/dataset, hdbscan_min_cluster_size=40) : succès rapide

Sur suggestion de `long-term-memory-builder` : `--max_docs_per_dataset 50` (au lieu de 200) ET `--hdbscan_min_cluster_size 40` (au lieu de 10 par défaut) combinés -- les deux réductions ensemble, pas une seule. **Terminé en 16.8s au total**, HDBSCAN lui-même en 6.3s (contre >27min bloqué avant). Confirme que le coût explosait bien avec le nombre de points/la granularité des clusters, pas un bug.

**Résultat (50 docs/dataset)** :

| dataset | n_docs | dispersion intra-doc moyenne | profondeur GMM moyenne | n clusters-feuille distincts moyen |
|---|---|---|---|---|
| general | 50 | 0.1395 | 5.00 | 1.02 |
| reasoning | 50 | 0.1260 | 5.00 | 1.00 |
| retrieval | 50 | 0.1475 | 5.00 | 1.00 |

**Lecture prudente** : `mean_depth_gmm=5.00` partout (= `max_depth` par défaut) et `mean_n_distinct_leaf≈1.00` -- à cette échelle réduite (50 docs), la récursion GMM atteint systématiquement la profondeur maximale sans discriminer entre datasets, peu informatif tel quel (pourrait indiquer soit une vraie richesse structurelle uniforme, soit simplement que `max_depth=5` est trop bas ou `min_node_size`/`bic_margin` mal calibrés pour ce volume de points -- pas encore distingué). Dispersion intra-document légèrement plus élevée pour `retrieval` (0.1475) que `reasoning` (0.1260)/`general` (0.1395), mais sur seulement 50 docs/dataset -- pas assez pour conclure. Sortie sauvegardée : `data/distill/token_clusters_smoke/clusters_50.npz`. Relayé à `long-term-memory-builder` -- décision sur la suite (plus de docs à `min_cluster_size` élevé fixe, ou paramètres GMM à ajuster) laissée à leur discrétion.
