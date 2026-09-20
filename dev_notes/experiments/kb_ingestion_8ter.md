# §8ter — KB ingérée dynamiquement par pass récurrent du modèle

## 2026-09-20 — Premier smoke test réel (MVP intra-batch): tourne sans crash, coût ~7x confirmé

Code de `ff2attn` (non committé au moment de la demande, relu en détail avant lancement -- diff additif confirmé, aucune régression sur les chemins existants sans `--ingest_kb`/`use_ingest_token`) : `Thinker._step`/`Thinker.ingest` (`core/indexed_thinker_model.py`), `HierarchicalMemory.clear`/`add_static_level` (`core/indexed_memory.py`), `ingest_documents()` + `--ingest_kb`/`--ingest_n_step` (`train_prompt_response.py`, retrieval uniquement). Committé localement (`bf8459a`) avant lancement.

**Test lancé** : `paradoxe-2` (CPU, ~50 cœurs libres au moment du lancement, routeur S3 16-couches occupant l'autre moitié), `d_model=256, n_step=4, batch_size=8, lr=3e-4`, `hotpotqa` réel (18000 train/2000 val), `max_time_minutes=12`, seed=0 identique aux deux runs :
- **Baseline §8** (projection directe k_proj/v_proj) : step 1 en 6.7s (0.148 pas/s), loss initiale 11.49, val step 1 = 11.08.
- **`--ingest_kb --ingest_n_step 3`** (§8ter MVP) : step 1 en 46.6s (0.021 pas/s), loss initiale 11.68 (comparable à la baseline, cohérent).

**Read (premier signal, pas encore de vraie comparaison de convergence -- budget de 12 min trop court pour ça en CPU)** :
- **Aucun crash, aucun NaN** sur les deux runs -- le MVP tourne correctement sur données réelles (pas seulement les tenseurs synthétiques déjà testés).
- **Coût par pas ~7x plus élevé** avec `--ingest_kb` (46.6s vs 6.7s) -- cohérent avec l'attente de la spec (`n_docs_max=10` documents à ingérer, chacun `ingest_n_step=3` itérations, avant même la passe QA) -- **attendu, pas un bug**, chiffre concret demandé par `ff2attn` maintenant disponible.
- Loss initiale du premier pas comparable entre les deux runs (11.49 vs 11.68) -- pas de signe d'instabilité numérique dès le départ.

**Pas encore fait** : une vraie comparaison de convergence (nécessite un budget de pas beaucoup plus long, et probablement du GPU plutôt que CPU vu le coût x7 -- à relancer avec plus de temps/sur GPU une fois qu'une capacité adaptée se libère). Runs laissés tourner en arrière-plan pour glaner quelques pas de plus avant la fin du budget de 12 min.

## 2026-09-20 — Résultats des tests précédents + nouveaux tests (checkpoint, step_size)

**Tests précédents (baseline §8 vs --ingest_kb, budget 12 min épuisé)** :
- `--ingest_kb --ingest_n_step 3` : 16 pas, `final_loss=9.58`.
- baseline §8 : 168 pas, `final_loss=7.23`.

Baseline fait ~10x plus de pas dans le même budget mural et atteint une loss nettement plus basse -- cohérent avec le ratio de coût déjà mesuré (~7-10x), mais c'est une comparaison à budget mural égal, pas à nombre de pas égal (confond attendu, signalé par `ff2attn` comme la vraie question à trancher séparément).

**Relu le nouveau diff de `ff2attn` (checkpointing + n_step dérivé de la longueur), additif et propre, commité (`2105c57`), synchronisé.**

**`--ingest_kb --ingest_checkpoint` (batch_size doublé, 16 au lieu de 8)** : step 1 en 41.3s -- comparable au test original à batch_size=8 (46.6s) malgré le batch deux fois plus gros. Signal cohérent avec l'attente (mémoire libérée permet un plus gros batch sans ralentissement proportionnel), mais pas encore poussé jusqu'à trouver la limite OOM réelle sans le flag pour confirmer le gain quantitativement.

**`--ingest_kb --ingest_step_size 8 --ingest_n_step_max 3`** : step 1 en 18.7s -- nettement plus rapide que `--ingest_n_step 3` fixe (46.6s), cohérent avec l'attente (documents courts coûtent moins cher).

**Toujours pas de vraie comparaison de convergence longue** -- les deux nouveaux tests tournent avec le même budget court (10 min CPU) pour une première vérification "ça tourne, coût cohérent". La priorité annoncée par `ff2attn` (comparaison de convergence sur un nombre de pas comparable) reste à faire, idéalement sur GPU vu le coût x7-10.

## 2026-09-20 — Comparaison de convergence longue lancée sur GPU (abacus11-1, priorité utilisateur)

`--ingest_kb --ingest_n_step 3` vs baseline §8, même config que le premier smoke test (`d_model=256, n_step=4, batch_size=8, lr=3e-4, seed=0`), cette fois `--max_steps 3000 --max_time_minutes 480` sur GPU (abacus11-1, partagent GPU1) au lieu de CPU -- ~5 pas/s pour ingest_kb (contre 0.021 pas/s en CPU), donc un budget de pas long devient réellement atteignable. Les deux confirmés actifs, `--val_data` inclus dès le départ (contrairement au premier test).

**Résultat (2026-09-20, comparaison à pas égal, `max_steps=3000` fixé pour les deux -- pas de confond temps/pas)** :

| variante | pas | temps | final_loss (train) | val_answer |
|---|---|---|---|---|
| baseline §8 (k_proj/v_proj) | 3000 | 247s (12.06 pas/s) | 5.006 | **5.529** |
| `--ingest_kb --ingest_n_step 3` | 3000 | 481s (6.20 pas/s) | 5.013 | **5.641** |

**Lecture** : à nombre de pas identique, les deux variantes convergent à une loss train quasi identique (5.006 vs 5.013, écart négligeable). En val, `--ingest_kb` est légèrement moins bon (5.641 vs 5.529, +0.11), pas d'avantage de généralisation à ce budget. Coût par pas ~2x plus élevé (481s vs 247s) sur GPU pour ce résultat comparable-à-légèrement-pire. **Conclusion provisoire** : sur cette config (`d_model=256`, `hotpotqa`, 3000 pas), l'ingestion dynamique de la KB via le pass récurrent du modèle n'apporte pas de gain de généralisation mesurable par rapport à la projection statique k_proj/v_proj, à un coût ~2x supérieur. Ne pas généraliser au-delà de cette taille de modèle/config sans re-tester -- possible que l'avantage attendu de l'ingestion dynamique (mémoire associative vs simple projection) se manifeste à plus grande échelle ou sur des tâches nécessitant plus de raisonnement inter-documents que du lookup HotpotQA à faible profondeur. Relayé à `long-term-memory-builder` et `ff2attn`.

## 2026-09-20 — Test à plus grande échelle lancé (réservation dédiée H100, abacus27-1)

Suite à la demande de `long-term-memory-builder` (l'hypothèse que le mécanisme d'ingestion récurrent a besoin de plus de capacité pour rivaliser avec une projection directe) : nouvelle réservation besteffort dédiée `oarsub -n ingest8ter-scale -l gpu=1,walltime=6:00:00 -t besteffort -t idempotent -p "cluster='abacus21' OR cluster='abacus26' OR cluster='abacus27'"` (job `4122747`) plutôt que d'attendre la libération des nœuds déjà utilisés -- démarrage immédiat sur `abacus27-1` (H100 NVL 100GB, 0 MiB utilisé).

Lancé (baseline §8 vs `--ingest_kb --ingest_n_step 3`), config passée à l'échelle sur les deux axes demandés :
- `d_model=256` → **`d_model=512, n_head=8`** (capacité du modèle).
- `block_size=64, n_docs_max=10` → **`block_size=128, n_docs_max=20`** (documents plus riches/plus nombreux).
- Budget identique : `--max_steps 3000 --max_time_minutes 480`, `--val_data` inclus, `seed=0` identique aux deux runs.

Logs : `logs/ingest8ter_scale_baseline.log`, `logs/ingest8ter_scale_ingest.log` sur `abacus27-1`. Confirmés actifs sur GPU (`nvidia-smi` avant lancement : H100 à 0 MiB, code/données déjà présents via NFS Rennes partagé).

**Résultat (2026-09-20, H100, 3000 pas fixés pour les deux)** :

| variante | pas | temps | final_loss (train) | val_answer |
|---|---|---|---|---|
| baseline §8, `d_model=512` | 3000 | 226s | **4.414** | **5.370** |
| `--ingest_kb --ingest_n_step 3`, `d_model=512` | 3000 | 295s | **4.665** | **5.484** |

**Lecture** : l'hypothèse de `long-term-memory-builder` (le mécanisme récurrent a besoin de plus de capacité pour rivaliser) n'est **pas confirmée** -- à `d_model=512` avec des documents plus riches (`block_size=128, n_docs_max=20` au lieu de `64/10`), l'écart baseline vs `--ingest_kb` **persiste et s'accentue légèrement en train** (0.251 d'écart vs 0.007 à `d_model=256`) et reste présent en val (+0.11, quasi identique à l'écart mesuré à petite échelle). Coût toujours ~1.3x supérieur ici (295s vs 226s, ratio plus faible qu'à `d_model=256` où c'était ~2x -- cohérent avec un coût fixe d'ingestion qui pèse relativement moins à mesure que le coût du reste du modèle augmente). **Conclusion renforcée** : sur ces deux échelles testées (`d_model=256` et `512`) et sur HotpotQA (lookup à faible profondeur), la projection statique k_proj/v_proj reste au moins aussi bonne que l'ingestion dynamique par pass récurrent, à moindre coût. Relayé à `long-term-memory-builder`.

## 2026-09-20 — Protocole synthétique composition (num_hops=2) vs contrôle (num_hops=1) : premier signal positif pour `--ingest_kb`

Constat préalable : `num_hops` (= `len(supporting_facts.title)`) vaut **2 au minimum sur tout le val set HotpotQA** (distribution {2:1394, 3:438, 4:143, 5:19, 6:5, 8:1}) -- structurel au format "distractor" du dataset, aucun exemple à 0/1 hop. La stratification `answer_hops_ge2`/`answer_hops_le1` (câblée dans `evaluate()`, `learn/indexed_attention/train_prompt_response.py`, commit `e798ed4`) ne peut donc rien montrer sur HotpotQA seul (`answer_hops_le1_n=0` confirmé sur tous les batches val d'un run réel). Généré un jeu synthétique dédié (`learn/distill/prepare_synthetic_composition_data.py`, commit `95ba141`) avec un vrai groupe à 1 hop :
- **composition** (`num_hops=2`) : Doc A relie une entité à un "pont", Doc B relie ce pont à la valeur finale -- la valeur finale n'apparaît JAMAIS avec l'entité directement, les deux docs sont strictement nécessaires, aucun raccourci lexical.
- **contrôle** (`num_hops=1`) : un seul doc énonce directement entité → valeur finale, même structure de question, mêmes distracteurs (même famille de templates, entités/lieux inventés par combinatoire préfixe+suffixe).

900 train (444 composition / 456 contrôle), 100 val (56/44). Lancé baseline §8 vs `--ingest_kb --ingest_n_step 3` (`d_model=512, n_head=8`, `block_size=64, n_docs_max=10`, `max_steps=3000`, seed=0) sur `abacus17-1` GPU0.

**Résultat (3000 pas fixés pour les deux)** :

| variante | final_loss (train) | val_answer (pooled) | val composition (`ge2`, n=56) | val contrôle (`le1`, n=44) |
|---|---|---|---|---|
| baseline §8 | 1.111 | 1.255 | **1.344** | 1.224 |
| `--ingest_kb --ingest_n_step 3` | 1.211 | **1.208** | **1.230** | 1.247 |

**Lecture** : c'est le premier signal positif pour `--ingest_kb` dans tout ce fil. Sur le groupe composition (celui qui nécessite structurellement de chaîner deux documents), `--ingest_kb` bat la baseline (1.230 vs 1.344, -0.114) -- exactement l'effet attendu si le pass récurrent d'ingestion permet une forme de composition/liaison inter-documents que la simple projection k_proj/v_proj ne capture pas aussi bien. Sur le groupe contrôle (1 doc suffit), c'est l'inverse mais plus faible (1.247 vs 1.224, +0.023) -- cohérent avec l'idée que l'ingestion n'apporte rien (voire coûte un peu) quand la composition n'est pas nécessaire. Le pooled val (moyenne des deux groupes) devient même favorable à `--ingest_kb` (1.208 vs 1.255) alors qu'il ne l'était jamais sur HotpotQA pur.

**Prudence nécessaire** : un seul seed, un seul budget, jeu synthétique de petite taille (900 exemples train, structure volontairement simple/template) -- signal à confirmer (plusieurs seeds, budget plus long, éventuellement un vrai jeu composition plus varié) avant de le traiter comme un résultat définitif, mais c'est la première fois que la stratification montre un effet différencié cohérent avec l'hypothèse de départ (`--ingest_kb` utile spécifiquement pour la composition inter-documents). Relayé à `long-term-memory-builder`.

## 2026-09-20 — Protocole proposé : synthèse inter-documents (stratification par num_hops)

**Motivation.** Les deux tests précédents (`d_model=256` et `512`, HotpotQA distractor, budget 3000 pas) sont négatifs sur une métrique de val_answer POOLÉE sur toutes les questions, quel que soit leur nombre de sauts réels. Or l'avantage attendu de l'ingestion dynamique (mémoire associative construite par le pass récurrent, capable de fusionner l'info de plusieurs documents dans le registre `R_t`) n'a de raison de se manifester QUE sur les questions qui exigent réellement de combiner ≥2 documents ingérés -- pas sur celles où une seule passe de projection directe suffit déjà à localiser le fait. En moyennant tout ensemble, un gain réel mais localisé sur le sous-ensemble multi-hop peut être noyé par la majorité des questions à faible profondeur (déjà noté : "HotpotQA lookup peu profond" dans le run `noctx`, où la mémorisation pure sans documents gagnait légèrement).

**Bonne nouvelle : la donnée existe déjà.** `learn/distill/prepare_retrieval_data.py::build_example` calcule déjà `num_hops = len(supporting_facts.title)` par exemple (HotpotQA fournit nativement ce label), mais ce champ n'est actuellement PAS propagé jusqu'à `data/prompt_response_dataset.py` ni `learn/indexed_attention/train_prompt_response.py` -- aucune évaluation stratifiée par hop-count n'existe. Pas besoin de régénérer de données, juste de brancher ce champ jusqu'à l'éval.

**Protocole.**
1. Propager `num_hops` de `prepare_retrieval_data.py` jusqu'à `RetrievalPromptDataset` (déjà lu dans le JSONL, juste ne pas le jeter) et jusqu'à `evaluate()` de `train_prompt_response.py`.
2. Construire deux sous-ensembles de val (même run, pas de nouvelle donnée) :
   - **A (multi-hop réel)** : `num_hops >= 2`.
   - **B (contrôle, faible profondeur)** : `num_hops <= 1`.
3. Reprendre EXACTEMENT la config du dernier test négatif (`d_model=512, block_size=128, n_docs_max=20, max_steps=3000, seed=0`, baseline §8 vs `--ingest_kb --ingest_n_step 3`) -- pas de nouveau réglage à calibrer, seul l'axe d'évaluation change.
4. Rapporter `val_answer` séparément sur A et B pour les deux variantes, et l'écart `ingest_kb - baseline` PAR sous-ensemble (pas seulement l'écart pooled déjà connu).
5. **Lecture attendue si l'hypothèse est vraie** : écart favorable à `ingest_kb` (ou au moins réduit) sur A, écart défavorable ou neutre sur B (cohérent avec le pooled déjà mesuré, dominé par B qui est numériquement majoritaire dans HotpotQA distractor).
6. **Repli si non concluant** : HotpotQA multi-hop est connu dans la littérature QA pour être souvent résoluble par raccourci lexical (chevauchement de mots entre question et un seul document) sans vraie combinaison -- si la stratification par `num_hops` natif ne révèle rien, construire un petit jeu SYNTHÉTIQUE de composition (2 documents, chacun porteur de la moitié de l'info nécessaire à la réponse, aucun raccourci lexical possible sur un seul document) pour un signal plus propre, avant de conclure que le mécanisme n'apporte aucun avantage même en présence de vraie synthèse.

**Coût** : quasi nul en plus des runs déjà faits -- même pipeline, même budget H100 (~250-500s/run à `d_model=512`), le seul travail est le branchement de `num_hops` + le split de l'éval en deux rapports au lieu d'un.
