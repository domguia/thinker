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

Logs : `logs/ingest8ter_scale_baseline.log`, `logs/ingest8ter_scale_ingest.log` sur `abacus27-1`. Confirmés actifs sur GPU (`nvidia-smi` avant lancement : H100 à 0 MiB, code/données déjà présents via NFS Rennes partagé). Résultat à suivre.
