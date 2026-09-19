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
