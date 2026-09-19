---
name: experiment-script-design
description: Conventions d'écriture pour tout script d'expérience du projet Thinker (learn/*/train_*.py, diagnose_*.py) — journalisation `progress:`/`summary:`, contrôles triviaux obligatoires, bornes co-spécifiées, intégration à la file de travail `tools/exp/`. À utiliser dès qu'on écrit, modifie ou lance un script d'entraînement ou de diagnostic de ce dépôt.
---

# Conception d'un script d'expérience — projet Thinker

Chaque règle ici vient d'un incident réel, journalisé dans
`dev_notes/experiment.log.md` / `dev_notes/grid5000_usage.log.md` et
synthétisé dans `dev_notes/experiment_tooling_requirements.md`. Ce n'est pas
du style : c'est ce qui rend un run lisible par `tools/exp/status.py` et
`tools/exp/collect.py`, et récupérable après une préemption.

## Les cinq règles

### 1. Sortie non bufferisée, toujours

`python -u`, ou `PYTHONUNBUFFERED=1`. `tools/exp/worker.py` le fixe déjà, mais
un lancement à la main doit le faire aussi.

**Pourquoi** : un log resté vide toute la durée d'un run, combiné à un
`SIGKILL` de walltime, a fait perdre une courbe de validation entière
(incident A5). Ce qui n'est pas écrit au fil de l'eau n'existe pas.

### 2. Utiliser `core.run_logging.RunLogger`, pas des `print` ad hoc

```python
from core.run_logging import RunLogger

logger = RunLogger(
    run_id=args.run_id,
    state_dir=args.state_dir,
    config=vars(args),            # les hyperparamètres COMPLETS, pas un extrait
    max_steps=args.max_steps,
    max_time_minutes=args.max_time_minutes,
    key_metric="loss",
)

with logger:
    for step in ...:
        ...
        if step % args.log_every == 0:
            logger.progress(step, loss=loss.item(), lr=current_lr)

    logger.finish(
        summary={"final_acc": acc, "best_loss": best, "num_steps": step},
        controls={"chance": baselines["conditional_chance"],
                  "margin": acc - baselines["conditional_chance"],
                  "leak_check": leak_rate},
    )
```

Ce que ça donne gratuitement : ligne `progress: {json}` avec `eta_s`, fichier
d'état réécrit atomiquement à chaque progression (donc l'historique survit à
une préemption), résumé final au format fixe, et statut `failed` renseigné si
le script lève.

**Pourquoi le JSON et pas une barre `tqdm`** : `tqdm` écrit via des retours
chariot — parfait en terminal, illisible une fois redirigé dans un fichier,
ce qui est le cas systématique sur Grid'5000. Une ligne préfixée + JSON se
grep avec une seule regex, quel que soit le script.

### 3. Jamais une accuracy sans ses contrôles triviaux

`finish()` **lève** si `summary` contient une métrique d'accuracy sans
`chance`, `margin` et `leak_check` dans `controls`. C'est volontaire.

**Pourquoi** : deux surinterprétations réelles sur ce projet (incident F2)
sont venues d'un chiffre d'accuracy lu sans son niveau de chance. Le garde-fou
est appliqué à l'écriture *et* à l'affichage (`tools/exp/collect.py` marque
`(SANS CONTRÔLE)` plutôt que d'afficher un chiffre nu).

Si un contrôle n'a pas de sens pour un run donné, le passer explicitement à
`None` — la décision reste tracée, au lieu d'être une omission silencieuse.

### 4. `--max_steps` ET `--max_time_minutes`, toujours co-spécifiés

Les deux, explicitement, à chaque lancement. `tools/exp/gridgen.py` refuse de
générer une grille sans les deux.

**Pourquoi** : un `--max_steps` resté à sa valeur par défaut (50) a produit
deux fois des runs vides qui ont eu l'air de tourner.

### 5. Arguments attendus par la file de travail

Tout script consommable par `tools/exp/worker.py` accepte :

```python
p.add_argument("--run_id", default=None)      # identité stable de la cellule
p.add_argument("--state_dir", default="runs/adhoc/state")
```

Le worker les passe systématiquement. Tout le reste de la config est passé
comme `--<clé> <valeur>` depuis le `grid.jsonl` ; les booléens sont passés en
drapeau nu quand vrais.

## Le flux complet

```bash
# 1. matérialiser la grille (une ligne = une cellule)
tools/exp/gridgen.py --out runs/kb_depth/grid.jsonl \
  --script learn/indexed_attention/train_kb_chain.py \
  --fixed max_steps=8000 max_time_minutes=45 \
  --sweep depth=2,3,4 seed=0,1,2

# 2. réserver et résoudre l'assignation RÉELLE (jamais la supposer)
tools/exp/reserve.py --tier A --nodes 2 --walltime 6 --name kb-depth --dry-run
tools/exp/reserve.py --resolve <JOB_ID>     # donne le plan de lancement exact

# 3. lancer les workers (le plan ci-dessus le fait déjà)
tools/exp/worker.py --grid runs/kb_depth/grid.jsonl --worker-id $i

# 4. observer, à tout moment, depuis n'importe où
tools/exp/status.py --grid runs/kb_depth/grid.jsonl --telemetry

# 5. collecter
tools/exp/collect.py --grid runs/kb_depth/grid.jsonl --sort final_acc
```

Après une préemption : **relancer exactement la même commande à l'étape 3**.
Les cellules `done` sont sautées, le travail reprend où il en était. C'est
l'unique raison pour laquelle le `run_id` est un hash de (script, config) et
pas un compteur.

## Ce que l'outil ne fait pas, et ne fera pas

Il détecte qu'un job est probablement mort, qu'une grille va dépasser son
walltime, qu'un doublon est sur le point de se lancer. Il ne tue jamais un
job, ne change jamais une priorité, ne classe jamais un résultat comme
concluant. Une ligne `SUSPECT` dans `status.py` est une hypothèse à vérifier,
pas un verdict.

C'est le seul garde-fou contre la rigidité : un outil qui se contente
d'observer et d'alerter ne peut pas devenir un carcan, même en grossissant.

## État de la conversion

Les huit scripts d'entraînement actifs sont convertis et consommables par la
file :

| Script | `chance` | `leak_check` |
|---|---|---|
| `indexed_attention/train_kb_chain.py` | `conditional_chance` | `pred_in_kb_rate` |
| `indexed_attention/train_kb_retrieval.py` | `conditional_chance` | `pred_in_kb_rate` |
| `indexed_attention/train_kb_chain_sharpened.py` | `1/vocab_size` | `None` (contrôle = `self_match_diagnostic`, non scalaire) |
| `indexed_attention/train_kb_chain_attn_supervised.py` | `1/vocab_size` | `None` (idem) |
| `toy_memory/train_toy_memory.py` | `mode_baseline.token_acc` | `leak_token_acc` |
| `toy_memory/train_associative_recall.py` | `vocab_chance` | `recency_match_excl_last` |
| `indexed_attention/train_real_text.py` | — (perplexité, pas d'accuracy) | — |
| `distill/train_sft.py` | — (pertes seules) | — |

Les deux `leak_check: None` sont des lacunes **déclarées**, pas des omissions :
`collect.py` les affichera en `(SANS CONTRÔLE)`. Les réduire à un scalaire est
un chantier ouvert.

Pour convertir un **nouveau** script :

1. `add_run_args(parser)` juste avant `parse_args()`, puis
   `logger = logger_from_args(args)` juste après ;
2. un `logger.progress(step, ...)` à côté du `print` existant de la boucle ;
3. un `logger.finish(summary=..., controls=...)` après le bloc de résumé.

Les `print` humains existants peuvent rester : l'outil ne lit que les lignes
`progress:` / `summary:`. Le seul point qui demande de réfléchir est le 3 —
c'est là qu'il faut nommer explicitement le niveau de chance et le contrôle de
fuite du run, comme dans le tableau ci-dessus.
