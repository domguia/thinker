# Pipeline prompt/thinking/answer pour Thinker (reasoning + retrieval)

(Migré depuis experiment.log.md le 2026-09-20 -- contenu original preserve tel quel, groupe par fil plutot que par date.)

## 2026-09-20 — Bug d'API réel: `target_input` dict vs tenseur unique attendu par `Thinker.forward()`

`train_prompt_response.py` (nouveau script de model-design, `ReasoningPromptDataset`/`RetrievalPromptDataset`) crashe systématiquement -- retrieval ET reasoning, avec ou sans contexte (`--depth 0/1`) -- avec `TypeError: embedding(): argument 'indices' must be Tensor, not dict` à `core/indexed_thinker_model.py:318`.

**Cause précise** : `Thinker.forward()` fait `self.embed(target_input)` en supposant `target_input` = un seul tenseur partagé, réutilisé par tous les streams `sequence_mode` (docstring : "embedded once here (shared self.embed) and reused by every such stream, rather than each stream re-embedding target_input independently"). Mais `train_prompt_response.py` construit `target_input = {"answer": ...}` (et ajoute `"thinking"` pour reasoning) -- un dict, pas un tenseur. Même le retrieval simple (1 seul stream) crashe puisque `self.embed()` reçoit le dict entier.

**Question de conception non résolue** : pour `reasoning` (2 streams, thinking+answer), le modèle ne supporte aujourd'hui qu'un `target_input` partagé pour tous les streams `sequence_mode` -- si thinking et answer ont besoin de teacher-forcing différent, il faut soit étendre `Thinker.forward()` pour accepter un dict par stream, soit restructurer le script. Diagnostiqué et transmis à `model-design`, aucun code touché en attendant leur lecture. Pas de perte de compute significative (crash immédiat, quelques secondes par tentative).

Datasets utilisés (générés cette nuit, prêts) : `data/distill/openr1_math/{train,val}.jsonl` (20k exemples, reasoning, champ `answer` présent après correction d'un script périmé), `data/distill/hotpotqa/{train,val}.jsonl` (20k exemples, retrieval, document-aware après le redesign `depth=1`/`block_size`/`n_docs_max`).

## 2026-09-20 — Correction: c'était bien un problème de synchronisation, pas un bug de code

`git log -- core/indexed_thinker_model.py` en local montre bien le commit `3f611b8` (support dict de `target_input`) présent. Le crash venait du fait que **`core/indexed_thinker_model.py` n'avait jamais été transféré vers Rennes** ce soir -- seuls les scripts `learn/*` et `data/*` l'avaient été. Diff systématique local vs Rennes sur tout `core/*.py` a aussi trouvé `core/model_families.py` désynchronisé (même cause probable). Les deux synchronisés, les trois runs (retrieval `depth=1`, ablation `depth=0`, reasoning) confirmés tournant réellement après relance (chargement des 18000 exemples, premiers pas de loss imprimés, pas de crash).

**Leçon opérationnelle** : quand un pair signale un commit sur un fichier `core/`, vérifier/synchroniser CE fichier explicitement -- transférer seulement les scripts de plus haut niveau qu'on pense avoir changé ne suffit pas si une dépendance partagée (`core/`) a aussi bougé sans qu'on le sache. Un diff `md5sum` systématique sur tout `core/*.py` local-vs-cluster est le moyen fiable de rattraper ce genre de trou avant de blâmer le code lui-même.

## 2026-09-20 — hotpotqa/train.jsonl sans `context_docs`: repli confirmé structurellement équivalent, run en cours conservé

`hotpotqa/train.jsonl` avait été généré avant l'ajout du champ `context_docs` (commit `81b4dc9`) -- `RetrievalPromptDataset` bascule donc sur son repli `context.split("\n")`. Vérifié empiriquement sur 50 exemples fraîchement générés avec le script à jour : **0/450 documents avec un `\n` littéral, 0 exemple où `context.split("\n") != context_docs`** -- le repli est structurellement identique à `context_docs`, pas juste probablement équivalent. Run retrieval en cours conservé tel quel, pas de régénération/relance nécessaire.

## 2026-09-20 — 3 baselines prompt/response complètes (retrieval, ablation sans contexte, reasoning)

`paradoxe-5` (CPU), `d_model=1024, n_step=6, lr=3e-4`, budget épuisé (~90 min chacun) :

| run | final_loss | num_steps |
|---|---|---|
| retrieval (`depth=1`, documents réels) | 6.06 | 551 |
| retrieval ablation (`depth=0`, aucun document) | **5.83** | 711 |
| reasoning (thinking+answer) | 7.81 | 319 |

**Read (surprenant) : l'ablation sans contexte (mémorisation paramétrique pure) atteint une loss PLUS BASSE que le run avec documents réels**, malgré ~1.3x plus de pas (confond partiel, pas un budget strictement égal). Sur cette configuration/budget, le contexte documentaire ne semble pas encore apporter de bénéfice net -- à recroiser avec le fait que ce sont des runs CPU non-KD, potentiellement sous-entraînés pour que la lecture de documents réels devienne utile.

**Extrapolation `n_step_test` sur reasoning (2 à 12, trained `n_step=6`)** : stream `answer` remarquablement plat (3.409 à 3.414, quasi aucune variation) sur toute la plage -- contraste avec stream `thinking`, plus variable et non-monotone (4.17 à 4.42, minimum à `n_step_test=6`, pas aux extrêmes). Lecture prudente : le stream `answer` semble robuste au nombre d'itérations à l'inférence, `thinking` moins.

Full logs : `paradoxe-5:~/thinker/logs/prompt_response_{retrieval_v3,noctx_v2,reasoning_v2}.log`.

## 2026-09-20 — CORRECTION: le "noctx bat retrieval" était en partie le même confond pas-de-temps-égal que Piste A

`model-design` a eu raison de demander vérification -- le `final_loss` rapporté plus haut (6.06/5.83) était le **train loss**, pas le val, et surtout **retrieval fait moins de pas que noctx au même budget mural** (551 vs 711 -- traiter des documents coûte plus cher par pas, exactement le confond déjà identifié sur Piste A/A-vs-C).

**Sur VAL, à pas d'entraînement comparable (step=400)** : retrieval=6.22, noctx=6.17 -- écart réduit à 0.05, bien plus faible que l'écart train (0.23) qui avait motivé la lecture initiale. **Retrieval n'a même pas atteint de checkpoint VAL après step 400** (s'est arrêté à 551 pas avant le prochain palier à 600) -- comparaison au-delà de step 400 impossible avec les données actuelles.

**Correction de lecture : pas de conclusion "mémorisation bat récupération" à tirer de ces deux runs tels quels** -- l'écart observé est largement expliqué par le confond pas-de-temps-égal, pas par une différence de capacité de généralisation. Nécessiterait soit un budget en nombre de pas fixé (pas en temps), soit un budget mural nettement plus long pour que retrieval atteigne des checkpoints VAL comparables. Pas relancé pour l'instant, en attente d'arbitrage sur la priorité de cette vérification vs le reste du fil Piste A/C.
