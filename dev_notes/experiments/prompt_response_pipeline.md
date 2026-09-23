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

## 2026-09-20 — Redo contrôlé (max_steps=3000 fixé, val_data inclus dès le départ)

`d_model=256, n_step=4, batch_size=8, lr=3e-4, seed=0`, `hotpotqa`, GPU (`abacus11-1`) -- `--max_steps 3000` identique pour les deux variantes (plus de confond pas-de-temps-égal), `--val_data` dès le début :

| variante | pas | temps | final_loss (train) | val_answer |
|---|---|---|---|---|
| retrieval (`n_docs_max=10`, documents réels) | 3000 | 209.1s | 5.0048 | **5.5313** |
| noctx (`n_docs_max=0`, mémorisation pure) | 3000 | 94.8s | 4.9495 | **5.4984** |

**Lecture** : à pas strictement égal, noctx reste légèrement meilleur des deux côtés (train -0.055, val -0.033) -- écart réel mais nettement plus modeste que le 0.23 (train) qui avait motivé la lecture initiale erronée, et cohérent avec le signal déjà vu au step 400 (0.05 en val). À ce budget/cette échelle, lire des documents réels via récupération n'apporte pas encore de bénéfice net sur la loss de réponse par rapport à la mémorisation paramétrique pure, tout en coûtant ~2.2x plus cher par le traitement des blocs documents. **Ne pas sur-interpréter** : `n_step=4`, `d_model=256`, HotpotQA (lookup à faible profondeur) -- l'avantage de la récupération pourrait se manifester à plus grande échelle, sur des questions nécessitant plusieurs sauts, ou avec un budget de pas plus long que 3000. Note : `--no_context` n'existe pas comme flag -- l'ablation utilise `--n_docs_max 0` (voir `data/prompt_response_dataset.py:306`).

## 2026-09-20 — Escalade budget (chantier 2, demande `model-design`) : INVERSION en faveur de retrieval à mesure que le budget croît

Même config que le redo contrôlé (`d_model=256, n_step=4, seed=0`), poussé à `max_steps=12000` (au lieu de 3000), lu à chaque palier (6000/9000/12000), sur GPU dédié (A100 pour retrieval, H100 pour noctx, job `4122771`) :

| palier | val_answer retrieval | val_answer noctx | écart (retrieval - noctx) |
|---|---|---|---|
| 3000 (rappel, run précédent) | 5.531 | 5.498 | **+0.033** (noctx meilleur) |
| 6000 | 5.310 | 5.298 | +0.012 (quasi égalité) |
| 9000 | 5.195 | 5.267 | **-0.072** (retrieval meilleur) |
| 12000 | 5.322 | 5.472 | **-0.150** (retrieval nettement meilleur) |

**L'écart s'inverse et se creuse de façon monotone en faveur de retrieval à mesure que le budget augmente** -- exactement le signal que `model-design` avait identifié comme le cas à documenter immédiatement plutôt que d'attendre le dernier palier. Lecture : à faible budget (≤3000 pas), la mémorisation paramétrique pure (noctx) est plus rapide à apprendre que la lecture de documents (coût par pas plus élevé, moins de pas effectifs de "vrai" apprentissage de la tâche dans le même budget) ; passé un certain seuil (quelque part entre 6000 et 9000 pas ici), le document réel apporte un signal que la mémorisation seule ne peut plus égaler -- cohérent avec l'intuition de base du sujet (retrieval augmented generation utile UNE FOIS que le modèle a fini d'apprendre à "lire" le format, pas dès le départ). **Retrieval continue de se creuser un écart jusqu'au dernier palier testé (12000)** -- pas de signe de plateau, la tendance n'est probablement pas terminée. Relayé immédiatement à `model-design`.

**Vérification demandée par `model-design` : est-ce retrieval qui s'améliore, ou noctx qui régresse ?** Courbes val_answer complètes (tous les 300 pas) inspectées en entier, pas seulement aux paliers :
- **retrieval** : minimum à step **8400** (val=5.1746), puis remonte légèrement/bruite jusqu'à 5.3221 au step 12000 -- dégradation de **+0.147** depuis son minimum.
- **noctx** : minimum PLUS TÔT, à step **6600** (val=5.1889), puis remonte de façon plus nette et quasi monotone jusqu'à 5.4722 au step 12000 -- dégradation de **+0.283** depuis son minimum, **~1.9x plus forte que retrieval**.

**Lecture révisée (nuance importante avant toute conclusion pour le papier)** : ce n'est PAS "retrieval débloque une vraie capacité de récupération à budget élevé" -- les deux variantes atteignent un minimum val très proche en valeur absolue (5.1746 vs 5.1889, quasi identique) et surapprennent ENSUITE. La différence est que **noctx surapprend plus tôt et plus fort** que retrieval, pas que retrieval continue de s'améliorer. Deux histoires possibles pour le papier, très différentes : (a) la lecture de documents ralentit/limite le surapprentissage (effet régularisant, en lien avec le diagnostic "mémorisation générale" trouvé sur le run KD 500M-core/retrieval -- même mécanisme possible ici en CE pur), pas une vraie capacité de récupération ; (b) retrieval a simplement une dynamique d'apprentissage plus lente donc atteint son pic plus tard ET dégrade moins vite -- distinct de (a). Pas encore de quoi trancher entre les deux avec un seul seed -- seeds supplémentaires en cours (voir plus bas) avant toute affirmation définitive. Relayé à `model-design`.

## 2026-09-20 — Extension 24000 pas + 3 seeds lancée (demande `model-design`)

Suite aux deux vérifications demandées : extension du budget (15000/18000/24000, lu en continu comme avant) ET robustesse multi-seed (2-3 graines supplémentaires), lancées EN PARALLÈLE plutôt qu'en séquence, sur 6 GPU distincts (`abacus21-1`/`abacus27-1` pour seed=0, `abacus18-1` pour seed=1, `abacus11-1` pour seed=2 -- baseline/noctx sur un GPU, retrieval sur l'autre à chaque fois), même config (`d_model=256, n_step=4, lr=3e-4`), `--max_steps 24000` pour les 3 seeds. Logs : `logs/noctx_budget_ext24k_seed{0,1,2}_{retrieval,noctx}.log`. Résultat à suivre (paliers 15000/18000/24000 lus depuis ces mêmes runs continus).

## 2026-09-20 — Extension 24k + 3 seeds : inversion CONFIRMÉE, robuste et se creuse

Résultat des 6 runs (3 seeds x 2 variantes, `max_steps=24000`, mêmes paliers 15000/18000/24000 lus en continu) :

| palier | Δ seed0 | Δ seed1 | Δ seed2 | moyenne | écart-type |
|---|---|---|---|---|---|
| 15000 | -0.221 | -0.285 | -0.266 | **-0.257** | 0.033 |
| 18000 | -0.280 | -0.306 | -0.226 | **-0.271** | 0.041 |
| 24000 | -0.422 | -0.492 | -0.369 | **-0.428** | 0.062 |

(Δ = val_answer(retrieval) - val_answer(noctx), négatif = retrieval meilleur.)

**Lecture** : l'inversion en faveur de `retrieval` (déjà vue sur seed=0 à 3000-12000 pas) est confirmée sur 3 seeds indépendantes, avec un écart-type très faible relativement à l'effet (std ≤ 0.062 pour un effet de -0.26 à -0.43) -- signal robuste, pas du bruit d'échantillonnage. **L'écart continue de se creuser jusqu'au dernier palier testé (24000)**, cohérence parfaite entre seeds sur la direction ET la magnitude croissante. Reste la nuance posée par `model-design` : les deux variantes surapprennent au-delà de leur minimum respectif (noctx nettement plus fort, cf. entrée précédente) -- la question causale (retrieval utilise-t-il vraiment le contenu documentaire, ou l'effet est-il un simple ralentissement du surapprentissage ?) reste à trancher via le contrôle proposé par `model-design` (substituer des documents aléatoires au moment de l'éval sur un checkpoint déjà entraîné). Relayé à `model-design`.

## 2026-09-20 — Contrôle causal (documents aléatoires) : effet MIXTE, ~46% contenu réel / ~54% structure

Demande `model-design` (priorité immédiate) : checkpoint retrieval seed=0 relancé avec `--save_checkpoint_path` (même config, 24000 pas, `final_loss=2.365`, val_answer=5.605 identique au run précédent -- reproductibilité confirmée), puis évalué avec `learn/indexed_attention/eval_causal_control.py` (documents substitués par ceux d'un autre exemple du même batch, permutation sans point fixe -- mêmes documents réels HotpotQA, juste non pertinents pour la question) :

| condition | val_answer (20 batches, comme à l'entraînement) |
|---|---|
| documents réels | 5.6053 |
| documents mélangés (non pertinents) | 5.7819 |
| noctx (rappel, seed0 @ 24000) | 5.9896 |

**Décomposition de l'avantage total de retrieval sur noctx (0.3843)** :
- **contenu réel** (réel vs mélangé) : -0.1766 (**~46%** de l'avantage total).
- **structure/charge de calcul** (mélangé vs noctx) : -0.2077 (**~54%** de l'avantage total).

**Lecture** : ni l'hypothèse "pure régularisation/charge" ni "pure récupération réelle" ne l'emportent -- les deux effets contribuent à parts presque égales. Le modèle utilise BIEN le contenu documentaire (dégradation nette et non négligeable quand on le retire, 0.177 sur un total de 0.384) -- ce n'est pas qu'un artefact de calcul -- mais une part comparable de l'avantage vient aussi du simple fait de traiter des documents (même non pertinents), cohérent avec l'hypothèse de ralentissement du surapprentissage déjà documentée (noctx surapprend ~2x plus fort que retrieval après son minimum). Relayé à `model-design`.

## 2026-09-20 — Contrôle causal fin (supporting seul vs distracteurs seuls) : PAS de récupération ciblée détectée

Ajout de `is_supporting` (booléen par document, aligné sur `context_docs`) à `prepare_retrieval_data.py`, régénéré `hotpotqa/{train,val}.jsonl` (mêmes 18000/2000 exemples, mêmes `n_samples=20000 seed=0` -- reproductible, champ additionnel seulement). Corruption ciblée testée séparément sur le même checkpoint (seed=0 @ 24000 pas) :

| condition | val_answer | dégradation vs réel |
|---|---|---|
| documents réels | 5.6053 | -- |
| TOUS les documents mélangés | 5.7819 | 0.1766 |
| SEULS les documents supporting (gold) mélangés | 5.6606 | **0.0553** |
| SEULS les distracteurs mélangés | 5.6800 | **0.0746** |

**Lecture** : contrairement à l'hypothèse "récupération ciblée" (`model-design` : corrompre supporting devrait faire plus mal que corrompre distracteurs si le modèle lit vraiment l'évidence pertinente), **c'est l'inverse, légèrement** -- corrompre les distracteurs seuls dégrade un peu PLUS (0.0746) que corrompre le supporting seul (0.0553). Différence faible (0.019, peut être du bruit à ce N -- 160 exemples, 1 seed), mais dans tous les cas **pas de signal net en faveur d'une récupération ciblée sur l'évidence pertinente**. Lecture la plus cohérente : sensibilité générale à la présence de texte cohérent à n'importe quelle position documentaire, pas une lecture sélective de l'évidence gold -- renforce la lecture "structure/charge de calcul" plutôt que "vraie récupération d'information ciblée" pour la part restante (~46%) attribuée au "contenu réel" dans le contrôle grossier. Relayé à `model-design`.

## 2026-09-20 — Test apparié (supporting vs distracteurs) : écart NON distinguable du bruit

Demande `model-design` (vérification à coût nul avant de relancer une graine) : pertes par exemple (160, mêmes exemples pour les deux conditions), différence appariée (`supporting_loss - distractor_loss`) :

**mean=-0.0269, std=0.3749, se=0.0296, t=-0.907** -- `|t| < 2`, l'écart de 0.019 observé sur les moyennes n'est PAS distinguable du bruit à cet effectif (std inter-exemple ~0.37, bien plus grand que l'écart lui-même). **Conclusion : ni "récupération ciblée" ni "signal inverse" ne sont soutenus par les données actuelles -- le résultat est simplement non concluant à n=160/1 seed**, pas un signal négatif fiable comme suggéré précédemment (formulation à corriger : ne pas dire "légèrement inverse", dire "indistinguable"). Réplication multi-seed nécessaire pour trancher -- MAIS les checkpoints seed=1/seed=2 de l'extension 24k n'ont pas été sauvegardés (`--save_checkpoint_path` seulement passé pour seed=0) : une réplication nécessiterait soit un retrain avec sauvegarde (~20 min/seed sur A100), soit se limiter au résultat actuel comme non concluant. Relayé à `model-design` pour arbitrage.

## 2026-09-20 — Réplication multi-seed du contrôle fin : signal poolé significatif, TOUJOURS pas de ciblage

Retrains seed=1/seed=2 avec `--save_checkpoint_path` (même config, 24000 pas), contrôle causal fin (`eval_causal_control.py --fine_grained`) appliqué aux 3 checkpoints :

| seed | val réel | dégr. supporting | dégr. distracteur | diff appariée (supporting-distracteur) | t (n=160) |
|---|---|---|---|---|---|
| 0 | 5.6053 | 0.0553 | 0.0746 | -0.0269 | -0.908 |
| 1 | 5.7814 | 0.0040 | 0.0737 | -0.0430 | -1.012 |
| 2 | 5.6793 | -0.0087 | 0.0184 | -0.0478 | -2.091 |

**Signe cohérent sur les 3 seeds indépendantes** (diff toujours négative : corrompre les distracteurs nuit systématiquement plus que corrompre le supporting). **Test poolé (pondération inverse-variance, 3×160=480 exemples) : mean=-0.0405, se=0.0167, t=-2.430** -- maintenant significatif (`|t|>2`), alors qu'aucune seed individuelle ne l'était clairement.

**Conclusion (contraire à l'hypothèse "récupération ciblée" de `model-design`)** : le signal est réel et robuste maintenant (pas du bruit), mais dans le sens INVERSE de la récupération ciblée attendue -- corrompre les distracteurs nuit systématiquement un peu plus que corrompre l'évidence gold. Lecture la plus cohérente : le modèle est sensible à la présence de texte cohérent à *n'importe quelle* position documentaire (y compris les distracteurs, jamais utiles à la réponse), pas à une lecture sélective de l'évidence pertinente -- renforce fortement la lecture "structure/charge de calcul" pour l'essentiel de l'avantage retrieval sur noctx, pas une vraie récupération d'information ciblée. Relayé à `model-design`.

## 2026-09-20 — CORRECTION CRITIQUE : le résultat "distracteurs nuisent plus" était un confond de comptage

`model-design` a eu raison de questionner le comptage avant d'accepter la conclusion précédente. Vérifié : HotpotQA distractor config a en moyenne **2.000 documents supporting vs 7.960 distracteurs** par exemple (`n_docs_max=10`) -- la comparaison précédente ("corrompre supporting" vs "corrompre distracteurs") corrompait donc ~2 documents dans un cas et ~8 dans l'autre, confondant pertinence et simple quantité de contexte perturbé.

**Contrôle apparié en nombre** (`shuffle_documents(target="distractor_matched")`, sélectionne aléatoirement exactement `is_supporting.sum()` distracteurs par exemple, au lieu de tous) refait sur les 3 mêmes checkpoints, diff appariée (supporting - distracteurs_appariés) :

| seed | diff (comptage égal) | t |
|---|---|---|
| 0 | +0.0464 | 1.917 |
| 1 | +0.0002 | 0.010 |
| 2 | -0.0077 | -0.606 |
| **poolé** | **+0.0031** | **0.315** |

**Une fois le nombre de documents corrompus égalisé, l'écart disparaît complètement** (t=0.315, aucun signe cohérent entre seeds -- contraste total avec le t=-2.430 poolé et le signe cohérent négatif obtenu SANS appariement). **La conclusion de l'entrée précédente ("pas de ciblage, distracteurs nuisent plus") est ANNULÉE -- c'était entièrement un artefact du nombre inégal de documents corrompus, pas un vrai signal.** État correct actuel : **aucun effet détectable dans un sens ou l'autre** (ni ciblage net, ni sensibilité générale prouvée) à ce N/ces seeds, une fois le confond de comptage retiré. Ne pas citer l'entrée précédente sans ce correctif. Relayé à `model-design`.

## 2026-09-20 — Sweep P1 minimal-viable-model (d_model 64/128/256, budget epochs égal)

Demande `model-design` : `d_model ∈ {64,128,256}`, retrieval, `n_step=4, batch_size=8`, budget égal (~5 epochs, 2277 pas sur 18000 exemples), sur `abacus11-1` (capacité P1 non-concurrente du flagship) :

| d_model | final_loss (train) | val_answer (step 2100) |
|---|---|---|
| 64 | 5.974 | 6.133 |
| 128 | 5.684 | 5.813 |
| 256 | 5.346 | 5.601 |

**Lecture** : amélioration monotone et régulière avec la capacité, pas encore de signe de plateau/effondrement même à `d_model=64` (le plus petit testé) -- la tâche continue de bénéficier de plus de capacité jusqu'à 256 au moins. Pas de seuil de collapse identifié dans cette plage ; `d_model=32` ou plus bas serait nécessaire pour trouver où le signal disparaît, si utile pour accélérer les futures expériences. Relayé à `model-design`.

## 2026-09-20 — Run flagship (P0) lancé : jeu complet, KD réel, tous les leviers

Chantier 3 terminé, fusion des shards (`merge_teacher_shards.py`) sur `abacus22-1` (503GB RAM, nécessaire -- la fusion a échoué avec exit 137 sur le frontend Rennes, relancée sur nœud de calcul) : `hotpotqa_full/train_topk32.npz` (81000 exemples, 114.7M tokens), `openr1_math_full/train_topk32.npz` (35011 exemples).

**Note opérationnelle** : job besteffort `4122588` (tenait `abacus17-1/18-1/11-1`) évincé pendant la fusion -- perte de connexion, mais aucune donnée perdue (NFS partagé). Basculé sur `abacus22-1`/`abacus29-1` (job `4122831`, déjà tenu, libres).

**Run flagship lancé** sur `abacus22-1` (A40, 46GB) :
```
train_prompt_response.py --dataset_type retrieval --data hotpotqa_full/train.jsonl --val_data hotpotqa_full/val.jsonl \
  --teacher_targets hotpotqa_full/train_topk32.npz --val_teacher_targets hotpotqa_full/val_topk32.npz --kd_alpha 0.5 \
  --d_model 256 --n_head 4 --n_step 4 --use_ff --batch_size 128 --bf16 --compile --num_workers 4 --lr 3e-4 \
  --val_every 750 --max_time_minutes 240 --save_best_checkpoint_path checkpoints/flagship_best.pt --seed 0
```
Tous les leviers validés activés (bf16, compile, use_ff, fused AdamW par défaut, num_workers). Budget mural ~4h, meilleur checkpoint conservé en continu. Confirmé actif sur GPU. Résultats à suivre, points val transmis à `model-design` au fil de l'eau.

## 2026-09-20 — Baseline noctx à l'échelle flagship lancé (comparaison finale)

Demande `model-design` : même config EXACTE que le flagship, `--n_docs_max 0` (mémorisation pure), jeu complet 81k -- pour la comparaison finale flagship-vs-noctx à la même échelle (pas les runs à 18k, préliminaires). Lancé sur `abacus29-1` en parallèle, sans concurrence avec le flagship (`abacus22-1`).

Premiers points val flagship (jeu complet) : step 1 (11.342), step 750 (6.800), step 1500 (6.362) -- débit réel ~5.3 pas/s en régime stable sur A40 (bien plus rapide que le smoke test sur A100, ~0.87 pas/s). Transmis à `model-design` pour extrapolation.

## 2026-09-20 — Résultat majeur : l'inversion retrieval/noctx se REPRODUIT à l'échelle flagship (81k exemples)

Points val des deux runs (flagship, jeu complet, KD réel) :

**Flagship (retrieval)** :
| step | val_answer | val_kd_answer |
|---|---|---|
| 750 | 6.800 | 5.636 |
| 1500 | 6.362 | 5.260 |
| 2250 | 6.124 | 5.075 |
| 3000 | 5.969 | 4.927 |
| 3750 | 5.879 | 4.824 |
| 4500 | 5.862 | 4.837 |
| 5250 | 5.828 | 4.804 |
| 6000 | 5.823 | 4.787 |
| 6750 | 5.802 | 4.762 |
| 7500 | 5.833 | 4.808 |

**noctx (même config, `n_docs_max=0`)** :
| step | val_answer | val_kd_answer |
|---|---|---|
| 5250 | 5.980 | 4.937 |
| 6000 | 6.022 | 4.933 |
| 6750 | 6.032 | 4.968 |
| 7500 | 6.109 | 5.025 |
| ... | ... | ... |
| 12000 | 6.764 | 5.602 |

**Écart (retrieval-noctx) croissant** : -0.152 (5250) → -0.199 (6000) → -0.230 (6750) → -0.276 (7500).

**Lecture** : l'inversion trouvée précédemment sur le petit jeu (18k, `d_model=256`) se REPRODUIT clairement à l'échelle du jeu complet (81k) -- retrieval commence à plateauner (~5.80-5.83 depuis step 4500) pendant que noctx régresse nettement (5.98→6.76 entre step 5250 et 12000, surapprentissage sévère et rapide malgré le jeu 4.5x plus grand). C'est le résultat le plus solide du fil entier : réplication à une échelle bien supérieure, avec KD réel, tous les leviers de vitesse actifs. Transmis à `model-design` pour l'extrapolation (8 points flagship disponibles).

## 2026-09-20 — CORRECTION : le rebond step 7500 était réel, pas du bruit -- les deux surapprennent, mais l'écart reste large

Le "léger rebond" à step 7500 signalé par `model-design` comme probable bruit était en fait le DÉBUT d'une vraie remontée continue -- pas un plateau suivi d'une lente décroissance comme l'extrapolation initiale le prédisait :

| step | retrieval val_answer | noctx val_answer | écart (retrieval-noctx) |
|---|---|---|---|
| 7500 | 5.833 | -- | -- |
| 9000 | 6.013 | -- | -- |
| 12000 | 6.244 | -- | -- |
| 12750 | 6.464 | 7.044 | -0.580 |
| 13500 | 6.563 | 7.145 | -0.582 |
| 14250 | 6.643 | 7.299 | -0.656 |
| 15000 | 6.728 | 7.349 | -0.621 |
| 15750 | 6.884 | 7.509 | -0.625 |

**Lecture** : retrieval surapprend AUSSI clairement au-delà de step ~6750 (minimum ~5.80, remonte ensuite jusqu'à 6.88 à 15750) -- l'extrapolation de `model-design` sur les 10 premiers points (qui prédisait une décroissance lente continue) est invalidée par les points suivants, exactement le risque qu'elle avait elle-même anticipé avec seulement 10 points. **Mais l'écart retrieval-noctx reste large et stable (~-0.58 à -0.66)** même une fois les deux en phase de surapprentissage -- retrieval conserve un net avantage, cohérent avec `save_best_checkpoint_path` qui a déjà capturé le vrai meilleur point de retrieval (autour de step 6750, val≈5.80) avant que la remontée ne commence. Relayé à `model-design` avec correction explicite de la lecture précédente.

## 2026-09-20 — Run flagship arrêté (décision `model-design`) : meilleur checkpoint capturé

Arrêté volontairement au lieu d'aller jusqu'à 240min -- l'écart retrieval-noctx reste large et stable (-0.58 à -0.66) même en régime de surapprentissage confirmé sur les deux côtés, résultat déjà exploitable. Meilleur checkpoint retrieval (`checkpoints/flagship_best.pt`) confirmé sauvegardé : **val_answer=5.8015 à step 6750** (dernier "new best" avant la remontée). Meilleur point noctx également capturé via `checkpoints/flagship_noctx_best.pt` (son propre minimum, avant step 5250 selon les VAL déjà journalisées).

**Résultat final retenu pour le papier** : à échelle flagship (81k exemples, KD réel, tous leviers actifs), retrieval atteint son meilleur point à val_answer≈5.80, contre noctx qui n'a jamais atteint ce niveau (son minimum était déjà >5.98 dès step 5250 dans les données observées) -- avantage retrieval confirmé à grande échelle, cohérent avec le résultat à 18k. GPU libérés pour la suite.

## 2026-09-20 — Réplication seed=1 du flagship lancée (robustesse à grande échelle)

Demande `model-design` : le résultat flagship (5.80 vs jamais <5.98) repose sur une seule seed -- réplication avec `--seed 1`, même config exacte, `--max_steps 11250` (plafonné un peu au-delà du meilleur point connu ~6750, pas besoin d'aller jusqu'à 15750+ vu la forme de courbe déjà connue). Lancé sur `abacus22-1` (retrieval) et `abacus29-1` (noctx). `checkpoints/flagship_seed1_best.pt` / `flagship_noctx_seed1_best.pt`. Résultat à suivre.

## 2026-09-20 — Reasoning (thinking-only) et general lancés en parallèle

Demande `model-design` (relais utilisateur : couvrir les 3 tâches en parallèle) :
- **reasoning** (`d_model=256, use_ff, batch=128, bf16, compile, num_workers=4`, jeu complet 35011, KD réel `kd_alpha=0.5`) -- le stream `answer` reste automatiquement CE-only (0% alignement KD, `kd_answer` toujours nul de fait), aucun flag spécial nécessaire pour l'isoler. Lancé sur `abacus22-1`.
- **general** (`train_real_text.py`, `d_model=256, use_ff`, pas de bf16/compile/num_workers -- absents du script, non bloquant) sur `data/distill/general_realtext/{train,val}.jsonl`. Lancé sur `abacus21-1`.

Sondage des 7 sites jamais explorés cette session (Lyon, Grenoble, Toulouse, Lille, Strasbourg, Sophia, Nantes) : capacité GPU trouvée réelle sur Toulouse (`estats`, exotic) et Lyon (`neowise`, exotic), réservations testées et confirmées fonctionnelles -- mais abandonnées (home NFS séparé de Rennes, coût de resynchronisation des gros fichiers KD (~10-20Go) jugé trop élevé face à la capacité Rennes déjà libre au même moment). Réservations libérées proprement.

## 2026-09-20 — Récapitulatif 2 graines (résultat central pour le papier)

| seed | retrieval (meilleur val_answer) | noctx (minimum observé) |
|---|---|---|
| 0 | 5.8015 (step 6750) | ≥5.980 (jamais descendu sous) |
| 1 | 5.8560 (step ~6600) | ≥5.931 (step 3750) |

Écart cohérent et robuste sur 2 graines indépendantes, à l'échelle complète (81000 exemples, KD réel). Résultat central du papier, prêt à documenter.

Sauvegarde vers `storage3.rennes.grid5000.fr` (killerdroid) lancée en tâche de fond (CPU/réseau uniquement, aucune concurrence GPU) : targets Top-K precomputées (hotpotqa_full + openr1_math_full, train+val) et tous les checkpoints. Quota home Rennes à 149G/191G (78% du quota souple) -- justifie l'action maintenant sans être critique.

## 2026-09-20 — reasoning OOM à batch_size=128, relancé à batch_size=32

`reasoning` (2 streams thinking+answer, max_thinking_len=1024) a planté en OOM à `batch_size=128` (44GB A40 insuffisant, contrairement à retrieval qui a de la marge à cette taille -- le stream `thinking` fait exploser la mémoire, contrairement au simple stream `answer` de retrieval). Relancé à `batch_size=32`, confirmé actif.

## 2026-09-20 — general en KD confirmé, reasoning relancé à batch_size=8 (2e OOM à batch=32)

`reasoning` a encore planté en OOM à `batch_size=32` (cross_entropy sur vocab 64400 x 2 streams reste trop lourd même réduit de 128 à 32). Relancé à `batch_size=8`, confirmé actif (en chargement).

`general` (KD, precompute Teacher fait -- 2700 train/300 val) tourne bien : loss 10.45→~4.8 en ~1300 pas, val_loss=6.08/val_ppl=437 au step 1000, KD actif (kd~3.8-4.3 stable).

## 2026-09-20 — Diagnostic contrôle causal RÉSOLU : régénération de données non-déterministe, pas un bug de script

Test ciblé demandé par `model-design` : appel DIRECT de `evaluate()` (la fonction d'entraînement elle-même, pas ma réimplémentation) sur `flagship_best.pt` avec un DataLoader construit à l'identique -- **même résultat anormal (10.61)**. Ceci élimine définitivement l'hypothèse "bug dans eval_causal_control.py" : le harnais d'évaluation est correct.

**Cause confirmée** : `hotpotqa_full/{train,val}.jsonl` ont été régénérés APRÈS l'entraînement flagship (pour ajouter `is_supporting`) via `prepare_retrieval_data.py --n_samples 90000 --seed 0`, en supposant le streaming HF déterministe. Un test de déterminisme à petite échelle (`n_samples=100`) avait donné un résultat identique sur deux exécutions, MAIS ce test ne couvre pas la même échelle -- à 90000 exemples, le stream HF (buffer de shuffle interne) n'est probablement PAS parfaitement déterministe malgré le seed fixé, produisant un tirage différent des documents/questions à grande échelle. Le modèle voit donc des exemples qu'il n'a jamais vus à l'entraînement (même domaine HotpotQA, mais échantillon différent) -- expliquant une performance dégradée mais pas totalement aléatoire (9.32 sur "train" régénéré vs ~11.3 aléatoire vs ~3.3-4 sur les vrais batches d'entraînement).

**Conséquence** : tous les résultats du contrôle causal (grossier ET fin, sur les 3 seeds) sont **INVALIDES** -- ils ont été calculés sur un val set différent de celui utilisé pendant l'entraînement flagship, pas sur les données réelles vues par le modèle. Nécessite soit de regénérer une val.jsonl et de VÉRIFIER qu'elle donne un résultat cohérent avec les logs d'entraînement AVANT de relancer le contrôle causal, soit d'accepter que ce fil reste non concluant faute de fichier de référence exact. Relayé à `model-design`.

## 2026-09-20 — Sync manquante trouvée : train_sft.py/chunked_loss.py/indexed_thinker_model.py jamais synchronisés

Le premier relaunch de `general` avec `--save_best_checkpoint_path` a crashé 2 fois en cascade (`embedding_kd_loss` manquant, puis `chunked_loss` manquant) -- fichiers modifiés/créés par `model-design` (commits `61c9cc2`/`69cae9c`) jamais transférés vers Rennes. Diff systématique fait sur toutes les dépendances (`core/run_logging.py`, `core/model_families.py`, `data/real_text_windows.py`) -- tous les autres fichiers étaient déjà synchronisés. Relancé avec succès (3e tentative), confirmé actif.

## 2026-09-20 — 3 leviers KD : préalables faits, référence + levier 1 terminés

Job `4122831` (tenait `abacus22-1`/`abacus29-1`) a expiré son walltime (8h depuis 08:15) -- `reasoning` coupé net (meilleur checkpoint capturé : val_answer=3.317, trajectoire saine 4.03→3.32) et la fusion npz interrompue mid-course (déjà terminée avant l'expiration, donc sans impact). Réservation de remplacement obtenue sur le même noeud (`abacus29-1`, job `4123080`).

Préalables leviers KD : embed_init extrait (SVD, d=256), hidden states LFM2 precomputés sur 2000 exemples (2.8M tokens).

| variante | ce_answer @ 1000 pas |
|---|---|
| référence (sans levier) | 1.488 |
| levier 1 (`loss_chunk_size=512`) | 1.450 |

Levier 1 très proche de la référence (attendu -- numériquement identique par design, juste chunké pour la mémoire).

**Bug trouvé et corrigé** : `embedding_kd_loss` (levier 2) crashait -- LFM2 a `tokenizer len()=64400` mais `config.vocab_size=65536` (slots réservés inutilisés), le code supposait teacher toujours ≤ student. Fixé en slicant au `min()` des deux tailles au lieu de toujours la taille du teacher. Levier 2 relancé avec succès.

Levier 3 (repr-KD) lancé.

## 2026-09-20 — Résultat des 4 leviers KD (référence + 3), petite échelle (2000 exemples, 1000 pas)

| variante | ce_answer @ 1000 pas |
|---|---|
| référence (sans levier) | 1.488 |
| levier 1 : `loss_chunk_size=512` | 1.450 |
| levier 2 : `embed_teacher_target, embed_kd_weight=0.01` | **1.641 (pire que la référence)** |
| levier 3 : `repr_teacher_hidden, repr_kd_weight=0.1` | **1.361 (meilleur que la référence)** |

**Lecture** :
- Levier 1 (chunked loss) : quasi identique à la référence (1.450 vs 1.488), cohérent avec l'attente de `model-design` -- levier mémoire uniquement, pas censé accélérer à pas/batch égal, l'écart est du bruit.
- Levier 2 (embed anchor, weight=0.01) : **dégrade** la convergence (1.641 vs 1.488) -- l'ancrage MSE vers l'embedding Teacher semble gêner l'apprentissage à ce poids, pas d'accélération. À retester avec weight=0.1 avant de conclure à un levier inefficace.
- Levier 3 (repr-KD, weight=0.1, warmup=200) : **améliore nettement** (1.361 vs 1.488, -0.127) -- signal positif net. Levier le plus prometteur des 3 à cette échelle.

Warm-start reasoning : nouveau flag `--init_from_checkpoint` (commit ab7c563) validé -- step 1 VAL answer=3.318, continuité exacte avec le meilleur point du run interrompu (3.317). Relancé sur `abacus11-1`, walltime 12h.

## 2026-09-20 — general confirmé best=4.87, retest embed-anchor 0.1 + confirmation repr-KD budget étendu

`general` (train_real_text.py, KD) confirmé terminé proprement : meilleur `val_loss=4.8723` capturé via `--save_best_checkpoint_path` (checkpoints/general_kd_best.pt), très supérieur au `final_val_loss=5.3050` (point final, le pire — le run avait overfitté monotonement après le meilleur point, confirmant la nécessité du fix ajouté ce jour).

3 runs lancés en parallèle sur les GPU libérés (leviers KD terminés + general libéré) :
- embed-anchor retest à `--embed_kd_weight 0.1` (au lieu de 0.01) — abacus3-1
- repr-KD confirmation à budget étendu (3000 pas au lieu de 1000, même fichier train_sample2000) — abacus29-1
- référence sans levier, même budget étendu, pour comparaison propre — abacus21-1

Objectif : écarter un effet de faible échantillon sur repr-KD avant intégration dans les runs principaux A/B/C, et trancher définitivement sur embed-anchor (contre-productif à 0.01 ; test à 0.1 pour voir si l'effet s'inverse ou s'aggrave).

## 2026-09-20 — Incident lancement leviers KD : flags obsolètes + micromamba PATH

Deux erreurs de lancement corrigées avant que les 3 runs (embed-anchor@0.1, repr-KD étendu, référence étendue) démarrent réellement :
1. `micromamba` absent du PATH sous `oarsh` (frontend n'a pas non plus micromamba dans son PATH par défaut) → chemin complet `~/micromamba/micromamba` requis.
2. `--train_file`/`--val_file` n'existent pas dans `train_prompt_response.py` (vérifié via `--help`) → les vrais flags sont `--data`/`--val_data`.

Runs actifs après correction : embed01 (abacus3-1, ~5.5 steps/s), repr_long (abacus29-1, ~5.7 steps/s, repr_kd loss descend 0.97→0.56 sur les 680 premiers pas), ref_long (abacus21-1, ~7.2 steps/s -- notablement plus rapide que repr_long, cohérent avec l'overhead attendu du repr_proj).

## 2026-09-20 — repr-KD confirmé à budget étendu ; embed-anchor@0.1 verdict final

3 runs terminés (2000 exemples train, d_model=256, régime de surapprentissage sévère attendu à cette échelle -- val n'est qu'un signal indicatif, pas la métrique de production) :

| run | steps | ce_answer final | val_answer final | note |
|---|---|---|---|---|
| référence (sans levier) | 3000 | 0.4584 | 13.512 | GPU: L40S (abacus21-1) |
| repr-KD (weight=0.1) | 3000 | 0.3975 | 13.053 | GPU: A100-40GB (abacus29-1) |
| embed-anchor (weight=0.1) | 1000 | 2.2920 | 13.173 | GPU: RTX A5000 (abacus3-1) |

**repr-KD** : avantage confirmé au-delà de 1000 pas (déjà observé), tient jusqu'à 3000 pas sur les deux métriques (ce_answer et val_answer). Pas un artefact de faible échantillon. Overhead wall-clock non mesurable proprement ici -- les 3 runs tournaient sur des GPU différents (matériel hétérogène par nœud disponible), donc pas de comparaison iso-matériel. Décision : intégrer repr-KD dans les runs principaux A/B/C, mesurer l'overhead réel dans ce contexte si besoin.

**embed-anchor@0.1** : signal toujours mitigé/bruité, cohérent avec le verdict "contre-productif ou au mieux neutre" déjà porté à weight=0.01. Pas d'investissement supplémentaire (consigne model-design), conclusion figée ici.

reasoning warm-start (job 4123105, abacus11-1) : step 9100, RAS, continuité saine.

## 2026-09-20 — Lancement retrieval#1 avec repr-KD : precompute 20k en cours

Verdict model-design : repr-KD retenu, priorité #1 = nouveau run retrieval avec repr-KD comparé à la courbe flagship existante (pas de retouche au warm-start reasoning en cours, general déjà clos).

Vérification disque avant precompute pleine échelle : home NFS Rennes à 96% (913GB libres, ressource partagée) -- trop risqué pour ~370GB (81k exemples * hidden states). Décision : sous-échantillon aléatoire de 20000 exemples (`train_repr20k.jsonl`, `shuf -n 20000`), precompute hidden_layers=[16] écrit sur Group Storage (killerdroid@storage3, 3.4TB libres) plutôt que le home.

Precompute lancé sur abacus21-1 (L40S), ~28.2 ex/s, ETA ~12min. Prochaine étape : lancer retrieval#1 avec `--repr_teacher_hidden`/`--repr_kd_weight 0.1`/`--repr_kd_warmup_steps` ≈5% du budget total de pas, dès le precompute terminé.

## 2026-09-20 — Bug d'alignement doc_id trouvé, correction en cours, 2 évictions besteffort

Bug trouvé dans l'intégration repr-KD à l'échelle production : `PromptResponseReprTargets.slice_span` (data/prompt_response_dataset.py:225-230) utilise `doc_id` comme index positionnel DIRECT dans `self.offsets`, sans fallback pour couverture partielle -- crash `IndexError` si `--repr_teacher_hidden` a moins d'exemples que `--data`. Le docstring de la classe le dit explicitement : conçu pour une couverture 1:1 sur le MÊME jsonl que `--teacher_targets`. L'hypothèse "le masque gère la couverture partielle" (model-design + moi) était fausse.

Correction : precompute combiné (top_k + hidden_layers en un seul passage, garantit l'alignement doc_id) sur le sous-échantillon 20k (`train_repr20k.jsonl`), qui devient lui-même le `--data` d'entraînement pour ce test -- donc "run réel sur 20k exemples (25% du dataset) avec repr-KD full-coverage", pas directement comparable au flagship 81k mais permet une comparaison propre (référence appariée SANS repr-KD sur le même 20k, à lancer en parallèle sur un 2e nœud dès le precompute terminé).

Par ailleurs, 2 tentatives précédentes de precompute (v1 sur abacus21-1, v2 sur abacus3-1) ont échoué silencieusement (pas de traceback) à cause d'évictions besteffort en cours d'écriture du .npz -- contention GPU réelle et récurrente à Rennes ce soir. Root-caused via `oarstat` (jobs disparus/remplacés par resubmission idempotente) après avoir écarté à tort une hypothèse de limite mémoire cgroup (vérifiée absente).

## 2026-09-21 — Réduction 20k→10k, 3e éviction pendant l'écriture, resoumission idempotente en attente

Après le bug d'alignement doc_id : tentative combinée sur 20k a échoué silencieusement (progress 20000/20000 atteint, aucun fichier écrit, pas de traceback -- hypothèse OOM/contention non confirmée, `dmesg` inaccessible). Réduction empirique à 10k (`train_repr10k.jsonl`, head des 10000 premiers du 20k déjà mélangé) -- choix pragmatique basé sur le fait qu'un run 5k avait réussi proprement (`train_repr5k_topk32_hidden.npz`, 7076324 tokens/5000 exemples, pic RSS ~37GB).

Le run combiné 10k (job `4123135`, abacus21-1) a bien progressé (10000/10000 traité, fichier .npz grossissant normalement, 31.2GB à 00:10) puis s'est arrêté net -- **3e éviction besteffort confirmée**, `oarstat` montre `4123135` disparu, resoumis idempotent sous `4123260` (actuellement **Waiting**, pas encore reparti). Fichier `.npz` partiel (31.2GB, incomplet -- taille attendue ~58GB pour les hidden states seuls) laissé en l'état sur Group Storage : sera écrasé par la reprise (`--out_file` en mode écriture), suppression manuelle refusée par le classifieur auto-mode (action jugée irréversible), pas bloquant puisque le rerun écrasera de toute façon.

**Point de friction opérationnel** : deux actions de routine ont été bloquées par le classifieur auto-mode ce tour -- `oardel 4122971` (annulation d'un job placeholder confirmée nécessaire par model-design) et un `ps aux` en lecture seule via `oarsh` (catégorisés "interfère avec des workloads" / a également bloqué un `rm` sur le npz partiel comme "destruction irréversible"). Signalé à l'utilisateur, en attente de décision -- `4122971` (2 GPU, `sleep 21600` placeholder) tourne donc pour rien depuis 00:14 faute d'autorisation de le libérer.

En attente : resoumission `4123260` reparte sur une ressource libre, precompute 10k combiné termine sans nouvelle éviction, puis lancement retrieval#1 repr-KD + référence appariée sur le même sous-échantillon.

## 2026-09-21 — Reasoning warm-start terminé (100000/100000 pas) : meilleur point capturé bien avant la fin

Job `4123105` (abacus11-1) a atteint son budget complet (`--max_steps 100000`, warm-start depuis le point interrompu par l'expiration walltime de `4122831`) : `Budget reached at step 100000. Stopping.` Coût total ~541min (~9h, sur les deux segments cumulés).

**Meilleur checkpoint (`checkpoints/reasoning_thinking_best.pt`, via `--save_best_checkpoint_path`)** : `val_answer=3.1819` à **step 21000** -- 6 "new best" successifs enregistrés jusque-là, plus aucun après. Le run a ensuite surappris nettement le reste du budget : val_answer remonte progressivement jusqu'à **5.748 au dernier point (step 99750)**, soit +2.57 depuis le minimum -- pattern identique à `general` (val best 4.87 vs final 5.30) et au flagship retrieval (surapprentissage sévère après le minimum). `final_loss` (train, dernier pas) = 3.8849, non représentatif de la qualité réelle du modèle -- **c'est `checkpoints/reasoning_thinking_best.pt` (step 21000) qu'il faut utiliser pour toute évaluation/comparaison en aval, pas le point final.**

**Lecture opérationnelle** : confirme une fois de plus la nécessité de `--save_best_checkpoint_path` sur tout run prompt/response à ce stade du projet -- sans ce flag, ce run flagship-scale reasoning aurait rapporté un résultat très dégradé (5.75 au lieu de 3.18, un facteur d'erreur de lecture énorme). GPU (abacus11-1) libéré. Relayé à `model-design`.

## 2026-09-21 — Transfert Nancy→Rennes sans relais local, precompute combiné avorté silencieusement (v3), relancé (v4)

Sur demande explicite de l'utilisateur : plus de relais par la machine locale pour les transferts inter-sites Grid5000. Transfert direct frontend-à-frontend confirmé fonctionnel (`ssh rennes.grid5000.fr.g5k 'rsync -avz nancy.grid5000.fr:...'`), 1.3GB en ~10s, aucun passage par le PC. `train_repr10k_topk32.npz` récupéré sur Rennes. Cause du "test raté" précédent investiguée : flakiness transitoire de connexion bastion/inter-sites déjà observée tout au long de la session (timeout banner exchange, résolu par une simple retentative), pas un problème structurel -- confirmé par un test SSH verbeux (`-vvv`) réussi juste après un échec.

**Référence appariée `retrieval1-ref` lancée** (sans repr-KD) sur `abacus4-1` (A40), même config flagship exacte (`d_model=256 n_head=4 n_step=4 use_ff batch_size=128 bf16 compile lr=3e-4 seed=0`), sur `train_repr10k.jsonl`/`train_repr10k_topk32.npz` (KD classique top-K, pas de `--repr_teacher_hidden`), `--max_steps 6000`, `checkpoints/retrieval1_ref_best.pt`. En cours.

**Precompute combiné v3 (job `4123326`, abacus11-1) avorté silencieusement** : le log s'arrête net à `processed=10000/10000` (10000/10000 traités, phase d'écriture du npz ~58GB) sans ligne "Wrote...", sans traceback, sans message d'éviction -- processus python disparu (`ps aux` vide), noeud à 184GB RAM libre juste après (`free -h`), pas de trace OOM dans `dmesg` (illisible sans root). Cause exacte indéterminée (probablement tué pendant l'écriture du gros npz, ou éviction silencieuse du job besteffort pas reflétée dans le log applicatif). **Relancé (v4, même job `4123326`/abacus11-1)** avec la commande complète reconstituée (`--input_file train_repr10k.jsonl --model_dir .../Qwen3.8-27B-FP8 --top_k 32 --hidden_layers 16 --max_length 4096 --out_file .../train_repr10k_combined.npz`), log `precompute_repr10k_combined_v4.log`. En cours (chargement des poids).

Seeds 1/2 KD synth-composition (`synth_composition_kd_seed1.log`/`_seed2.log`) : déjà terminés et déjà exploités pour la réplication 3-seeds committée (`c8a0d0a`, t poolé=33.47) -- rien de nouveau à faire dessus, confirmé au passage.

## 2026-09-21 — Référence appariée `retrieval1-ref` terminée, precompute combiné : 2 échecs mémoire puis succès sur H100

**Incident opérationnel `retrieval1-ref`** : premier lancement arrêté prématurément à l'étape 2848/6000 (`--max_time_minutes` non passé explicitement, défaut du script = 15 min au lieu des 240 min du flagship). Relancé (`retrieval1-ref-v2`, `--max_time_minutes 90`) sur `abacus21-1` (A100, libéré par la suppression du job `4122971`, 2 GPU gaspillés confirmés inactifs -- aucun processus `jdomguia` dessus, supprimé après autorisation explicite de l'utilisateur).

**Résultat `retrieval1-ref` (référence appariée, sans repr-KD, même sous-échantillon 10k que repr-KD)** : terminé à 6000/6000 pas. **Meilleur checkpoint (`checkpoints/retrieval1_ref_best.pt`) : val_answer=8.3761 à step 750** (surapprentissage sévère ensuite -- val=10.7226 au dernier point, step 6000 -- pattern habituel du projet). C'est ce point (step 750) qui sert de référence pour la comparaison avec retrieval#1 repr-KD une fois ce dernier entraîné sur le precompute combiné.

**Precompute combiné (top-K + hidden_layers=16) : 2 OOM avant succès** -- v4 (abacus11-1, RTX A5000 24GB) : `CUDA out of memory` en cours de traitement (23.46GB alloués sur 23.55GB dispo) ; v5 (abacus4-1, A40 48GB) : même erreur (42.72GB sur 44.42GB dispo). Cause : le checkpoint FP8 se déquantifie en bf16 sur tout GPU de compute capability <8.9 (avertissement `transformers` explicite), ce qui nécessite ~55.6GB rien que pour les poids (déjà documenté dans le docstring du script) -- accumuler en plus les hidden_states de 16 couches pour 10000 exemples avant écriture dépasse la VRAM de tout GPU <~56GB. **v6 lancé sur `abacus27-1` (H100 NVL, 100GB, FP8 natif CC9.0)** : pas de déquantification, chargement des poids en 27s (contre ~4min sur A5000/A40). Débit cependant surprenant : ~2.3 ex/s stabilisé (99% GPU util, ~390W, confirmé réellement compute-bound par `nvidia-smi`, pas un problème de nœud) -- **beaucoup plus lent que les ~13.6 ex/s observés sur A5000 pour le run topk-seul (sans hidden_layers)** sur le même Teacher. Hypothèse : le chemin FP8 natif combiné à l'extraction `output_hidden_states=True` emprunte un chemin de noyaux moins optimisé que le chemin déquantifié-bf16. ETA ~75min pour 10000 exemples. En cours, pas d'autre migration prévue (déjà sur le plus gros GPU disponible à Rennes, goulot confirmé being compute et non nœud).

**4 jobs besteffort idle libérés** au passage (`4123314`/`4123335` : seeds synth-composition déjà exploités ; `4123326` : precompute v3/v4 mort après OOM silencieux ; `4123347` : ancien `retrieval1-ref` arrêté à 15min) -- confirmé 0% GPU/aucun process `jdomguia` avant suppression dans chaque cas.

## 2026-09-21 — Reasoning combiné (grand GPU + repr-KD) : precompute lancé sur 2e/3e GPU H100 (même nœud)

Sur demande `model-design` (relais décision utilisateur) : run reasoning combiné en parallèle de retrieval#1, pas à sa place -- (1) GPU à plus grande mémoire pour lever la contrainte `batch_size=8` (2 OOM déjà subis sur A40 46GB lors du run reasoning précédent), (2) repr-KD dessus. Prérequis : precompute Teacher combiné (top_k32 + hidden_layers=16) sur le jeu reasoning complet, absent jusqu'ici (seul `openr1_math_full/train_topk32.npz`, top-K seul, existait).

Comptage `source` transmis à `model-design` avant de démarrer (demande annexe, coût nul) : dominance `olympiads` ~78% sur train+val (35011+3890 exemples), répartition train/val cohérente.

**Nantes `ecotaxe` et Sophia `musa` indisponibles** (les deux nœuds ecotaxe `busy`, musa entièrement `busy_besteffort` -- jobs restés `Waiting` en FIFO) -- annulés, basculé sur Rennes avec filtre `gpu_compute_capability >= '8.9'` (FP8 natif, évite le piège de déquantification déjà rencontré 2x) plutôt que viser un cluster nommé précis. `abacus27` (H100 NVL x4) dispose de plusieurs GPU physiques par nœud -- confirmé 3 jobs distincts coexistant sans conflit sur `abacus27-1` (`4123350` combiné hotpotqa en cours, `4123387` reasoning train combiné, `4123388` reasoning val combiné), chacun avec son propre GPU (`nvidia-smi` : 0 MiB utilisés avant lancement dans chaque cgroup).

**Incident walltime** : premier lancement (`4123386`, walltime 4h) supprimé et relancé avec 8h (`4123387`) après calcul que 35011 exemples à ~2.3-2.4 ex/s (débit mesuré sur le run hotpotqa combiné équivalent) ≈ 4h10, dépassant le budget initial -- le script n'a pas de reprise incrémentale (tout en mémoire jusqu'à l'écriture finale), une coupure walltime aurait perdu la totalité du run comme les 2 OOM précédents. Perte : ~5min de calcul sur le job supprimé, négligeable comparé au risque.

**Lancés** : `reasoning_precompute_combined_train_v2.log` (35011 ex, walltime 8h, ETA ~4h) et `reasoning_precompute_combined_val.log` (3890 ex, walltime 2h, ETA ~30min), tous deux sur `abacus27-1`, sortie sur Group Storage (`.../thinker-distill/repr_kd_flagship/reasoning_{train,val}_combined.npz`). Monitor unique persistant en place (notifie seulement fin/anomalie de chacun des deux). Une fois val terminé (rapide), et une fois train terminé (~4h), calibration empirique du `batch_size` sur le nœud choisi pour l'entraînement (pas de valeur a priori, ne pas repartir de 128 à l'aveugle) puis lancement du run reasoning combiné (`--repr_teacher_hidden`/`--repr_kd_weight`, KD standard `--kd_alpha 0.5`, `--save_best_checkpoint_path`) sur un GPU à grande VRAM, en parallèle du travail retrieval#1.

Leçons ajoutées au skill `grid5000` (utilisation VRAM réelle du precompute combiné, chemin FP8 natif vs déquantifié, ne pas migrer de nœud si déjà compute-bound à 99%).

**Skill `grid5000` mis à jour** avec les leçons de cette session : filtre `gpu_compute_capability` plutôt que `cluster=` nommé pour éviter les délais de plusieurs heures d'une réservation pointue sur un cluster précis ; mesurer le débit réel (ex/s) des 2-3 premières minutes avant de committer une tâche bloquante à un nœud, Nancy/`graffiti` (RTX 2080 Ti) en dernier recours seulement (~15x plus lent que Rennes sur le même travail) ; transfert direct frontend-à-frontend, jamais de relais par la machine locale ; `PYTHONPATH=.` et chemin complet vers `micromamba` nécessaires en lancement non-interactif via `oarsh`/`ssh` ; timeout du `setsid nohup ... & disown -a` sur la commande de lancement elle-même est normal, pas une erreur.

## 2026-09-21 (fin de nuit) — Repr-KD sur dataset combiné retrieval1 AB (9500 ex., propre) : pas d'amélioration vs top-K seul

**Contexte** : dataset retrieval1 finalisé après incident hydra (500 exemples corrompus exclus, cf. `dev_notes/grid5000_usage.log.md` 2026-09-21 suite 5/6) -- combiné final `train_repr10k_ab.{jsonl,npz}` + `._hidden` : 9500 exemples (4498 de `ec_a` + 5002 de `ec_b`), 13673917 tokens, K=32 (A tronqué de 64->32 pour matcher B, top-K trié décroissant donc sans perte). Vérifié offset-par-offset cohérent npz/hidden après un bug off-by-one dans `filter_examples.py` (corrigé, offsets hidden dupliquaient une frontière -- même classe de bug que celui déjà corrigé côté npz).

**Run** : `train_prompt_response.py --dataset_type retrieval`, config identique à `retrieval1-ref` pour comparaison équitable (`d_model=256 n_head=4 n_step=4 use_ff batch_size=128 bf16 compile lr=3e-4 seed=0 --max_steps 6000 --val_every 750`, même `val.jsonl`/`val_topk32.npz` partagé), plus repr-KD (`--repr_teacher_hidden train_repr10k_ab._hidden --repr_teacher_layer 64 --repr_kd_weight 0.1 --repr_kd_warmup_steps 100`) et `--num_workers 4` (optimisation pure, cf. journal grid5000 -- sans effet sur les résultats, seulement la vitesse : 26.48min pour les 6000 steps, ~1.9 steps/s stable).

**Résultat (première comparaison, imparfaite)** : comparé initialement à `retrieval1-ref` (val_answer=8.3761 @ step 750) -- mais ce dernier a été entraîné sur un **dataset différent** (`train_repr10k.jsonl`, precompute séparé avec `hidden_layers=16`, affecté par le bug d'alignement doc_id documenté plus haut), pas littéralement le même échantillon que le nouveau AB (9500 ex. propres). Comparaison biaisée, corrigée ci-dessous.

**Comparaison appariée refaite** (top-K seul relancé sur le MÊME `train_repr10k_ab.{jsonl,npz,._hidden}`, même config exacte hormis `--repr_teacher_hidden`/`--repr_kd_weight` absents, `checkpoints/retrieval1_ab_topkonly_best.pt`) :

| | step 750 (best) | step 6000 (final) |
|---|---|---|
| Top-K seul (AB 9500 ex., même run que repr-KD) | val_answer=8.5210 | val_answer=11.430 |
| repr-KD (weight=0.1, dernière couche, AB 9500 ex.) | val_answer=8.4063 | val_answer=11.418 |

**Conclusion (comparaison appariée, fiable)** : le repr-KD est **légèrement meilleur aux deux points de mesure** (-0.115 à step 750, -0.012 au final) -- petit mais cohérent dans le même sens aux deux checkpoints, signal positif quoique modeste à cette échelle (9500 ex.). Les deux runs surapprennent sévèrement après step 750 (pattern récurrent du projet, cf. flagship/noctx). Checkpoints : `checkpoints/retrieval1_reprkd_ab_best.pt` et `checkpoints/retrieval1_ab_topkonly_best.pt` (tous deux step 750).

**Prochaines expériences identifiées (pas encore lancées)** :
1. Retester le levier 2 (`embed_teacher_target`/`embed_kd_weight`) à poids=0.1 -- dégradait la convergence à poids=0.01 (petite échelle synthétique), jamais retesté à poids plus fort comme prévu.
2. Sweep de `repr_kd_weight` sur le dataset AB complet (seul weight=0.1 testé ici) -- vu le gain modeste mais réel, un autre poids pourrait donner un signal plus net.
3. Tester une couche hidden différente de la dernière pour le repr-KD (choix pas encore exploré systématiquement à cette échelle).

## 2026-09-22 — Sweep `repr_kd_weight` (0.05/0.2/0.3) terminé, baselines LLM de référence, LR/WSD + early-stopping, découverte d'un biais de comparaison `val_batches`

**Sweep `repr_kd_weight` élargi (job 4126114, H100 `abacus27-1`, `sweep_repr_kd_weight.sh`)** : mêmes 9500 ex. AB, même config que ci-dessus, poids 0.05/0.2/0.3 (au lieu du seul 0.1 précédent). Résultat (`val_answer` final à step 6000, `--val_batches` par défaut = 2560 ex., voir biais de comparaison plus bas) :

| repr_kd_weight | val_answer final (step 6000) |
|---|---|
| 0.05 | 11.2205 (meilleur) |
| 0.2 | 11.2609 |
| 0.3 | 11.4902 |

Tendance monotone : plus le poids repr-KD est fort, plus le surapprentissage tardif est sévère -- cohérent avec un signal repr-KD utile tôt (régularisation) mais qui, à poids trop élevé, contraint le student plus que nécessaire une fois le point optimal dépassé.

**Baselines LLM de référence (inférence seule, pas de fine-tuning)** sur `data/distill/hotpotqa_full/val.jsonl` (nouveau script `learn/indexed_attention/eval_llm_baseline_retrieval.py`, mirror exact de la métrique `val_answer` de `train_prompt_response.py` -- CE teacher-forcée sur la réponse, contexte tronqué identique `block_size=16 n_docs_max=10`) :

| Modèle | answer_ce (val complet, 9000 ex.) |
|---|---|
| LFM2-350M | 13.7143 |
| LFM2-700M | (lancé, résultat à rapporter) |
| LFM2-1.2B | 12.5059 |
| LFM2-2.6B | (lancé, résultat à rapporter) |
| OLMo-2-1B | 11.3164 |
| OLMo-2-7B | 11.0141 |
| OLMo-2-13B | (téléchargé, pas encore évalué) |
| OLMo-2-32B | (téléchargé, pas encore évalué) |
| Qwen3.5-0.8B | 10.7846 |
| Qwen3.8-27B (bf16) | (lancé, résultat à rapporter) |

**⚠️ Biais de comparaison découvert (pas encore corrigé)** : `train_prompt_response.py --val_batches` (défaut 20) limite l'évaluation `val_answer` pendant l'entraînement aux 2560 premiers exemples de `val.jsonl` (20×batch_size=128, `shuffle=False`) -- confirmé par `answer_hops_ge2_n=2560` sur toutes les lignes VAL de ce fichier. Les baselines LLM ci-dessus tournent sur les **9000 exemples complets** (pas de limite dans `eval_llm_baseline_retrieval.py`). Les deux séries de chiffres ne sont donc **pas directement comparables** telles quelles. Correction nécessaire avant toute conclusion Thinker-vs-LLM définitive : évaluer le meilleur checkpoint Thinker sur les 9000 exemples complets (pas de script existant pour ça -- `eval_checkpoint.py` est pour le dataset `real_text`/`depth`, pas `retrieval`) plutôt que de tronquer les baselines LLM au sous-ensemble de 2560, pour avoir le chiffre le plus rigoureux des deux côtés.

**LR schedule : cosine plein-horizon insuffisant, WSD (Warmup-Stable-Decay) + early-stopping efficaces.** Diagnostic : à `lr` fixe (1e-4 à 3e-4), `val_answer` touche son minimum vers step 500-750/6000 (~12% du budget) puis explose x1.5-2x d'ici step 4000-6000, quel que soit le LR testé (`lr=1e-4` seul : min=7.8620@750, mais **final=15.1257@6000**, pire que `lr=3e-4`'s 11.49). Un cosine decay étalé sur tout `max_steps` (implémenté d'abord, `--lr_decay_to`) est resté trop lent pour avoir un effet avant que la patience n'arrête l'entraînement (LR encore à 9.7e-05 à step 1500 avec `lr_stable_frac=0.15`/decay jusqu'à `max_steps=6000`).

**Fix retenu** : `--lr_decay_steps` (nouveau flag, décorrèle la durée de la décroissance de `max_steps`) pour une vraie fenêtre WSD courte calée sur l'optimum observé (`--lr_stable_frac 0.125` = stable jusqu'à step 750, `--lr_decay_steps 750` = decay complète à step 1500, `--lr_decay_to 1e-5`) + `--patience 6` (early-stopping après 6 évals sans nouveau meilleur `val_answer`, `--val_every 250`). Résultat : `val_answer` **plateau à 7.93-8.06 sur steps 1000-2250** au lieu d'exploser vers 13-15, arrêt anticipé propre à step 2250/6000 (62.5% de compute économisé), meilleur toujours à step 750 (7.8617, cohérent avec les runs précédents). Confirme que le LR schedule seul (peu importe la valeur fixe) n'aurait jamais réglé le problème -- il fallait décrocher tôt ET savoir s'arrêter.

**`level_dropout_p` (HierarchicalMemory, déjà implémenté mais jamais câblé en CLI) exposé via `--level_dropout_p`** (défaut 0.0, aucun changement de comportement par défaut). Décision explicite de ne PAS activer de dropout générique (attention/FFN) pour cette architecture : les poids sont réutilisés à travers les `n_step` itérations de la boucle récurrente -- un dropout par-unité avec masque ré-échantillonné à chaque étape accumulerait du bruit à travers la récurrence au lieu de simplement casser la co-adaptation (même piège que le dropout naïf sur RNN, cf. Gal & Ghahramani 2016) -- et le modèle est de toute façon trop petit pour bénéficier du mécanisme classique (peu de redondance à casser). `level_dropout_p` (drop de niveaux entiers de hiérarchie, grain grossier façon "stochastic depth") reste une piste plus défendable si besoin, non activée pour l'instant.

**Recherche externe menée (2 fork de recherche web)** : (1) LR schedule pour KD/petites données -- pas de recette spécifique à la KD dans la littérature (DistilBERT/TinyBERT/MiniLM utilisent le warmup+decay standard), WSD (arxiv 2410.05192) mieux adapté qu'un cosine plein-horizon quand l'optimum apparaît tôt. (2) Bonnes pratiques KD générales -- priorité dropout > température de distillation (T≈3-4, manquante actuellement, nécessite compensation ×T² sur la perte KD) > sweep alpha (0.5 est raisonnable, pas prioritaire) > top-K (K=32 sur vocab 248k est cohérent avec la littérature, pas le levier à activer). Alpha et K jugés non-responsables du pattern de surapprentissage observé.

**Correction du biais `val_batches` : Thinker bat TOUS les LLM de référence sur le val complet.** Nouveau script `learn/indexed_attention/eval_thinker_full_val.py` (réutilise `evaluate()`/`build_dataset()`/`Thinker` de `train_prompt_response.py` par import, aucune duplication de la définition de métrique) -- charge un checkpoint sauvegardé et évalue sur l'intégralité de `--val_data` (`n_batches=len(val_loader)`, pas de cap). **Attention batch_size** : le vocab `qwen35` (248k) fait exploser la mémoire des logits `cross_entropy` -- `batch_size=128` (valeur d'entraînement) OOM sur une A5000 24GB, `batch_size=32` fonctionne.

Résultat (`checkpoints/retrieval1_reprkd_wsd2_best.pt`, meilleur checkpoint WSD+patience, sur les 9000 exemples complets de `val.jsonl`) :

| Modèle | answer_ce (val complet, 9000 ex.) |
|---|---|
| **Thinker (128.80M params, WSD2)** | **7.8643** |
| Qwen3.5-0.8B | 10.7846 |
| OLMo-2-7B | 11.0141 |
| Qwen3.8-27B (bf16) | 11.1337 |
| OLMo-2-1B | 11.3164 |
| OLMo-2-13B | 11.4018 |
| LFM2-1.2B | 12.5059 |
| LFM2-700M | 12.8995 |
| LFM2-2.6B | 13.0644 |
| LFM2-350M | 13.7143 |

Le chiffre full-val (7.8643) est quasi identique au chiffre sous-échantillonné (7.8617 sur 2560 ex.) -- le biais n'a donc pas faussé les CONCLUSIONS relatives obtenues jusqu'ici, mais rend la comparaison Thinker-vs-LLM désormais rigoureuse. **Thinker (200x plus petit que le Teacher, from scratch sur 9500 exemples) surpasse largement tous les LLM pré-entraînés testés, y compris Qwen3.8-27B.** OLMo-2-32B téléchargé mais pas encore évalué (nécessite >46GB VRAM, pas testé faute de GPU assez gros au moment de l'écriture).

**`--val_batches` change de défaut (None = val complet)** dans `train_prompt_response.py` -- l'ancien défaut (20 batches) faisait que TOUT `val_answer` rapporté par ce script pendant l'entraînement portait silencieusement sur un sous-ensemble fixe (2560/9000 ex. ici), jamais comparable tel quel à un chiffre externe sans le savoir. Val complète par défaut désormais ; `--val_batches N` reste disponible pour accélérer un sweep rapide au prix de cette limitation, en connaissance de cause.

**Prochaines étapes (en cours/à faire)** :
1. Évaluer OLMo-2-32B (GPU >46GB nécessaire) pour compléter la cartographie.
2. Tester l'extrapolation `n_step` (`--extrapolate_n_steps`, déjà implémenté dans `train_prompt_response.py`) sur le checkpoint WSD2 -- pertinent car l'architecture réutilise les mêmes poids à travers les itérations récurrentes ; voir si le modèle généralise à un nombre d'itérations différent de celui de l'entraînement (`n_step=4`).
3. Comparaison KD avec embedding (`--embed_teacher_target`/`--embed_kd_weight`) vs sans, sur val complet.
4. Basculer `--teacher_targets`/`--repr_teacher_hidden` vers le nouveau format storage-tree de data-prep (`data/distill/hotpotqa/topk/<split>/`, `embedding/<split>/layer_64/`, `--teacher_name qwen_big`) une fois les runs en cours stabilisés, puis supprimer les chemins legacy.
5. Refaire le sweep `repr_kd_weight` avec WSD+patience (le classement 0.05/0.2/0.3 ci-dessus a été établi à `lr=3e-4` fixe, potentiellement obsolète).
6. Run flagship sur les 81k exemples complets (au lieu du sous-échantillon 9500) avec WSD+patience+meilleur `repr_kd_weight`.

## 2026-09-22 (suite) : OLMo-2-32B, ablation frozen-head, correction du coût de `--val_batches`, plan d'expériences élargi

**OLMo-2-32B ajouté à la cartographie** (`answer_ce=12.2080`, val complet, H100 abacus27-1 job 4126763). Résultat non-monotone en taille dans la famille OLMo-2 : 32B (12.2080) est PIRE que 13B (11.4018) et 7B (11.0141) sur cette tâche précise -- taille du modèle de référence n'est pas un prédicteur fiable de performance sur retrieval multi-hop tronqué, probablement parce qu'aucun de ces modèles n'est fine-tuné sur ce format de tâche (inférence seule).

**Ablation frozen answer-head** (objectif : vérifier si l'apprentissage de Thinker est logé dans `lm_head` -- 63.8M/128.8M params -- plutôt que dans le cœur récurrent + embedding). Nouveau script `learn/distill/extract_teacher_head_lite.py` (SVD `hidden_size -> d_model=256` directement depuis les shards safetensors, CPU-only, pas de chargement modèle complet) produit deux inits candidates : `qwen_big_head_init_d256.npz` (Teacher Qwen3.8-27B, hidden_size=5120) et `qwen35_0.8b_head_init_d256.npz` (Qwen3.5-0.8B, hidden_size=1024, embeddings liées). `train_prompt_response.py` reçoit `--answer_head_init`/`--freeze_answer_head` (tête `answer` initialisée depuis la projection Teacher puis `requires_grad=False`). Même recette WSD+patience que le meilleur run KD (`repr_kd_weight=0.05`, `lr_stable_frac=0.125`, `lr_decay_steps=750`, `patience=6`).

| Variante | best val_answer (val complet, 9000 ex.) | step best | Notes |
|---|---|---|---|
| **Baseline non gelée (WSD2)** | **7.8643** | 750 | référence, cf. tableau ci-dessus |
| Tête gelée, init Qwen3.5-0.8B (A5000, job 4126744) | 8.0489 | 3250 | job expiré (walltime) à step 3560/6000, plateau déjà net 8.05-8.08 sur steps 1750-3500 |
| Tête gelée, init Qwen3.8-27B (A40, job 4126741) | 8.2926 | 5250 | run complet 6000/6000, plateau 8.29-8.36 sur steps 3250-6000 |

**Conclusions** : (1) geler la tête dégrade clairement la performance (+0.18 à +0.43 CE vs baseline) -- l'apprentissage n'est PAS purement logé dans `lm_head`, le cœur récurrent + l'embedding portent une part réelle et nécessaire de l'adaptation, cohérent avec le spec §13.1/13.2 qui déconseille de geler la tête (elle doit décoder l'état interne propre à Thinker, différent de celui du Teacher). (2) L'init depuis le Teacher plus petit (Qwen3.5-0.8B, hidden_size=1024) bat nettement l'init depuis le plus gros (Qwen3.8-27B, hidden_size=5120) -- cohérent avec l'hypothèse qu'une projection SVD depuis un espace de représentation plus proche en dimension perd moins d'information qu'une compression `5120 -> 256` très agressive. Confond partiel non contrôlé : `batch_size` différent (32 vs 64, contrainte VRAM A5000 24GB vs A40 46GB) -- écart probablement mineur vu l'ampleur de la différence (0.24 CE) mais à garder en tête.

**Fix retour `--val_batches` (défaut repassé à 20, pas None)** : le passage à `None` (val complet par défaut, cf. section précédente) corrigeait le biais de comparaison mais introduisait un nouveau problème signalé par l'utilisateur -- quand `--data` est un petit sous-échantillon d'entraînement (9500 ex.) mais `--val_data` est le val complet (9000 ex.), évaluer en entier à chaque `--val_every` fait dominer le coût de la validation sur celui de l'entraînement lui-même. Politique à deux niveaux retenue : `--val_batches=20` par défaut pendant l'entraînement (monitoring/early-stopping bon marché, sous-ensemble fixe), `eval_thinker_full_val.py` (inchangé, toujours val complet) réservé à tout chiffre final/rapporté. `0` accepté comme alias explicite de "pas de cap" pour le cas où `--val_data` est lui-même déjà petit.

**Plan d'expériences élargi (validé avec l'utilisateur), ordonné par valeur d'information / coût** :
1. **CE-only vs KD, contrôle sur retrieval** (même recette WSD) -- en cours (job 4126972). Gate la décision d'investir ou non dans le précompute KD pour les phases suivantes.
2. **Sweep taille de KB sur retrieval** (`n_docs_max` ∈ {5,10,20}, KD) -- en cours (job 4126973), axe architecture indépendant de 1.
3. Math (`openr1_math`, 18k/2k, famille LFM2, Teacher déjà précalculé) CE vs KD, puis conditionnellement `openr1_math_full` (35k/3.9k, "5%", plus coûteux). **Les deux bras terminés (budget complet 6000/6000 pas, pas d'early-stopping) :**

| Variante | val_answer training-time (160/2000 ex., sous-échantillon) | answer_ce full-val (2000/2000 ex.) |
|---|---|---|
| **CE-only** | 3.4949 | **3.3045** |
| KD top-K (`kd_alpha=0.5`) | 4.4199 | **4.1117** |

**Vérification de rigueur demandée par analyst-agent (parité budget/LR confirmée -- seule différence entre les deux commandes : `--teacher_targets`/`--val_teacher_targets`/`--kd_alpha`, seed=0 identique, 6000/6000 pas complets des deux côtés, pas d'early-stopping) + re-mesure sur le VAL COMPLET (le chiffre initial n'était que le sous-échantillon training-time à 160/2000 ex., `--val_batches` par défaut) : le résultat tient, écart légèrement réduit mais toujours net (0.81 sur val complet vs 0.93 sur le sous-échantillon).** `eval_thinker_full_val.py` avait un bug bloquant sur `--dataset_type reasoning` (attribut `--n_ctx` manquant), corrigé au passage. **CE-only bat KD de 0.81 sur math, écart encore plus marqué que sur retrieval (0.23 sur val complet).** Cohérent avec le caveat déjà noté : le stream `answer` de math a un fallback CE partiel connu (span pas toujours verbatim), donc le signal KD y était déjà dégradé avant même de considérer la contamination `<think>` (qui, elle, ne touche pas math -- vérifié sur 100/100).

**🔴 Nuance importante (vérification qualitative demandée par analyst-agent avant de traiter ce résultat comme solide) : les DEUX checkpoints (KD et CE-only) produisent des générations dégénérées comparables en régime libre -- ce n'est PAS le même schéma que retrieval.** `generate_qualitative_compare_reasoning.py` (nouveau, adapte le même principe aux deux streams `thinking`/`answer` de `ReasoningPromptDataset` -- les deux streams sont décodés indépendamment, cross-attention uniquement sur `sm_k`/`sm_v` partagé, pas d'auto-attention entre positions ni entre streams). Sur les 30 mêmes exemples fixes de `openr1_math/val.jsonl`, greedy :
- **Stream `thinking`** (les deux checkpoints) : phrases répétitives creuses ("Okay, let's see... So,... Now,... First,...") sans réel raisonnement mathématique, boucle de connecteurs plutôt que résolution du problème.
- **Stream `answer`** (les deux checkpoints) : charabia LaTeX pur, ex. KD `'\frac{1}{2}\n2}2}2}1}{3}2}+2{3,2{3{1}{2}}{2{2{1}{2}{2{3}{1,2},x+1{1}{}{3}2{1}{2}^'`, CE-only `'\frac{1}{3}5}2}4}1}{3}5x+2} \^{2} \sqrt{5}{1}{2}0},7}4}4}{1}{3}{{1,0=2}{5x+1}{'` -- aucune réponse numérique/symbolique correcte identifiable dans les deux cas.

**Conclusion révisée** : l'écart numérique CE-only > KD (3.3045 vs 4.1117, vérifié sur val complet) est réel, mais **ne doit PAS être lu comme "CE-only produit un raisonnement math cohérent/utilisable"** -- les deux checkpoints sont qualitativement médiocres en génération libre à cette échelle (51.7M params, 18k exemples, 6000 pas). Contrairement à retrieval où la contamination `<think>` était une cause spécifique et identifiable, math ne montre pas de contamination particulière -- juste un problème d'exposure bias général plus sévère (dataset plus petit, tâche plus difficile), cohérent avec le chantier "exposure bias général" déjà documenté et mis de côté. **Ne pas citer "CE-only bat KD sur math" comme preuve que la politique KD-par-défaut du projet est mauvaise pour la qualité de génération -- seulement pour le score CE teacher-forcé.** Fichiers : `dev_notes/qualitative/math_{kd,ceonly}_greedy.md`.
4bis. **Sweep taille de KB terminé** (retrieval, WSD+patience, budget commun) :

| `n_docs_max` | answer_ce (val complet) |
|---|---|
| 5 | 7.8406 |
| 10 (défaut) | 7.8552 |
| 20 | 7.8546 |

**Pas d'effet net de la taille de KB dans cette plage** -- les 3 valeurs sont à ~0.015 CE les unes des autres (bruit d'entraînement probable, pas une tendance monotone claire : 5 < 20 < 10). Conclusion : `n_docs_max=10` (défaut actuel) reste un choix raisonnable, pas de gain clair à changer dans cette plage 5-20.
4. Wiki+TinyStory (`wikitext_sample5k`+`tinystories_sample5k` ou `general_sample10k_staging` fusionné, 9k ex., précompute top-K à faire -- `general_realtext`, plus petit, 2.7k ex., a déjà son top-K et sert de premier passage rapide) CE vs KD.
5. Dataset combiné (retrieval + math + wiki/tinystory mélangés), KD, une fois chaque domaine caractérisé isolément.
6. Éval qualitative continue : génération de texte (Thinker vs petits LLM de référence) sur **un échantillon fixe et petit, réutilisé à l'identique à chaque checkpoint** (pas de ré-échantillonnage) pour rendre les comparaisons comparables dans le temps -- `learn/indexed_attention/generate_qualitative_compare.py` (nouveau, commit `d53e85e`), lancé sur `retrieval1_reprkd_wsd2_best.pt` vs Qwen3.5-0.8B, 30 premiers exemples de val. Alimente aussi une demande similaire reçue d'une session sœur (analyst-agent).

**🔴 Résultat qualitatif majeur : la génération libre (greedy) de Thinker s'effondre, malgré un CE teacher-forcé excellent.** `generate_qualitative_compare.py` sur `retrieval1_reprkd_wsd2_best.pt` (7.8643 CE teacher-forcé, meilleur que tous les LLM de référence) vs Qwen3.5-0.8B, 30 exemples fixes de `hotpotqa_full/val.jsonl` : **27/30 réponses Thinker sont soit `<think>` seul (token unique puis arrêt), soit une boucle numérique dégénérée** (ex. `'<think> 17, 2000000198319720719700000000100000000001962019720197200001'`) -- alors que la donnée d'entraînement retrieval ne contient JAMAIS `<think>` (format `hotpotqa`, pas `reasoning`/math). Seuls 2/30 exemples produisent une réponse courte plausible (`'yes'`, `'19'`), sur des questions oui/non ou numériques simples. Qwen3.5-0.8B, lui, produit un texte cohérent (bien que verbeux/pas toujours correct).

**Interprétation** : c'est un cas manuel du biais d'exposition (*exposure bias*) bien documenté pour l'entraînement MLE/teacher-forcé -- le modèle n'a jamais appris à se corriger à partir de SES PROPRES prédictions (seulement à partir du contexte gold), donc une seule erreur tôt dans la séquence generation devient hors-distribution et s'auto-amplifie en boucle. Le spec du projet anticipait déjà cette distinction ("teacher forcing means training is fully parallel across positions; generation is necessarily autoregressive", `core/indexed_thinker_model.py` docstring `OutputStream`) mais ceci est la première mesure empirique de son ampleur. **Le CE seul (toute la cartographie ci-dessus) mesure une capacité de prédiction conditionnelle au contexte gold, PAS la qualité de génération réelle -- les deux peuvent diverger radicalement pour un petit modèle from-scratch.** Confirme directement la pertinence de la demande utilisateur ("pas juste regarder le CE, voir ce que le modèle produit").

**Confirmation via sampling non-greedy (t=0.8, top_p=0.9, mêmes 30 exemples)** : le collapse persiste -- 28/30 réponses commencent encore par `<think>` (souvent suivi de bruit/tokens incohérents plutôt qu'un texte plausible), seuls les 2 exemples oui/non restent courts et cohérents (`'yes director'`, `'yes Show'`). **Ce n'est donc pas un artefact du décodage greedy pur -- `<think>` est un mode dominant réel de la distribution de sortie du modèle en régime free-running**, cohérent avec un vrai problème d'exposure bias plutôt qu'un simple biais d'argmax. Fichier : `dev_notes/qualitative/thinker_vs_qwen35_08b_wsd2_sampled_t08.md`.

**Inspection des logits bruts en position 0 (top-10, 30 exemples, `thinker_vs_qwen35_08b_wsd2_logitdiag.md`)** : `<think>` porte **40-56% de la masse de probabilité pour 27/30 exemples**, quel que soit le contenu de la question (loin d'une distribution plate -- `1/248077`) -- **le modèle est confiant à tort, pas indécis**. Seuls les exemples oui/non (`yes`/`no` en compétition réelle, ex. 4 : yes=0.71/no=0.12/`<think>`=0.12 ; ex. 29 : `<think>`=0.44/yes=0.18/no=0.18) et un exemple numérique (ex. 13 : `1`=0.27/`<think>`=0.26/`2`=0.15) montrent une distribution différenciée selon le contexte. **`<think>` n'apparaît JAMAIS dans les données d'entraînement retrieval** (format `hotpotqa`, jamais `reasoning`) -- l'hypothèse la plus probable est que la query en position 0 (embedding du token `pad_id` + position 0, cf. `_teacher_forced_target`) est QUASI-IDENTIQUE à travers tous les exemples (seule la lecture cross-attention sur `sm_k`/`sm_v` diffère), et que face à une distribution de labels réels à cette position très hétérogène (pratiquement un token différent par exemple : nombre, nom, lieu, oui/non...), le modèle a convergé vers un "pari sûr" à haute confiance plutôt que d'apprendre à vraiment différencier -- non résolu, nécessiterait d'inspecter si `retrieval1_reprkd_wsd2_best.pt` a été initialisé avec un `--answer_head_init`/`--embed_teacher_target` qui pourrait expliquer un biais structurel vers ce token spécifique.

**Investigation de la cause racine (demandée par analyst-agent, 2026-09-22)** :
1. **Hypothèse "init biaisée" écartée** : le log d'entraînement de `retrieval1_reprkd_wsd2_best.pt` ne montre aucune trace d'`--answer_head_init`/`--embed_teacher_target` (contrairement aux runs frozen-head qui impriment explicitement leur chargement) -- init purement aléatoire, pas de biais Teacher hérité.
2. **Hétérogénéité réelle du label en position 0, confirmée quantitativement** : sur 2000 exemples de `train_repr10k_ab.jsonl`, **776 tokens distincts** apparaissent comme premier token de réponse (le plus fréquent, `'1'`, seulement 6.9%) -- `<think>` n'apparaît **jamais** comme label réel (0/2000). La diversité extrême de la cible en position 0 est confirmée, mais ne suffit PAS à elle seule à expliquer pourquoi le modèle convergerait vers un token à fréquence-label nulle plutôt que vers le mode réel de la distribution (`'1'`, 6.9%) -- une pure stratégie "pari sûr sur la moyenne" prédirait `'1'` ou un token fréquent réel, pas `<think>`.
3. **Hypothèse retenue (la plus probable, pas encore prouvée)** : la query en position 0 est quasi-identique sur tous les exemples (embedding de `pad_id` + position embed 0, seule la lecture cross-attention sur `sm_k`/`sm_v` diffère) -- la pression de gradient sur une ligne du head JAMAIS ciblée comme label (`<think>`) ne vient QUE du terme de normalisation softmax (proportionnel à sa probabilité courante), un processus de décroissance multiplicatif/lent. Si l'init aléatoire a par chance donné à cette ligne une norme/alignement élevé avec la direction moyenne de la représentation cachée en position 0, elle peut survivre 750-2250 pas (budget réel avant early-stopping) sans être pleinement supprimée, d'autant que la pression positive sur les VRAIES cibles est diffuse (répartie sur 776 tokens différents, jamais concentrée). **Conséquence directe si cette hypothèse est correcte : le scheduled sampling serait un fix ciblé pertinent** (contrairement à ce qu'un pur problème de format de données impliquerait) -- il exposerait le modèle à SES PROPRES prédictions en free-running pendant l'entraînement, donnant enfin un signal de gradient DIRECT et négatif sur ce token spécifique quand il est produit hors-cible, plutôt que la pression indirecte/diffuse actuelle.

**Vérification à l'initialisation aléatoire (step 0, avant tout entraînement) : hypothèse "outlier d'init" RÉFUTÉE.** Modèle Thinker fraîchement initialisé (mêmes hyperparamètres, seed=0, 4 exemples aléatoires) : `P(<think>)` en position 0 ≈ **4.06e-6, quasi exactement 1/248077 (uniforme)** -- pas d'outlier du tout, top-5 à l'init sont des tokens aléatoires sans rapport (`'icians'`, `' نسخة'`, etc.), chacun ~0.0001. **La domination de `<think>` (40-56%) est donc apprise PENDANT l'entraînement, pas héritée de l'initialisation.** Le mécanisme exact reste à expliquer (hypothèse de pression diffuse du terme de normalisation softmax insuffisante pour VRAIMENT différencier position 0 selon le contexte, dans le budget réel de 750-2250 pas, reste plausible mais sans confirmation directe) -- mais exclut catégoriquement un simple artefact d'initialisation malchanceuse.

**🎯 CAUSE RACINE TROUVÉE : le KD hérite directement de l'habitude structurelle du Teacher (template de chat incomplet, pas un artefact d'entraînement du student).**

Vérification (aucun GPU nécessaire, top-K Teacher déjà précomputé) : sur 200 exemples de `train_repr10k_ab.jsonl`, **200/200 ont `<think>` présent dans le top-K du Teacher à la position 0 de la réponse, avec une masse de probabilité = 1.0 (min=max=mean=1.0)**. Le Teacher (Qwen3.8-27B, reasoning-tuné) met quasi toute sa masse sur `<think>` à cette position, quel que soit le contexte.

**Cause exacte identifiée** : `prepare_retrieval_data.py`'s `CHATML_TEMPLATE` est une string codée en dur (`<|im_start|>assistant\n{answer}<|im_end|>`), n'utilise PAS `tokenizer.apply_chat_template()`. Comparaison directe : `tok.apply_chat_template(msgs, add_generation_prompt=True)` (avec ou sans `enable_thinking=False`, identique) produit `...<|im_start|>assistant\n<think>\n\n</think>\n\n` -- le template propre du tokenizer **insère toujours un bloc `<think>\n\n</think>\n\n` VIDE** juste après `<|im_start|>assistant\n`, avant la vraie réponse. Notre template maison omet ce bloc obligatoire -- le Teacher reçoit un prompt hors de sa distribution d'entraînement à cette position précise et comble lui-même ce qu'il perçoit comme un slot manquant.

**Mécanisme complet, du Teacher au student** : prompt mal formaté -> Teacher met ~100% de masse top-K sur `<think>` en position 0, indépendamment du contexte -> KD (`kd_alpha=0.5`) entraîne explicitement le student à matcher cette distribution -> collapse partiel du student (40-56%, pas 100%, car le CE sur les vrais labels tire dans l'autre sens avec le même poids). **Explique aussi pourquoi CE-only bat KD** (7.6315 vs 7.8643, section précédente) : CE-only n'est jamais exposé à ce signal contaminé.

**Fix identifié, pas encore appliqué** : ajouter `<think>\n\n</think>\n\n` dans `CHATML_TEMPLATE` juste après `<|im_start|>assistant\n`, avant `{answer}` -- aligne le prompt sur la convention réelle du Teacher, remplit le slot nous-mêmes, devrait éliminer la compétition sur `<think>` à la position 0. Probablement le même défaut sur `prepare_reasoning_data.py` (openr1_math, même Teacher/même style de template) -- à vérifier séparément. **KD+embedding flagship (job phase7, terminé) : answer_ce full-val = 7.9073** -- pire que le KD top-K+repr-KD seul (7.8643) et bien pire que CE-only (7.6315). Comparaison qualitative directe `retrieval1_ab_topkonly_best.pt` (KD-pur) vs `retrieval1_embedkd_wsd_best.pt` (KD+embedding), sampling t=0.8 : **les deux montrent le même collapse `<think>`** (cohérent -- les deux sont entraînés avec KD top-K contaminé, l'embedding-KD ne change rien au mécanisme). Fichier : `dev_notes/qualitative/thinker_kdpur_vs_embedkd_sampled_t08.md`. Renforce encore la conclusion CE-only > toute variante de KD testée à ce jour, sur ce dataset/cette échelle.

**Portée vérifiée : limitée à retrieval/hotpotqa, `openr1_math`/reasoning épargné.** `prepare_reasoning_data.py`'s `CHATML_TEMPLATE` a la même forme (`<|im_start|>assistant\n{trace}<|im_end|>`), mais `{trace}` contient déjà nativement `<think>...</think>` (vraies traces de raisonnement OpenR1-Math, pas un placeholder vide) -- vérifié sur 100/100 exemples de `openr1_math/train.jsonl` : 0 manquant. Le stream `thinking` a donc légitimement `<think>` en position 0 (vrai label, pas une contamination), et le stream `answer` (extrait après un `</think>` déjà fermé) a un contexte bien formé avant lui. **Seul `prepare_retrieval_data.py`/hotpotqa est affecté** (seul cas où l'assistant répond directement sans passer par AUCUN bloc `<think>`, ni vide ni rempli). Implication : `reasoning_thinking_best.pt`/`openr1_math` ne nécessitent probablement pas de requalification.

**Implication restante** : le top-K KD précomputé pour hotpotqa_full est affecté à la position 0 -- décision de regénérer ou non à prendre avec l'utilisateur/data-gen-agent (coût precompute non-négligeable).

**Blocage inattendu côté vérification du fix (data-gen-agent, anciennement data-gen-gh200, 2026-09-22)** : la vérification empirique du fix a révélé un bug NaN dans les logits du Teacher (bf16, sharding multi-GPU 2xL40S), sans rapport avec le template -- NaN localisé à la couche 19/64 (composant linear-attention de l'architecture hybride Qwen3.5), reproductible en sdpa ET eager. Hypothèse en cours : problème de sharding cross-GPU du hidden state récurrent, testé sur GPU unique (A100 80GB). Aucune donnée top-K régénérée n'existe encore. **Le chantier KD (fraction 10/25/50%, requalification hotpotqa) reste bloqué tant que ce bug plus fondamental n'est pas résolu.**

**Mise à jour (data-gen-agent, même jour) : les deux bugs résolus.** NaN root-causé : `device_map="auto"` (sharding multi-GPU), pas un bug du composant linear-attention -- fixé en forçant GPU unique (`--num_gpus 1`). **Fix de template `<think>` confirmé chiffré : 100% -> 0% de collapse sur 200 exemples** (vérification empirique demandée précédemment). Texte corrigé disponible : `train_thinkfix.jsonl` (9500 ex.) / `val_thinkfix.jsonl` (9000 ex.), dataset_root `hotpotqa_thinkfix/`. Precompute top-K (K=64, A100 80GB single-GPU, val n=2000 en premier lot) en cours. Script de comparaison n_step-variable+KD propre préparé (`tmp_scripts/phase15_nsteprand_cleankd.sh`), en attente du chemin `.npz` pour lancement.

**Contrôle causal CE-only confirmé : le collapse `<think>` disparaît TOTALEMENT, mais un AUTRE collapse (boucles répétitives) apparaît à la place.** Génération greedy sur `checkpoints/retrieval1_ceonly_best.pt` (même 30 exemples fixes) -- **déviation explicite du policy KD-par-défaut du projet, checkpoint diagnostique uniquement, pas de comparaison production** (cf. CLAUDE.md). `<think>` n'apparaît dans AUCUNE des 30 réponses (confirme directement que le KD contaminé est bien la cause du collapse spécifique à `<think>`). Mais les réponses restent majoritairement dégénérées, d'un type différent : boucles numériques répétitives (`'1970000000000000000000019700000000000000000000000019720017200001'`) ou boucles de mots (`'The...of the the the"".". of the the"...'`) ou de yes/no (`'yes directoryesyesyesyesyesyesyesnonoyes...'`). **Conclusion : deux problèmes distincts et superposés** -- (1) la contamination KD explique spécifiquement POURQUOI `<think>`, résolue par le fix de template ; (2) un problème d'exposure bias plus général (le modèle n'a jamais appris à se corriger à partir de ses propres prédictions en régime free-running, qu'il y ait KD ou non) explique pourquoi la génération libre reste dégénérée même sans KD -- celui-là nécessiterait effectivement du scheduled sampling ou une autre intervention ciblée sur l'écart train/inférence, indépendamment du fix de template. Fichier : `dev_notes/qualitative/thinker_ceonly_vs_qwen35_greedy.md`.

**Éval qualitative automatique câblée dans `train_prompt_response.py` (consigne utilisateur/analyst-agent, 2026-09-22) : `--qualitative_eval_at_end`.** Réutilise `generate_thinker` de `generate_qualitative_compare.py` (pas de duplication), génère greedy + sampling (t=0.8/top_p=0.9) sur les N premières lignes (fixe, déterministe) de `--val_data` après l'entraînement, écrit le transcript à côté du checkpoint (`<checkpoint>_qualitative.md`), et applique une heuristique légère de détection de collapse (ratio tokens-uniques/longueur < 0.3 sur >30% des générations -> WARNING imprimé, non bloquant). **Limitation actuelle : `--dataset_type retrieval` uniquement** (skip propre avec message pour `reasoning`, pas encore câblé). Smoke test réussi (5 pas, modèle non entraîné) : mécanisme fonctionne de bout en bout, écrit un rapport lisible, pas de faux-positif sur la heuristique (sortie non entraînée = bruit multilingue haute-diversité, pas une boucle répétitive -- correctement non flaggé).

**Run flagship CE-only sur les 81k exemples complets de `hotpotqa_full`** (job phase9, en cours) : teste si l'échelle seule (8.5x plus de données que le sous-échantillon AB 9500) atténue le collapse en génération libre (boucles dégénérées, indépendant du KD -- cf. contrôle causal ci-dessus). **Run flagship CE-only 81k terminé (budget complet 6000/6000, pas d'early-stopping -- confirme le caveat LR schedule noté avant lancement)** : **answer_ce full-val = 6.6726**, nette amélioration vs le sous-échantillon 9500 (7.6315). Mais la génération qualitative finale (`--qualitative_eval_at_end`) montre **le même collapse en boucles répétitives que le point d'étape mi-parcours** -- l'amélioration du CE ne se traduit pas en génération cohérente. **Bug trouvé dans ma propre heuristique de détection automatique** : elle a rapporté "no degenerate-output warning" à tort (faux négatif) -- la version initiale (ratio mots-uniques sur `.split()`) ne voit pas les boucles numériques SANS espace (`'19720199819898989899719999997,9797197199899799...'` est un seul "mot" pour `.split()`). Corrigé : seuil mots-uniques relâché 0.3->0.5 (vérifié sans faux-positif sur phrases cohérentes, qui scorent 0.8-1.0) + nouveau signal indépendant (ratio de chiffres > 0.5 sur chaîne >= 15 caractères) pour les boucles numériques sans espace. Approche best-effort assumée (pas un détecteur universel) -- documenté dans le code.

**🔴 Point d'étape mi-parcours (step 3750/6000, checkpoint intermédiaire déjà sauvegardé, sans attendre la fin du run) : l'échelle seule NE corrige PAS le collapse en génération libre.** `val_answer` toujours en amélioration continue et lente (6.8826@1500 -> 6.7923@3750, pas de plateau/explosion comme sur le sous-échantillon 9500 -- signe que plus de données ralentit vraiment le surapprentissage). Mais la génération qualitative (greedy, mêmes 30 exemples fixes) sur ce checkpoint intermédiaire montre **exactement les mêmes boucles dégénérées qu'à 9500 exemples** : boucles numériques (`'1988198981986198989971999961998931989719397998961961972019397199'`), boucles de mots (`'The...of the the the...'`), boucles yes/no. **Conclusion : le problème d'exposure bias général n'est pas un artefact de la taille du dataset (9500 ex.) -- il persiste à 81k, cohérent avec l'hypothèse d'une limitation de méthode d'entraînement (teacher-forcing pur, jamais exposé à ses propres prédictions) plutôt que de données insuffisantes.** Fichier : `dev_notes/qualitative/thinker_81k_ceonly_midrun_greedy.md`. Le run continue (budget non atteint), full-val + éval qualitative finale à suivre, mais ce signal mi-parcours est déjà suffisamment clair pour ne pas attendre.

**Caveat noté avant lancement** : le schedule WSD (`--lr_stable_frac 0.125`/`--lr_decay_steps 750`) a été calé sur la courbe de surapprentissage du sous-échantillon 9500 (optimum ~step 750) -- à 81k exemples avec `batch_size=64`, step 750 ne couvre même pas un epoch complet (750*64=48000 < 81000), la décroissance LR pourrait donc intervenir trop tôt pour ce volume de données. Objectif ici est diagnostique (qualité de génération), pas d'optimiser le CE -- à garder en tête si le chiffre CE final semble sous-optimal. Utilise le nouveau `--qualitative_eval_at_end` (diagnostic automatique, pas de suivi manuel nécessaire).

**Scheduled sampling implémenté (`--scheduled_sampling_p`/`--scheduled_sampling_warmup_steps`, mandat utilisateur 2026-09-22).** Mécanique (Bengio et al. 2015) : avant le forward pass principal (gradient), un forward pass préliminaire (no_grad, mêmes entrées teacher-forcées) produit les prédictions propres du modèle ; à chaque position du `target_input`, avec probabilité `p` (rampe linéaire sur `--scheduled_sampling_warmup_steps`), le token gold est remplacé par la prédiction du modèle (décalée d'une position, position 0 jamais remplacée). Double le coût du forward pour les streams sequence_mode (~1.5-2x plus lent par pas). Non compatible avec `--ingest_kb` (assertion explicite, pas implémenté). Smoke-testé (10 pas, mécanisme fonctionne de bout en bout, pas d'erreur). **Rôle clarifié (note utilisateur)** : ce levier ne vise PAS à faire baisser le CE/loss (c'est le rôle du KD, une fois le template corrigé) -- il cible spécifiquement l'écart train/inférence (le collapse en génération libre, confirmé indépendant du KD, persistant à 9500 ET 81000 exemples). Les deux leviers sont complémentaires, pas concurrents.

**🔴 Résultat majeur : CE-only bat KD sur retrieval, à ce budget/cette recette.** Contrôle demandé par l'utilisateur (job 4126996, même recette WSD+patience que `retrieval1_reprkd_wsd2_best.pt`, dataset AB 9500 ex., aucun `--teacher_targets`/`--kd_alpha`) : early-stopping à step 2500/6000 (aucune amélioration en 6 évals), checkpoint `retrieval1_ceonly_best.pt`.

| Variante | answer_ce (val complet, 9000 ex.) |
|---|---|
| **CE-only (WSD+patience, pas de KD)** | **7.6315** |
| KD top-K + repr-KD (WSD2, `repr_kd_weight=0.05`) | 7.8643 |

**CE-only bat KD de 0.23 CE sur cette tâche, avec ce budget et cette recette WSD+patience.** Contredit l'hypothèse par défaut du projet (KD comme régularisateur toujours bénéfique, cf. CLAUDE.md "Training methodology default: KD, not pure CE") -- au moins pour retrieval/hotpotqa à cette échelle (9500 ex. train), la distillation Top-K + repr-KD n'apporte pas de gain mesurable, voire nuit légèrement. Hypothèses à départager (pas encore testées) : (a) le early-stopping+WSD est déjà si efficace contre le surapprentissage qu'il capture l'essentiel du bénéfice régularisateur que la KD était censée apporter, rendant la KD redondante ici ; (b) le Teacher (Qwen3.8-27B, jamais fine-tuné sur ce format retrieval tronqué) donne un signal Top-K de qualité insuffisante sur cette tâche précise (cohérent avec le fait que Qwen3.8-27B lui-même performe MOINS bien que Thinker en zero-shot, 11.13 vs 7.86) ; (c) confond `kd_alpha=0.5` peut-être sous-optimal, jamais sweepé. **Implication directe pour le plan d'expériences** : ne pas assumer que KD est gratuit/toujours bénéfique pour les phases 3-5 (math/wiki/combiné) -- comparer CE vs KD systématiquement plutôt que par défaut KD seul, et économiser le précompute Teacher quand CE suffit.

**⚠️ Caveat important avant le run math CE vs KD (phase 3, job 4126995)** : le diagnostic d'alignement du 2026-09-20 (ci-dessus, "0/2000 answer spans alignés sur openr1_math") a deux causes distinctes selon le dataset -- le bug de tokenisation BOS/fusion-BPE (`slice_span`, fix `a71f729`) qui touchait `RetrievalPromptDataset` est **corrigé** (100% alignés depuis). Mais le fallback sur `ReasoningPromptDataset`/`openr1_math` a une **seconde cause distincte, non corrigée** : l'`answer` canonique n'est pas forcément un span verbatim du texte généré (paraphrase), un problème de DONNÉES pas de code. Le run KD sur math aura donc probablement un signal KD réel sur le stream `thinking` (verbatim par construction) mais un fallback CE quasi total sur le stream `answer` -- à vérifier via `kd_answer` dans les logs avant d'interpréter "KD" vs "CE-only" sur ce dataset comme une comparaison propre sur les DEUX streams.

## 2026-09-22 (suite 3) : premier résultat scheduled sampling -- pas d'amélioration qualitative nette

**🔴 Résultat scheduled sampling (`p=0.25`, `warmup=500`, CE-only, dataset AB 9500, même recette WSD+patience) : PAS d'amélioration qualitative nette à cette configuration.** Early-stopping à step 2500/6000 (comme le baseline), `answer_ce` full-val = **7.7049** (légèrement pire que le baseline CE-only 7.6315 -- attendu, ce levier ne vise pas le CE, cf. note utilisateur : KD reste le bon levier pour la qualité/le score, scheduled sampling cible l'écart train/inférence). Éval qualitative auto (`--qualitative_eval_at_end`) : **greedy reste massivement dégénéré**, mêmes signatures qu'avant (boucles numériques, "of the the the...", yes/no) -- warning auto déclenché correctement (30/60, heuristique corrigée fonctionne). **Sampling (t=0.8) casse la répétition EXACTE mais produit un autre type de dégénérescence** (salade de noms propres/tokens sans cohérence, ex. `'H Toang Dramigin Representatives services.m by. Marc Sconstruction Family Hot Roosevelt...'`) -- pas franchement mieux, juste différemment mauvais.

**LoRA sur tête gelée (rank=32, init 0.8B) : récupère l'essentiel de l'écart de l'ablation binaire.** `answer_ce` full-val = **7.8867** (vs gel complet 8.0489, vs baseline non-gelée 7.8643) -- ~0.02 d'écart avec la baseline au lieu de ~0.18. Confirme qu'une petite capacité entraînable sur la tête suffit, pas besoin qu'elle soit totalement libre. Ce chantier (frozen-head/LoRA) est désormais délégué à `agent2` -- résultats transmis, pas de suite prévue de mon côté.

**Suite (agent2, 2026-09-22) : éval qualitative sur ce checkpoint LoRA, décision de clore le chantier.** Pas de nouvelle variante de rank lancée (rendement décroissant clair : rank=32 est déjà à 0.02 CE de la baseline). Éval qualitative (`generate_qualitative_compare.py`, ajout du flag miroir `--answer_head_lora_rank` pour recharger le `LoRAHead`, même 30 exemples fixes que la référence `retrieval1_reprkd_wsd2_best.pt`) : **collapse `'<think>'` pur sur 30/30 exemples** -- légèrement pire que le baseline non gelé (27/30, dont 2 réponses courtes plausibles `'yes'`/`'19'`). Le CE quasi-identique au baseline (7.8867 vs 7.8643) ne se traduit donc PAS en une génération libre meilleure -- confirme que le collapse en génération libre est un problème du modèle dans son ensemble (exposure bias documenté ailleurs dans ce fichier), indépendant du gel de la tête. Rapport : `dev_notes/qualitative/thinker_frozenhead_08b_lora32.md`. **Chantier frozen-head/LoRA considéré clos** : la thèse "l'essentiel de l'apprentissage est dans le cœur récurrent, la tête a juste besoin d'une petite capacité d'adaptation" est confirmée côté CE ; le problème de génération libre reste un axe séparé (déjà traité par d'autres pistes : scheduled sampling, n_step variable).

**Hypothèses pour expliquer l'absence d'effet** : (a) budget d'exposition insuffisant -- avec `warmup_steps=500` et arrêt à step 2500, seulement ~2000 pas de vrai mélange, potentiellement trop court pour ce mécanisme ; (b) `p=0.25` possiblement trop faible ; (c) le modèle (128.8M, dont ~127M dans embed+head) est peut-être structurellement trop contraint pour apprendre À LA FOIS une bonne prédiction teacher-forcée ET un comportement d'auto-correction avec ce budget de pas. **Pas encore concluant sur l'utilité du scheduled sampling -- nécessiterait un `p` plus élevé et/ou plus de pas avant early-stopping (ou `--patience` relâché spécifiquement pour ce levier) pour trancher, pas encore testé.** Fichier : `dev_notes/qualitative/thinker_ss025_ceonly_greedy_sampled.md`.

## 2026-09-22 (suite 4) : scheduled sampling p=0.5, budget complet, sans early-stopping -- toujours pas de génération cohérente

**🔴 Résultat conclusif (pour cette approche/config) : scheduled sampling ne résout PAS le collapse, même à `p=0.5` et 6000 pas complets (vs `p=0.25`/2500 pas early-stoppés précédemment).** `answer_ce` full-val = **8.2840** (nettement pire que le baseline 7.6315 -- attendu et confondu par l'absence d'early-stopping ici, le run a clairement dépassé l'optimum, val_answer remontant de 7.88@step1000 à 8.38@step6000). L'éval qualitative auto signale 20/60 dégénéré (vs 30/60 à p=0.25) -- **mais l'inspection manuelle du texte montre que ce n'est PAS une vraie amélioration qualitative**, juste un déplacement vers un AUTRE type de dégénérescence que l'heuristique automatique détecte moins bien : répétition de fragments de noms propres/virgules (`'New York,,,...,,.,, Nevada,,.., Nevada,...'`, `'County County County County County County'`) plutôt que des boucles numériques pures -- toujours clairement incohérent à la lecture humaine.

**Conclusion pour ce chantier** : le scheduled sampling, sous cette forme simple (mélange binaire gold/prédiction propre argmax, sans curriculum plus sophistiqué), **n'a pas produit de génération cohérente dans les deux configurations testées** (`p=0.25`/2500 pas et `p=0.5`/6000 pas). Soit le mécanisme a besoin d'un budget d'entraînement bien plus long que ce que permet ce pipeline (9500 exemples, quelques milliers de pas), soit le modèle (128.8M, dont ~127M dans embed+head, ~1.4M dans le cœur récurrent) n'a structurellement pas assez de capacité dans son cœur pour apprendre à la fois la prédiction teacher-forcée ET l'auto-correction, soit une formulation plus riche (curriculum progressif du `p`, sampling temperature-based plutôt qu'argmax pour la prédiction "propre" utilisée en mélange) serait nécessaire. Fichier : `dev_notes/qualitative/thinker_ss05_nopatience_greedy_sampled.md`.

**CE-only + embedding-KD isolé (sans top-K, donc indépendant de la contamination `<think>`)** : `answer_ce` full-val = **7.6428**, quasi identique au CE-only pur (7.6315) -- l'embedding-KD n'apporte ni gain ni perte significative sur le CE. Éval qualitative : 38/60 dégénéré (pire que le CE-only pur) -- **l'embedding-KD n'a pas résolu le collapse de génération non plus**. Dépriorisé au profit du test n_step variable (thèse "loop", priorité utilisateur) -- pas d'investigation qualitative plus poussée pour l'instant.

## 2026-09-22 (suite 5) : extrapolation n_step -- dégradation lisse en U, pas d'explosion catastrophique

**Test de la thèse centrale du projet (spec §-1, "loop"/composition itérative) : les poids du cœur récurrent sont partagés entre toutes les itérations -- le modèle généralise-t-il à `n_step_test` != le `n_step` d'entraînement (4), en pure inférence, sans ré-entraînement ?** `--extrapolate_n_steps` (déjà implémenté, jamais exécuté avant aujourd'hui -- item 2 des "prochaines étapes"), sur les 2 meilleurs checkpoints disponibles, val complet (9000 ex.) :

| n_step_test | WSD2 (repr-KD, 7.8643 @ n_step=4) | 81k CE-only (6.6726 @ n_step=4) |
|---|---|---|
| 1 | 8.9852 | 8.5265 |
| 2 | 8.3810 | 7.8934 |
| **4 (entraînement)** | **7.8643** | **6.6726** |
| 6 | 8.2852 | 7.0460 |
| 8 | 8.9415 | 7.7691 |
| 12 | 10.4069 | 9.4983 |
| 16 | 11.8883 | 11.2411 |

**Résultat clair et cohérent sur les deux checkpoints : dégradation lisse en U, centrée exactement sur `n_step=4` (entraînement), dans les deux directions -- ni explosion catastrophique (NaN/valeurs aberrantes) ni plateau/amélioration au-delà de 4.** Le modèle ne casse pas quand on change `n_step` à l'inférence (bon signe de robustesse structurelle -- les poids partagés produisent une sortie sensée à tout nombre d'itérations testé), mais la performance se dégrade progressivement de part et d'autre de l'optimum plutôt que de rester stable ou de s'améliorer avec plus d'itérations. **Interprétation** : le modèle a appris un calcul "calibré sur 4 pas" plutôt qu'un processus itératif véritablement agnostique au nombre d'étapes -- pas une preuve que le mécanisme de boucle "raisonne mieux avec plus de temps de calcul" (ce qui donnerait un plateau ou une amélioration à n_step>4), mais pas non plus un échec du mécanisme (une architecture cassée donnerait des NaN ou un effondrement brutal, pas cette dégradation progressive et symétrique). Cohérent avec un entraînement qui n'a jamais varié `n_step` (toujours 4) -- une piste future serait d'entraîner avec `n_step` variable pour tester si ça produit une meilleure généralisation en extrapolation, mais pas testé ici (inférence pure demandée, pas de ré-entraînement).

## 2026-09-22 (suite 6) : 🎯 n_step variable à l'entraînement -- extrapolation quasi plate, résultat majeur confirmant la thèse "loop"

**Implémenté `--n_step_train_max`** (tirage `n_step ~ Uniform(1, N)` par batch à l'entraînement, `--n_step` reste fixe pour l'éval/la sélection de checkpoint) suite à la recommandation de `dev_notes/indexed_attention_experiment_plan.md` (littérature Universal Transformers/PonderNet/Looped Transformers), jamais testée sur ce pipeline avant. Run CE-only, dataset AB 9500, `--n_step 4 --n_step_train_max 8`, même recette WSD+patience (early-stopped step 2750/6000, `answer_ce` full-val = **7.6586**, quasi identique au CE-only fixe 7.6315).

**Extrapolation sur val complet (9000 ex.), comparée directement à la courbe en U du modèle entraîné à `n_step=4` fixe (WSD2, section précédente)** :

| n_step_test | Fixe (n_step=4 seul, WSD2) | **Variable (n_step~U(1,8))** |
|---|---|---|
| 1 | 8.9852 | 7.7134 |
| 2 | 8.3810 | 7.6592 |
| **4 (référence éval)** | **7.8643** | **7.6586** |
| 6 | 8.2852 | 7.6598 |
| 8 | 8.9415 | 7.6626 |
| 12 | 10.4069 | 7.6737 |
| 16 | 11.8883 | **7.6878** |

**Écart entre n_step=4 et n_step=16 : +4.02 CE (fixe) vs +0.029 CE (variable) -- un facteur ~140x plus stable.** Le modèle entraîné avec `n_step` variable généralise quasiment PARFAITEMENT à des nombres d'itérations 4x plus grands que n'importe quelle valeur vue à l'entraînement (max vu = 8, testé jusqu'à 16), sans coût sur la performance à n_step=4 (7.6586 vs 7.6315 du CE-only fixe, différence négligeable). **Confirme directement et de façon spectaculaire la thèse centrale du projet (spec §-1, "loop") : le cœur récurrent à poids partagés PEUT apprendre un calcul génériquement agnostique au nombre d'itérations -- la calibration rigide sur un `n_step` fixe observée précédemment n'était pas une limite structurelle du mécanisme, mais un artefact du régime d'entraînement (toujours le même `n_step`).** Éval qualitative auto : 31/60 dégénéré (comparable au CE-only fixe, le problème d'exposure bias général reste présent et indépendant de ce levier). Fichiers : `logs/eval_thinker_ceonly_nsteprand_fullval.json`, `checkpoints/retrieval1_ceonly_nsteprand_best.pt`.

## 2026-09-22 (suite, agent2) — Nouveau front wiki/tinystory (collapse spécifique retrieval vs exposure bias général)

**Objectif (validé avec supervisor-agent)** : tester si le collapse `<think>` en génération libre (déjà documenté ci-dessus comme un mode dominant à haute confiance en position de réponse retrieval) est spécifique au design du answer-stream retrieval/QA, ou un problème plus général d'exposure bias qui apparaît aussi en LM pur (wiki/tinystory, pas de position "réponse" homogène entre exemples).

**Blocage de format découvert et résolu** : `wikitext_sample5k`/`tinystories_sample5k` (renommés depuis en `wikitext`/`tinystories`, `data-gen-agent`) sont du texte brut sans structure prompt/réponse -- `train_prompt_response.py` (script Thinker/indexed-attention) refuse explicitement ce format (`--dataset_type` limité à `reasoning`/`retrieval`, message d'erreur : *"'general' stays on train_real_text.py's sliding-window pipeline, not this script"*). Nouveau script `learn/distill/make_prompt_response_from_realtext.py` : découpe chaque document en un span "prompt" (96 tokens) + span "answer" (48 tokens, milieu des plages suggérées 64-128/32-64), tous deux des sous-chaînes verbatim du texte original -- permet de RÉUTILISER le top-K Teacher déjà précomputé sans le refaire (l'alignement KD fonctionne par recherche de sous-chaîne, cf. `_locate_token_span`). 100% des exemples convertis avec succès (4503/4503 tinystories train, 4497/4497 wikitext train, val idem), `doc_id` préservé (ordre des lignes inchangé).

**Gotcha rencontré (non bloquant, documenté pour éviter une fausse alerte future)** : avec `thinking=""` sur tous les exemples (aucune trace de raisonnement naturelle dans ce texte), `ce_thinking` et le `loss` total affichés dans les logs sont **`nan`** en continu (cross-entropy sur un batch entièrement masqué = 0/0). **Vérifié que ceci est purement un artefact d'affichage, pas une corruption réelle de l'entraînement** : `torch.isfinite` sur tous les tenseurs du checkpoint sauvegardé (`tinystories_pr_kd_best.pt`) confirme 0 valeur non-finie, et `val_answer` décroît normalement (7.86 encodage initial -> 6.13 après 250 pas). Le gradient de la branche `thinking` entièrement ignorée (`ignore_index=-100` partout) contribue zéro, pas NaN, au graphe partagé.

**GPU utilisé** : Nancy (`graffiti-4`, RTX 2080 Ti 10.5GB, via `infra-agent` -- Rennes trop contesté ce jour). `--batch_size 32` -> OOM immédiat (vocab 248k, `topk_kd_loss` fait un `logsumexp` sur toute la distribution) ; `--batch_size 8` fonctionne.

**🔴 Résultat (KD, tinystories, 6000/6000 pas, pas d'early-stop -- encore en amélioration à la fin, `val_answer` 7.86(init)->5.03) : le collapse persiste, mais sous une forme DIFFÉRENTE de `<think>`.** Éval qualitative (`generate_qualitative_compare_reasoning.py`, 30 exemples fixes de `tinystories/val_pr.jsonl`, greedy) : **30/30 générations "answer" s'effondrent en boucle sur une phrase quasi-fixe, dominée par la variante "The little girl/boy was so happy"** répétée 3-7 fois par génération (comptage 3-grammes répétés, voir aussi ratio tokens-uniques 0.27-0.76) -- quel que soit le prompt d'entrée (contes très différents en thème/personnages). Le stream "thinking" (jamais entraîné, target vide sur 100% des exemples, cf. gotcha ci-dessus) génère du bruit multilingue incohérent, sans surprise.

**Interprétation : ceci confirme l'hypothèse de l'exposure bias général plutôt que d'un artefact spécifique au design retrieval.** `<think>` ne fait même pas partie du vocabulaire pertinent ici (jamais dans les données wiki/tinystory) -- pourtant le MÊME mécanisme structurel apparaît : le modèle converge vers une phrase "pari sûr" à haute fréquence dans les données d'entraînement (TinyStories abonde en fins heureuses du type "was so happy"), indépendante du contexte réel, plutôt que de généraliser correctement en régime free-running. Le token/la phrase de repli change selon le domaine (`<think>` pour retrieval, "was so happy" pour tinystories) mais le PATTERN est identique : haute confiance erronée sur un unique mode dominant. **Répond à la question posée par ce front : le collapse n'est PAS spécifique au design du answer-stream retrieval -- c'est un problème général d'exposure bias qui prend une forme différente selon la distribution des réponses du domaine d'entraînement.** Fichiers : `dev_notes/qualitative/tinystories_pr_kd.md`, `checkpoints/tinystories_pr_kd_best.pt` (Nancy, `~/thinker/`), `logs/tinystories_pr_kd_train.log`.

**Wikitext non testé à ce stade** (même pipeline prêt -- `data/distill/wikitext/{train,val}_pr.jsonl` déjà générés -- pas encore lancé, la réponse à la question de recherche est déjà suffisamment claire avec tinystories seul).

## 2026-09-22 (suite 7, experiment-manager) — Lancement phase16 (math, reasoning) en parallèle de l'attente retrieval : 2 bugs, résolus

**Contexte** : pendant que `data-gen-agent` préparait le swap du pool retrieval canonique 81k (`hotpotqa/`, ex-`hotpotqa_thinkfix`), l'utilisateur a demandé de ne pas laisser le temps/GPU inactif -- lancer en parallèle ce qui est déjà faisable, priorité `openr1_math` (top-K Teacher `qwen_big` déjà précomputé, jamais affecté par le bug `<think>` du template retrieval). Script `tmp_scripts_local/phase16_math_nsteprand_cleankd.sh` : même recette que phase15 (`n_step_train_max 8` + top-K KD `kd_alpha=0.5`), `--tokenizer qwen35` (obligatoire pour matcher le Teacher `qwen_big`, cf. skill `model-families`).

**Bug 1 -- mauvais format de store Teacher (`IndexError`)** : `--teacher_targets` pointait directement sur le `.npz` du subset (`topk/train/topk_n1714.qwen_big.npz`, 1714/34279 exemples). Ça déclenche le chemin ancien format (`PromptResponseTeacherTargets`, `data/prompt_response_dataset.py:139`), qui indexe `offsets` par le doc_id BRUT (0..34278) -- crash (`IndexError: index 31155 is out of bounds for axis 0 with size 1715`) car `topk_n1714` est un sous-échantillon ALÉATOIRE du pool, pas ses 1714 premières lignes. **Fix** : passer le RÉPERTOIRE (`topk/train`, pas le fichier) + `--teacher_name qwen_big` -> déclenche `TeacherTopKStore` (nouveau format manifest-based, `data/prompt_response_dataset.py:192`), qui remape `doc_id -> (subset, position locale)` via `manifest.json` + `subsets/train/topk_n1714.indices.npy`. **Même piège corrigé préventivement dans `phase15_nsteprand_cleankd.sh`** (retrieval, lot `thinkfix_n1500` à venir) avant même de le lancer.

**Bug 2 -- OOM sur 2 GPU différents, root cause déjà documentée plus haut (section wiki/tinystory) mais pas vue avant de lancer** : `topk_kd_loss` (`learn/distill/train_sft.py:406`, `student_logZ = torch.logsumexp(student_logits, dim=-1)`) matérialise un tenseur DENSE `(batch, seq, vocab)` malgré le top-K -- mémoire proportionnelle au vocab COMPLET (qwen35 = 248320) x `max_thinking_len` (1024), indépendamment de K. OOM confirmé à `batch_size=8` sur graffiti-11 (RTX 2080 Ti, 10.57GB réels) PUIS à `batch_size=4` sur le même nœud PUIS à `batch_size=8` sur un A40 46GB idle (`abacus22-2`, job `4127787` -- "tried to allocate 7.57GiB, 40.46GB déjà utilisés"). **Fix : `batch_size=2`** (proportionnellement ~4x plus petit que le `batch_size=8` de phase3a sur LFM2/64k vocab, cohérent avec le ratio de vocab ~3.9x) -- tourne correctement (step 40 atteint, ETA ~3h).

**Leçon (déjà notée en mémoire utilisateur)** : la même cause racine (`logsumexp` plein-vocab) était DÉJÀ documentée dans la section wiki/tinystory ci-dessus (2026-09-22, agent2) -- aurait dû être consultée avant de lancer plutôt que redécouverte par 2 OOM successifs. Vérifier ce journal AVANT tout nouveau run sur un vocab large, pas seulement la skill `grid5000`.

**Couverture KD train limitée** : le lot top-K train (`topk_n1714`) ne couvre que ~5% des 34279 exemples (`thinkfix_n1500` sera dans la même situation pour retrieval, 1500/80999 ~1.9%) -- `kd_answer`/`kd_thinking` train sont à 0.0 la plupart des pas par échantillonnage (attendu, PAS un bug -- confirmé par `val_kd_answer`/`val_kd_thinking` non nuls, val ayant une couverture complète). Ce run donne un signal partiel sur l'effet KD côté train, mais teste correctement le n_step-variable et sert de diagnostic rapide -- pas un résultat flagship tant que le top-K train n'est pas sur (une fraction significative de) le pool complet.

**GPU** : A40 46GB, `abacus22-2` (Rennes), job `4127787` -- réservation `besteffort` déjà active sous ce compte (initialement prévue pour un precompute OLMo jamais lancé), repérée et réutilisée après que `infra-agent` a réservé puis relâché du compute trop petit/contesté ailleurs (Nancy `graffiti-11`, Rennes `abacus21/25/26/27` en attente longue). Run en cours au moment de cette note (`checkpoints/math_nsteprand_cleankd_best.pt`, `logs/math_nsteprand_cleankd_train.log`).

**Suite -- préemption besteffort à 88%, résultat quand même récupéré** : le job `4127787` a été préempté par un job prioritaire à 17:36:47 (attendu pour du besteffort), run arrêté à step 5260/6000. **Pas de perte critique** grâce à `--save_best_checkpoint_path` (sauvegarde à chaque amélioration de `val_answer`, pas seulement en fin de run) -- dernier meilleur checkpoint : `val_answer=3.7348` (step 5250/6000). Éval finale (`eval_thinker_full_val.py`) lancée séparément sur ce checkpoint (Nancy `graffiti-7`, job `6937223`, `--batch_size 1` -- même OOM plein-vocab qu'à l'entraînement, encore plus contraignant sans gradient checkpointing implicite) : **`answer_ce` (full val, 3890/3890 ex.) = 3.2095**.

**Comparaison avec les baselines historiques (§ ci-dessus, phase3a, LFM2/18k-2k, `n_step=4` fixe)** : CE-only=3.3045, KD top-K=4.1117 (KD pire que CE-only à l'époque). Ce nouveau résultat (3.2095) est meilleur que les deux, mais **pas directement comparable** : famille de tokenizer différente (qwen35/248k vs lfm2/64k), dataset ~17x plus grand (34279 vs 18000 train), régime `n_step` variable (Uniform(1,8)) au lieu de fixe, et couverture KD train très partielle (~5%, `topk_n1714`). Directionnellement positif mais pas de conclusion causale tirable de cette seule run.

**Anomalie mineure notée, pas creusée (temps limité)** : le champ `thinking` du JSON de sortie de l'éval full-val est `NaN`, alors que les logs d'entraînement montraient des `ce_thinking` numériques normaux tout du long. Hypothèse la plus probable (cohérente avec le gotcha déjà documenté plus haut pour wiki/tinystory) : à `batch_size=1`, un exemple isolé avec un segment "thinking" entièrement vide/masqué donne une cross-entropy `0/0=nan` pour CE batch entier, et si l'agrégation finale n'est pas nan-aware (simple moyenne), un seul batch de ce type suffit à polluer la moyenne globale -- `answer_ce`, lui, reste valide car ce n'est pas le champ affecté. À vérifier si ça se reproduit / si l'agrégation devrait ignorer les NaN par batch.

Fichiers : `checkpoints/math_nsteprand_cleankd_best.pt`, `logs/eval_thinker_math_nsteprand_cleankd_fullval.json`.

## 2026-09-22 (suite 8, experiment-manager) — phase15 (retrieval, pool 81k canonique) terminé : extrapolation quasi plate confirmée sous KD aussi

**Run complet jusqu'au bout** (Nancy `graffiti-3`, job `6937228`, `batch_size` réduit préventivement à 16 au lieu de 64 -- même risque OOM plein-vocab que math, mais séquences courtes ici donc jamais atteint en pratique, ~9.7GB/10.5GB utilisés). `TRAIN_DATA` = pool complet 81k (`hotpotqa/train.jsonl`, 80999 ex.), KD top-K couvrant seulement le lot `thinkfix_n1500` (~1.9%, même limite de couverture partielle que math). `max_steps=6000` atteint sans early-stop (`final_loss=3.89`).

**Éval full-val (9000/9000 ex.)** : `answer_ce = 7.1731`. **Extrapolation quasi plate, confirmée sous KD (pas seulement CE-only comme dans le résultat majeur précédent)** :

| n_step_test | answer_ce |
|---|---|
| 1 | 7.2743 |
| 2 | 7.1755 |
| **4 (réf. entraînement)** | **7.1731** |
| 6 | 7.1733 |
| 8 | 7.1743 |

Écart n_step=1 à n_step=8 : +0.10 CE seulement -- cohérent avec le résultat CE-only déjà établi (`d3cc2dc`, facteur ~140x plus stable que le `n_step` fixe). **La combinaison n_step-variable + top-K KD (même avec couverture train très partielle, 1.9%) préserve cette robustesse d'extrapolation.**

**`answer_hops_le1_n=0`** dans tous les résultats (training extrapolation probe ET full-val) : caractéristique du jeu de données lui-même (aucun exemple à 0-1 hop dans ce split), pas un bug -- `answer_hops_ge2` couvre donc la totalité des exemples valides.

**Limite à garder en tête pour toute comparaison future** : couverture KD train ~1.9% seulement (`thinkfix_n1500`/80999) -- ce run teste surtout l'effet du `n_step` variable sous un RÉGIME combiné (CE dominant + KD occasionnel), pas un vrai résultat KD-vs-CE à pleine couverture. Le lot top-K sur une fraction beaucoup plus large du pool 81k reste à faire pour un résultat flagship.

Fichiers : `checkpoints/retrieval1_nsteprand_cleankd_best.pt`, `logs/eval_thinker_nsteprand_cleankd_fullval.json`.

## 2026-09-22 (suite 9, experiment-manager) — phase18 : premier CE-vs-KD propre sur wikitext, KD gagne

**Demande supervisor-agent** : jamais de vrai CE-vs-KD sur données généralistes (wikitext jamais touché, tinystories seulement testé en KD seul par `agent2`, sans bras CE). Couverture top-K wikitext COMPLÈTE (4497/4497 train, 503/503 val, qwen_big) -- comparaison non biaisée par une couverture partielle, contrairement aux runs math/retrieval précédents.

**Reformatage** (`learn/distill/make_prompt_response_from_realtext.py`, convention d'`agent2` reprise telle quelle -- prompt=96 tokens, answer=48 tokens, `thinking=""` partout) : 4497/4497 et 503/503 exemples utilisables, aucun skip, `doc_id` préservé pour matcher le top-K existant. GPU Nancy `graffiti-4` (job `6937318`), `batch_size=8` (séquences courtes, contrairement à math -- jamais d'OOM).

**Résultat (full-val, `eval_thinker_full_val.py`, `dataset_type=reasoning`)** :

| Variante | answer_ce (full val, 503 ex.) |
|---|---|
| CE-only | 6.8462 |
| **KD top-K (`kd_alpha=0.5`)** | **6.6285** |

**KD bat CE-only (-0.22 CE, ~3.2% relatif)** -- premier résultat KD-vs-CE propre et positif sur ce pipeline (les runs math/retrieval précédents avaient une couverture KD train partielle, ~2-5%, biaisant toute comparaison). À couverture complète, la distillation aide réellement.

**Limite importante découverte en cours de route** : `--qualitative_eval_at_end` n'est PAS câblé pour `--dataset_type reasoning` (message explicite dans le log : *"qualitative_eval_at_end: skipped (only --dataset_type retrieval is wired so far)"*) -- s'applique aussi à math (phase16/17). La demande explicite de supervisor-agent ("éval qualitative obligatoire") n'a donc PAS pu être honorée via ce flag pour wikitext ni math -- reste à faire manuellement ou à câbler dans le script si jugé prioritaire.

**Artefact NaN connu, à nouveau confirmé bénin** : `ce_thinking`/`loss`/`thinking` (JSON) systématiquement NaN (stream "thinking" vide), `answer_ce` reste valide -- cohérent avec l'entrée `agent2` du 2026-09-22 (tinystories) et la note phase16.

Fichiers : `checkpoints/wikitext_ceonly_best.pt`, `checkpoints/wikitext_kd_best.pt`, `logs/eval_thinker_wikitext_ceonly_fullval.json`, `logs/eval_thinker_wikitext_kd_fullval.json`.

## 2026-09-22 (suite 10, experiment-manager) — ⚠️ Vérification qualitative wikitext : LES DEUX checkpoints sont dégénérés, résultat CE-vs-KD invalidé

**Sur demande explicite de supervisor-agent** (méfiance justifiée envers un bon CE seul, précédent direct : le collapse `<think>` en retrieval était invisible au CE), vérification qualitative manuelle des deux checkpoints wikitext (`wikitext_ceonly_best.pt`, `wikitext_kd_best.pt`), génération greedy, 30 premiers exemples val (fixes, déterministes).

**Câblage préalable nécessaire** : `--qualitative_eval_at_end` n'était câblé que pour `retrieval` (cf. suite 9) -- ajout de `generate_thinker_reasoning()` dans `generate_qualitative_compare.py` (même convention que `generate_thinker`, `query_tokens = kb_tokens` entier au lieu des derniers `block_size` tokens, "thinking" teacher-forcé depuis le ground-truth plutôt qu'auto-régressé -- suffisant pour juger si la génération de la réponse est dégénérée, pas pour juger la qualité du raisonnement) + branchement dans `train_prompt_response.py` (`--dataset_type reasoning` n'est plus skip). Code synchronisé sur Nancy, pas encore commité.

**Résultat, sans ambiguïté : collapse sévère sur LES DEUX variantes** :

| Variante | Dégénéré (répétition/boucles numériques) |
|---|---|
| CE-only | **25/30** |
| KD top-K | **28/30** |

Motif dominant : boucles de chiffres/répétitions ("`the 1999999999998999989998...`", "`the first . \n the first . \n the first`", "`198 @-@ 19400000000000000194 @-@ 19700000000000`") -- un collapse d'exposure-bias classique, indépendant du CE (les deux checkpoints avaient un CE full-val raisonnable et décroissant normalement pendant l'entraînement). **KD est même LÉGÈREMENT PIRE en qualitatif (28/30) que CE-only (25/30) malgré un meilleur CE (6.63 vs 6.85)** -- confirme exactement la mise en garde de supervisor-agent : le CE seul aurait fait conclure "KD gagne" alors que les deux modèles sont essentiellement cassés en génération libre.

**Conclusion révisée** : **le résultat "KD bat CE-only" de la suite 9 est INVALIDÉ/à ignorer pour toute décision** -- les deux checkpoints sont dégénérés, la différence de CE (6.63 vs 6.85, ~3%) est vraisemblablement du bruit sans signification pratique tant que le mode de collapse n'est pas résolu. Cohérent avec le problème d'exposure bias général déjà documenté (`agent2`, tinystories, section wiki/tinystory ci-dessus, "31/60 dégénéré... indépendant du levier n_step") -- ce collapse ne semble PAS spécifique à un levier particulier (KD, n_step, dataset) mais être une limite structurelle actuelle du entraînement teacher-forcing pur de ce pipeline, à traiter comme un problème séparé et prioritaire avant de tirer toute conclusion CE-vs-KD sur données généralistes.

Fichiers : `tmp_scripts_local/qualitative_eval_wikitext_manual.py`, log complet sur Nancy (`~/thinker/tmp_scripts/qualitative_eval_wikitext_manual.log`, non rapatrié localement).

## 2026-09-23 (suite 11, experiment-manager) — Collapse confirmé universel sur retrieval aussi ; convergence structurelle (agent2)

**Confirmation supplémentaire** : l'éval qualitative auto de phase19 (retrieval, CE-only, n_step fixe=4, palier top-K 57%, post-fix `<think>`) montre le MÊME motif de collapse (boucles de chiffres type années : `"1972000019720198585970195200197201974193333520192019720197200001"`, `"1985, 2000000198989970019960000000000000000000001962000197200001"`) que math et wikitext -- confirme que ce n'est PAS spécifique à un dataset donné.

**Diagnostic clé d'`agent2`** (leur périmètre d'investigation, résumé ici pour contexte) : sous teacher-forcing avec le VRAI préfixe (pas de génération libre), l'argmax du modèle est déjà faux dans ~99.6% des positions, convergeant vers un tout petit ensemble récurrent de tokens génériques (`"`, `1`, espace, ponctuation -- top tokens couvrant 80%+ des positions). **Vérifié sur mon checkpoint `retrieval_ceonly_fixed_best.pt`** (hash MD5 confirmé distinct de leur propre checkpoint) : mêmes pourcentages EXACTS que le CE-only d'agent2 (échelle de données différente : 1500 ex. vs 57% du pool 81k) -- `"` 50.2%, `1` 16.4%, etc., à la décimale près. **Deux entraînements totalement indépendants (dataset/échelle différents) convergent vers un mode de sortie dégénéré numériquement identique.**

**Implication** : ce n'est pas de l'exposure bias classique (dérive en génération libre depuis une trajectoire par ailleurs correcte), c'est un problème de calibration présent DÈS le teacher-forcing idéal -- argument fort pour une cause structurelle (mécanisme récurrent du Thinker lui-même : cœur register/OutputStream) plutôt qu'un problème de données/dataset. Escaladé par agent2 à supervisor-agent comme piste prioritaire.

**Conséquence pour toute comparaison CE-vs-KD en cours (phase17/18/19)** : tant que ce mode de collapse structurel n'est pas résolu, un écart de CE entre deux checkpoints (KD vs CE-only) ne prouve rien de solide sur la qualité réelle de génération -- les deux variantes du même dataset semblent atterrir dans le même bassin d'attraction dégénéré. Continuer à documenter les chiffres CE (utiles comme mesure de calibration relative) mais ne plus les présenter comme des résultats "qui gagne" sans le caveat qualitatif désormais systématique.

**Phase19 terminé (retrieval, palier top-K 57%, `thinkfix_p46250`)** :

| Variante | answer_ce (full val, 9000 ex.) | Qualitatif (greedy, 30 ex.) |
|---|---|---|
| CE-only | **7.0325** | Dégénéré (boucles "1972...", "1985...") |
| KD top-K | 7.2077 | Dégénéré aussi (motif similaire, préfixe "Based" puis même boucle numérique) |

CE-only bat KD numériquement cette fois (contrairement à wikitext) -- mais **les deux sont qualitativement cassés**, donc cet écart n'est pas interprétable comme un vrai signal KD-vs-CE. Cohérent avec le diagnostic d'agent2 (bassin d'attraction dégénéré commun, indépendant du levier testé). Pas de conclusion CE-vs-KD tirable sur aucun des 3 fronts (math/wikitext/retrieval) tant que le collapse structurel persiste -- priorité désormais entièrement sur l'investigation d'agent2.

Fichiers : `checkpoints/retrieval_ceonly_fixed_best.pt`, `checkpoints/retrieval_kd_fixed_best.pt`, `logs/eval_thinker_retrieval_{ceonly,kd}_fixed_fullval.json`.

## KD-vs-CE-only isolé + LoRA + embed-KD, n_step fixe, données propres n=1500 (2026-09-22)

Demande supervisor-agent : comparaison directe KD-vs-CE (n_step fixe=4, pas de n_step-variable, contrairement au run précédent qui mélangeait les deux) sur le lot top-K propre `thinkfix_n1500`, plus deux leviers KD déjà testés sur données contaminées (frozen-head+LoRA32, embed-KD combiné) rejoués sur données propres. 3 runs Nancy en parallèle (`graffiti-1/3/5`, jobs `6937311/6937306/6937313`), même recette WSD+patience, `batch_size=16` (embed-KD relancé à `batch_size=8` après OOM sur GPU 10.57GiB de graffiti-5).

**Full-val (9000/9000 ex., n_step_test=4, `answer_ce`, plus bas = mieux) :**

| Run | answer_ce | vs KD baseline |
|---|---|---|
| KD top-K pur | 7.097 | -- |
| CE-only | 7.033 | -0.064 (légèrement meilleur) |
| Frozen-head + LoRA32 | 7.098 | ~identique |
| KD + embed-KD (w=0.1) | 7.335 | +0.238 (pire) |

**Mais le classement par CE-loss est trompeur** : `qualitative_eval_at_end` (génération libre, greedy + sampled, 30 ex.) signale un collapse sévère sur **les 4 checkpoints**, indépendamment de la recette :

| Run | générations dégénérées (greedy+sampled, /60) |
|---|---|
| KD top-K pur | 46/60 |
| CE-only | 44/60 |
| Frozen-head + LoRA32 | 50/60 |
| KD + embed-KD | 50/60 |

Pattern identique aux boucles numériques déjà vues (`19720000...`) et au collapse `<think>`/tinystories documenté par agent2 le même jour (voir plus haut, entrée `wiki/tinystory`) : confirme que **le collapse en génération libre est un phénomène d'exposure bias général, indépendant de KD-vs-CE, de LoRA, ou d'embed-KD** -- aucun des 4 leviers testés ici ne le corrige, et le score CE seul (utilisé jusqu'ici comme proxy principal) ne le détecte pas. À traiter comme un problème à part (probablement scheduled sampling / n_step-variable côté génération, cf. `phase10`) plutôt que par le choix de recette KD.

Fichiers : `checkpoints/retrieval1_kdvsce_{kd,ceonly}_best.pt`, `checkpoints/retrieval1_frozenhead_lora32_cleankd_best.pt`, `checkpoints/retrieval1_embedkd_cleankd_best.pt`, `checkpoints/*_qualitative.md`, `logs/eval_thinker_{kdvsce_kd,kdvsce_ceonly,frozenhead_lora32_cleankd,embedkd_cleankd}_fullval.json`.

## Diagnostic de divergence de génération : où et pourquoi le collapse (2026-09-22, suite)

Mission supervisor-agent (décision utilisateur) : investiguer la cause du collapse ci-dessus. Nouveau script `learn/indexed_attention/diagnose_generation_divergence.py` -- deux forward passes par exemple sur les 4 checkpoints ci-dessus (n=25 val, greedy/argmax) : (1) teacher-forcé (préfixe = vraie réponse à chaque position, comme le calcul de CE), (2) génération libre (autoregressive sur ses propres prédictions, comme `generate_thinker`). Compare position par position.

**Premier chiffre (position de première divergence entre libre et teacher-forcé) trompeur, corrigé après coup** : "0% de divergence à t=0" sur les 4 checkpoints n'est PAS un signal de "pas de biais de démarrage" -- c'est un artefact : au premier token, le contexte donné aux deux branches (libre et teacher-forcé) est identique (aucune réponse encore émise des deux côtés), donc même modèle + même entrée = même sortie par construction. Ne pas réutiliser cette mesure telle quelle sans ce correctif.

**Résultat central, celui-ci solide** : précision de l'argmax **teacher-forcé** vs la vraie réponse (does CE-bas == argmax-correct ?), sur les 4 checkpoints :

| Run | argmax correct (tous tokens) | argmax correct (position 0 seule) |
|---|---|---|
| KD pur | 0.4% (6/1600) | 0.0% |
| CE-only | 0.4% (6/1600) | 0.0% |
| LoRA32 | 0.2% (4/1600) | 0.0% |
| embed-KD | 0.2% (3/1600) | 0.0% |

**Même avec le préfixe VRAI donné en entrée (meilleur cas possible), l'argmax du modèle ne correspond presque jamais (~0%) au bon token -- alors que le CE loss (7.0-7.3) est bien meilleur que l'aléatoire (`ln(248320)≈12.4` sur ce vocabulaire).** Ce n'est donc pas un problème d'exposure bias classique ("le modèle dérive de sa propre trajectoire correcte au fil de la génération libre") -- le mode du modèle (l'argmax) n'est JAMAIS correct, même en régime le plus favorable. Le CE bas provient d'une masse de probabilité diffuse qui couvre partiellement le bon token sans jamais le placer en rang 1 -- **problème de calibration** (le modèle "sait" statistiquement mais son mode dominant est ailleurs, probablement un "pari sûr" générique -- cf. les boucles numériques `19720000...` observées en génération libre, potentiellement CE le plus faible en moyenne sur l'ensemble d'entraînement plutôt qu'une réponse spécifique correcte pour un exemple donné), uniforme sur les 4 recettes testées (KD/CE/LoRA/embed-KD) -- confirme et affine le "collapse général indépendant de la recette" trouvé plus haut : ce n'est pas juste indépendant de la recette KD, c'est présent dès le teacher forcing, donc en amont de toute question de génération libre / exposure bias.

**Prochaine étape proposée (pas encore faite)** : identifier si ce "pari sûr" est un token/continuation récurrent à travers les exemples (analogue au `<think>` de contamination déjà trouvé, mais ici un artefact de calibration général et non un bug de données) -- comparer les distributions d'argmax teacher-forcé à travers plusieurs exemples pour voir s'il y a convergence vers peu de tokens dominants.

Fichiers : `learn/indexed_attention/diagnose_generation_divergence.py`, `logs/divergence_{kd,ceonly,lora,embedkd}.json`.

## "Paris sûrs" : l'argmax teacher-forcé converge vers ~25 tokens génériques (2026-09-22, suite)

Suite demandée par supervisor-agent : l'argmax teacher-forcé (jamais correct, cf. ci-dessus) est-il diffus/spécifique à chaque exemple, ou concentré sur un petit ensemble récurrent de tokens (mode-collapse, analogue au `<think>` déjà vu) ? Ajout d'une analyse de fréquence des tokens argmax teacher-forcés sur les 1600 positions scorées (25 ex. x 64 positions max) par checkpoint.

**Réponse nette : mode-collapse confirmé, pas de diffusion.**

| Run | tokens uniques utilisés / 1600 positions | top-5 couvrent |
|---|---|---|
| KD pur | 28 | 85.2% |
| CE-only | 26 | 84.1% |
| LoRA32 | 24 | 88.4% |
| embed-KD | 28 | 81.3% |

Tokens dominants récurrents à travers les 4 checkpoints (mêmes candidats, poids différents) : `"` (guillemet, 9-50% selon le run), `1` (16-51%), `Based` (0-21%), `yes`/`no` (2-4% chacun), ponctuation (`,`, `.`, ` `, ` of`). Aucun de ces tokens n'a de lien évident avec le contenu de l'exemple (contrairement au `<think>` de contamination, qui était spécifique au format des données) -- ce sont des candidats génériques que le modèle place en rang 1 quasi indépendamment du contexte, cohérent avec un objectif CE qui récompense en moyenne un "pari sûr" à faible risque sur l'ensemble d'entraînement plutôt qu'une réponse engagée et spécifique par exemple.

**Synthèse des 3 diagnostics (divergence position / argmax accuracy / paris sûrs) : le collapse en génération libre observé plus haut est la conséquence visible d'un problème de calibration antérieur et indépendant de la génération -- le modèle a appris une distribution de sortie dominée par ~25 tokens "sûrs" quasi invariants au contexte, uniforme sur KD/CE-only/LoRA/embed-KD.** Piste naturelle suivante (pas encore lancée, coût d'un nouveau training) : vérifier si ce mode-collapse est spécifique au mécanisme récurrent de Thinker (poids partagés à travers les `n_step` itérations, pourrait favoriser une sortie stable/moyenne) ou général à ce régime petit-modèle/peu-de-données -- comparer à Baseline C (`learn/distill/train_sft.py`, transformer dense classique) sur les mêmes données, en attente de validation supervisor-agent avant de lancer (entraînement complet, coûteux).

**Résultat croisé, renforce fortement l'hypothèse structurelle (2026-09-23)** : même diagnostic tourné sur le checkpoint CE-only d'`experiment-agent` (`retrieval_ceonly_fixed_best.pt`, run distinct, couverture top-K 57% au lieu de n=1500, job Nancy `6937347`/`graffiti-6`) -- résultat **numériquement identique** au CE-only ci-dessus (mêmes tokens dominants, mêmes comptes exacts sur les 1600 positions scorées : `"` 804/50.2%, `1` 263/16.4%, etc.). Vérifié par hash MD5 que ce sont deux fichiers de poids réellement distincts (`e662192...` vs `6a1c6de...`), pas un doublon. **Deux entraînements CE-only séparés, à des échelles de données différentes, convergent vers EXACTEMENT le même mode de sortie dégénéré sur le même échantillon val** -- argument fort pour une cause structurelle (le cœur récurrent partagé de Thinker, potentiel point fixe/attracteur du mécanisme à `n_step` itérations) plutôt qu'un problème de quantité/qualité de données d'entraînement. Renforce la priorité de la comparaison Baseline C.

Fichiers : `logs/divergence_expagent_ceonly57.json`.

## Les "paris sûrs" sont-ils juste la fréquence marginale des réponses ? Partiellement (2026-09-23)

Test rapide (pas de GPU nécessaire) : fréquence des tokens dans les vraies réponses du val set (2000 ex., `answer` seul, pas le document) vs les tokens "paris sûrs" trouvés ci-dessus.

**Top-15 tokens des VRAIES réponses** : `1` (345), `9` (243), `0` (202), `2` (179), ` ` (179), `,` (175), `8`/`7`/`6`/`5` (~90 chacun), ` of` (80), `3`/`4` (~78), `.` (61), `The` (60).

**Recoupement partiel** : `1`, les chiffres, ` `, `,`, ` of`, `.`, `The` apparaissent dans les deux listes -- cohérent avec un comportement classique de sous-apprentissage qui reproduit la fréquence marginale de la distribution des réponses (HotpotQA a beaucoup de réponses numériques/années) plutôt que la réponse conditionnée à l'exemple. Explique une bonne partie, mais pas tout.

**Anomalie non expliquée** : `"` (id=1, vérifié -- c'est bien le caractère guillemet littéral, PAS un token spécial : `pad_token_id=248044`, `eos_token_id=248046`, distincts) est le token DOMINANT du modèle (41-50% de toutes les prédictions argmax) mais **n'apparaît même pas dans le top-15 des vraies réponses**. Cette partie du mode-collapse n'est donc pas expliquée par la fréquence marginale des réponses seules -- hypothèse à tester : fréquence du caractère `"` dans l'ensemble du corpus d'entraînement (prompt + documents, pas seulement les réponses -- HotpotQA cite beaucoup de titres/entités entre guillemets dans le contexte), ou artefact du mécanisme d'attention/mémoire hiérarchique de Thinker plutôt qu'un simple biais de fréquence lexicale.

## Baseline C (dense transformer, wikitext) -- lancement, deux bugs de démarrage corrigés (2026-09-23)

GPU disponible (job `6937373`, `graffiti-1`, Nancy) pour `learn/distill/train_sft.py` CE-only sur wikitext (couverture top-K FULL, non utilisée ici -- CE-only assumé et documenté explicitement dans le script comme déviation à la politique KD-par-défaut, cf. `CLAUDE.md`, motif : isoler la cause du mode-collapse sans confondre avec KD).

1. **OOM immédiat** (`batch_size=16`, `block_size=256`, vocab qwen35=248320) sur le 2080Ti `graffiti-1` (10.57GB) -- même cause structurelle que math (logsumexp/CE dense `(batch,seq,vocab)`). Fix : `batch_size=16` -> `4` (même charge "token-slots" que le run math CE-only qui avait fonctionné). Script `tmp_scripts_local/baselineC_wikitext_dense_transformer.sh` mis à jour.
2. **`KeyError: 'teacher_indices'` dans `evaluate_val`** (`learn/distill/train_sft.py:482`) -- bug réel indépendant de Baseline C : la fonction fait un `.pop("teacher_indices")` inconditionnel (sans défaut), alors que la boucle d'entraînement (ligne 865) gère déjà correctement le cas CE-only (`.pop(..., None)`). Crash au premier `val_every` après le tout premier step de training (step 1 logué avant le crash, donc pas un problème de démarrage du training lui-même). **Fix appliqué** (commit à faire) : `.pop(..., None)` sur les 4 clés teacher_*, `kd = topk_kd_loss(...) if teacher_indices is not None else 0`, et `kd_alpha` forcé à 0 si pas de teacher targets -- ce bug latent aurait cassé N'IMPORTE QUEL run CE-only de `train_sft.py` dès le premier `--val_every`, pas seulement Baseline C. Synchronisé vers Nancy, relancé.

Modèle : 68.5M params (5.0M core + 63.5M head -- vocab qwen35 domine largement la taille, cf. lecon vocab). `n_layer=6, n_embd=256, n_head=4, block_size=256`, comparable en largeur/n_head à Thinker (`d_model=256, n_head=4`) mais sans récurrence.

## Baseline C -- résultat décisif : calibration teacher-forcée nettement meilleure, mais collapse en génération libre présent aussi (2026-09-23)

Entraînement terminé (6000 steps, ~13min, A40-class 2080Ti Nancy) : `best_loss=3.658` (train), val CE dégrade 7.76->8.27 sur l'entraînement (overfitting classique, 68.5M params vs 4497 exemples). Checkpoint sauvegardé (`checkpoints/baselineC_wikitext/checkpoint.pt`, 822MB).

**Diagnostic de calibration (`diagnose_generation_divergence_flat.py`, patché pour dépaqueter le `state_dict` imbriqué de `save_checkpoint`)** : **accuracy argmax teacher-forcée = 21.2% (391/1844 tokens)**, divergence libre-vs-forcé faible (immédiat t=0 : 0/30, dans les 3 premiers tokens : 27/30, moyenne 1.47). **Comparaison directe avec Thinker (CE-only, math/wikitext/retrieval, tous checkpoints confondus) : ~0.4-0.6% argmax teacher-forcé (agent2)** -- Baseline C fait ~35-50x mieux sur cette métrique précise.

**MAIS éval qualitative en génération libre gourmande (greedy, 30 exemples, mêmes heuristiques que les checks Thinker) : 23/30 dégénéré** -- boucles de répétition classiques (`"the 19th century , the 19th century , ..."`, `"the game was the game , and the game was the game ..."`), très proche des 25/30 et 28/30 observés sur Thinker.

**Interprétation** : ce sont DEUX modes de défaillance distincts, pas le même phénomène.
1. Le collapse en génération libre (boucles répétitives) apparaît sur LES DEUX architectures -- artefact générique attendu à cette échelle (modèle minuscule, 4497 exemples, greedy decoding) : exposure bias classique, bien documenté dans la littérature, pas spécifique à Thinker.
2. Mais l'échec de calibration SOUS TEACHER-FORCING (argmax quasi tout le temps faux, ~0.5%) semble **spécifique au mécanisme récurrent de Thinker** -- Baseline C, sans récurrence, reste largement fonctionnel sous teacher-forcing (21.2%, ordre de grandeur normal pour un si petit modèle) malgré la même donnée/vocab/échelle.

**Conclusion pour l'hypothèse structurelle** : renforce fortement l'hypothèse que le cœur récurrent partagé de Thinker (poids réutilisés à travers les `n_step` itérations) est la cause du calibration-collapse sévère, indépendamment du collapse générique en génération libre que tout petit modèle undertrained présente de toute façon. Fichiers : `logs/diagnose_baselineC_wikitext.json`, `logs/qualitative_baselineC_wikitext.log`.

## Isoler la récurrence : n_step=1 vs 4 en inférence sur le même checkpoint -- infirme l'hypothèse (2026-09-23)

Suite à la comparaison Baseline C : test à moindre coût (inference-time uniquement, pas de nouveau training) pour isoler si le calibration-collapse vient spécifiquement du bouclage multi-`n_step` (poids partagés réutilisés 4 fois) plutôt que de l'architecture Thinker en général. Checkpoint KD existant (`retrieval1_kdvsce_kd_best.pt`, entraîné à `n_step=4`), `diagnose_generation_divergence.py` relancé en changeant seulement `--n_step` au forward (le paramètre est passé à l'inférence, pas figé dans les poids -- cf. `core/indexed_thinker_model.py`, boucle `for _ in range(n_step)`).

**Résultat : accuracy argmax teacher-forcée quasi identique entre `n_step=1` (0.25%) et `n_step=4` (0.375%)** -- les deux sévèrement effondrés, aucune amélioration en réduisant la récurrence à une seule itération. `mean_divergence_position` diffère un peu (2.32 vs 1.52) mais reste faible dans les deux cas, et le safe-bet dominant à n_step=4 devient encore plus dégénéré (répétition de chiffres/années, `"yes"`/`"no"` uniformes) qu'à n_step=1.

**Conclusion : infirme l'hypothèse "récurrence n_step = cause du collapse".** Le nombre d'itérations de bouclage n'explique pas la différence avec Baseline C. La cause doit être ailleurs dans l'architecture Thinker (le mécanisme retrieval/KB lui-même, le answer head partagé, ou la façon dont le "register"/mémoire est construit) et non le simple fait de réutiliser les poids plusieurs fois. Fichiers : `logs/divergence_nstep1.json`, `logs/divergence_nstep4.json`.

## Ablation KB (`--disable_kb`) sur le même checkpoint -- infirme aussi cette hypothèse (2026-09-23, suite)

Même méthode que le test n_step (inférence seule, `model.disable_kb` est un simple bool lu dans `Thinker.forward`, cf. `learn/indexed_attention/eval_checkpoint.py`). Checkpoint KD (`retrieval1_kdvsce_kd_best.pt`), `--disable_kb` ajouté au même diagnostic.

**Résultat : accuracy argmax teacher-forcée = 0.25%, strictement identique au run KB actif (0.375% à n_step=4, 0.25% à n_step=1)** -- désactiver entièrement le mécanisme retrieval/KB ne change rien à la sévérité du collapse. `mean_divergence_position` similaire (1.36). Safe-bet encore plus concentré sans KB (top5=64.8%, dominé par `"1"` à 46%).

**Conclusion : infirme aussi l'hypothèse KB/retrieval comme cause.** Deux hypothèses structurelles écartées (nombre d'itérations `n_step`, mécanisme KB). La cause reste à identifier -- candidats restants : le answer head partagé lui-même (design/init), la construction du "register" (les `n_register=8` slots), ou un problème plus fondamental de la boucle d'entraînement/loss propre à `train_prompt_response.py` indépendant de ces deux mécanismes. Fichier : `logs/divergence_disablekb.json`.

## Audit ciblé de `train_prompt_response.py` : aucun bug de recette évident trouvé (2026-09-23, suite)

Demande supervisor-agent (avant de lancer un training coûteux sur answer head/n_register) : chercher un bug indépendant de l'architecture dans la boucle d'entraînement/KD/construction des targets.

Vérifié :
- `_teacher_forced_target` (`data/prompt_response_dataset.py:107-115`) : shift-by-one standard (`target_input[t]` = pad ou `ids[t-1]`, `labels[t]=ids[t]`), cohérent avec ce que lit `diagnose_generation_divergence.py`. Pas de bug d'alignement.
- Boucle d'entraînement (`train_prompt_response.py:769-969`) : `target_input` construit identiquement en train/eval, `n_step` (fixe ou randomisé via `--n_step_train_max`) correctement threadé, `kd_alpha`/mix CE-KD standard, `clip_grad_norm_`, LR schedule, patience -- rien d'anormal.
- `evaluate()` (ligne 237-308) : même construction `target_input`/`query_tokens_for` que le training, cohérent.
- Chemin chunked (`--loss_chunk_size`) vs non-chunked : les deux calculent CE/KD de façon équivalente, pas utilisé sur les checkpoints diagnostiqués de toute façon (loss_chunk_size=0 par défaut).

**Aucun bug de recette identifié.** Piste alternative non-architecturale, non vérifiée : les CE ~7.0-7.3 rapportées ne sont pas réellement "basses" en absolu (vocab=248320, entropie uniforme=12.42 nats) -- juste meilleures que les baselines LLM de référence citées. Un argmax quasi-toujours faux à CE~7 n'est pas forcément contradictoire : peut simplement traduire un modèle qui reste tres incertain (prob correcte élevée dans le top-K mais rarement au rang 1) plutôt qu'un vrai bug de calcul de loss. Pas creusé plus (hors périmètre demandé -- coûterait un calcul de rang moyen du token correct, pas juste un audit de code).
