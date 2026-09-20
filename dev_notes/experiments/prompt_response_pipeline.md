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
