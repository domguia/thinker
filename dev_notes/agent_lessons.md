# Leçons méthodologiques transversales — à lire avant de lancer une expérience

Document vivant, pas un journal d'expérience (voir `experiment.log.md`/`grid5000_usage.log.md` pour ça) — l'objectif ici est de capturer les leçons **comportementales/méthodologiques** qui se sont répétées assez de fois, sur des chantiers différents, pour valoir une règle générale plutôt qu'une note locale à une seule expérience. Écrit après une session (2026-09-13/14) où plusieurs de ces leçons se sont répétées 4 à 6 fois sur deux chantiers indépendants (Indexed Attention, toy-memory) avant d'être vraiment intégrées.

**Toute session (agent ou humain) qui reprend ce projet devrait lire ce document avant de lancer un balayage d'hyperparamètres ou de conclure quoi que ce soit sur un résultat.**

## 1. Un LR (ou tout hyperparamètre sensible à l'échelle) ne transfère JAMAIS automatiquement à travers un changement d'axe

Vérifié à répétition, sur des axes différents, toujours avec le même schéma (le LR "connu" échoue silencieusement, ressemble à un problème de mécanisme/capacité, et un balayage dédié le résout) :

| Axe changé | LR qui ne transférait pas | Symptôme avant diagnostic |
|---|---|---|
| `batch_size` (64→256) | réutiliser `lr` sans mise à l'échelle linéaire | performance dégradée, lu à tort comme un problème d'architecture |
| `d_model`/largeur | `lr` calibré à petite largeur | échec total à grande échelle (Phase -1, EXP-007) |
| `decouple_kv` (nouveau paramètre `query_v`) | `lr=1,2e-3` hérité du pooling partagé | "le découplage seul ne bat pas le raccourci" — faux, LR périmé |
| `n_latent` (8→32, toy-memory) | `lr` validé à `n_latent=8` | "cumsum a un mur de capacité" — faux, LR mal calé à cette largeur |
| Tâche (`copy`→`cumsum`) | `lr` validé sur `copy` | curriculum bloqué dès l'étape 1 |
| `seq_len` (8→32, même largeur) | `lr=3e-3` validé à `seq_len=8` | une seed diverge après progrès réel — signature d'instabilité, pas de mur |

**Règle** : dès qu'**un seul** de ces axes change (architecture, largeur, batch, tâche, longueur de séquence), considérer le LR précédent comme **non validé**, pas comme "probablement encore bon". Balayer au moins 3-4 valeurs espacées d'un facteur ~2-3× avant toute conclusion — jamais réutiliser une valeur "de mémoire" en la présentant comme acquise.

## 2. Avant d'accepter "limite réelle du mécanisme/de la tâche", épuiser dans l'ordre : budget → LR → confusion de config héritée d'un autre axe

Schéma répété quasi identique sur Indexed Attention (`n_hops=2`) et toy-memory (`cumsum@32`) la même nuit : un résultat qui ressemble à un mur est presque toujours l'un de ces trois, dans cet ordre de probabilité décroissante observée ce soir :
1. **Budget/nombre de pas insuffisant** — vérifier d'abord si la loss/l'accuracy progressait encore à la coupure (gratuit à relire dans les logs existants, pas besoin de relancer).
2. **LR non revalidé** (voir §1).
3. **Un paramètre hérité d'une configuration différente** sans le vouloir (ex. `n_latent` fixé à la valeur cible tout au long d'un curriculum, alors qu'un sweep isolé avait testé une autre valeur par défaut du script).

Ne conclure à un vrai mur qu'après avoir épuisé les trois, et même alors, documenter la lecture comme nuancée si les symptômes sont mixtes (ex. une seed réussit, une diverge, une stagne — trois signatures différentes, pas une seule "capacité insuffisante").

## 3. Ne jamais comparer une accuracy au hasard sur tout le vocabulaire — calculer le hasard conditionnel et les raccourcis triviaux spécifiques à la tâche

La leçon la plus coûteuse de ce projet à ce jour (plusieurs mois de conclusions invalidées, cf. `experiment.log.md` "Contre-expertise") : un plateau à 25-34% avait été lu comme "bien au-dessus du hasard (~1,5-3%)" en comparant au vocabulaire entier, alors que le modèle ne choisit jamais dans tout le vocabulaire — il recopie une valeur de la base de connaissance de l'épisode. La vraie référence (`1/n_facts`) donnait une correspondance exacte avec le plateau observé : le modèle n'avait rien appris.

**Règle, redevenue nécessaire une seconde fois la même nuit** (Phase 2-redo, `n_hops=4` avec `n_distractors=0` faisant saturer le raccourci `non_key` à 100%) : ne jamais citer une accuracy sans son bloc `chance-level`/`margin_over_shortcut` (`learn/indexed_attention/eval_metrics.py`, `learn/toy_memory/eval_metrics.py`). Une comparaison qui a l'air favorable peut être en réalité en dessous du meilleur raccourci sans aucune capacité réelle testée.

## 4. Un résultat trop propre (variance nulle sur une grille large et diverse) est un signal d'alerte, pas un motif de satisfaction

La fuite de mesure du toy-model (96/96 cellules à exactement 100%, sur tout le balayage `read_step`, les deux échelles, les deux tâches) a été repérée précisément *parce que* c'était trop parfait pour être crédible — vérifié en une seule expérience ciblée (remplacer l'entrée par une séquence sans rapport, voir si l'accuracy tient quand même). Même réflexe que la contre-expertise du compresseur K/V (§3) : douter d'un résultat qui semble trop bon avant de le rapporter comme acquis, surtout sur une grille censée montrer de la variation.

## 5. Une contrainte structurelle assouplie à un endroit doit être vérifiée partout où le même motif existe

`HierarchicalMemory.build()`'s assertion `N == block_size**depth` (égalité stricte, forçant une racine unique) a été assouplie en divisibilité (`N % block_size**depth == 0`) le 2026-09-13. Le **même** motif d'égalité stricte a ensuite été réintroduit par erreur, la même nuit, dans **trois endroits indépendants** : `learn/indexed_attention/train_kb_retrieval.py`, `learn/indexed_attention/train_kb_chain.py`, et un nouveau script (`learn/indexed_attention/train_real_text.py`) écrit *après* le correctif d'origine, par la même session qui l'avait pourtant écrit.

**Règle** : après avoir assoupli/changé une contrainte dans un module central (`core/`), chercher (`grep -rn` du motif) tous les autres endroits du dépôt qui pourraient dupliquer la même vérification — scripts existants ET tout nouveau script écrit ensuite, qui peut halluciner l'ancien motif "plus familier" sans le vouloir.

## 6. Dans un curriculum multi-étapes, un budget global unique peut être entièrement consommé par une étape précoce lente, sans rien laisser à l'étape cible

Trouvé deux fois la même nuit (une fois sur la dernière étape, une fois — après un premier correctif trop étroit — sur une étape intermédiaire). Protéger **chaque** étape avec un plancher de pas minimal garanti (quitte à dépasser le budget global demandé), pas seulement la dernière — sinon le budget "généreux" en apparence peut laisser certaines étapes avec presque rien.

## 7. Vérifier dans les logs existants avant de relancer une expérience

Plusieurs questions cette nuit ("le plancher par étape était-il suffisant ?", "le sweep LR isolé a-t-il convergé avant 1500 pas ?") avaient déjà leur réponse dans des logs déjà produits — gratuit à relire, pas besoin de consommer de la compute pour le redécouvrir. Réflexe à adopter systématiquement avant de proposer un nouveau run diagnostique.

## Pointeurs

- Discipline détaillée (seeds, résultat ambigu, transférabilité petite→grande échelle) : `indexed_attention_experiment_plan.md`, section « Méthodologie commune ».
- Traces complètes des incidents résumés ici : `experiment.log.md` (entrée « Contre-expertise », 2026-09-13) et les sections correctives de `indexed_attention_experiment_plan.md`/`toy_memory_experiment_plan.md` datées 2026-09-13/14.
