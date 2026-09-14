# Plan d'expérimentation & de recherche — Indexed Attention (au-delà du MVP)

Objectif : passer du MVP testé uniquement en local sur CPU (`dev_notes/indexed_attention_spec.md` §11) à une validation progressive sur GPU (Grid'5000), alignée sur la vision complète du projet (séparer raisonnement et mémoire, cf. §-1 de la spec) plutôt que de s'arrêter à la mécanique de base.

Ce document a été révisé après un audit critique (3 agents indépendants, angles code/math, méthodologie, validité scientifique) qui a trouvé des trous réels : absence de contrôle statistique, absence de vérification d'attribution causale, généralisation jamais testée, et surtout — la décision la plus débattue de cette session (supprimer tout FF, §-1 de la spec) n'était testée nulle part. Ce plan corrige ça : chaque phase porte maintenant une **hypothèse explicite**, le **raisonnement** qui la sous-tend, et une **table observation → action** (y compris pour les résultats ambigus, pas juste les deux extrêmes).

## ⚠️ PRIORITÉ ABSOLUE — révision post-contre-expertise (2026-09-13). À lire avant toute exécution.

Une contre-expertise indépendante (voir `dev_notes/experiment.log.md`, entrée « Contre-expertise ») a invalidé la conclusion principale de cette branche. **Trois corrections sont déjà dans le code ; les re-runs restent à faire.** Aucun run GPU n'a été lancé pour cette révision — c'est le travail décrit ci-dessous.

### Ce qui est invalidé (ne plus citer ces conclusions)

1. **« (c) une vraie limite du mécanisme de composition à 2+ sauts »** — **RÉFUTÉ**. Le plateau valait exactement `1/n_facts` (33,3 % à `n_distractors=1`, 25,0 % à `n_distractors=2`, deux correspondances exactes), et 97-98 % des prédictions tombaient sur une valeur de la KB : le modèle recopiait une valeur au hasard, il n'avait rien appris de la chaîne. Il était même **en dessous** du meilleur prédicteur sans récupération (`non_key` = 0,500 à `n_distractors=1`).
2. **Cause réelle** : `LevelCompressor` poolait `parent_k` et `parent_v` avec **le même softmax**, ce qui rend une association clé→valeur structurellement irreprésentable. Corrigé : `decouple_kv=True` par défaut (une seconde requête apprise, `n_slots × d_model` paramètres). Preuve CPU : `n_hops=2` passe de 26-35 % à **100 % (loss 0,000)** sur 2 seeds.
3. **La supervision d'attention visait la mauvaise cible** (la feuille KEY, dont la valeur est la clé elle-même) — d'où un self-match parfait sans gain d'accuracy. Corrigé : `--supervise node` par défaut.
4. **Le go/no-go Phase 0 (« hiérarchie 99,6 % vs plat 6,0 % »)** ne démontre pas ce qu'on lui a fait dire — voir Phase 0bis ci-dessous.
5. **Les verdicts `use_ff` / `n_register` / `N_step` / LR sur `n_hops=2`** ont été rendus sur une architecture incapable de réussir la tâche. Ils ne sont ni vrais ni faux : ils sont **sans objet**. Ne pas les recycler comme acquis ; ne pas non plus se précipiter à tous les relancer (voir « Ce qu'il ne faut PAS faire »).

### Règle d'évaluation désormais obligatoire pour tout run sur les tâches synthétiques

`learn/indexed_attention/eval_metrics.py` est branché dans `train_kb_chain.py` et `train_kb_retrieval.py` : chaque run imprime un bloc « chance-level report ». **Aucune accuracy ne doit être rapportée, citée ou comparée sans ce bloc.** Points de contrôle :

- comparer à `conditional_chance` (= `1/n_facts`), **jamais** à `vocab_chance` ;
- vérifier `pred_in_kb_rate` : proche de 1,0 ⇒ c'est bien `conditional_chance` la référence ;
- exiger `margin_over_shortcut > 0` : un modèle qui ne bat pas `non_key`/`random_kb` n'a rien démontré ;
- `probe/first_hop` n'est pas un raccourci (c'est le solveur exact à 1 saut) — il est exclu de la marge, il sert de marqueur de progression.

La tâche `data/kb_chain_retrieval.py` contient de vrais raccourcis structurels (la réponse finale n'apparaît jamais comme clé). Un durcissement du générateur est souhaitable (chaînes leurres, distracteurs dont les valeurs sont aussi des clés) — voir Phase 2-redo, étape 4.

### Ordre d'exécution recommandé

**Étape 1 — Phase 2-redo (la plus importante, à lancer en premier).** Reproduire sur GPU le résultat CPU du découplage, à budget et échelle réels.
- Grille : `--shared_kv_pooling` (ablation) vs défaut découplé × `n_hops ∈ {2, 3, 4}` × **≥3 seeds** (le CPU montre une seed sur deux qui échoue à 3 sauts — caractériser cette variance est l'objet principal de l'étape).
- Config de départ : `batch_size=256`, `lr=1.2e-3` (couple déjà validé), `depth=2`, `block_size=4`, `n_step=12`, `d_model=256`.
- Critère de réussite : le découplé bat `non_key` d'une marge nette à `n_hops=2` sur les 3 seeds ; l'ablation reste collée à `conditional_chance`.
- Si `n_hops=3` reste instable : **d'abord** un balayage LR propre pour la variante découplée (son paysage d'optimisation a changé, l'ancien LR n'est plus forcément calibré), **ensuite seulement** un curriculum `--hop_curriculum 1,2,3`.

**Étape 2 — Phase 0bis : une baseline plate honnête.** `depth=0` n'a aucun compresseur : ses feuilles sont des K/V par *token*, donc attendre sur un token-clé retourne ce token-clé. Une mémoire plate de feuilles brutes **ne peut représenter aucune association clé→valeur, quel que soit le budget** — les 5-7 % observés sont structurels, pas un déficit d'indexation. Le gap 99,6 % vs 6,0 % ne prouve donc pas la supériorité de l'indexation hiérarchique ; il prouve que le groupement en blocs est le seul chemin vers une entrée associative.
- La vraie baseline « mémoire **sans index** » est **`depth=1`** (un seul niveau de compression, un nœud par fait, aucune hiérarchie multi-niveaux, attention dense sur les nœuds). C'est ce qu'il faut comparer à `depth≥2`.
- Grille : `depth ∈ {0, 1, 2, 3}` × `n_facts ∈ {16, 64, 256}` × 3 seeds, avec le découplage activé partout.
- Hypothèse : `depth=1` doit *égaler* `depth≥2` tant que la KB est petite ; l'intérêt de la hiérarchie n'apparaît qu'en coût/passage à l'échelle (nombre de faits grand), pas en accuracy. **Un `depth=1` qui égale `depth=3` à petite échelle n'est pas un échec de la thèse** — c'est le résultat attendu, et il indique à quelle taille de KB il faut monter pour que l'indexation paie.
- Conséquence pratique (question de l'utilisateur, 2026-09-13) : **oui, on peut démarrer l'intégration sur texte réel avec une mémoire sans index** (`depth=1`). L'indexation hiérarchique est motivée par la taille de mémoire visée, pas par la capacité d'association — elle peut donc être introduite plus tard, une fois la mécanique validée à `depth=1`.

**[CORRECTIF 2026-09-13, `core/indexed_memory.py`] — la grille ci-dessus n'était pas exécutable telle quelle.** `experiment-manager` a trouvé que `HierarchicalMemory.build()` exigeait `N == block_size ** depth` (égalité stricte), ce qui force la hiérarchie à toujours se réduire à **un seul nœud racine** — avec `n_facts` fixé et `block_size=4` (la largeur d'un bloc-fait), ça ne laisse aucune valeur de `depth` produisant « un nœud par fait » pour `n_facts>1` (`depth=1` avec `block_size=4` ne donne qu'1 nœud total pour 4 feuilles = 1 fait, pas `n_facts` nœuds). Contourner avec un `block_size` non-4 mélange plusieurs faits par bloc — une variable confondue différente, pas un test de profondeur.

Rien en aval (le softmax unifié de `attend()`, §5.2) ne suppose une racine unique — concaténer une "forêt" de `N / block_size**depth` nœuds de tête fonctionne exactement comme concaténer un seul nœud. **Assertion assouplie** : `N % (block_size ** depth) == 0` (divisibilité, pas égalité) — 46 tests existants toujours verts (aucun ne dépendait de la racine unique). Avec `block_size=4` fixé et `N = 4 × n_facts` (aucun padding requis), les profondeurs valides deviennent :

| `n_facts` | profondeurs valides (`block_size=4`) | nœuds au niveau `depth` |
|---|---|---|
| 16 | {0,1,2,3} | depth=1→16, depth=2→4, depth=3→1 (racine native) |
| 64 | {0,1,2,3,4} | depth=1→64, depth=2→16, depth=3→4, depth=4→1 (native) |
| 256 | {0,1,2,3,4,5} | depth=1→256, depth=2→64, depth=3→16, depth=4→4, depth=5→1 (native) |

**`depth ∈ {0,1,2,3}` est donc directement exécutable pour les 3 valeurs de `n_facts`, avec `block_size=4` partout, sans padding ni `block_size` par cellule.** Reste à faire côté `learn/indexed_attention/train_kb_retrieval.py` : son assertion (`block_size**depth == max_facts*4`, ligne ~214) doit être assouplie à l'identique (`%`, pas `==`) — fichier `experiment-manager`, pas de coordination nécessaire au-delà de ce message.

**[CORRECTIF 2026-09-13] — Étape 1 auto-corrigée par `experiment-manager` : aucun résultat de cette grille ne bat le raccourci, à aucun `n_hops`.** Premier passage lu à tort comme un signal positif croissant (`n_hops=4` à 89-92%, présenté comme "surprenant") — en réalité une comparaison au seul `conditional_chance`, sans vérifier `margin_over_shortcut` comme la méthodologie l'exige. Repris intégralement avec le bloc chance-level complet sur les 21 runs terminés (`n_hops` 2/3/4, découplé + ablation partagée) :

| | `n_hops=2` (5 seeds) | `n_hops=3` (5 seeds) | `n_hops=4` (3 seeds) |
|---|---|---|---|
| découplé | 34,6/32,9/33,5/25,5/33,8 % — marge ±0,08 pt max, jamais nette | 47,2/39,4/46,3/44,9/38,2 % — marge toujours négative (-3 à -13 pts) | 92,0/89,0/91,3 % — marge négative (-8 à -11 pts) |
| partagé (ablation) | 24,7/23,4/23,8 % — au niveau de `conditional_chance`, comme attendu | 16,4/22,8/24,8 % — marge très négative (-26 à -33 pts) | 68,6 % — marge -31,5 pts |

**Cause du malentendu sur `n_hops=4`** : la grille garde `max_facts=4` fixe (pour préserver `block_size=4`/`depth=2`), ce qui force `n_distractors=0` à `n_hops=4` — sans aucun distracteur, le raccourci `non_key` (deviner parmi les valeurs qui n'apparaissent jamais comme clé) sature à **100%** (à ce `n_hops`, la seule valeur non-clé de tout l'épisode *est* la bonne réponse, sans qu'aucun chaînage soit nécessaire). Les "89-92%" étaient donc en dessous, pas au-dessus, de ce plafond dégénéré — l'amélioration apparente avec `n_hops` croissant était la hausse du plafond du raccourci, pas un progrès du modèle. Même lecture, plus discrète, à `n_hops=3` (`non_key` ≈ 49-52%, jamais dépassé).

**Conclusion révisée (à ce moment du plan, supersédée ci-dessous par le balayage LR)** : avec le compresseur découplé, **aucune configuration testée à ce jour (2, 3 ou 4 sauts) ne démontre un chaînage réel au-delà d'un raccourci sans récupération** — ce n'est ni un problème spécifique à `n_hops=2` (lecture précédente), ni un signal positif à `n_hops≥3` (lecture initiale erronée, rétractée).

**[RÉTRACTÉ 2026-09-13] — la lecture "2e fois qu'un correctif CPU ne transfère pas au GPU" était prématurée.** Formulée ici initialement en rapprochant ce résultat de celui de la supervision d'attention (CPU → 72,9-99,2%, GPU → ~25%). Après le balayage LR ci-dessous, les deux cas ont en fait des explications **mondaines, pas un phénomène d'échelle réel** : (a) le test GPU de supervision d'attention datait d'**avant** le correctif `decouple_kv` — le compresseur était structurellement cassé à ce moment-là, donc bien sûr que rien ne pouvait fonctionner au-delà du matching, peu importe l'échelle ; (b) le "découplage seul ne bat pas le raccourci à `n_hops=2`" mesuré juste au-dessus utilisait `lr=1,2e-3`, calibré pour un tout autre mécanisme (pooling partagé) à un tout autre point d'échelle — pas une vraie tentative à l'échelle GPU avec un LR correctement calibré pour le découplage. **Aucun des deux cas ne constitue une preuve de non-transfert petite→grande échelle** — retenir la leçon opérationnelle (toujours revalider le LR après un changement de mécanisme) plutôt que la lecture "pattern d'échelle", qui n'est pas soutenue une fois qu'on regarde d'où viennent réellement ces deux résultats.

**[RÉSULTAT 2026-09-14] — balayage LR dédié, découplé seul, `n_hops=2, n_distractors=2`, 3 seeds/point** :

| lr | seed0 | seed1 | seed2 | marge sur raccourci |
|---|---|---|---|---|
| 3e-4 | 100,0% | 100,0% | 100,0% | +0,66 à +0,68 — victoire nette, 3/3 seeds |
| 6e-4 | 32,5% | 32,8% | 32,0% | -0,002 à -0,019 — échoue, collé au raccourci |
| 1,2e-3 (ancien défaut) | 33,6% | 33,0% | 29,5% | +0,009 à -0,044 — échoue pareil |
| 2,4e-3 | 0,0% | 0,0% | diverge (loss NaN) | -0,33 à -0,34 — diverge franchement |

**Le découplage seul, sans aucune supervision, résout `n_hops=2` proprement** (100% sur 3/3 seeds) une fois le LR correctement calibré — mais la fenêtre est **très étroite** : `6e-4` (2× `3e-4`) retombe déjà au niveau du raccourci, `2,4e-3` (8×) diverge complètement. Cohérent avec la fenêtre de LR déjà documentée comme étroite ailleurs dans ce projet (Phase -1, `n_facts=16`) — un motif qui se répète, pas une anomalie isolée cette fois.

**Conséquence pour Étape 3 (supervision d'attention)** : le résultat "découplé + supervision = 100%" utilisait très probablement le même `lr=1,2e-3` mal calibré — la supervision compensait un LR faux, pas un manque structurel une fois le découplage en place. Lecture retenue : **la supervision est redondante à `n_hops=2`, une fois le LR correct**. Reste **ouvert** : est-ce que cette redondance tient aussi à `n_hops≥3` une fois le générateur durci (Étape 4), ou est-ce que la supervision redevient nécessaire sur des chaînes plus longues ? Ne pas généraliser avant ce test spécifique.

**Prochaines étapes, dans cet ordre** :
1. ~~Balayage LR dédié à la variante découplée~~ **fait, voir résultat ci-dessus** — `lr=3e-4` devient la référence pour `n_hops=2, n_distractors=2, decoupled`, à ne pas réutiliser à d'autres `n_hops`/tailles sans revérifier (la fenêtre est trop étroite pour supposer un transfert).
2. **Durcir le générateur avant de retester `n_hops≥3`** (voir Étape 4 ci-dessous, priorité relevée) : garder `n_distractors` fixe et non nul (ex. 2) à travers tout balayage `n_hops`, en laissant `max_facts` (et donc `depth`, maintenant libre grâce à l'assouplissement `N % block_size**depth == 0` ci-dessus) croître avec `n_hops` plutôt que de figer `max_facts`. Sans ça, `n_hops≥3` avec peu/pas de distracteurs reste ininterprétable quel que soit le LR.
3. Une fois (2) fait, refaire un balayage LR fin à `n_hops=3` (ne pas supposer que `lr=3e-4` transfère) puis retester la question "supervision redondante ou nécessaire" à ce `n_hops`.

**Étape 3 — Supervision d'attention, cible corrigée.** `--attn_supervised --supervise node`, avec découplage, `n_hops=2`, 3 seeds, à budget comparable à l'ancienne grille (qui donnait 25 %).
- Observation de fumée à ne pas citer comme résultat : 72 % en 434 pas CPU (15 s).
- Rapporter `node_selection_diagnostic` (sélection du bon nœud) **et non** le self-match de feuille, qui s'est révélé maximisable sans lien avec la tâche.
- Question à trancher : la supervision reste-t-elle utile une fois le compresseur réparé, ou devient-elle redondante ? Une réponse « redondante » est un bon résultat (moins de machinerie à porter).

**[RÉSULTAT 2026-09-13, GPU, validé contre le raccourci] — décisif, positif.** `n_hops=2, n_distractors=2, decoupled` + `--supervise node` : **100,0 % exact sur 3/3 seeds.** Vérifié contre `shortcut/non_key` (~33 % à cette config exacte, mesuré dans la grille Étape 1 corrigée ci-dessus) — 100 % est donc un résultat réel, sans ambiguïté possible, pas un artefact de raccourci.

**Réponse tranchée par le balayage LR (voir Étape 1 ci-dessus)** : le découplage **seul**, à `lr=3e-4` (jamais testé avant ce balayage — la grille précédente n'utilisait que `lr=1,2e-3`, mal calibré), bat le raccourci tout aussi proprement que découplage+supervision (100% sur 3/3 seeds dans les deux cas). **La supervision est donc redondante à `n_hops=2`** — le résultat "100%" de cette étape reflétait très probablement le même LR mal calibré compensé par la supervision, pas un besoin structurel. Bon résultat (moins de machinerie à porter), au sens où la question de §"Étape 3" l'anticipait. **Reste ouvert** : cette redondance tient-elle aussi à `n_hops≥3` une fois le générateur durci (Étape 4) et un LR propre re-swept à ce `n_hops` ? Ne pas généraliser sans ce test spécifique — la supervision pourrait redevenir utile sur des chaînes plus longues même si elle ne l'est plus ici.

**Étape 4 — Durcir le générateur de tâche** — **[priorité relevée 2026-09-13]**, n'est plus seulement souhaitable en parallèle : bloquant pour toute conclusion sur `n_hops≥3` (voir le correctif de l'Étape 1 ci-dessus, où `n_distractors=0` a fait saturer `non_key` à 100% et rendu les résultats `n_hops=3/4` ininterprétables). Concrètement : garder `n_distractors` fixe et non nul à travers tout balayage `n_hops` (`max_facts`/`depth` variables à la place, cf. l'assouplissement `N % block_size**depth == 0` plus haut), et durcir `data/kb_chain_retrieval.py` contre les raccourcis structurels restants (chaînes leurres, distracteurs dont les valeurs sont aussi des clés).

### Ce qu'il ne faut PAS faire maintenant (décision explicite de l'utilisateur, 2026-09-13)

- **Ne pas relancer `use_ff` / `n_register` / les balayages `N_step`/LR de `n_hops=2`.** Leurs verdicts sont sans objet, mais les re-tester à l'aveugle est du travail à faible valeur : ces variantes n'ont d'intérêt que si le modèle réparé bute à nouveau quelque part. Les garder en réserve, comme diagnostics conditionnels, pas comme file d'attente.
- **Ne pas geler le chantier distillation.** Voir « Statut des chantiers » ci-dessous — il a une fonction propre que la contre-expertise avait sous-estimée.
- **Ne pas lancer un script `learn/indexed_attention/*` sans relire son `--lr` par défaut.** Piège déjà rencontré au moins 3 fois sur ce projet (EXP-007, Phase 0bis 2026-09-13) : un défaut de script resté à une valeur d'une autre échelle (souvent `3e-3`) diverge silencieusement à `d_model` plus grand — la perte augmente au lieu de baisser, ce qui peut se lire à tort comme une régression d'architecture. Toujours passer `--lr` explicitement à la valeur déjà validée pour la config testée (voir §13 pour les valeurs de référence par échelle), jamais compter sur le défaut du script.

### Statut des chantiers du projet (2026-09-13, cadrage utilisateur)

| Chantier | Statut | Rôle |
|---|---|---|
| **LLM-as-Compressor** (`core/compressor/`, `notebooks/`, `docs/compression/`) | **ARRÊTÉ** | Travaux gelés. Ne pas y consacrer de ressources ni de temps GPU. Le `README.md` le met encore en avant — à lire comme un historique, pas comme un chantier actif. |
| **Distillation** (`learn/distill/`) | **ACTIF, en parallèle, volontairement** | Deux fonctions : (a) produire une **baseline** de référence sur une architecture standard ; (b) **acquérir l'expérience de la distillation** (pipeline Teacher, KD Top-K, muP, checkpoint/resume, précision) *avant* de l'appliquer au Thinker. Ce n'est pas une diversion : c'est le pré-requis assumé de la Phase 4. Le fait que `train_sft.py` utilise un transformer dense est **intentionnel** à ce stade. |
| **Indexed Attention** (`core/indexed_*.py`, `learn/indexed_attention/`) | **ACTIF, chantier principal** | L'architecture de la thèse. Priorités ci-dessus. |

### Deux questions ouvertes de l'utilisateur, instruites ici

**Q1 — « Peut-on commencer avec une attention sur la mémoire sans index ? »** Oui, et c'est même recommandé : c'est `depth=1` (voir Étape 2). L'indexation hiérarchique est une réponse au **coût** d'une mémoire massive, pas à la capacité d'association — cette dernière vient du compresseur au niveau du fait, qui existe déjà à `depth=1`. Démarrer sans index réduit le nombre de mécanismes non validés simultanément, et fournit la baseline qui manquait pour justifier l'index.

**Q2 — « Peut-on utiliser les embeddings intermédiaires pour accélérer la distillation, et gagner de l'expérience sur leur usage ? »** Deux effets à ne pas confondre :
- **Coût de stockage : c'est plus cher, pas moins.** Top-K32 ≈ 192 o/token (32 indices int32 + 32 valeurs fp16) ; un hidden state du Teacher en fp16 (`hidden_size=5120`) ≈ 10 240 o/token, soit **~53×**. Précalculer les états intermédiaires bruts alourdirait le pipeline au lieu de l'accélérer.
- **Vitesse de convergence : c'est là que le gain est réel.** La distillation de représentations (FitNets, MiniLM, layer-wise KD) donne bien plus de signal par token que des logits Top-K, et converge en moins de tokens — le levier pertinent quand le budget est en jours-GPU.
- **Compromis recommandé** : stocker une **projection réduite** des états du Teacher (PCA/SVD 5120 → 256-512 dims, ~512-1024 o/token, soit 3-5× les Top-K, acceptable), en réutilisant exactement la machinerie SVD déjà proposée en spec §13.1 pour l'initialisation de l'embedding. Superviser par **perte cosinus** plutôt que MSE brute (spec §11bis : une MSE non normalisée est le risque d'instabilité identifié), avec montée en poids progressive.
- **Bénéfice secondaire, qui est le vrai argument** : c'est exactement le mécanisme du stream `thinking` en embedding (spec §11bis, plan Phase 1ter). Le faire côté distillation dense, c'est acquérir l'expérience du composant avant de le porter sur le Thinker — cohérent avec le rôle assigné au chantier distillation.
- **Levier de vitesse plus direct, déjà identifié par le projet et non implémenté** : la **loss KD chunkée** (façon Liger-Kernel / « Cut Your Losses »), qui lève le plafond mémoire imposé par le tenseur de logits `(B, T, 248077)` — c'est lui qui bloque le batch à 6, indépendamment de la précision. À faire avant ou en parallèle de l'idée des embeddings intermédiaires.

## Méthodologie commune à toutes les phases (corrige les trous trouvés par l'audit)

- **Seeds** : chaque condition testée avec **≥3 seeds**, moyenne ± écart-type reportée. Un écart entre deux conditions n'est traité comme un signal réel que s'il dépasse ~2σ ; sinon → **"non concluant"**, jamais "pas d'effet".
- **Généralisation obligatoire** : split train/held-out dès la Phase 0 — la KB et les paires clé/valeur de l'évaluation sont régénérées avec un seed disjoint de l'entraînement, jamais vues pendant l'entraînement. Le test de sur-apprentissage CPU actuel (`tests/test_indexed_memory.py`, même batch fixe train=eval) reste un test de **plomberie** (le pipeline tourne), pas un test de **mécanisme** — ne pas confondre les deux, et ne pas répéter cette faiblesse aux échelles GPU.
- **Diagnostic d'attribution causale** : en plus de l'accuracy, logguer où se concentre le poids d'attention du softmax unifié pour l'exemple cible (le nœud de la hiérarchie contenant la bonne réponse doit recevoir le poids dominant). Sans ça, une accuracy élevée peut venir d'un raccourci (le registre ou la SM mémorisant l'info sans jamais passer par la KB) plutôt que du mécanisme qu'on prétend valider.
- **Résultat ambigu** : si l'intervalle ± écart-type de deux conditions se chevauche, la règle par défaut est **+3 seeds supplémentaires, ou +1 palier d'échelle**, jamais conclure sur un signal ambigu.
- **Résultat contradictoire à une hypothèse (consigne explicite de l'utilisateur, s'applique à toutes les phases)** : un résultat qui contredit une hypothèse de départ est un signal diagnostique sur **comment** atteindre l'objectif du projet (séparer raisonnement et connaissance, cf. spec §-1), pas un référendum sur **si** on continue dans cette direction. Ne pas abandonner la direction générale à cause d'un résultat isolé — investiguer d'abord ce qui, dans l'implémentation actuelle, empêche le résultat attendu, avant de remettre en cause la thèse elle-même. Exemple concret déjà traité : Phase 1bis, "avec-FF gagne" ne réfute pas §-1 (voir la table de décision de cette phase et la nuance ajoutée dans la spec).
- **Transférabilité échelle petite → grande (question explicite de l'utilisateur, à garder en tête pour toutes les phases 1/1bis/2)** : la littérature est sans ambiguïté sur le risque — Narang et al. 2021 ("Do Transformer Modifications Transfer Across Implementations and Applications?", EMNLP) montrent que la plupart des modifications architecturales qui gagnent à petite échelle **ne transfèrent pas** à plus grande échelle ; une réplication 2020-2026 à l'échelle 1-3B (arXiv 2605.20798) confirme la même conclusion et ajoute un point aggravant : deux modifications testées convergent à ±2-3% de loss de validation (indiscernable en loss) mais perdent 6-16 points sur les tâches downstream — **un classement basé sur la seule loss/accuracy de petite échelle peut activement mentir**, pas juste être bruité. Symétriquement, la littérature sur les "capacités émergentes" (Wei et al. 2022 ; débat Schaeffer et al. 2023 sur le caractère possiblement artefactuel de la métrique) montre qu'un résultat **plat/nul à petite échelle** n'exclut pas non plus l'apparition d'une capacité à plus grande échelle (transition de phase) — donc le risque est symétrique : ni un "gain net" ni un "résultat nul" à petite échelle ne garantissent quoi que ce soit à plus grande échelle.
  - **Ce qui reste valide à l'échelle actuelle, malgré ce risque** : la correction de plomberie (le pipeline tourne, le gradient circule, forme/shapes correctes — cf. distinction déjà actée ligne 10), la localisation grossière d'une fenêtre de LR/dimensionnement (µP — Yang & Hu, *Tensor Programs*, et µTransfer, Yang et al. 2022 — sont conçus explicitement pour ce transfert et la transférabilité est justifiée en largeur, et pour la profondeur sous Depth-µP, mais reste sensible à des détails d'implémentation — normalisation entraînable, choix d'optimiseur — qui peuvent casser le transfert si mal alignés, cf. la mésaventure Depth-muP déjà vécue dans ce projet en distillation), et les diagnostics d'attribution causale (où l'attention se concentre) qui testent un **mécanisme**, pas un score.
  - **Ce qui devient activement à risque à l'échelle actuelle** : tout classement de variantes dont l'écart dépend de la thèse elle-même (ex. `use_ff` vs sans-FF) — la thèse §-1 postule justement que le bénéfice de "connaissance en KV plutôt qu'en poids" ne se manifeste qu'à partir d'un **volume/diversité de faits suffisant à travers les exemples** (mémoire associative = phénomène statistique cross-exemple, cf. spec §-1) ; `n_facts=16` est probablement *sous* ce seuil (déjà observé : Phase 1bis non concluant à n_facts=16, cf. `[NOTES]` associée). Conclure quoi que ce soit sur `use_ff` à cette échelle serait exactement le type de comparaison que Narang et al. montrent peu fiable.
  - **Mitigation adoptée (pas d'arrêt de l'échelle actuelle, mais discipline de lecture)** : (1) ne jamais transformer un résultat de petite échelle en conclusion architecturale définitive — seulement en hypothèse à re-tester à l'échelle suivante ; (2) pour les comparisons dont la thèse prédit un effet dépendant de l'échelle (use_ff, hiérarchie vs plat), prioriser un **balayage à plusieurs points d'échelle** (mini scaling-curve : n_facts/d_model croissants avec le même classement de variantes à chaque point) plutôt qu'une conclusion à un seul point — un renversement de classement en cours de balayage est en soi une information précieuse (et attendue par la thèse), un classement stable à travers l'échelle est un bien meilleur argument qu'un classement à un seul point, quel qu'il soit ; (3) avancer Phase 6 (µP) plus tôt que "en dernier" pour au moins les axes largeur/profondeur qui ont un besoin de transfert connu, plutôt que de la traiter comme une passe finale de mise à l'échelle isolée du reste.
  - **Cas concret, `decouple_kv` vs `--shared_kv_pooling` (question explicite de l'utilisateur, 2026-09-13)** : le découplage K/V corrige un problème que l'analyse mathématique montre **structurel, pas de capacité** — un unique jeu de poids d'attention ne peut pas simultanément concentrer son poids sur la position clé (pour que $\tilde K_p$ soit trouvable) et sur la position valeur (pour que $\tilde V_p$ soit correcte), quelle que soit la largeur de $K$/$V$ ; augmenter `d_model` élargit le vecteur produit par cette pondération, pas la pondération elle-même. Mon évaluation : peu probable que ça "émerge" en augmentant seulement les dimensions, par nature différente du cas `use_ff` (où la thèse elle-même prédit un effet dépendant de l'échelle). Mais **ce n'est qu'un raisonnement, pas une garantie** — la littérature "capacités émergentes" citée juste au-dessus dit explicitement qu'un résultat nul à petite échelle n'exclut pas une transition de phase plus loin, donc l'hypothèse de l'utilisateur reste légitime à vérifier, pas à écarter sur la seule base de cet argument. **Un point de données GPU existe déjà et va dans le sens du raisonnement** : la grille Phase 2-redo Étape 1 (corrigée) inclut `--shared_kv_pooling` à `d_model=256` (8× le `d_model=32` du test CPU original) — aucune amélioration observée, toujours en dessous du raccourci sans récupération à `n_hops` 2/3/4. Pas concluant en soi (256 reste loin d'un "modèle réaliste"), mais cohérent avec l'attente. **Règle adoptée pour la suite** : garder `--shared_kv_pooling` comme **colonne d'ablation permanente** dans toute grille future impliquant un changement d'échelle (Phase 3 texte réel, Phase 6 muP, etc.) — coût nul (flag déjà implémenté), et exactement le protocole "mini scaling-curve" du point (2) ci-dessus appliqué à cette variante spécifiquement. Un renversement de classement à une échelle donnée serait en soi un résultat important, pas un incident à ignorer.

## Parallélisation — comment aller plus vite

La plupart des runs listés ci-dessous sont mutuellement indépendants (le résultat de l'un ne conditionne pas la conduite de l'autre) et n'ont **aucune raison d'être séquentiels** :

- **Au sein d'une phase** : chaque combinaison (variante × seed) est un run indépendant. Ex. Phase 1bis avec 5 variantes × 3 seeds = 15 runs lançables simultanément si les ressources le permettent.
- **Multi-GPU sur un même nœud** : `abacus26` (2×L40S), `chuc` (4×A100) — un run indépendant par GPU sur un seul job OAR.
- **Plusieurs jobs OAR simultanés** : le palier dry-run (`abacus3/10`, `abacus22`) est peu demandé — plusieurs réservations besteffort/courtes peuvent coexister sans attente de queue significative.
- **Plusieurs sessions Claude Code en parallèle** : déjà en pratique sur ce projet (`thinker-e9` exécute la Phase 0 pendant que cette session travaille sur l'audit/la vérification). Le même modèle se réplique pour la Phase 1bis : une session par lot de variantes plutôt que tout faire séquentiellement dans une seule conversation. Utiliser `ListAgents`/`SendMessage` pour coordonner et éviter les doublons (déjà fait une fois avec succès dans cette session).
- **Ce qui reste séquentiel** : les dépendances inter-phases. Phase 3 (données réelles) a besoin d'un signal de Phase 2 ; Phase 4 (Teacher réel) a besoin de Phase 3 ; Phase 6 (muP) ne dépend que de la taille du modèle, pas du contenu des données — peut tourner **en parallèle** de Phase 4/5, pas après.

Référence cluster : skill `grid5000` + `dev_notes/grid5000_usage.log.md`. Paliers déjà validés dans le chantier de distillation (`dev_notes/experiment.log.md`) : **dry-run** (`abacus3/10` A5000, `abacus22` A40), **KD/échelle moyenne** (`abacus26` 2×L40S, `chuc` 4×A100), **à éviter** (`abacus1/2` P100, `drac`, `chiclet` — pas de BF16/FlashAttention).

### Consigne de rythme (temps limité pour ce projet — priorité à la vitesse d'itération)

Les modèles testés à ce stade sont minuscules (`d_model` 32-256, quelques Mo, batch 16-64) — un seul run sur un A100/L40S entier laisse le GPU largement sous-utilisé. Directives concrètes pour l'agent d'exécution (`experiment-manager`) :

- **Empiler plusieurs runs indépendants sur le même GPU.** À cette taille de modèle, rien n'empêche de lancer plusieurs processus Python concurrents sur un seul GPU (mémoire/compute largement disponibles) — ce n'est déconseillé que quand la charge sature déjà la carte, ce qui n'est pas le cas ici. Utiliser ça pour paralléliser les seeds/variantes d'une même phase plutôt que de les faire les uns après les autres sur le même GPU.
- **Augmenter le batch size pour absorber le temps de calcul inutilisé** plutôt que de le laisser inoccupé — accélère la convergence en wall-clock sans coût matériel supplémentaire, tant que ça ne change pas la dynamique d'apprentissage de façon confondante (garder une trace de quel batch size a été utilisé par run, pour ne pas mélanger ça avec un vrai signal architectural).
- **Lancer large maintenant plutôt qu'attendre une confirmation "propre" séquentielle avant le prochain lot.** Le temps disponible pour ce projet est limité — préférer lancer en parallèle tout ce qui est déjà spécifié dans ce plan (Phase 0 seeds restants, balayage Phase 1bis, curriculum n_facts, sondage Phase 1ter, diagnostics du blocage n_facts=64) plutôt que d'attendre qu'une phase soit entièrement "verte" avant de démarrer la suivante — les critères d'ambiguïté déjà définis plus haut (±σ, ≥3 seeds) trient le signal du bruit après coup, pas besoin d'attendre avant de lancer.
- **Réserver activement d'autres nœuds** (pas seulement ceux déjà utilisés) dès qu'un lot de runs supplémentaire est prêt à partir — le palier dry-run est peu demandé, plusieurs réservations simultanées ne se bloquent pas mutuellement.
- **Pendant les temps morts** (attente de résultats, nœuds en cours de libération) : profiling/optimisation du pipeline de données, tests de configurations plus larges (modèle/`d_model` plus grand, batch plus large) pour préparer les paliers suivants du plan à l'avance plutôt que de les découvrir seulement une fois arrivé dessus.

**Retour explicite de l'utilisateur (2026-09-13) — la sous-utilisation mesurée reste "terrible" malgré les directives ci-dessus** : même en empilant 4-8 runs par nœud, on mesure 5-30% calcul / 2-7% VRAM par process (modèles à 0,7-1,6M params, largement trop petits pour saturer un A5000/L40S même cumulés). Les directives précédentes ("empiler des runs", "augmenter le batch") n'ont pas suffi en pratique — durcir la consigne :
- **Le dimensionnement (batch_size, et taille de modèle quand la question de recherche le permet) doit être choisi *pour* saturer le GPU visé, pas ajusté après coup si ça reste trop petit.** Avant de lancer un lot, estimer combien de processes/quelle taille de batch il faut pour que l'empilement mange réellement la capacité dispo — pas juste "quelques runs en parallèle à la petite échelle habituelle". `batch_size=256` (déjà validé, cf. Phase 2) est le nouveau défaut à essayer en premier pour tout nouveau sweep sur ces tâches synthétiques, pas `4`/`16`/`64` sauf si la question posée exige spécifiquement un petit batch.
- **`experiment-manager` peut indiquer la marge GPU restante à un instant donné** — s'en servir activement pour décider combien de configurations empiler par nœud avant de lancer, pas seulement après coup pour constater le gaspillage.
- Concrètement pour la suite immédiate (sweep `N_step` en cours) : lancer à `batch_size=256`/LR scalé en conséquence par défaut, et empiler autant de couples (N_step × seed) que la VRAM/calcul du nœud le permet réellement, pas juste 4-8 runs à l'ancienne petite échelle.

### Graphe de dépendances des tâches en cours (ce qui peut tourner maintenant, en parallèle, sans rien attendre)

Répartition des rôles : cette session (design) identifie quelles tâches sont indépendantes et sur quels leviers jouer pour maximiser l'utilisation des ressources ; `experiment-manager` décide de la mécanique (nœuds, nombre de processus par GPU, ordonnancement OAR) — cette session ne maîtrise pas assez l'infrastructure pour trancher ça.

**Totalement indépendantes entre elles, lançables toutes en même temps dès maintenant :**
1. Phase 0 — seeds `depth=0` restants (déjà en cours).
2. Phase 1bis — chaque variante × seed est un run isolé : avec/sans FF (priorité #1), `M=1` vs `M>1`, stop-gradient SM on/off, dropout de niveaux on/off. 4 variantes × 3 seeds = 12 runs, tous indépendants les uns des autres et du reste de cette liste.
3. Curriculum `n_facts` (16→32→64) — thread indépendant du reste, mais **interne au thread : séquentiel** (le palier N+1 dépend de l'accuracy held-out atteinte au palier N, pas question de lancer 64 avant que 16 ait promu).
4. Diagnostics du blocage `n_facts=64` (balayage `n_register`, batch size) — indépendant de 1-3.
5. Phase 1ter — le **sondage** (cosinus/CKA couche par couche) peut démarrer **immédiatement**, sans attendre la fin du curriculum : il suffit d'un core déjà entraîné sur le stream `answer` seul (les checkpoints de la Phase 0 conviennent déjà) + un forward pass du Teacher (GPT-2 124M, léger — peut même tourner CPU-only sur un nœud sans GPU pendant que les GPU sont occupés ailleurs).

**Dépendances réelles (à respecter) :**
- Phase 1ter, étape "brancher le stream" → dépend de la fin du sondage (étape précédente du même thread), pas des autres threads.
- Phase 2 (multi-sauts) → peut démarrer en parallèle dès maintenant avec la config déjà connue comme fonctionnelle (`lr=3e-4`, `n_facts=16`) plutôt que d'attendre un signal "propre" de Phase 1bis — le risque (mauvais hyperparamètre) est faible vu qu'on a déjà un point de fonctionnement validé.
- Phase 3 (données réelles) → attend un vrai signal de Phase 2 (changement de tokenizer/vocabulaire, coût de mise en place plus élevé, moins rentable de lancer à l'aveugle).
- Phase 4 (distillation) → attend Phase 3. Phase 5 (stratégie KB) et Phase 6 (muP) → indépendantes l'une de l'autre et peuvent tourner **en parallèle** de la Phase 4, pas après.

**Leviers pour maximiser l'utilisation (ce que cette session peut spécifier sans connaître l'infra) :**
- **Batch size** : les runs actuels (16-64) sont petits pour la mémoire disponible sur ces GPU — passer à 128-256+ absorbe du calcul autrement perdu, accélère la convergence en wall-clock. Ne pas mélanger un changement de batch size avec une conclusion architecturale sans le noter explicitement dans le log du run.
- **Nombre de configs simultanées par GPU** : pas de règle fixe à donner sans mesurer — `experiment-manager` peut tester en pratique (2, puis 4, puis 8 processus concurrents) et reculer dès que le temps par run commence à se dégrader significativement.
- **Utiliser les leftovers CPU/non-GPU** pour ce qui ne nécessite pas de GPU (sondage Phase 1ter avec un petit Teacher type GPT-2 124M, analyses/agrégation de résultats déjà tombés).

### Protocole de modification de code

`experiment-manager` a les mains libres pour modifier ses propres fichiers (`learn/indexed_attention/*`) sans validation préalable. **Pour toute modification touchant `core/` (architecture du modèle) : proposer d'abord à cette session plutôt que de committer directement** — cette session valide l'alignement avec les décisions de design déjà prises (spec, plan) avant que ça parte en exécution. Motif : `core/` porte les décisions architecturales tracées dans `indexed_attention_spec.md` ; un changement non coordonné là risquerait de créer une divergence entre ce que la spec documente et ce que le code fait réellement — exactement le genre de dérive silencieuse que la discipline `[NOTES]/[CONFIRMÉ]/[DÉFAUT]/[OUVERT]` du document cherche à éviter.

---

## Phase -1 — Dimensionnement (identifier les bonnes dimensions avant de juger l'architecture)

**Pourquoi cette phase existe** : trouvée nécessaire *en direct* pendant cette session — `thinker-e9` a rapporté un échec total à `depth=4`/64 facts (loss bloquée à `ln(vocab)`, accuracy au hasard). Reproduit localement en quelques minutes (CPU, `Thinker`) : avec KB **resamplée à chaque batch** (signal réel, pas mémorisation), 4 facts → apprend (~93% acc), 64 facts → n'apprend plus **du tout**, et ce **indépendamment de la profondeur** (`depth=4` et `depth=2` échouent pareil à 64 facts, à `d_model=64`). Ça élimine "la hiérarchie est cassée" comme explication première et pointe vers un **sous-dimensionnement** (`d_model`, `n_register`) — exactement le type de confusion que l'audit redoutait (attribuer à l'architecture un échec qui est en fait un problème de taille).

**Deux dimensions à distinguer, pas une seule** :
- **`d_model` (K/V de la mémoire)** : doit rester assez grand pour que les clés soient séparables (a) au niveau des feuilles, en fonction du nombre total de faits, ET (b) à chaque nœud compressé, en fonction de `block_size` (un parent doit rester discriminable parmi ses frères). Sous-dimensionner ici est indiscernable, en accuracy seule, d'un problème d'architecture — d'où l'urgence de cette phase avant Phase 0.
- **`n_register`** (largeur du registre latent) : doit croître avec le **parallélisme de raisonnement nécessaire** (nombre de sous-objectifs/sauts actifs simultanément), pas directement avec la taille de la KB — l'attention est adressable par contenu, un seul vecteur de requête suffit en principe à cibler n'importe quel fait si `d_model` est suffisant.

**Hypothèse** : à `n_facts` fixé, il existe un `d_model` (et éventuellement un `n_register`) minimal en-dessous duquel l'apprentissage échoue totalement (pas juste dégradé), et au-dessus duquel il réussit — une transition de phase plutôt qu'une dégradation progressive.

**Protocole** : à `n_facts` fixe (ex. 64, resamplé à chaque batch — jamais un seul batch fixe), balayer `d_model` (32/64/128/256) × `n_register` (1/2/4) indépendamment de `depth`/`block_size` (fixés à une valeur qui marche déjà, ex. `depth=2`), 3 seeds par point. Reporter la courbe accuracy vs `d_model` pour trouver le seuil de transition.

| Observation | Action |
|---|---|
| Transition nette identifiée (échec en dessous, succès au-dessus) | Utiliser ce `d_model`/`n_register` minimal (+ marge) comme config de référence pour toutes les phases suivantes, y compris Phase 0 — **ne pas** conclure quoi que ce soit sur `depth`/`block_size` avec un `d_model` sous-dimensionné |
| Pas de transition claire (échec à toutes les tailles testées) | **Déjà observé** (voir note ci-dessous) — le problème n'est pas dimensionnel, passer directement au diagnostic "signal d'entraînement" plutôt que continuer à augmenter `d_model`/`n_register` |
| Transition très tardive (`d_model` très grand nécessaire même pour 64 facts) | Signal que le compresseur/la mémoire sont peu efficaces en information par dimension — révisateur candidat : le pooling sans aucun poids appris (cf. Phase 1bis "avec/sans FF") |

**Relation avec le reste du plan** : cette phase doit être refaite (ou son résultat revalidé) à chaque fois que `n_facts`/le nombre de leaves change significativement dans une phase ultérieure (Phase 2 multi-sauts, Phase 3 données réelles) — le dimensionnement n'est pas une constante universelle, il dépend de la taille de la tâche.

**Correction importante, trouvée en direct pendant cette session** : une première lecture des résultats CPU (échec à `n_facts=16` et `64`, non résolu par `d_model=128`+`n_register=4` ni par `lr=1e-3`) avait conduit à soupçonner une instabilité structurelle du looped transformer (cf. spec §11bis, hypothèse maintenant infirmée pour ce cas). **`thinker-e9` a identifié la vraie cause par un balayage de LR plus complet** : un **plateau de convergence étroit autour de `lr=3e-4`** (`1e-4`, `1e-3`, `3e-3`, `1e-2` échouent tous ; `3e-4` fait passer l'accuracy de ~10% à 98-100% entre 800 et 1000 steps) — confirmé indépendamment en local avec les mêmes hyperparamètres exacts. **Leçon méthodologique directement actionnable : le balayage de `lr` doit être fait AVANT/EN MÊME TEMPS que celui de `d_model`/`n_register`, jamais après ou en le survolant avec 2-3 valeurs espacées d'un facteur 10** — un plateau de convergence peut être bien plus étroit qu'un facteur 10, et un LR mal choisi est indiscernable d'un problème de capacité ou de mécanisme si on ne teste pas assez finement autour de la bonne valeur.

**Protocole corrigé** : le balayage de cette phase doit inclure `lr` comme troisième axe (pas juste `d_model`×`n_register`) — grille fine autour d'un ordre de grandeur (ex. `{1e-4, 3e-4, 1e-3, 3e-3}` minimum) à chaque point `d_model`/`n_register`, pas une seule valeur fixe par défaut.

**Hypothèse de supervision intermédiaire (spec §11bis) : reportée, pas écartée.** Elle reste une piste légitime si un balayage LR complet échoue à trouver un plateau stable à plus grande échelle (ex. `n_facts=64`+, profondeur plus grande) — mais elle ne doit être testée qu'après avoir exclu un simple problème de LR, pas avant.

**Nouveau blocage confirmé, isolé proprement par `thinker-e9`** : à `n_facts=64`, ni `d_model` (128 vs 256), ni `lr` (1e-4 à 3e-3), ni `n_step` (3 vs 6) ne débloquent l'apprentissage (échec uniforme, ~0.2-0.6% acc) — et **`depth=0` échoue identiquement à `depth>0`**, donc ce n'est pas spécifique à la hiérarchie : c'est le mécanisme registre+SM+fusion entier qui ne parvient pas à apprendre à cette échelle avec les budgets testés (5-15 min, 2000-9000 steps). `n_facts=16` reste la seule échelle confirmée où le protocole 3-seeds de la Phase 0 peut produire un signal go/no-go réel — décision prise de lancer Phase 0 à cette échelle et de documenter `n_facts=64` comme blocage ouvert distinct, plutôt que de forcer des runs qui ne produiraient que du bruit des deux côtés.

**Précédent direct dans l'historique de ce projet, à tester en priorité avant les deux autres hypothèses de `thinker-e9`** (capacité de `n_register`, budget d'entraînement plus long) : `dev_notes/experiment.log.md` (21-22 déc. 2023) documente **exactement ce pattern** sur la toute première version de ToyThinker (tâche copy) : *« I design a basic curriculum learning and it make the copy task much easier to learn, the model reach 100% accuracy with that approach but plateau 60% accuracy without it »* — en faisant varier progressivement la longueur de séquence plutôt que de l'entraîner directement à la longueur cible. Conclusion du journal, directement applicable ici : *« having a plateau doesn't mean that the model is at capacity »*. **Action concrète à tester en premier** : curriculum sur `n_facts` (entraîner d'abord à 16, augmenter progressivement vers 32 puis 64) plutôt que d'entraîner directement à 64 — c'est la solution déjà validée empiriquement dans ce projet pour un plateau structurellement similaire, moins coûteux à tester qu'un budget d'entraînement 10x plus long ou qu'un changement de `n_register`.

## Phase 0 — Sanity check GPU du MVP actuel

**Hypothèse** : `HierarchicalMemory` (depth>0) égale ou dépasse l'attention plate (depth=0) en accuracy, à un coût FLOPs/mémoire inférieur, une fois l'échelle assez grande pour que l'attention plate commence à diluer le signal entre les distracteurs.

**Raisonnement** : le softmax unifié (spec §5.2) est censé arbitrer nativement entre résolution grossière et fine. À trop petite échelle, l'attention plate n'a structurellement aucune raison d'être pire (elle voit tout directement) — l'avantage de la hiérarchie n'est testable que si le nombre de feuilles dépasse ce qu'une attention plate gère sans diluer le signal (le "piège de dilution" déjà documenté en §6.2 de la spec, ici côté échelle plutôt que côté input/KB).

**Protocole** : étendre `data/kb_retrieval.py` (64-256 facts distracteurs, depth 3-4, d_model 256-512), split train/held-out, 3 seeds, palier dry-run, budget ~15-30 min.

| Observation | Action |
|---|---|
| depth>0 ≥ depth=0 (± σ) et coût inférieur | Continuer vers Phase 1bis, depth>0 en config par défaut |
| depth>0 < depth=0 significativement | Revoir la conception du compresseur avant de continuer (candidat : capacité insuffisante, cf. Phase 1bis "avec/sans FF") |
| Écart dans la marge de bruit | Ambigu — augmenter le nombre de distracteurs avant de conclure, pas de verdict |

## Phase 1 — Baselines A/B/C

**Hypothèse** : Baseline B (boucle sans mémoire) échoue sur la récupération ; Baseline A (T=1) réussit sur récupération simple mais plafonnera sur le multi-sauts (Phase 2) ; le modèle complet réussit partout.

**Raisonnement** : ce n'est pas qu'une comparaison empirique — c'est un test de **cohérence interne du design**. Si plus aucune connaissance factuelle ne peut être stockée dans les poids du core (plus de FF, §-1), Baseline B ne peut structurellement pas réussir la récupération : elle n'a nulle part où mettre l'information. Si elle réussit quand même, c'est un signal d'alarme, pas juste un résultat négatif.

| Observation | Action |
|---|---|
| B échoue, A et modèle complet réussissent | Comme attendu, continuer |
| B réussit partiellement, au-dessus du hasard | **Signal d'alarme** — chercher une fuite d'information (candidats : `intrablock_pos`/`source_bias` partagés qui pourraient encoder des faits par accident, cf. audit) avant de continuer |
| A échoue déjà sur récupération simple | Revoir `HierarchicalMemory` avant la Phase 2 |

## Phase 1bis — Balayage des variantes architecturales

**Priorité #1 (nouvelle, suite à l'audit) — avec FF vs sans FF.**

**Hypothèse** : supprimer le FF (compresseur, boucle, streams — décision de §-1) ne dégrade pas l'accuracy sur récupération pure, mais peut dégrader la capacité de **composition/calcul** sur les valeurs récupérées (ex. une tâche demandant une opération simple sur deux faits extraits, pas juste les recopier).

**Raisonnement** : c'est exactement le point que l'audit scientifique a identifié comme jamais testé — la prémisse §-1 (FF = stockage de faits, donc à bannir partout) conflate "stocker des faits" et "disposer d'une capacité de calcul non-linéaire générale". Le modèle actuel n'a plus aucune non-linéarité apprise hors du softmax ; cette expérience est le premier test empirique de si c'est un problème réel ou juste théorique.

**Précision importante (retour de l'utilisateur, à ne pas perdre de vue — voir spec §-1 "Comment interpréter un résultat avec-FF gagne")** : un résultat où `use_ff=True` gagne significativement **n'est pas un référendum sur la thèse de §-1** — il ne doit pas conduire à l'abandonner ni à remettre des FF partout par défaut. La question n'est pas "présence vs absence de non-linéarité" mais "où et à quelle échelle statistique" : un FF appliqué au contenu déjà récupéré pour l'exemple courant (composition/calcul) n'est pas la même chose qu'un FF qui mémoriserait des associations génériques à travers la distribution d'entraînement (stockage de faits). On a une direction ; la question posée par cette expérience est *comment* y arriver (quelle forme de capacité de calcul ajouter, où, sous quelle contrainte), pas *si* on continue dans cette direction.

| Observation | Action |
|---|---|
| avec-FF ≈ sans-FF sur récupération pure ET sur une tâche demandant un calcul simple sur les valeurs récupérées | Prémisse §-1 supportée, garder sans-FF |
| sans-FF nettement pire sur la tâche de calcul mais pas sur la récupération pure | Prémisse supportée dans son principe — réintroduire un FF minimal dans la **boucle principale seulement** (pas le compresseur, pas les streams), pour la composition, pas le stockage. Ajouter un diagnostic de non-mémorisation (le FF répond-il différemment selon le contenu de $O_{kb}$ pour deux exemples de même structure, ou converge-t-il vers une fonction fixe ?) avant de le considérer validé |
| sans-FF pire même sur récupération pure | Ne pas abandonner §-1 — investiguer d'abord si le problème vient d'ailleurs (dimensionnement, LR, profondeur du compresseur) avant de conclure que la thèse elle-même est en cause ; ne remettre en question l'architecture complète qu'après avoir épuisé ces pistes |

**Autres variantes déjà câblées, zéro code à ajouter :**
- **$M=1$ vs $M>1$** (`n_slots`). Hypothèse : $M>1$ aide si les blocs contiennent plusieurs "aspects" distincts (littérature Slot Attention). Observation nulle → ambigu à cette échelle (cf. clause de résultat ambigu ci-dessus), pas "aucun bénéfice".
- **Confirmation, pas une variante à trancher** : $Q_{KB}$/$Q_{SM}$ sont déjà découplés par défaut.

**Variantes nécessitant un petit flag :**
- **Stop-gradient sur les clés SM**. Hypothèse : pas de différence visible à cette échelle — son rôle (éviter la "moving target" pour l'indexeur, cf. comparison.md DSA/QSA) est surtout pertinent à grande échelle/longue durée d'entraînement. Résultat nul ici → reporter le test à la Phase 3/4, ne pas conclure.
- **Dropout stochastique des niveaux hauts**. Hypothèse : bénéfice attendu en généralisation **hors distribution de profondeur**, pas en accuracy in-distribution — le protocole doit donc évaluer à une profondeur différente de l'entraînement pour être probant, sinon le test ne teste rien.

**Reportées à une comparaison dédiée plus lourde** : softmax unifié vs gating façon NSA, routeur explicite vs implicite (nécessiteraient une seconde implémentation quasi complète).

**Protocole** : tâche de la **Phase 0** (pas Phase 2 — correction d'une erreur de séquencement de la version précédente de ce plan, qui référençait une phase pas encore atteinte), 3 seeds par variante.

## Phase 1quater — Balayage de $N_{\text{step}}$ (nombre d'itérations), à tester dès maintenant

**Décision de l'utilisateur** : tester l'effet d'un nombre important d'itérations dès le début plutôt que d'attendre — c'est le mécanisme central de la thèse du projet (extraction itérative KB→SM puis traitement sur ce qui a été retenu, permettant un raisonnement plus long à budget de calcul inférieur puisque le core reste léger). L'utilisateur anticipe que plusieurs expériences seront nécessaires pour stabiliser ce balayage — ne pas s'arrêter à un premier résultat instable ou peu concluant.

**Hypothèse** : l'accuracy sur une tâche qui *nécessite* du chaînage (Phase 2, multi-sauts) croît avec $N_{\text{step}}$ jusqu'à un plateau correspondant au nombre de sauts réels de la tâche ; au-delà, pas de dégradation. Sur la tâche à un seul saut (Phase 0/1bis), $N_{\text{step}}$ au-delà de 2 ne devrait apporter aucun gain — ce n'est **pas** le bon banc d'essai pour cette question (la tâche sature déjà à $N_{\text{step}}$ faible), Phase 2 est le vrai test.

**Raisonnement** : le nombre d'itérations n'est pas un hyperparamètre secondaire ici — c'est directement ce que la séparation raisonnement/mémoire est censée permettre (§-1). Un échec à en tirer un bénéfice mesurable serait un signal important sur la thèse centrale, pas un détail d'optimisation.

**Point de vigilance, déjà rencontré empiriquement dans cette session** : la fenêtre de LR stable s'est révélée étroite et sensible (plateau autour de `lr=3e-4` à `n_step=2`, cf. Phase -1) — rien ne garantit que cette même fenêtre reste valide à `n_step` plus grand (chaque itération ajoute un résidu supplémentaire à $R$, la dynamique d'accumulation change). **Balayer le LR à chaque valeur de $N_{\text{step}}$ testée, ne pas réutiliser un LR trouvé bon à un autre $N_{\text{step}}$ sans le revérifier** — exactement la leçon déjà tirée en Phase -1 pour `d_model`/`n_register`, applicable ici à l'identique.

**Protocole** : sur la tâche multi-sauts de la Phase 2 (2 à 4 sauts), balayer $N_{\text{step}} \in \{1, 2, 4, 8, 16\}$ (au moins), avec un balayage LR fin à chaque valeur (pas un LR fixe reporté d'un autre point), ≥3 seeds par point une fois une fenêtre de LR stable identifiée.

**Extrapolation à l'inférence, à ajouter dès qu'un premier checkpoint multi-sauts existe (précédent direct dans ce projet, `dev_notes/future_experiments.md` §3 : *"Step Extrapolation: demonstrate that increasing N_step at inference time continues to improve accuracy... beyond the steps seen during training"*, déjà prévu pour l'architecture Thinker d'origine)** : entraîner à un $N_{\text{step,train}}$ fixe (ex. 4, le nombre de sauts réels de la tâche), puis évaluer **en inférence seulement** (pas de ré-entraînement — les poids sont partagés entre itérations, donc c'est immédiat) à $N_{\text{step,test}} > N_{\text{step,train}}$ (ex. 8, 16). Peu coûteux (inférence pure) et à lancer en parallèle du reste dès qu'un checkpoint existe, sans attendre la fin du balayage complet ci-dessus.

| Observation (extrapolation) | Action |
|---|---|
| Accuracy stable ou continue de s'améliorer au-delà de $N_{\text{step,train}}$ | Signal fort que le mécanisme généralise réellement au-delà de ce qu'il a vu — argument solide pour la thèse "raisonnement plus long à budget de calcul inférieur" |
| Accuracy se dégrade nettement au-delà de $N_{\text{step,train}}$ | Le modèle a probablement appris une heuristique calée sur le nombre d'itérations d'entraînement plutôt qu'un mécanisme d'extraction généralisable — creuser avant de considérer $N_{\text{step}}$ comme un simple hyperparamètre de budget ajustable librement à l'inférence |

| Observation | Action |
|---|---|
| Accuracy croît avec $N_{\text{step}}$ jusqu'au nombre de sauts réels, plateau ensuite, LR stable trouvé à chaque palier | Confirme le mécanisme central — utiliser cette relation ($N_{\text{step}}$ ≈ nombre de sauts + marge) comme règle de dimensionnement pour les phases suivantes |
| Pas de LR stable trouvé à grand $N_{\text{step}}$ malgré un balayage fin | **Ne pas conclure à un échec du mécanisme** — c'est potentiellement l'hypothèse d'instabilité du looped transformer (spec §11bis, déjà rencontrée puis écartée pour le cas `n_facts`, mais jamais testée spécifiquement pour un grand `n_step`) qui redevient pertinente ici. Tester la supervision intermédiaire (même section) avant d'abandonner |
| Accuracy plafonne bien en-dessous du nombre de sauts réels même avec LR stable | La fusion $\Delta$ (projection linéaire, §6.1) ou la capacité du registre (`n_register`) sont candidats — revoir Phase 1bis (avec-FF) et Phase -1 (dimensionnement) en conjonction, pas isolément |

### Intuition sur l'ordre de grandeur de $N_{\text{step}}$ nécessaire (note de l'utilisateur, à garder en tête pour dimensionner le balayage)

Un layer transformer standard fait en une seule étape deux choses à la fois : l'attention **va chercher** l'information (routage/agrégation content-based) et la FF qui suit **calcule** dessus (composition, mémoire associative). Dans cette architecture, la FF a été retirée de la boucle principale (§-1) — chaque itération de notre boucle ne fait donc plus qu'une seule de ces deux choses à la fois (attention pure), pas les deux fusionnées comme le fait un layer standard. Intuition de l'utilisateur, à retenir explicitement pour la suite : pour reproduire ce qu'un layer standard fait en une étape, on doit s'attendre à avoir besoin d'**au moins ~2 de nos itérations** (une pour aller chercher/ramener l'information dans le registre, une pour calculer dessus) — et comme la tâche multi-sauts demande en plus de répéter ce cycle **par saut de la chaîne** (aller chercher → traiter → repartir chercher le saut suivant), le ratio total attendu par rapport à un empilement de $L$ layers standards équivalents est plutôt de l'ordre de **$N_{\text{step}} \gtrsim 2\text{-}4 \times L$** (minimum, pas un plafond) — pas un simple $N_{\text{step}} \approx L$.

**Conséquence pratique pour le balayage ci-dessus** : ne pas se limiter à $N_{\text{step}} \in \{1,2,4,8,16\}$ si la tâche a $n_{\text{hops}}$ sauts réels — pousser explicitement jusqu'à des valeurs de l'ordre de $4 \times n_{\text{hops}}$ (voire plus) avant de conclure à un plateau ou à un échec du mécanisme itératif, cohérent avec l'attente déjà actée par l'utilisateur que "plusieurs expériences seront nécessaires pour stabiliser" ce balayage — ce n'est pas juste une question de bruit statistique, c'est aussi probablement une question de ne pas avoir simplement testé assez loin.

### Entraîner pour l'extrapolation en nombre de pas (comment stabiliser le modèle à un $N_{\text{step,test}}$ jamais vu)

Contrairement à un empilement de layers indépendants (où chaque profondeur a ses propres poids), ici c'est **le même bloc, aux mêmes poids, appliqué de façon répétée** — situation directement comparable aux *Universal Transformers* (Dehghani et al. 2018, avec ACT — Adaptive Computation Time, Graves 2016), à *PonderNet* (Banino et al. 2021), et plus récemment aux travaux sur les *Looped Transformers* (ex. "Looped Transformers as Programmable Computers" ; travaux 2024 sur la généralisation en longueur/nombre de boucles). Le point commun de toute cette littérature, et la réponse à la question posée : **la technique standard et la mieux supportée empiriquement est d'entraîner avec un nombre d'itérations aléatoire, tiré à chaque batch (ou exemple)**, ex. $N_{\text{step}} \sim \text{Uniform}(1, N_{\max})$ — pas un $N_{\text{step}}$ fixe. La raison documentée dans ces travaux : entraîner à un $N_{\text{step}}$ fixe pousse le modèle à converger vers un comportement qui n'est correct qu'à *exactement* ce nombre d'itérations (une sorte de point fixe local calé sur l'entraînement), sans aucune garantie de stabilité avant ou après — alors qu'un tirage aléatoire du nombre d'itérations force le modèle à rester correct (ou à converger vers un point fixe stable) à toutes les valeurs intermédiaires vues, ce qui est justement ce qui permet l'extrapolation au-delà.

Concernant les deux autres idées proposées par l'utilisateur (stop-gradient aléatoire en cours de boucle, bruit aléatoire injecté dans l'embedding intermédiaire) :
- **Stop-gradient aléatoire en milieu de boucle** (BPTT tronqué) : utile surtout pour contrôler le coût mémoire/gradient sur de longues boucles (comme pour les RNN), mais ne cible pas directement l'extrapolation en soi — combinable avec le $N_{\text{step}}$ aléatoire (tronquer le gradient quand $N_{\text{step}}$ tiré est grand), pas un substitut.
- **Bruit dans l'embedding intermédiaire** : plus spéculatif, sans précédent direct dans la littérature "extrapolation du nombre de boucles" citée ci-dessus — apparenté plutôt aux techniques de robustesse de point fixe (proche des *Deep Equilibrium Models*, Bai et al. 2019, qui entraînent explicitement à converger vers un point fixe indépendant du budget d'itérations du solveur). À garder comme ablation de second rang si le $N_{\text{step}}$ aléatoire seul ne suffit pas, pas comme premier essai.

**Recommandation** : ajouter au protocole de cette phase un entraînement avec $N_{\text{step}} \sim \text{Uniform}(1, N_{\max})$ tiré par batch (au lieu d'un $N_{\text{step}}$ fixe), et comparer son extrapolation à $N_{\text{step,test}} > N_{\max}$ contre le modèle entraîné à $N_{\text{step}}$ fixe déjà prévu ci-dessus — l'écart entre les deux est en lui-même une mesure directe de la valeur de cette technique.

### Autres axes d'extrapolation évoqués — profondeur de hiérarchie, taille de KB, vocabulaire jamais vu

- **Profondeur de la hiérarchie (extrapolation intéressante, approche proposée)** : dans `HierarchicalMemory.build`, la profondeur n'est pas un hyperparamètre choisi indépendamment — elle est dérivée du nombre de feuilles et de la taille de bloc ($\text{depth} = \lceil \log_{\text{block\_size}}(n_{\text{leaves}}) \rceil$). Une KB plus grande à l'inférence implique donc *automatiquement* une profondeur plus grande, avec le même compresseur à poids partagés appliqué simplement plus de fois — aucun changement de code n'est nécessaire pour "activer" une profondeur non vue. Protocole : entraîner à une taille de KB fixe (profondeur fixe, ex. depth=3), tester en inférence sur une KB nettement plus grande (profondeur non vue, ex. depth=5). Exactement par symétrie avec le $N_{\text{step}}$ ci-dessus, la même recommandation s'applique pour la robustesse à l'entraînement : **randomiser la taille de la KB (donc la profondeur) par batch pendant l'entraînement**, pas seulement à taille fixe — à tester conjointement avec le $N_{\text{step}}$ aléatoire plutôt que comme deux expériences indépendantes, puisque les deux touchent au même risque (sur-spécialisation à un budget de calcul figé).
- **Taille de KB à l'inférence, doute de l'utilisateur (partagé)** : contrairement à $N_{\text{step}}$ et à la profondeur (qui sont des budgets de calcul, où l'extrapolation a un sens direct pour la thèse), faire varier la taille de la KB *elle-même* à l'inférence n'est pas vraiment l'objectif du projet — l'objectif est plutôt une KB de taille donnée, fixe pour un déploiement donné. Il est en effet plausible que le modèle soit plus fragile à ce type de changement (la distribution du nombre de niveaux, la dilution du softmax unifié, etc. changent avec la taille de KB) sans que ce soit un signal négatif sur la thèse centrale. **Recommandation : ne pas en faire un axe de généralisation prioritaire à démontrer** — le garder seulement comme sous-produit du contrôle profondeur ci-dessus (qui, lui, teste quelque chose de réellement pertinent : la réutilisation du compresseur à un nombre de niveaux non vu), pas comme un objectif de robustesse en soi.
- **Vocabulaire (K/V) jamais vu pendant l'entraînement — doute de l'utilisateur, partagé** : cette expérience teste surtout la généralisation de la table d'embedding de tokens elle-même — une propriété qui existerait identiquement dans n'importe quel transformer standard avec une couche d'embedding, indépendamment de tout mécanisme d'indexation hiérarchique. Elle ne discrimine donc pas ce qui est spécifique à cette architecture (la hiérarchie, la compression, le softmax unifié) : un échec ou un succès sur ce test ne dirait presque rien sur *ces* mécanismes-là, seulement sur la qualité de l'espace d'embedding. **Recommandation : ne pas en faire une phase dédiée.** Le split train/held-out déjà imposé par la méthodologie commune (identifiants de clé/valeur tirés avec un seed disjoint train/eval) couvre déjà implicitement le cas où les *combinaisons* clé→valeur exactes n'ont jamais été vues, ce qui est la version pertinente de "généralisation" ici (mémorisation d'associations spécifiques vs mécanisme de récupération content-based) — un partitionnement dur et supplémentaire du vocabulaire lui-même n'ajoute pas grand-chose de plus, et détournerait du temps limité du projet vers un test peu diagnostique.

## Phase 1ter — Stream `thinking` en embedding (alignement Teacher, avancé plus tôt sur demande explicite)

**Décision** : initialement prévu en Phase 8 (après données réelles + distillation), avancé ici sur demande explicite de l'utilisateur — tester tôt comment le mécanisme se comporte, avant d'investir dans les phases suivantes.

**Hypothèse** : un stream `thinking` supervisé par alignement (cosinus, pas MSE brute — voir raisonnement plus bas) sur une couche intermédiaire d'un vrai Teacher pré-entraîné donne un signal d'entraînement supplémentaire cohérent (loss qui décroît proprement), sans dégrader la convergence du stream `answer` sur `kb_retrieval`.

**Raisonnement** : §11bis de la spec formalise maintenant le critère de choix de couche (éviter 0-25% et la dernière couche, cibler ~40-65%, valider par un protocole de sonde plutôt qu'un choix a priori) et le risque d'instabilité réel (perte non normalisée + conflit multi-objectif sur des poids partagés, pas "distance à l'initialisation" comme argument invoqué au départ — corrigé après discussion avec l'utilisateur). Les deux causes réelles d'instabilité sont atténuables (perte cosinus normalisée, montée en poids progressive), donc testables sans risque disproportionné.

**Ce qui est déjà en place, aucun changement d'architecture requis** : `Thinker(stream_dims={'answer': ..., 'thinking': teacher_hidden_dim})` fonctionne tel quel. `core/compressor/model_wrapper.py::HFModelWrapper.get_hidden_states()` (ajouté cette session) extrait les couches intermédiaires d'un Teacher HF déjà supporté (GPT-2/SmolLM/Qwen).

**Protocole** :
1. Choisir un petit Teacher déjà supporté (GPT-2 124M par défaut, le moins cher) et un petit corpus réel (`data/wiki_samples.json`, déjà présent).
2. Protocole de sonde d'abord (spec §11bis) : entraîner le core avec seulement le stream `answer`, mesurer la corrélation cosinus/CKA entre l'état SM à chaque itération et chaque couche candidate du Teacher (~40-65% de profondeur) — retenir la couche la plus stable/monotone, pas un choix fixé.
3. Brancher le stream `thinking` sur cette couche, perte cosinus (pas MSE brute), poids de perte croissant progressivement (pas à pleine intensité dès le départ).
4. Comparer convergence du stream `answer` avec vs sans le stream `thinking` actif.

| Observation | Action |
|---|---|
| `thinking` converge proprement (loss cosinus décroît), `answer` pas dégradé | Garder le stream, envisager de l'étendre en Phase 8 (curriculum d'extinction) |
| `thinking` erratique (loss qui oscille/ne baisse pas) | Vérifier d'abord le choix de couche (protocole de sonde bâclé ?) avant de conclure à une instabilité structurelle |
| `answer` dégradé par la présence de `thinking` | Signal de conflit de gradient réel (cf. §11bis) — réduire le poids de la perte `thinking` ou geler le core pendant l'entraînement du stream (l'un des deux est explicitement indépendant de l'autre par construction, spec §11bis) |

## Phase 2 — Tâche synthétique multi-sauts

**Hypothèse** : l'accuracy du modèle complet croît avec $N_{\text{step}}$ (1 à 6) ; Baseline A plafonne tôt (un seul passage ne peut pas chaîner) ; Baseline B échoue partout.

**Raisonnement** : c'est le test direct de la thèse centrale du projet — le raisonnement itératif n'a de valeur que s'il chaîne des faits extraits un à un de la mémoire externe, pas si l'information est déjà toute disponible en un passage.

| Observation | Action |
|---|---|
| Accuracy croît avec $N_{\text{step}}$ pour le modèle complet, plafonne pour A, échoue pour B | Comme attendu, passer à la Phase 3 |
| Accuracy plafonne aussi pour le modèle complet dès 2-3 sauts | La fusion $\Delta$ (projection linéaire simple, §6.1) est probablement le goulot — reconsidérer avant Phase 3 |
| Signal ambigu (chevauchement ±σ entre $N_{\text{step}}$ croissants) | +3 seeds ou tâche à sauts plus nombreux avant de conclure |

**Générateur de données disponible** : `data/kb_chain_retrieval.py::KBChainDataset(n_hops, n_distractors, vocab_size, max_facts=None, seed=None)` — chaîne de `n_hops` faits (valeur du fait $i$ = clé du fait $i+1$) + `n_distractors` faits non liés, blocs mélangés (pas de raccourci positionnel), même convention `max_facts`/masque que `KBRetrievalDataset`. Testé pour la correction (`tests/test_kb_chain_retrieval.py`) : la chaîne décodée mène bien du `query` au `label` en suivant exactement `n_hops` sauts.

**Résultat préliminaire (CPU, non concluant, à ne pas sur-interpréter — cf. le principe de non-abandon ci-dessus)** : test rapide à `n_hops=2`, `n_distractors=2`, `vocab_size=32`, `d_model=64` : ni `N_step=1` ni `N_step=4` ne dépassent ~25% d'accuracy en tenue (KB resamplée) après 3000 pas à `lr=1e-3` (le meilleur trouvé sur un balayage rapide {1e-4, 3e-4, 1e-3, 3e-3}) — pas de différenciation nette entre les deux, mais aucun des deux n'est non plus clairement convergé (chance ≈ 3%, donc apprentissage partiel réel des deux côtés, pas un blocage total comme celui rencontré en Phase -1). **Ne pas conclure à ce stade** — hypothèses à tester avant toute conclusion, par ordre de coût croissant :
1. **Curriculum sur `n_hops`** (1→2→3...), même logique que le curriculum `n_facts` déjà validé en Phase -1/0 — commencer par `n_hops=1` (qui se réduit exactement à la tâche `kb_retrieval` déjà bien maîtrisée) avant d'augmenter.
2. Balayage LR plus fin et plus de pas (la tâche chaîne est structurellement plus dure que la récupération simple — pas anormal qu'elle demande plus de budget).
3. Revoir la fusion $\Delta$ / la capacité du registre si 1 et 2 n'aident pas (cf. table de décision ci-dessus).

**Mise à jour GPU (experiment-manager, job 4104848, 2026-09-12/13)** : curriculum `n_hops` 1→2 exécuté (`n_hops=1` ~98%, comme attendu). À `n_hops=2` : `N_step=12` fixe et `N_step ~ Uniform(1,12)` **plafonnent tous deux à 32-34%** après 16-22k+ pas — au-dessus du hasard (~2-3%) mais loin de la maîtrise, et toujours **aucune différenciation** fixe vs. aléatoire. Le `lr=3e-4` utilisé vient tel quel de la tâche de récupération simple (`n_hops=1`), **jamais re-sweepé spécifiquement pour `n_hops=2`** — exactement le pattern déjà rencontré deux fois cette session (`n_facts=64`, et maintenant potentiellement ici) où un LR mal calé imite un plafond de mécanisme. **Décision : sweep LR dédié à `n_hops=2` avant toute conclusion sur (a) `N_step` insuffisant, (b) LR, ou (c) une vraie limite de composition à 2+ sauts** — ne pas descendre vers (c) tant que (b) n'est pas exclu, cohérent avec la discipline déjà actée dans ce plan (ne jamais réutiliser un LR d'une autre échelle sans le revérifier).

**(b) écarté proprement, à budget complet (~16-17k steps, comparable aux runs originaux)** : `lr=6e-4` → 25,2% (dans/en dessous du plateau), `lr=1e-3` → instable (oscille 22-28% en cours d'entraînement, lecture finale 4,6%, cohérent avec le fait que 1e-3 est juste en dessous de 3e-3 qui diverge). Le `lr=3e-4` original (celui des runs à 32-34%) reste donc le meilleur LR trouvé — la fenêtre de LR stable a bien été explorée, pas de gain caché par un LR mal calé. **Sonde d'extrapolation à l'inférence (`N_step_test`=16/20/24)** sur les deux configs : plat ou légèrement pire que `N_step=12` d'entraînement — aucun signe qu'ajouter des itérations à l'inférence débloque quoi que ce soit (résultat négatif pour l'extrapolation à ce stade, cf. Phase 1quater). Reste à trancher entre (a) et (c) : **prochaine étape décidée — isoler `N_step` directement** (sweep `N_step ∈ {8,16,24,32}` à `lr=3e-4` fixe, même méthodologie que le sweep `d_model` de la Phase -1).

**(a) écarté aussi — résultat net (`batch=256, lr=1.2e-3`, 2 seeds/point)** : accuracy complètement plate sur `N_step ∈ {8,16,24,32}`, toutes dans une bande de 0,8 point (24,7-25,5%), aucune tendance — sur un facteur 4× en $N_{\text{step}}$ (soit 4-16× le nombre de sauts réels, bien au-delà de l'heuristique ~2-4×). **(a) et (b) sont maintenant tous deux écartés proprement (budget complet, sweep dédié) — (c) une vraie limite du mécanisme de composition à 2+ sauts devient l'explication la mieux soutenue.**

**Application du principe de non-abandon (méthodologie commune)** : ce résultat est diagnostique sur **où** vit la capacité de composition manquante, pas un référendum sur la thèse centrale — le mécanisme gère un seul saut quasi parfaitement (`n_hops=1` ~98%), il ne compose pas de façon fiable à 2 sauts, indépendamment du budget de calcul. Deux candidats déjà identifiés dans la table de décision de cette phase, à tester **en priorité et en premier** avant toute remise en cause plus large :
1. **`use_ff=True` sur `n_hops=2`** (flag déjà implémenté, Phase 1bis) — c'est exactement le test que la Phase 1bis n'avait pas pu trancher à `n_facts=16` (trop petit pour différencier). La composition à 2 sauts (retrouver fait 1 → combiner dans $R$ → utiliser $R$ pour retrouver fait 2) est plausiblement le genre de calcul non-linéaire sur le contenu déjà récupéré que `fuse_proj` (projection linéaire simple, §6.1) ne peut pas faire — cohérent avec la nuance déjà actée en spec §-1 ("où et à quelle échelle" la non-linéarité opère, pas présence/absence). Si `use_ff=True` débloque nettement la composition, ça **ne réfute pas** §-1 (composition par-exemple ≠ stockage de faits cross-exemples) — voir spec §-1 "avec-FF gagne".
2. **`n_register`** (capacité du registre latent) — si `use_ff` n'aide pas, tester une largeur de registre plus grande, sur l'hypothèse que le registre ne peut pas tenir assez d'état intermédiaire pour composer 2 sauts.

**Décision : lancer (1) en premier** (le plus rapide à tester, flag déjà prêt), (2) en parallèle si les ressources le permettent plutôt qu'en séquentiel.

**Diagnostic mécanistique préalable, sans FF (2026-09-13, CPU local, priorité explicite de l'utilisateur avant (1)/(2) ci-dessus)** : `learn/indexed_attention/diagnose_no_ff_composition.py` — reproduit le plateau à petite échelle ($d_{\text{model}}=32$, $N_{\text{step}}=16$, `n_hops=2`/`n_distractors=2`/`vocab_size=32`) pour instrumenter directement le modèle, sans passer par Grid5000/experiment-manager (même philosophie "test local d'abord" que `tests/test_indexed_memory.py`).

1. **Contrôle positif (`n_hops=1`)** : converge à 100% (`acc=1.000` après 4000 pas) — confirme que le protocole d'entraînement/évaluation du script lui-même est sain.
2. **Sonde par plus-proche-voisin brut** (similarité cosinus entre $R$ moyen et la table d'embedding non transformée, à chaque itération) : **invalidée par son propre contrôle positif** — reste quasi nulle (0-2%) même quand la tâche est résolue à 100%. Conclusion méthodologique : cette sonde ne mesure rien d'utile (la représentation vit probablement dans une base tournée par `k_proj`/`v_proj`/`fuse_proj`, pas alignée avec les vecteurs d'embedding bruts) — **toute conclusion "absence de représentation" basée sur cette sonde doit être ignorée.**
3. **Test causal d'intervention** (le plus fiable des trois) : à `n_hops=2` (plateau reproduit, 23,0% puis 31,6% sur une deuxième graine — cohérent avec la bande 24-34% déjà observée), forcer $R$ à la valeur intermédiaire correcte de la chaîne, à différentes itérations (0, milieu, pic de sonde, fin) :
   - **v1** (naïf, $R := \text{embed}(\text{mid\_val})$) : 18,0-30,5% selon l'étape — aucune remontée.
   - **v2** (corrigé, $R := \text{register\_init} + \text{embed}(\text{mid\_val})$, pour rester dans la distribution naturelle d'amorçage du registre) : 21,9-29,7% — **toujours aucune remontée**, alors que cette version corrige le biais méthodologique identifié après v1.
   - **Conclusion** : donner explicitement la bonne valeur intermédiaire au modèle ne débloque pas la tâche. Ça déplace la suspicion de "dériver la valeur intermédiaire" (ce qui semblait être le problème naïvement) vers **l'utilisation d'une clé déjà connue** — soit la récupération du second fait (`memory.attend`/`q_proj`) à partir d'un $R$ correctement formé, soit la chaîne SM→stream de sortie en aval.
4. **Audit des poids d'attention bruts** (top-1 argmax sur `memory.attend`, sans entraînement supplémentaire, à partir du checkpoint sauvegardé) : précision top-1 (la bonne feuille reçoit le poids maximal) = **0,000** aussi bien pour le premier saut au $R$ naturel ($t=0$) que pour le second saut au $R$ forcé — alors que la précision de tâche finale (~23-32%) est très au-dessus du hasard (~3%, vocab 34). **Pas concluant en l'état** : la résolution est explicitement **implicite/soft** par conception (softmax unifié, spec §5.3, pas de sélection dure) — un poids maximal ailleurs n'exclut pas qu'une combinaison pondérée diffuse sur plusieurs feuilles/niveaux extraie quand même une valeur utile. Le bon test de suivi serait une mesure **douce** (corrélation entre `o_kb` et l'embedding de la vraie valeur), pas encore fait.

**Statut** : (3) est le résultat le plus solide de ce diagnostic — un résultat négatif propre et reproduit deux fois (v1/v2) sur le test causal le plus direct disponible. Recommandation avant de lancer (1)/(2) ci-dessus : ce résultat suggère que le problème n'est **pas** localisé au registre/à la fusion seuls (`fuse_proj` linéaire) mais possiblement aussi (ou plutôt) dans la boucle retrieval→SM→stream — `use_ff=True` (1) ne corrigerait que la fusion, pas ce chemin aval ; garder cette réserve à l'esprit en interprétant le résultat déjà obtenu pour (1) (neutre/une seed divergente, §ci-dessus) plutôt que d'y voir une réfutation de l'hypothèse fusion-linéaire. Prochaine étape suggérée : mesure douce (4) plutôt que de nouvelles variantes architecturales à l'aveugle.

**Mesure douce, corrigée deux fois (2026-09-13, même checkpoint CPU)** : la première tentative de mesure douce (corrélation cosinus entre `o_kb` et `self.embed(valeur)` brut) souffrait du **même biais méthodologique** que la sonde de (2) — `o_kb` vit dans l'espace transformé par `v_proj` (+ biais de source), pas dans l'espace brut des embeddings, donc cette comparaison était invalide (corrigé avant d'en tirer une conclusion). **Version corrigée** : comparaison de `o_kb` aux vecteurs $V$ réellement stockés par `HierarchicalMemory.build()` pour chaque fait candidat de l'épisode (même espace, `memory._levels_v[0]`), avec un rang/top-1 calculé uniquement parmi les faits réellement présents dans l'épisode (4 candidats, hasard=25%) plutôt que sur tout le vocabulaire :
- **Saut 1** (registre naturel $R_0$, la requête que le modèle produit réellement) : top-1 = 30,9% (hasard 25%), rang moyen 1,336 (hasard 1,50) — **léger signal au-dessus du hasard, mais faible** (à $n=256$, l'écart-type binomial est ≈2,7 points, donc ~2 σ, à la limite du bruit).
- **Saut 2** (registre forcé à la bonne valeur intermédiaire, requête "propre" par construction) : top-1 = 24,2%, rang moyen 1,500 — **exactement au niveau du hasard, aucune discrimination**.
- **Conclusion** : même en corrigeant la méthodologie (bon espace vectoriel, bon ensemble de candidats), le mécanisme de récupération lui-même **ne discrimine pas** la bonne feuille pour le second saut, même avec une requête correcte et propre fournie directement — ça confirme et affine le résultat causal (3) : la panne est spécifiquement dans `HierarchicalMemory.attend`/`q_proj` à ce second point de récupération (matching clé↔requête), pas seulement dans la fusion ou le registre en amont. Le premier saut montre un signal réel mais faible sur ce même modèle entraîné à 2 sauts — cohérent avec une accuracy de tâche finale (~23-32%) largement au-dessus du hasard (~3% sur 34 tokens) mais très loin de la maîtrise.

**Recommandation mise à jour avant (1)/(2)** : le point de départ le plus informatif n'est plus "tester `use_ff`/`n_register` à l'aveugle" mais comprendre **pourquoi** `q_proj`/`k_proj` ne discriminent pas la bonne clé au second saut alors que `k_proj` reçoit exactement le bon type de contenu au premier saut (avec un signal certes faible). Pistes non testées, par coût croissant : (a) vérifier si le problème est un simple défaut d'échelle/norme entre la requête forcée et les requêtes vues à l'entraînement (RMSNorm calibrée sur la distribution réelle de $R$ après plusieurs itérations de fusion, pas sur `register_init + embed(x)` brut) ; (b) élargir `n_head`/`dim(K)` (§5.4) si le goulot est une capacité de discrimination insuffisante à `d_model=32` sur ce jouet ; (c) seulement ensuite, `use_ff`/`n_register`.

**Piste (a) testée et écartée (2026-09-13, même checkpoint)** : découverte en cours de route — $\|R_t\|$ **croît sans borne** sur la boucle non forcée (aucune normalisation n'est appliquée à $R$ lui-même entre itérations, seule la concaténation avant `fuse_proj` passe par RMSNorm) : $\|R_0\|\approx 6{,}7 \to \|R_{15}\|\approx 29{,}0$, croissance quasi linéaire, un facteur ${\sim}4$-5× sur la boucle. L'intervention "replace" (§ci-dessus) forçait $R$ à une norme ${\sim}5{,}8$-$5{,}9$ (échelle de $t{=}0$), très en dessous de la norme naturelle à mi-boucle (${\sim}19$-25) — un décalage d'échelle qui pouvait, à lui seul, expliquer l'échec sans que le matching soit en cause. **Test corrigé** : deux variantes additives (au lieu de remplacer $R$, on **ajoute** $\text{embed}(\text{mid\_val})$ par-dessus la trajectoire naturelle — brute, ou mise à l'échelle de $\|R\|$ courant) à `force_step` $\in\{0,4,8,11,15\}$. **Résultat : toujours aucune remontée** (23-29% pour la version mise à l'échelle, 25-29% pour la version brute — même bande que le plateau non forcé). **(a) est donc écarté** : le problème n'est pas un simple défaut d'échelle entre la requête forcée et la distribution naturelle de $R$.

**Triangulation complète, conclusion de ce round de diagnostic** : trois mesures indépendantes (intervention causale §ci-dessus, rang doux en espace $V$ correct, et un test en espace $K$ — cosinus de `q_proj(R)` contre les clés candidates réellement stockées, restreint aux `n_facts` faits de l'épisode) convergent : le second saut de récupération ne discrimine pas la bonne clé, **même quand on lui fournit une requête correcte et bien mise à l'échelle par construction**. Le test en espace $K$ montre même une similarité **en dessous du hasard** pour la bonne clé (12,9%/17,2% de top-1 contre 25% de hasard, aux deux sauts) — pas juste un signal nul/bruité, plutôt une anti-corrélation systématique, qui évoque une forme de "mode collapse" du matching `q_proj`/`k_proj` (le mécanisme aurait appris un motif générique plutôt qu'une vraie discrimination par contenu) plutôt qu'un simple manque de capacité. **Hypothèse à tester ensuite** (nouvelle piste, pas encore explorée) : inspecter directement les poids/la structure de `q_proj`/`k_proj` appris (rang effectif, alignement avec `source_bias`) pour caractériser ce possible effondrement, avant de réessayer `use_ff`/`n_register`/`dim(K)` — un changement de capacité pourrait ne rien changer si le problème est un optimum local de matching dégénéré plutôt qu'un manque de paramètres.

**Inspection des poids (2026-09-13, même checkpoint) — image plus nuancée que prévu, aucune des hypothèses de "collapse" simples ne tient** :
- **Rang effectif** (ratio de participation des valeurs singulières) : 21-29 sur 32 pour `q_proj`/`k_proj`/`v_proj`/`sm_q_proj`/`fuse_proj` — pas d'effondrement en rang faible, les matrices utilisent la quasi-totalité de leur dimensionnalité.
- **Biais partagé** (`k_proj(source_bias[KB=1])`, la composante additive identique à toutes les clés KB) : norme ${\approx}2{,}90$, comparable (pas dominante) à la variation liée au contenu (${\approx}3{,}31$, ratio $0{,}88$) — le biais n'écrase pas le signal de contenu.
- **`register_init`** : norme ${\approx}0{,}8$-$1{,}2$, très inférieure à celle d'un embedding de token (${\approx}5{,}8$) — similarité de direction $0{,}983$ entre `q_proj(\text{register\_init}+\text{embed}(x))$ et $q_\text{proj}(\text{embed}(x))$ seul : `register_init` ne noie pas l'identité du token injecté.
- **Clés distinctes** : similarité cosinus moyenne de $0{,}39$ (écart-type $0{,}21$) entre 200 clés KB réelles distinctes (post `level_norm`) — pas proches de $1$, donc pas de collapse des clés entre elles non plus.
- **Test d'auto-appariement, tentative initiale — [ERREUR DÉTECTÉE ET CORRIGÉE, 2026-09-13]** : annoncé comme "sur 300 tokens aléatoires isolés", rang moyen 21/300 (hasard 150) — **ce chiffre était faux**. Bug : `torch.randperm(vocab_size)[:300]` avec `vocab_size=34` (le nombre réel de tokens de ce modèle jouet — `VOCAB_SIZE=32` du script + 2 marqueurs `KEY_MARK`/`VAL_MARK`) ne peut renvoyer que 34 éléments (`randperm` ne produit jamais plus d'éléments qu'il n'y a de valeurs possibles, et le slicing Python au-delà de la longueur ne lève pas d'erreur — il renvoie silencieusement ce qui existe). Le test comparait donc en réalité **34 tokens, pas 300** — le seul ensemble de tokens qui existe dans ce jouet, il n'y a pas de vocabulaire plus grand à échantillonner.
- **Test corrigé** (même calcul, bon effectif) : rang moyen de l'auto-appariement = **21,4 sur 34** (hasard correct = 16,5, pas 150) — c'est en fait **légèrement pire que le hasard**, pas mieux.

**Conclusion corrigée, plus simple que ce qui avait été rapporté** : il n'y a **pas de tension/contradiction** entre un "matching isolé qui fonctionne" et un "matching en épisode qui échoue" — les deux étaient en réalité au niveau du hasard ou en dessous, la différence rapportée précédemment venait uniquement du bug de comptage. Image consolidée et cohérente sur l'ensemble des tests de ce round : le matching `q_proj`/`k_proj` de ce checkpoint est à peu près au niveau du hasard partout où on le mesure — pas de piste supplémentaire à investiguer de ce côté avec les outils utilisés jusqu'ici. **Point d'arrêt de ce round de diagnostic.**

**Contrôle décisif ajouté (2026-09-13)** : le même test d'auto-appariement appliqué au modèle `n_hops=1` (celui qui atteint 100% de précision de tâche) donne un résultat **net et fort** : rang moyen 0,089 sur 3 candidats (hasard 1,00), top-1 = 91,8% (hasard 33,3%). Ça valide la méthode de test elle-même (elle sait détecter un vrai matching quand il existe) et confirme que l'architecture linéaire `q_proj`/`k_proj` **peut** apprendre une discrimination quasi parfaite — le problème à `n_hops=2` n'est donc ni un artefact de méthode ni une limite d'expressivité de la forme linéaire, mais quelque chose qui se dégrade spécifiquement quand la tâche devient plus dure.

**Test des correctifs littérature-guidés (température apprise + normalisation de requête), échelle CPU — résultat négatif (2026-09-13)** : implémentation de `SharpenedHierarchicalMemory` (§ recherche bibliographique : NTM "key-strength" β, Product-Key Memory query-normalization contre le "catastrophic drift"). Deux placements testés :
- **v1** : `RMSNorm` appliquée à l'**entrée** de `q_proj` (avant projection), température multipliant la sortie.
- **v2, corrigée** : `RMSNorm` appliquée à la **sortie** de `q_proj` (après projection) — placement fidèle à la recommandation Product-Key Memory (BatchNorm sur la sortie du réseau de requête, pas son entrée).

Les deux donnent le **même résultat négatif** : accuracy finale 19,7-20,1% (v1/v2), en réalité **légèrement pire** que le plateau de base (~25-32%) ; auto-appariement toujours en dessous du hasard (rang 1,69-1,75 sur 4, hasard 1,50 ; top-1 15,5-20% contre 25%) ; la température apprise bouge à peine de son initialisation (4,0 → 4,1-4,2).

**Réinterprétation importante** : un paramètre de sharpening **amplifie** un signal de préférence existant mais trop plat — il ne peut rien créer si le signal brut est **absent** (voire légèrement anti-corrélé, comme mesuré précédemment). Puisque le rang d'auto-appariement était déjà au niveau du hasard ou en dessous *avant* toute tentative de correction, l'hypothèse "il manque juste un facteur d'amplification" ne tient probablement pas — le vrai manque semble être en amont : rien ne pousse `q_proj`/`k_proj` à *représenter* une préférence correcte en premier lieu, pas seulement à l'exprimer plus fort. Ça oriente vers les recommandations 3-4 du rapport de recherche (perte contrastive InfoNCE, ou supervision explicite de l'attention) plutôt que vers un réglage du même type de correctif.

**Sur `use_ff` et la non-linéarité, en réponse à une question directe de l'utilisateur** : `use_ff=True` ne cible pas ce problème — il rend `fuse_proj` non-linéaire (l'étape de **combinaison** après récupération), pas `q_proj`/`k_proj` (l'étape de **comparaison** elle-même, en amont). Rendre `q_proj`/`k_proj` non-linéaires ne semble pas non plus prometteur : le contrôle `n_hops=1` ci-dessus prouve que la forme **linéaire** actuelle peut déjà discriminer quasi parfaitement (91,8% top-1) — la limite n'est pas une question d'expressivité architecturale de la fonction de matching, mais de ce qui pousse cette capacité à émerger/se maintenir pendant l'entraînement sur la tâche plus dure.

**Test GPU en cours (experiment-manager, A100 idle, Nantes)** : la même grille (`--sharpened` vs baseline × 3 graines, `d_model=128`) tourne à plus grande échelle — prévenu du résultat CPU négatif ci-dessus pour contextualiser, mais le test continue (échelle différente, LR pas encore re-sweepé pour la nouvelle paramétrisation, donc pas encore concluant en soi).

**Supervision explicite de l'attention (recommandation 3-4 du rapport de recherche) — résultat décisif, positif (2026-09-13)** : `learn/indexed_attention/diagnose_attention_supervision.py`. Perte auxiliaire de type contrastif (CE sur les scores `q_proj(R)·k_proj(clé_candidate)`, restreinte aux `n_facts` candidats réels de l'épisode, **pas** de nouveaux paramètres — seule la fonction de perte change) à deux points :
- **Saut 1** ($t=0$) : $R_0$ est exactement la requête naturelle du premier saut (vérité terrain = le fait dont la clé égale la requête) — un seul terme, bien défini.
- **Saut 2** ($t=1 \ldots N_{\text{step}}-1$) : pas d'étape fixe où le second saut *devrait* se résoudre — la perte est moyennée sur **toutes** les étapes restantes, pour superviser quelle que soit l'étape que le modèle choisit naturellement, sans avoir à la deviner/forcer.

**Résultat** (4000 pas, même échelle CPU que tous les tests précédents, `aux_weight=1.0`) :
- **Auto-appariement** : top-1 = **88,7%** (hasard 25%), rang moyen 0,152 sur 4 (hasard 1,50) — quasiment au niveau du contrôle `n_hops=1` (91,8%). Le matching `q_proj`/`k_proj` **se répare** avec cette supervision.
- **Accuracy de tâche finale** : **72,9%**, contre ~25-32% pour le plateau de base — toujours en hausse au moment où le budget d'entraînement s'arrête (68,8% à l'étape 4000, tendance encore montante), pas encore un plafond confirmé.

**Conclusion** : ça confirme directement l'hypothèse retenue après l'échec du sharpening — le problème n'était pas un manque d'amplification d'un signal existant, mais l'**absence totale d'un signal d'entraînement direct** poussant `q_proj`/`k_proj` vers la bonne discrimination ; le gradient dilué venant uniquement de la perte de tâche finale (à travers `fuse_proj`→SM→`OutputStream`) n'y suffit pas seul, mais fonctionne très bien une fois complété par une supervision directe et bon marché (aucun paramètre ajouté).

**Budget plus long (12 000 pas au lieu de 4000), même échelle CPU — le plateau est complètement résolu (2026-09-13)** :
- **Accuracy de tâche finale : 99,2%** (contre ~25-32% en baseline, et 72,9% au budget précédent, encore montant) — la tâche `n_hops=2` est essentiellement **résolue**.
- **Auto-appariement : top-1 = 100,0%, rang moyen = 0,000** (hasard 25%/1,50) — discrimination **parfaite**, au niveau (voire au-delà) du contrôle `n_hops=1` (91,8%).

**Ce que ça clôt (à l'échelle CPU seulement — [NUANCÉ], voir résultat GPU ci-dessous)** : à `d_model=32`, le plateau de composition `n_hops=2` — sweep `N_step` (écarté), sweep LR à budget complet (écarté), `use_ff` (neutre), `n_register` (neutre/pire), sharpening/normalisation de requête (échec) — était en réalité un problème de **signal d'entraînement**, pas de capacité architecturale ni d'hyperparamètre. Une supervision contrastive bon marché, sans paramètre supplémentaire, suffit à le résoudre complètement à cette échelle (99,2%/self-match parfait).

**Résultat GPU (`d_model=128`, experiment-manager, 3 graines chacun) — [CORRECTIF IMPORTANT, ne reproduit PAS le résultat CPU]** :

| variant | seed 0/1/2 final_acc | mean_rank | top1_rate |
|---|---|---|---|
| baseline | 0,246 / 0,264 / 0,256 | 1,25 / 1,26 / 1,59 | 0,34 / 0,40 / 0,27 |
| attn_supervised | 0,247 / 0,245 / 0,257 | **0,000 / 0,000 / 0,000** | **1,00 / 1,00 / 1,00** |

Le diagnostic d'auto-appariement est **parfait sur les 3 graines** (mean_rank=0, top1=100% — la perte auxiliaire corrige totalement et de façon robuste le problème mécanistique visé), **mais `final_acc` ne bouge quasiment pas** (~0,25 dans les deux cas) — le plateau de tâche persiste **malgré** un matching clé↔requête parfaitement réparé.

**Conclusion révisée, plus importante que le résultat CPU seul** : le matching `q_proj`/`k_proj` cassé était un **vrai** problème (confirmé, corrigé, robuste) mais **pas LE** goulot qui limite l'accuracy de tâche à cette échelle — il est **nécessaire mais pas suffisant**. Le vrai facteur limitant se trouve ailleurs, probablement en aval : le chemin SM (comment le fait retrouvé au second saut est effectivement écrit/lu dans la mémoire court-terme) ou `OutputStream` (comment ce contenu SM est traduit en réponse finale) — cette hypothèse était déjà envisagée plus haut dans ce document. **Pourquoi le CPU (d_model=32) a-t-il montré une remontée à 99,2% alors que le GPU (d_model=128) non ?** Non résolu — hypothèses à départager : (a) une échelle de $d_{\text{model}}$ différente change qualitativement le comportement (peu probable a priori, mais pas exclu), (b) un budget d'entraînement différent (12000 pas CPU vs 20000/24min GPU, débit différent — vérifier le nombre de pas réellement atteints côté GPU avant walltime), (c) un effet de graine/instabilité non contrôlé. **Prochaine étape, avant toute nouvelle piste architecturale** : reproduire précisément la config CPU qui a marché (mêmes `d_model=32`, mêmes hyperparamètres) à un budget de pas identique sur GPU, pour confirmer si l'écart vient de l'échelle ou du budget — écarter cette confusion avant d'aller chercher le "vrai" goulot en aval.

**Trou méthodologique identifié par l'utilisateur dans le sweep `N_step` ci-dessus** : le sweep n'a fait varier que $N_{\text{step}}$ en gardant `n_hops=2` **fixe** — il ne dit donc que "plus de budget ne débloque pas *ce* problème à 2 sauts précis", pas si $N_{\text{step}}$ et `n_hops` **composent** ensemble en général. Un test plus complet, à ajouter (pas bloquant pour (1)/(2) ci-dessus, complémentaire) : une **grille jointe** $(n_{\text{hops}}, N_{\text{step}})$ plutôt qu'un balayage 1D — augmenter `n_hops` (3, 4...) en même temps que $N_{\text{step}}$, **et** inclure explicitement des points où $n_{\text{hops}} > N_{\text{step}}$ (moins d'itérations que de sauts nécessaires — cas dégénéré, échec attendu, sert de plancher de calibration). Deux lectures possibles :
- Si le plateau (~24-25%) se retrouve identique **quel que soit** le rapport $n_{\text{hops}}/N_{\text{step}}$ (y compris quand $n_{\text{hops}} \le N_{\text{step}}$ largement) → renforce (c) : ce n'est vraiment pas une question de budget, à aucune combinaison testée.
- Si une combinaison $(n_{\text{hops}}, N_{\text{step}})$ **différente** de $(2, 8\text{-}32)$ débloque un signal net → nuance (c), le problème serait plus spécifique qu'une limite générale de composition (à creuser lequel).
À lancer après/en parallèle de (1)/(2) selon les ressources disponibles — pas urgent devant `use_ff`, mais à ne pas oublier.

**Confirmation batch/LR co-scaling (même session, à `n_hops=1`/tâche simple)** : `batch_size=256` avec `lr=1.2e-3` (scaling linéaire depuis la config de référence, ×4 batch → ×4 LR) atteint 99,9% acc ; `lr=2.4e-3` diverge (7,1%, loss 4,3) ; `lr=6e-4` légèrement en dessous (99,55%). La règle de scaling linéaire a tenu exactement à ce saut de ×4 — confirme la recommandation donnée précédemment (rescaler puis revérifier par un sweep étroit, pas supposer). `batch_size=256, lr=1.2e-3` devient la config de référence pour les prochains runs à cette échelle.

## Phase 3 — Passage à des données textuelles réelles

**Hypothèse** : `HierarchicalMemory` tient sur un vocabulaire/texte réel sans dégradation qualitative majeure par rapport au vocabulaire synthétique minuscule des phases précédentes.

**Raisonnement** : le compresseur n'a aucun poids appris hors la query de pooling et l'embedding de position (audit : "est-ce assez expressif pour du texte réel ?"). Le risque principal est qu'un pooling aussi simple ne capture pas une sémantique utile sur un contenu réellement varié, contrairement au jouet synthétique à vocabulaire de 32-64 tokens.

| Observation | Action |
|---|---|
| Performance comparable (ajustée à la difficulté de la tâche) | Continuer vers Phase 4 |
| Dégradation nette non expliquée par la difficulté de la tâche elle-même | Revisiter le résultat de Phase 1bis "avec/sans FF" spécifiquement pour le compresseur — c'est le candidat le plus probable pour un manque d'expressivité sur données réelles |

**Protocole** : réutiliser `scripts/fetch_wiki.py`/`data/wiki_samples.json`, tokenizer réel, Baselines A/B/C reconduites.

## Phase 4 — Introduction d'un vrai Teacher (distillation, Option A d'abord)

**Hypothèse** : la distillation Top-K KD (Option A, black-box) sur le stream `answer` converge au moins aussi bien qu'un entraînement CE pur, avec moins de données.

**Raisonnement** : hypothèse standard de la distillation (signal du Teacher plus riche que des labels durs) — mais spécifique à ce projet, il faut vérifier en plus que le stream reste capable de lire correctement la SM/KB plutôt que d'imiter le Teacher en surface (le stream n'a pas de FF, cf. Phase 1bis).

| Observation | Action |
|---|---|
| Convergence ≥ baseline CE, diagnostic d'attribution causale toujours correct | Continuer |
| Convergence OK mais diagnostic causal dégradé (le stream "triche" en imitant le Teacher sans vraiment lire la KB) | Ajouter une supervision partielle sur l'attention elle-même (Option B légère) avant de continuer |

Réutiliser `learn/distill/precompute_teacher_targets.py` et `topk_kd_loss()` (déjà validés EXP-003 à EXP-006). Vérifier le piège FP8 déjà rencontré sur ce projet (`dev_notes/experiment.log.md`) avant de faire confiance aux cibles précalculées.

## Phase 5 — Stratégie de construction de la KB : espace unifié vs Sub-KB par batch

**Hypothèse** : la stratégie 2 (Sub-KB par batch + distracteurs) réduit le risque de dilution mesuré en Phase 0/3, au prix d'un coût de calcul par batch plus élevé.

| Observation | Action |
|---|---|
| Stratégie 2 significativement meilleure | Basculer par défaut, absorber le coût |
| Pas de différence significative | Garder la stratégie 1 (plus simple, moins chère) |

## Phase 6 — Montée en échelle avec muP (parallélisable avec Phase 4/5)

**Hypothèse** : le LR optimal au plus petit palier transfère aux paliers plus larges/profonds **si** Depth-muP est appliqué dès le départ.

**Raisonnement** : leçon déjà apprise dans le chantier de distillation (`dev_notes/experiment.log.md`) — le transfert échoue si profondeur et largeur augmentent simultanément sans correction de profondeur. Ne pas reproduire cette erreur ici.

| Observation | Action |
|---|---|
| Transfert stable sur 2-3 paliers | Adopter comme discipline par défaut pour tout scaling futur |
| Transfert instable malgré Depth-muP | Isoler profondeur/largeur comme deux scans séparés avant de re-tenter un scaling conjoint |

## Phase 7 — Benchmark multi-sauts réel (HotpotQA / MuSiQue)

**Hypothèse** : le pattern Baseline A/B/complet observé en synthétique (Phase 2) se reproduit sur données réelles.

**Raisonnement** : c'est le test qui valide (ou invalide) que la tâche synthétique était un proxy prédictif, pas un artefact du jouet.

| Observation | Action |
|---|---|
| Pattern reproduit | Le proxy synthétique était valide, confiance accrue dans les décisions prises via Phases 0-2 |
| Pattern non reproduit | Les tâches synthétiques étaient trompeuses — revoir leur conception avant de les réutiliser pour de futures itérations |

## Phase 8 — Multi-output streams pour de vrai

**Hypothèse** : un stream `thinking` en embedding aligné sur une couche médiane du Teacher améliore la qualité du raisonnement mesurée sur le stream `answer`, par rapport à l'absence de guidage latent.

| Observation | Action |
|---|---|
| Amélioration mesurable sur `answer` | Garder le stream thinking, définir le schedule d'extinction (§11bis, non formalisé) empiriquement |
| Pas d'effet ou effet négatif | Le guidage latent n'est pas nécessaire à ce stade — ne pas ajouter de complexité non justifiée |

## Phase 9 — Sanity check généraliste

**Hypothèse** : le modèle reste un générateur de texte cohérent, pas seulement un moteur de lookup+chaînage.

**Raisonnement** : lié directement à la préoccupation de Phase 1bis (zéro non-linéarité hors softmax) — si le modèle échoue ici, la cause la plus probable est celle déjà identifiée, pas quelque chose de nouveau à diagnostiquer de zéro.

| Observation | Action |
|---|---|
| Comportement d'assistant cohérent maintenu | Le design "sans FF" n'a pas sacrifié la fluidité de base |
| Dégradation nette de fluidité/grammaire | Revenir sur le résultat de Phase 1bis "avec/sans FF" — probable que la capacité de calcul non-linéaire manquante affecte aussi la génération de surface, pas seulement le raisonnement multi-sauts |

## Phase 10 — KB persistante apprise (Product-Key-Memory-style) — nouvelle capacité, pas encore implémentée

**Décision de l'utilisateur (2026-09-13)** : au-delà de la récupération par épisode déjà testée (Phase 1bis/2), le modèle doit aussi pouvoir **construire sa propre KB long-terme avec des embeddings appris** — accumuler de la connaissance à travers les exemples d'entraînement, pas seulement récupérer ce qui est donné en contexte. Cf. spec §8bis pour la proposition d'implémentation (niveau de mémoire additionnel, $(K,V)$ appris comme paramètres, façon Product-Key Memory — Lample et al. 2019 / "Memory Layers at Scale", Meta 2024).

**Hypothèse** : sur un corpus où les faits **récurrent réellement à travers les exemples** (contrairement à `kb_retrieval`/`kb_chain_retrieval`, délibérément à faits uniques par épisode), le modèle apprend à écrire dans la mémoire persistante l'information réutilisable plutôt que de la re-dériver à chaque fois — testable en mesurant si la performance sur des faits récurrents s'améliore avec le nombre de fois qu'ils ont été vus en entraînement (signature d'accumulation), à distinguer d'une performance plate qui indiquerait que tout passe par la KB par épisode sans accumulation réelle.

**Corpus proposés (question de l'utilisateur)** : `TinyStories` (déjà dans `learn/distill/prepare_general_data.py`, faits/relations simples récurrents) pour un premier test peu coûteux, puis `open-r1/OpenR1-Math-220k` (déjà dans le pipeline) pour un test plus proche de la thèse (formules/identités factuelles réutilisables, pas seulement du vocabulaire général).

**Protocole (esquisse, à affiner)** : entraîner sur un sous-ensemble à faits limités (nombre contrôlé de faits/relations distincts, répétés across exemples avec une fréquence variable), comparer accuracy/perplexité par fréquence d'occurrence du fait en entraînement (bucket rare/moyen/fréquent), ≥3 seeds. Diagnostic d'attribution causale (déjà standard dans ce plan) adapté : vérifier que l'attention se concentre sur les slots de la mémoire persistante pour les faits récurrents fréquents, pas sur la KB par épisode ni sur le registre.

**Dépendance** : nécessite que `Thinker` (pas le transformer dense de `train_sft.py`, qui ne sert qu'à valider le pipeline de distillation) soit le modèle réellement entraîné — cette phase est donc postérieure au branchement effectif de l'architecture Indexed Attention dans le pipeline de distillation, pas bloquante pour les phases 0-9 qui testent le mécanisme indépendamment de ce branchement.

**Ne remplace aucune phase existante** : la KB par épisode (Phase 1bis/2) reste le bon banc d'essai pour la récupération ; cette phase teste l'**accumulation**, une capacité distincte et complémentaire.

**Extension possible, à considérer seulement après un premier résultat PKM-style (pas en même temps)** : **Titans / Infini-Transformer** (Google, déjà dans `raw/Branch-•-Indexed-Attention.md` ligne 543, cf. spec §8bis) — mémoire différentiable mise à jour **en continu, y compris à l'inférence**, via une règle d'apprentissage associatif basée sur une métrique de "surprise", plutôt qu'une mémoire figée après l'entraînement (comme le PKM-style ci-dessus). Plus complexe (règle d'écriture différentiable dédiée à concevoir et valider), donc à ne tester qu'une fois la version PKM-style (accumulation par gradient d'entraînement seul) validée ou clairement insuffisante — pas une alternative à essayer en parallèle dès le départ.

## Phase 11 — Intégration réelle sur texte long : fenêtrage + sortie multi-position **[PROPOSITION 2026-09-13, pas encore implémentée]**

**Contexte** : distincte de la Phase 3 (données/vocabulaire réels, mais forme toujours "une requête → une réponse" façon `kb_chain_retrieval.py`, cf. `wiki_samples.json`). Ici, il s'agit de brancher pour de vrai `Thinker` (pas le transformer dense de `train_sft.py`, qui ne sert que de véhicule pipeline pour la piste distillation) sur un objectif LM **par position** sur des documents longs — la tâche explicitement demandée ("intégration sur texte réel"), voir spec §14 pour la conception complète (fenêtrage glissant, récurrence de $R$ entre fenêtres avec SM remise à zéro, généralisation multi-position d'`OutputStream`).

**Hypothèse** : le mécanisme (cœur récurrent + KB hiérarchique + registre reporté entre fenêtres) atteint au moins la perplexité d'un transformer dense de taille comparable sur le même découpage de document, avec l'argument d'efficacité du projet (§13) portant sur le budget de paramètres/profondeur de calcul, pas sur la qualité brute à budget égal.

| Observation | Action |
|---|---|
| Perplexité comparable (à budget de paramètres/FLOPs ajusté) | Continuer vers Phase 4/9 sur ce pipeline réel plutôt que sur le transformer dense placeholder |
| Dégradation nette | Diagnostiquer séparément chaque nouveauté de cette phase : récurrence de $R$ (`detach_register_across_windows` on/off), $T_{\text{local}}$/$T_{\text{tgt}}$ trop petits ou trop grands, sortie multi-position (comparer à la sortie mono-position sur une tâche de contrôle équivalente à `kb_chain_retrieval.py` pour isoler si le bug vient du nouveau mécanisme de sortie ou du fenêtrage) |

**Protocole (esquisse)** : implémenter `data/real_text_windows.py` (§14.5) + les changements d'interface `Thinker.forward` (§14.6), valider d'abord sur CPU à toute petite échelle (quelques documents courts, $N_{\text{ctx}}$/$T_{\text{tgt}}$ minuscules) avant tout passage Grid5000, dans le même esprit que le MVP initial (tests locaux avant GPU). ≥3 seeds une fois à l'échelle GPU.

**Dépendance** : bloquante pour la Phase 10 (KB persistante apprise) et pour tout entraînement KD réaliste sur l'architecture Indexed Attention elle-même (le run KD réaliste actuellement en cours côté experiment-manager tourne sur le transformer dense placeholder, pas sur ce pipeline).

**Ne remplace pas** : les Phases 1/1bis/1quater/2 (mécanisme testé sur tâches synthétiques courtes, cadre plus simple pour isoler les effets) restent le bon outil de diagnostic mécanistique — cette phase est un passage à l'échelle/à la réalité, pas un remplacement du protocole de diagnostic.

## Ce que ce plan laisse volontairement de côté

No-Op/largeur adaptative, curriculum de largeur "large→étroit" — reportées après la Phase 7. Les introduire plus tôt ajouterait des variables libres avant d'avoir une base fiable pour juger si elles aident.

**[RÉTRACTÉ le 2026-09-13]** : l'unification ci-dessus (Sentinel = No-Op + largeur variable + step compute-only) était une erreur — le Sentinel ne couvre que le cas binaire No-Op, pas un contrôle gradué de largeur/précision ni un mécanisme de step "compute-only". **[OUVERT]** : la largeur de récupération variable (rendre le modèle plus précis selon la tâche courante) et les steps "compute-only" (itérations sans appel mémoire, pertinent pour le débat $N_{\text{step}} \gtrsim 2\text{-}4\times L$ de la Phase 1quater) restent deux besoins identifiés par l'utilisateur, qui soupçonne qu'une approche unifiée existe entre ces deux-là — pas encore trouvée, cf. spec §7.1 pour le détail et ne pas réintroduire le lien avec le Sentinel sans nouvelle justification.
