# Plan d'expérimentation & de recherche — Indexed Attention (au-delà du MVP)

Objectif : passer du MVP testé uniquement en local sur CPU (`dev_notes/indexed_attention_spec.md` §11) à une validation progressive sur GPU (Grid'5000), alignée sur la vision complète du projet (séparer raisonnement et mémoire, cf. §-1 de la spec) plutôt que de s'arrêter à la mécanique de base.

Ce document a été révisé après un audit critique (3 agents indépendants, angles code/math, méthodologie, validité scientifique) qui a trouvé des trous réels : absence de contrôle statistique, absence de vérification d'attribution causale, généralisation jamais testée, et surtout — la décision la plus débattue de cette session (supprimer tout FF, §-1 de la spec) n'était testée nulle part. Ce plan corrige ça : chaque phase porte maintenant une **hypothèse explicite**, le **raisonnement** qui la sous-tend, et une **table observation → action** (y compris pour les résultats ambigus, pas juste les deux extrêmes).

## Méthodologie commune à toutes les phases (corrige les trous trouvés par l'audit)

- **Seeds** : chaque condition testée avec **≥3 seeds**, moyenne ± écart-type reportée. Un écart entre deux conditions n'est traité comme un signal réel que s'il dépasse ~2σ ; sinon → **"non concluant"**, jamais "pas d'effet".
- **Généralisation obligatoire** : split train/held-out dès la Phase 0 — la KB et les paires clé/valeur de l'évaluation sont régénérées avec un seed disjoint de l'entraînement, jamais vues pendant l'entraînement. Le test de sur-apprentissage CPU actuel (`tests/test_indexed_memory.py`, même batch fixe train=eval) reste un test de **plomberie** (le pipeline tourne), pas un test de **mécanisme** — ne pas confondre les deux, et ne pas répéter cette faiblesse aux échelles GPU.
- **Diagnostic d'attribution causale** : en plus de l'accuracy, logguer où se concentre le poids d'attention du softmax unifié pour l'exemple cible (le nœud de la hiérarchie contenant la bonne réponse doit recevoir le poids dominant). Sans ça, une accuracy élevée peut venir d'un raccourci (le registre ou la SM mémorisant l'info sans jamais passer par la KB) plutôt que du mécanisme qu'on prétend valider.
- **Résultat ambigu** : si l'intervalle ± écart-type de deux conditions se chevauche, la règle par défaut est **+3 seeds supplémentaires, ou +1 palier d'échelle**, jamais conclure sur un signal ambigu.

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

**Pourquoi cette phase existe** : trouvée nécessaire *en direct* pendant cette session — `thinker-e9` a rapporté un échec total à `depth=4`/64 facts (loss bloquée à `ln(vocab)`, accuracy au hasard). Reproduit localement en quelques minutes (CPU, `IndexedThinker`) : avec KB **resamplée à chaque batch** (signal réel, pas mémorisation), 4 facts → apprend (~93% acc), 64 facts → n'apprend plus **du tout**, et ce **indépendamment de la profondeur** (`depth=4` et `depth=2` échouent pareil à 64 facts, à `d_model=64`). Ça élimine "la hiérarchie est cassée" comme explication première et pointe vers un **sous-dimensionnement** (`d_model`, `n_register`) — exactement le type de confusion que l'audit redoutait (attribuer à l'architecture un échec qui est en fait un problème de taille).

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

| Observation | Action |
|---|---|
| avec-FF ≈ sans-FF sur récupération pure ET sur une tâche demandant un calcul simple sur les valeurs récupérées | Prémisse §-1 supportée, garder sans-FF |
| sans-FF nettement pire sur la tâche de calcul mais pas sur la récupération pure | Prémisse partiellement supportée — réintroduire un FF minimal dans la **boucle principale seulement** (pas le compresseur, pas les streams), pour la composition, pas le stockage |
| sans-FF pire même sur récupération pure | Remettre en question §-1 dans son ensemble avant d'aller plus loin |

**Autres variantes déjà câblées, zéro code à ajouter :**
- **$M=1$ vs $M>1$** (`n_slots`). Hypothèse : $M>1$ aide si les blocs contiennent plusieurs "aspects" distincts (littérature Slot Attention). Observation nulle → ambigu à cette échelle (cf. clause de résultat ambigu ci-dessus), pas "aucun bénéfice".
- **Confirmation, pas une variante à trancher** : $Q_{KB}$/$Q_{SM}$ sont déjà découplés par défaut.

**Variantes nécessitant un petit flag :**
- **Stop-gradient sur les clés SM**. Hypothèse : pas de différence visible à cette échelle — son rôle (éviter la "moving target" pour l'indexeur, cf. comparison.md DSA/QSA) est surtout pertinent à grande échelle/longue durée d'entraînement. Résultat nul ici → reporter le test à la Phase 3/4, ne pas conclure.
- **Dropout stochastique des niveaux hauts**. Hypothèse : bénéfice attendu en généralisation **hors distribution de profondeur**, pas en accuracy in-distribution — le protocole doit donc évaluer à une profondeur différente de l'entraînement pour être probant, sinon le test ne teste rien.

**Reportées à une comparaison dédiée plus lourde** : softmax unifié vs gating façon NSA, routeur explicite vs implicite (nécessiteraient une seconde implémentation quasi complète).

**Protocole** : tâche de la **Phase 0** (pas Phase 2 — correction d'une erreur de séquencement de la version précédente de ce plan, qui référençait une phase pas encore atteinte), 3 seeds par variante.

## Phase 1ter — Stream `thinking` en embedding (alignement Teacher, avancé plus tôt sur demande explicite)

**Décision** : initialement prévu en Phase 8 (après données réelles + distillation), avancé ici sur demande explicite de l'utilisateur — tester tôt comment le mécanisme se comporte, avant d'investir dans les phases suivantes.

**Hypothèse** : un stream `thinking` supervisé par alignement (cosinus, pas MSE brute — voir raisonnement plus bas) sur une couche intermédiaire d'un vrai Teacher pré-entraîné donne un signal d'entraînement supplémentaire cohérent (loss qui décroît proprement), sans dégrader la convergence du stream `answer` sur `kb_retrieval`.

**Raisonnement** : §11bis de la spec formalise maintenant le critère de choix de couche (éviter 0-25% et la dernière couche, cibler ~40-65%, valider par un protocole de sonde plutôt qu'un choix a priori) et le risque d'instabilité réel (perte non normalisée + conflit multi-objectif sur des poids partagés, pas "distance à l'initialisation" comme argument invoqué au départ — corrigé après discussion avec l'utilisateur). Les deux causes réelles d'instabilité sont atténuables (perte cosinus normalisée, montée en poids progressive), donc testables sans risque disproportionné.

**Ce qui est déjà en place, aucun changement d'architecture requis** : `IndexedThinker(stream_dims={'answer': ..., 'thinking': teacher_hidden_dim})` fonctionne tel quel. `core/compressor/model_wrapper.py::HFModelWrapper.get_hidden_states()` (ajouté cette session) extrait les couches intermédiaires d'un Teacher HF déjà supporté (GPT-2/SmolLM/Qwen).

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

## Ce que ce plan laisse volontairement de côté

No-Op/largeur adaptative, curriculum de largeur "large→étroit" — reportées après la Phase 7. Les introduire plus tôt ajouterait des variables libres avant d'avoir une base fiable pour juger si elles aident.
