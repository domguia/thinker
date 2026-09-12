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

**Résultat préliminaire obtenu localement pendant cette session (CPU, 1 seed — à refaire dans les règles avec ≥3 seeds avant conclusion finale, mais assez net pour orienter la suite immédiatement)** : la branche "pas de transition dimensionnelle" ci-dessus s'est produite en pratique. À `n_facts=64`, ni `d_model=128`+`n_register=4` ni un LR plus bas (`1e-3`) n'ont débloqué l'apprentissage (loss reste au niveau du hasard). Pire : l'échec apparaît déjà à **`n_facts=16`**, et sur 3000 pas d'entraînement la loss reste plate sans tendance d'amélioration, même lente — un échec net et non-progressif, pas un problème de vitesse de convergence.

**Piste retenue en priorité avant de continuer à balayer les dimensions** (spec §11bis, motivation ajoutée cette session) : ce pattern est cohérent avec le phénomène documenté dans `raw/Distill-reasonning-stream.md` (« les transformers à poids partagés récurrents souffrent d'instabilité dynamique... contraindre les états latents du loop via une supervision intermédiaire agit comme régularisation »). Le stream `answer` seul ne fournit un gradient qu'après $N_{\text{step}}$ itérations + toute la hiérarchie — signal potentiellement trop indirect. **Action concrète à tester avant toute autre chose** : ajouter une supervision intermédiaire sur `kb_retrieval` (on connaît la vérité terrain de la récupération à chaque étape) — ex. une perte auxiliaire sur les poids d'attention du softmax unifié, forçant le nœud contenant la bonne valeur à recevoir le poids dominant à une étape donnée, en plus de la perte finale sur le stream `answer`. C'est une version sans Teacher de l'Option B ("White-Box Trajectory Distillation", déjà documentée dans la conversation source pour un contexte avec Teacher). Si cette supervision intermédiaire débloque l'apprentissage à `n_facts=16-64` sans changer les dimensions, ça confirme que le problème est bien un signal d'entraînement trop indirect, pas un problème de capacité ni un bug de mécanisme.

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
