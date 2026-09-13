# Ordonnancement du calcul — saturer les nœuds disponibles

Écrit le 2026-09-13 (session `thinker-5b`, clôture) en réponse à une question directe de l'utilisateur : comment planifier les expériences pour maximiser l'utilisation des GPU sur un nœud **et** sur l'ensemble des nœuds, et où placer le calcul CPU.

Faits d'infra repris de `dev_notes/grid5000_usage.log.md` et de la skill `grid5000` — ne pas les redécouvrir par essai-erreur. Ce qui est marqué **[à mesurer]** n'a pas été vérifié et doit l'être avant d'être traité comme acquis.

## 1. La contrainte réelle n'est pas la puissance GPU

Mesures déjà au journal : **5-30 % de calcul et 2-7 % de VRAM** par process avec un run par GPU, sur des modèles à 0,7-1,6 M paramètres. En passant à `batch_size=256` et 2 runs par GPU : 31-41 % de calcul, 8-14 % de VRAM, soit 1-2,6 Go sur 24. Deux conclusions :

1. **Le GPU est la mauvaise unité d'allocation** pour ces charges. Ce qui limite le débit, c'est le **nombre de processus lancés**, pas la carte. Augmenter `batch_size` ne suffit pas et rouvre un confondant batch/LR sur des grilles en cours.
2. **La préemption besteffort est le vrai coût.** Le groupe `wide` n'a aucune allocation GPU prioritaire à Rennes : tout job GPU y est `-t besteffort` et peut mourir à tout moment (observé : parfois en ~25 min). Une grille conçue comme « un job OAR par cellule » perd donc des cellules en permanence.

Le plan ci-dessous attaque ces deux points séparément : **étager les charges** (§2) et **découpler les cellules des réservations** (§3).

## 2. Étager les charges selon ce dont elles ont réellement besoin

| Tier | Charge | Besoin réel | Où l'envoyer |
|---|---|---|---|
| **A** | Grille toy-memory Exp. 1 (`learn/toy_memory/`, `d_model=64`, `seq_len` 8-32) — ~100 K paramètres | **Aucun GPU.** Ces modèles ne saturent rien ; le CPU est adapté | Rennes `paradoxe`, **queue normale** (pas besteffort) |
| **B** | Indexed Attention Phase 2-redo et Phase 0bis (`d_model=256`, 0,7-1,6 M paramètres) | Un GPU modeste, mais surtout **beaucoup de processus** | `abacus3`/`abacus10` (A5000), `abacus22`/`abacus4` (A40) — besteffort, empilés |
| **C** | Distillation : entraînement KD 810 M, précompute Teacher 27 B FP8 | **Vrai besoin GPU** : ≥56 Go de VRAM ou FP8 natif | `abacus26` (L40S), `abacus27` (H100), `abacus21` (A100) — besteffort |

**Le point qui change le plus les choses : le Tier A part sur CPU en queue normale.** Les jobs CPU à Rennes passent en queue normale (jobs 4091061, 4091111, 4091210 au journal), donc **sans préemption**. Une grille de 84 cellules qui tourne tranquillement sur un nœud CPU réservé 6 h est plus productive qu'une grille GPU besteffort qui se fait tuer trois fois. Et ça libère les GPU pour le Tier C.

**Corollaire à tenir** : ne jamais laisser le Tier A ou B occuper `abacus26`/`abacus27`. Ce sont les seules cartes qui font tourner le Teacher FP8 (Ada/Hopper ont le FP8 natif ; l'A100 80 Go passe par déquantification bf16 ~56 Go). Y mettre un modèle de 100 K paramètres, c'est bloquer la seule ressource non substituable du projet.

## 3. Le patron à adopter : une file de travail, pas un job par cellule

C'est la réponse commune à la préemption et au remplissage. Au lieu de soumettre un job OAR par cellule de grille :

1. **Matérialiser la grille en fichier** — une ligne par cellule (tous les arguments), généré une fois.
2. **Réserver des nœuds entiers** (`-l gpu=4` / un nœud CPU complet), plusieurs simultanément, sur plusieurs clusters.
3. **Sur chaque nœud, lancer N workers** qui tirent la cellule suivante de la file, l'exécutent, écrivent leur résultat, recommencent.
4. **Un fichier de résultat par cellule**, écrit atomiquement (écrire dans un temporaire puis `mv`). Une cellule dont le fichier existe déjà est sautée.

Ce que ça donne :

- **La préemption ne coûte qu'une cellule par worker**, pas la grille. Le job suivant reprend la file là où elle en est, sans rien réordonnancer à la main.
- **Plus besoin de deviner combien de runs tiennent** : la file s'équilibre seule. Un nœud rapide en prend plus.
- **Idempotence** : relancer la même commande après une préemption est sûr et reprend le travail.
- **Hétérogénéité gratuite** : on peut ajouter un nœud A40 en cours de route, il se met à consommer la même file.

Tirer une ligne de façon atomique entre workers concurrents peut se faire simplement avec `flock` sur le fichier de file. La granularité doit rester grosse (une cellule = un run complet de plusieurs minutes), donc la contention n'est pas un sujet.

**Alternative plus simple si la file paraît trop lourde** : découper la grille en autant de tranches fixes que de workers (`cellule_index % n_workers == worker_id`). Ça perd l'équilibrage et la reprise propre après préemption, mais ça se code en une ligne. À réserver aux grilles courtes.

## 4. Paramètres de remplissage

**Sur GPU (Tier B).** Viser ~6-8 processus par carte pour ces tailles de modèle, puis mesurer et ajuster — la VRAM n'est pas contraignante (1-2,6 Go par run sur 24), c'est le calcul qui sature vers 31-41 % à 2 runs. **[à mesurer]** le point où l'occupation calcul plafonne ; ne pas extrapoler linéairement depuis 2 runs.

**Sur CPU (Tier A) — le piège à éviter.** PyTorch ouvre par défaut autant de threads qu'il y a de cœurs. Lancer 30 processus sur un nœud à 30 cœurs sans rien régler donne 900 threads qui se battent, et le débit s'effondre. **Fixer `OMP_NUM_THREADS=1` (et `MKL_NUM_THREADS=1`) par worker**, puis lancer autant de workers que de cœurs physiques moins deux. Pour des modèles à 100 K paramètres, un thread par process est le bon réglage : le parallélisme utile est entre les cellules, pas à l'intérieur d'une.

**Réserver plusieurs nœuds à la fois.** Un job besteffort ne consomme pas de priorité d'usage : il n'y a aucune raison de se limiter à une réservation. En lancer plusieurs sur des clusters différents augmente mécaniquement le débit et amortit la préemption (elles ne tombent pas toutes en même temps).

**Vérifier la disponibilité avant de soumettre** : `https://rennes.grid5000.fr/drawgantt-svg/` ou `oarsub --dry-run`. Piège déjà rencontré, noté dans la skill : ne jamais conclure à la saturation d'un site depuis les compteurs agrégés — un `busy` global peut être un job de maintenance. Échantillonner `.reservations[].types` avant de renoncer à un site.

## 5. Écriture des résultats et survie à la préemption

- **Les artefacts volumineux vont dans `/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/`**, jamais dans le home (quota 25 Go souple / 100 Go dur, extension à 200 Go en cours) ni dans `/tmp` du nœud (**effacé à la fin du job**, y compris lors d'une préemption).
- `/tmp` reste le bon endroit pour les **données de travail** copiées en début de job (disque local rapide), mais tout ce qui doit survivre est recopié avant la fin.
- **Sortie des processus distants : écrire dans un fichier sur le nœud**, pas dans le tuyau SSH de la session qui orchestre. Leçon déjà payée deux fois (journal du 12 sept.) : une coupure de la connexion locale tue le tuyau et perd toute la sortie alors que le process distant survit. Le patron qui marche : `... > ~/chemin/log 2>&1 < /dev/null & disown`, lancé par un `oarsh` bref qui rend la main immédiatement.

## 6. Où va quoi, concrètement, pour les grilles en attente

| Grille | Cellules (ordre de grandeur) | Tier | Placement |
|---|---|---|---|
| Toy Exp. 1 — contrôle préalable `copy` à `read_step=n_step` | 3 (seeds) | A | CPU, à faire **en premier et seul** : si ~100 % d'exact-match n'est pas atteint, le reste n'est pas interprétable |
| Toy Exp. 1 — grille `read_step` × 3 seeds × {`seq_len` 8, 32} × {`copy`, `cumsum`} | ~84 | A | CPU queue normale, file de travail |
| Toy Exp. 1 — curriculum sur les cellules à falaise | variable, à la demande | A | idem, ajouté à la file après lecture des premiers résultats |
| Indexed Phase 2-redo — ablation × `n_hops` {2,3,4} × 3 seeds | ~18 | B | A5000/A40 besteffort, ~6-8 process/GPU |
| Indexed Phase 0bis — `depth` {0,1,2,3} × `n_facts` {16,64,256} × 3 seeds | ~36 | B | idem, même file |
| Distillation (KD 810 M, précompute Teacher) | peu, longues | C | L40S/H100/A100-80 uniquement |

Les Tiers A, B et C n'entrent pas en concurrence : ils visent des ressources différentes et peuvent tourner **tous les trois en parallèle**. C'est le gain principal de cet étagement — aujourd'hui tout se bouscule sur les mêmes GPU.

## 7. Ce qui reste à vérifier

- **[à mesurer]** Nombre de cœurs réellement disponibles sur un nœud `paradoxe`, et débit réel d'une cellule toy sur un cœur unique — détermine si 84 cellules tiennent dans une réservation ou s'il en faut plusieurs.
- **[à mesurer]** Le plafond d'occupation calcul en fonction du nombre de processus par GPU (§4), sur A5000 et A40.
- **[à vérifier]** Que les modèles toy à `d_model=64` sont bien plus rapides sur CPU que le coût d'attente d'un GPU besteffort. C'est très probable à cette taille, mais ça n'a pas été chronométré.
- **Le PC de l'utilisateur** reste le bon endroit pour les tests rapides (quelques centaines de pas, vérification qu'un script tourne) — pas pour les grilles. Tout ce qui dépasse quelques minutes part sur le cluster.
