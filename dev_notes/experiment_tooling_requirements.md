# Besoins d'un outil de gestion des expériences

Document de travail, pas une spec figée. Point de départ : la session du 2026-09-14
(voir `experiment.log.md` et `grid5000_usage.log.md`), où la quasi-totalité des
incidents d'infra provenait du même trou — aucune vue fiable de l'état réel entre
"on a lancé quelque chose" et "on a un résultat".

## 0. Principe directeur

**L'outil rend l'état vrai bon marché à voir et la dérive risquée bruyante à
ignorer. Il ne décide jamais à la place de l'agent qui le lit.**

Concrètement : l'outil peut *détecter* qu'un job est probablement mort, qu'une
grille va dépasser son walltime, qu'un doublon de calcul est sur le point de se
lancer — il ne doit jamais *décider seul* de tuer un job, changer une priorité,
ou classer un résultat comme concluant. Chaque capacité listée plus bas porte
cette distinction explicitement (colonne "Automatisable" / "Reste à l'agent").

C'est la seule garde-fou contre la rigidité que l'utilisateur a demandée : un
outil qui se contente d'observer et d'alerter ne peut pas devenir un carcan,
même s'il grossit beaucoup.

## 1. Ce qu'il faut monitorer — GPU **et** CPU, pas seulement GPU

Angle mort de la première passe de réflexion : ce projet fait tourner deux
familles de charges très différentes sur des files différentes (voir
`compute_scheduling.md`) —

- **Tier C (GPU, besteffort)** : Indexed Attention, Phase 3, distillation.
- **Tier A (CPU, file normale)** : `toy_memory`, petits modèles (`d_model=64`,
  quelques centaines de milliers de paramètres), lancés en dizaines de process
  parallèles sur un seul nœud (`OMP_NUM_THREADS=1` par worker).

La sous-utilisation et les plantages silencieux touchent **les deux**, avec des
signatures différentes :

| | GPU (Tier C) | CPU (Tier A) |
|---|---|---|
| Sous-utilisation | `nvidia-smi` : %util faible, VRAM presque vide (observé toute la session : 11-35%, <2 Go/24-46 Go) | Nombre de workers lancés < cœurs disponibles, ou l'inverse — sursouscription |
| Sur-souscription | Rare (VRAM sature avant le compute en général) | **Déjà rencontré ce projet** : PyTorch prend un thread par cœur par défaut ; N process sans `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` sur un nœud à N cœurs donne N² threads en contention |
| Signal de vie | `nvidia-smi` par job/GPU (cgroup-scopé) | `ps aux` + `uptime`/`load average` du nœud, par job |
| Piège d'assignation | Un `gpu=2` peut atterrir sur 1 ou 2 nœuds (voir friction log C2) | Un `host=1` est toujours un nœud entier — pas d'équivalent, mais le nombre de cœurs varie beaucoup d'un nœud `paradoxe` à l'autre (déjà vu : 16 à 52 cœurs) et change le nombre de workers à lancer |

**Besoin concret** : la boucle de réconciliation (friction log A2) doit avoir
deux branches de collecte, pas une — `nvidia-smi` **et** `uptime`/`ps`/charge
CPU par nœud, unifiées dans le même statut composite par job. Aujourd'hui je
fais ça à la main et seulement quand quelqu'un pose la question ("current
utilization ?") — jamais en continu, jamais pour les jobs CPU spécifiquement
(je n'ai vérifié la charge CPU des nœuds `paradoxe` qu'une seule fois ce soir,
sur demande, jamais de façon proactive).

## 2. Le chaînon manquant : la progression doit venir de la tâche, pas être devinée de l'extérieur

Question posée : est-ce qu'une tâche a sa propre estimation de fin, avec
quelque chose comme `tqdm` ? Réponse courte : **pas aujourd'hui, et `tqdm` tel
quel n'est pas le bon outil ici.**

- `tqdm` écrit une barre de progression via des retours chariot (`\r`) —
  parfait dans un terminal interactif, **illisible une fois redirigé dans un
  fichier de log** (`> log.txt 2>&1`, ce qui est notre cas systématique sur
  Grid'5000) : soit ça pollue le fichier de milliers de lignes partielles,
  soit ça n'écrit rien de lisible du tout selon le buffering.
- Le format qui marche déjà, utilisé sans le nommer dans certains scripts de ce
  projet (`train_real_text.py` : `step= 1200 elapsed=0.43m loss=... ppl=...`) :
  **une ligne texte simple, complète, à intervalle fixe** (`--log_every`), pas
  une barre. Ça grep bien, ça survit à une redirection, ça s'affiche pareil en
  interactif ou en batch.

**Ce qui manque à ce format aujourd'hui** : l'ETA elle-même. Le script sait son
`step` courant et son `max_steps`/`max_time_minutes` — rien ne l'oblige à
calculer et imprimer un temps restant. Ajouter ça au format standard
transformerait "combien de temps il reste" d'un calcul externe approximatif
(item B3 du friction log — extrapoler depuis un run passé "similaire") en une
donnée que l'outil n'a qu'à lire.

**Proposition concrète : un format de ligne de progression standard**, que
tout script d'expérience de ce projet devrait imprimer à `--log_every` :

```
progress: step=<n>/<max_steps> elapsed=<s>s eta=<s>s <métrique clé>=<valeur>
```

Simple, greppable par l'outil (une regex, pas un parseur par script), et
suffisant pour dériver l'ETA sans historique ni devinette.

### Est-ce que ça devrait influencer la conception des tâches elles-mêmes ?

Oui, et c'est exactement le bon réflexe. Ce n'est pas seulement l'ETA — au
moins trois incidents de ce soir viennent de scripts qui n'imprimaient pas la
même chose de la même façon :

- pas de sortie non bufferisée par défaut → un log resté vide toute la durée
  d'un run, catastrophique une fois combiné à un `SIGKILL` de walltime
  (courbe de validation perdue en entier, voir friction log A5) ;
- pas de résumé final au format fixe → j'ai dû regrep différemment
  `final_acc`/`final_exact_match`/`best_loss` selon le script ;
- pas de garde-fou "les métriques ne s'affichent jamais sans leur contrôle
  trivial attaché" (`margin_over_shortcut`, `leak_check`) → deux incidents de
  surinterprétation réels sur ce projet (friction log F2).

**Recommandation : formaliser ça en skill**, sur le modèle exact de
`.claude/skills/grid5000/` déjà dans ce dépôt — un
`.claude/skills/experiment-script-design/SKILL.md` qui fixe, pour tout nouveau
script `learn/*/train_*.py` :

1. `python -u` / sortie non bufferisée, toujours.
2. Une ligne `progress: ...` à `--log_every`, format fixe avec `eta=`.
3. Un bloc résumé final au format fixe (clé: valeur, une par ligne),
   incluant systématiquement les contrôles triviaux (chance level, marge,
   `leak_check`) juste à côté de toute métrique d'accuracy — jamais séparés.
4. `--max_steps` et `--max_time_minutes` toujours co-spécifiés explicitement
   dans tout script de lancement (le piège `--max_steps` par défaut à 50 a
   mordu deux fois ce projet, voir friction log D-adjacent).

Ça a un avantage que l'outil seul n'a pas : appliqué **à l'écriture** du
script, pas en aval, ça marche même pour un script que l'outil n'a jamais vu
tourner — pas besoin d'apprentissage empirique par incident.

## 3. Catalogue des besoins (dérivé du Cluster Friction Log, 22 items observés)

Repris et complété avec les items CPU. Détail complet, exemples concrets et le
découpage "outil / agent" par item : voir le Cluster Friction Log (artefact
publié le 2026-09-14, même session) ou `experiment.log.md`/
`grid5000_usage.log.md` pour les incidents bruts. Ici, juste la liste des
*capacités* à couvrir :

**Vérification & état réel**
- Réconciliation post-lancement (process vivant + log qui avance) avant de
  marquer un item "lancé" — GPU et CPU.
- Statut composite par job : état OAR + télémétrie (GPU *ou* CPU selon le
  tier) + fraîcheur du log + progression auto-rapportée (§2).
- Distinguer un artefact de connexion (SSH/oarsh qui se ferme après un
  `disown` réussi) d'un vrai échec de lancement.
- Retry automatique à court délai sur la latence de propagation NFS.

**Budget & planification**
- ETA continue par job = Σ (temps restant des items en cours, lu depuis leur
  propre `eta=`, §2) + (items pas encore démarrés × durée historique moyenne
  du même script/config) — vs. walltime restant du job. Alerte avant coupure.
- Recalcul de l'échéance à chaque relance effective, pas seulement à la
  soumission initiale du job (piège rencontré : relance à 11h40 visant 80 min,
  job mourant à 12h43).
- Journal de débit réutilisable par (script, config) — remplace le calcul
  d'ETA à la main par un historique consultable.
- Registre central du backlog (statut : en attente / en cours / fait /
  abandonné), unifiant ce qui vit aujourd'hui dispersé dans
  `dev_notes/*_experiment_plan.md`.

**Répartition des ressources**
- Suggestion (jamais imposition) de flux concurrents supplémentaires
  compatibles avec la marge mesurée (VRAM **ou** cœurs CPU libres).
- Résolution automatique d'assignation réelle (nœud, index GPU **ou** nombre
  de cœurs) avant de générer les commandes de lancement.

**Dérive de configuration**
- Empreinte de version des dépendances partagées (`core/`, `learn/*/eval_metrics.py`)
  au moment du lancement — alerte si le nœud a une version différente.
- Bibliothèque de contraintes de config connues, vérifiées avant lancement
  (`vocab_size` vs `n_facts`, `block_size**depth`, etc.), alimentée au fil des
  incidents plutôt que réécrite à chaque fois.
- Comparaison automatique de **tous** les hyperparamètres entre un diagnostic
  isolé (ex. un sweep LR) et la grille qu'il est censé éclairer — pas
  seulement ceux nommés dans le titre du run.

**Hygiène d'exécution**
- Traçage PID parent+enfant par item lancé, pour un arrêt propre garanti.
- Distinction automatique et journalisée entre préemption (contrainte externe)
  et walltime auto-infligé (`oarstat -j <id> -s` : `Error` vs `Terminated`).
- Gabarit de lancement standard qui fixe l'environnement une fois pour toutes
  (`PYTHONPATH`, chemins `micromamba` absolus, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
  pour les jobs CPU) plutôt que reconstruit à la main par script.
- Liste blanche de clusters compatibles (compute capability, vérifiée contre
  le build torch réellement installé) contrôlée avant réservation.

**Intégrité des résultats**
- Empreinte (script, config, seed) avant lancement — avertissement, pas
  blocage, en cas de doublon potentiel.
- Refus d'afficher une métrique d'accuracy sans ses contrôles obligatoires
  attachés (marge, `leak_check`) — appliqué à la fois par le skill de
  conception de script (§2) et par l'affichage de l'outil.

## 4. Portée d'un premier prototype

Pour ne pas tout vouloir d'un coup (et parce que l'exhaustivité ci-dessus est
un backlog, pas un cahier des charges v1) :

1. Boucle de réconciliation minimale : un job déclaré, son vrai statut
   (OAR + GPU/CPU + log), rien d'autre — remplace déjà la majorité des
   `oarstat`/`ps aux`/`nvidia-smi` faits à la main ce soir.
2. Garde walltime-vs-ETA, à partir du format `progress: ... eta=...` (§2) —
   nécessite d'abord d'adopter le format dans les scripts actifs.
3. Tout le reste (registre de backlog, empreinte anti-doublon, bibliothèque de
   contraintes de config) vient après, une fois que 1-2 tournent en vrai sur
   un chantier réel.

Le skill de conception de script (§2) peut avancer en parallèle et
indépendamment — c'est une convention d'écriture, pas un développement
d'outil, et il rend la capacité 2 possible pour n'importe quel script déjà
écrit dans ce style.
