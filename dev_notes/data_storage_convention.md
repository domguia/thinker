# Convention de stockage des données de distillation

Racine : `/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/<dataset>/`

## Règle : un nom de dataset canonique, stable, sans suffixe de variante

Le nom de dataset au premier niveau (`<dataset>`) est **propre et stable**
dans le temps : `hotpotqa`, `openr1_math`, `tinystories`, `wikitext`, etc.
Aucun qualificatif de variante ne va dans ce nom — ni taille d'échantillon
(`_sample5k`), ni statut de correction (`_thinkfix`, `_fixed`), ni portée
(`_full`). Ces informations sont des **subsets internes**, pas des
dossiers racine séparés.

**Pourquoi** : un nom de dataset avec suffixe crée deux vérités pour la
même donnée (ex. `hotpotqa` vs `hotpotqa_thinkfix` vs `hotpotqa_full` —
trois noms rencontrés en pratique le 2026-09-22 pour le même dataset à
différents stades), source de confusion et de risque de consommer par
erreur une version obsolète/buggée. Rencontré concrètement : suffixe
`_full` banni le 2026-09-22 (confusion sur la portée), puis `hotpotqa`/
`hotpotqa_thinkfix` fusionnés le même jour (deux racines pour un seul
dataset avant/après un fix de template).

## Où va la variante

- **jsonl `train.jsonl`/`val.jsonl` à la racine du dataset** : toujours le
  contenu canonique le plus à jour et le plus correct. Quand un fix (bug de
  template, etc.) est appliqué, il **remplace** le contenu canonique — pas
  de fichier `train_<fix>.jsonl` séparé qui traîne à côté.
- **Le nom de subset (`--subset_name`)** encode la variante de
  sélection/version du precompute Teacher : taille d'échantillon, seed,
  ou état du fix au moment du run (ex. `thinkfix_n2000`, `lastLayer_n4503`).
  C'est le système déjà en place (`topk/<split>/<subset_name>.<teacher>.npz`
  + `manifest.json` + `subsets/<split>/<subset_name>.indices.npy`) — la
  variante vit là, pas dans le nom du dossier racine.
- Si un artefact top-K a été généré à partir d'une version **buggée** du
  jsonl canonique (ex. avant un fix), il devient invalide dès que le jsonl
  canonique change — le supprimer plutôt que le garder à côté (source de
  confusion, pas de valeur de debug durable une fois le bug documenté dans
  `dev_notes/`).

## Avant tout renommage/fusion de dataset_root

1. Vérifier par checksum (pas par supposition) si les fichiers qui se
   recouvrent sont identiques ou divergent.
2. Coordonner avec les agents ayant des jobs actifs sur l'ancien chemin
   (`SendMessage`) avant de renommer/écraser — un run en cours peut
   dépendre d'un chemin qui va disparaître.
3. Mettre à jour tout script (`tmp_scripts_local/`, launch commands) qui
   référence l'ancien chemin en dur.
