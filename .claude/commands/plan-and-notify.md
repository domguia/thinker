---
description: Documente une décision/instruction dans le plan d'expérience puis la transmet à experiment-manager
---

Arguments reçus : $ARGUMENTS -- la décision, le résultat ou l'instruction à documenter et
transmettre. Si vide, demande à l'utilisateur ce qu'il veut consigner avant de continuer
(rien à faire sans contenu).

Deux actions à faire, dans l'ordre, à chaque fois que cette commande est invoquée :

## 1. Documenter dans le plan d'expérience

Fichier : `dev_notes/indexed_attention_experiment_plan.md` (le plan vivant du projet -- si le
sujet de $ARGUMENTS concerne clairement un autre chantier du dépôt, utilise le document le
plus pertinent à la place, ex. `dev_notes/model_selection_small_vocab_reasoning.md`).

- Trouve la section (`## Phase N — ...`) la plus pertinente pour le contenu de $ARGUMENTS --
  celle déjà en cours de discussion dans la conversation, sinon celle dont le sujet
  correspond le mieux. Ne crée pas de nouvelle Phase sans qu'on te l'ait demandé.
- Ajoute une entrée à la fin du dernier paragraphe pertinent de cette section (pas en tête
  de fichier), dans le style narratif déjà utilisé partout dans ce document : commence par
  un tag en gras `**[DÉCISION <date>]**`, `**[LANCÉ <date>]**` ou `**[RÉSULTAT <date>]**`
  (choisis celui qui correspond), suivi d'une phrase résumant le fait en une ligne, puis le
  détail. Date du jour via `date +%F`.
- Réutilise les noms de fichiers/flags/alias déjà établis dans le dépôt (ex.
  `core/model_families.py`, `train_real_text.py`, tags de baseline A/B/C) plutôt que de les
  reformuler -- ce document sert de référence exacte, pas de résumé approximatif.
- Si $ARGUMENTS laisse une question ouverte (pas encore tranchée par l'utilisateur), dis-le
  explicitement dans l'entrée ("point resté ouvert, non tranché") plutôt que de trancher à sa
  place.

## 2. Transmettre à experiment-manager

Envoie un message (SendMessage, `to: "experiment-manager"`) reformulant le contenu de
$ARGUMENTS en instruction autonome et actionnable -- ce destinataire est une session Claude
Code séparée, sans visibilité sur cette conversation : inclus tout le contexte nécessaire
(chemins de fichiers, valeurs concrètes, contraintes) plutôt que d'y faire référence de façon
elliptique. Si un point reste ouvert côté utilisateur, dis-le aussi à experiment-manager pour
qu'il ne tranche pas à sa place ni ne lance quelque chose de prématuré sur ce point précis.

Termine par un résumé bref (2-3 phrases) de ce qui a été écrit dans le plan et de ce qui a
été envoyé.
