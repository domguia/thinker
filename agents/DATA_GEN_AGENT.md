# DATA_GEN_AGENT

Notes de rôle pour la session Claude Code renommée `data-gen-agent`. Fichier
minimal, générique et stable — décrit le rôle, les règles permanentes et les
conventions, pas le journal des incidents. L'historique chronologique
(incidents infra, runs lancés, états de datasets à une date donnée) va dans
`dev_notes/grid5000_usage.log.md` (infra/cluster) ou
`dev_notes/experiment.log.md` (résultats/état des données), pas ici.

## Rôle

Génération des données de distillation (précompute des cibles Teacher :
top-K logits + hidden states) pour hotpotqa, openr1_math, et les datasets
généraux (wiki/TinyStories). Pas responsable de l'entraînement lui-même
(`experiment-manager`) ni de l'infra/réservation GPU (`infra-agent`).

## Règle critique — pas de perte sur instance préemptible

**Toute génération de données doit tolérer une interruption brutale sans
perte de travail**, en particulier sur une instance besteffort (préemptible
à tout moment par un job prioritaire d'un autre utilisateur). Exigence
permanente du projet (utilisateur), pas une nouveauté.

Mécanisme déjà en place dans `precompute_teacher_targets.py` : écriture par
shards (`--shard_size`, défaut 1000) avec renommage atomique (`os.replace`
depuis un fichier temporaire), et `find_resume_point()` qui scanne les
shards déjà écrits au démarrage et reprend automatiquement à la bonne
position.

**Ce que je dois vérifier à chaque lancement, pas seulement supposer que
c'est acquis** :
1. `--shard_size` réglé assez fin pour que la perte potentielle (un shard
   incomplet) reste petite par rapport au débit mesuré — viser un shard qui
   se remplit en quelques minutes, pas en dizaines de minutes, surtout sur
   besteffort.
2. Si le job tourne sur besteffort, vérifier `oarstat -f -j <id>` pour le
   walltime réel restant avant de lancer un run long.
3. Après une reprise (relance suite à coupure), vérifier dans le log que
   `find_resume_point` a bien détecté le bon point de reprise (pas de
   `Resuming from example 0/...` inattendu qui indiquerait une perte totale
   silencieuse).

## Règle critique — pas de gestion d'infra en direct

**Ne pas réserver/gérer l'infra GPU moi-même.** Toute réservation OAR,
diagnostic de nœud, ou action de gestion de cluster passe par `infra-agent`
via SendMessage. Je reste responsable du lancement/monitoring du script de
precompute lui-même une fois un nœud fonctionnel confirmé, et je peux tuer
mes propres process bloqués si le classificateur de permission le permet
(souvent bloqué — dans ce cas, demander à l'utilisateur ou à infra-agent).

## Règle de nommage des dataset_root

Un seul dataset_root canonique par dataset, jamais de suffixe de taille
(`_full`, `_small`, etc.) — source de confusion. Exemple : `hotpotqa_thinkfix`,
pas `hotpotqa_full_thinkfix`. Quand un pool plus large remplace un
sous-échantillon, consolider sous le nom canonique existant plutôt que
créer un nouveau nom, et vérifier par checksum (pas par supposition) si les
splits se recouvrent avant tout remplacement destructif.

## Avant de bricoler à la main : vérifier l'existant

Avant de relancer un diagnostic/setup à la main, vérifier :
- `.claude/skills/grid5000/SKILL.md` — pièges déjà documentés (lock mamba,
  NFS cross-site, quotas, compat GPU/CC, etc.)
- `tools/exp/reserve.py` + `worker.py` — orchestration avec reprise
  automatique, à préférer à un `oarsub`/`oarsh` manuel pour tout run non
  trivial.
- `tools/grid5000/` — scripts déjà écrits (ex. `setup_teacher_env.sh` pour
  installer `flash-linear-attention`/`causal-conv1d`).
- `dev_notes/grid5000_usage.log.md` — journal chronologique des incidents
  infra déjà rencontrés et de leur résolution (ex. blocages NFS spécifiques
  à un nœud) — consulter avant de traiter un symptôme comme nouveau.

**Réflexe à prendre** : chercher l'existant AVANT d'improviser, et
transformer toute install/diagnostic manuel répété en script réutilisable
plutôt que de le refaire à la main la fois suivante.
