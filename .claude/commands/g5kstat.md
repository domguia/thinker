---
description: Etat des jobs Grid'5000 (jdomguia) sur tous les sites + utilisation reelle des noeuds alloues
---

Execute `tools/g5kstat.sh` (lecture seule -- ne lance ni ne modifie rien) et presente son resultat.

```bash
tools/g5kstat.sh
```

Par defaut ca resout chaque job actif sur tous les sites et donne une ligne
d'utilisation reelle par noeud (GPU% + VRAM separee, ou charge CPU). Options
utiles selon ce que l'utilisateur demande :

- `tools/g5kstat.sh -g quick` -- juste la liste des jobs + walltime restant,
  aucun ssh vers les noeuds de calcul (rapide, pour un coup d'oeil).
- `tools/g5kstat.sh -g deep` -- ajoute le detail des processus par noeud
  (top process CPU, ou process GPU actifs) -- a utiliser si un job semble
  "Running" mais peu/pas utilise, pour voir ce qui tourne reellement dessus.
- `tools/g5kstat.sh -j <JOBID>` -- se limite a un seul job (cherche sur tous
  les sites de `-s`, ou passe `-s <site>` si le site est deja connu).
- `tools/g5kstat.sh -s "rennes nancy"` -- limite la recherche a certains sites
  (plus rapide que le balayage complet des 8 sites par defaut).
- `tools/g5kstat.sh -g kwollect` -- delegue a `tools/g5k_monitor.py` : CPU via
  l'API Kwollect (pas de ssh au noeud), GPU via dcgm-exporter, et surtout
  detection automatique des claims orphelins sur `runs/*/claims` de chaque
  site (le vrai gain par rapport a `-g deep`, valide sur l'incident reel
  `i3_step2_bothArms`). Pas encore le mode par defaut -- a preferer quand on
  veut aussi verifier les claims orphelins, pas seulement l'utilisation.

Si le script signale un noeud `unreachable` ou une grille a 0% alors que le
job est cense tourner, creuser avec `-g deep` sur ce job avant de conclure a
un probleme -- ne jamais se fier a l'etat `oarstat` seul.
