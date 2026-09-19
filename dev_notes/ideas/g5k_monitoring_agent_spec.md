# Spec : suivi à jour de l'utilisation des nœuds Grid'5000 -- **réutiliser l'existant, pas développer un agent maison**

## Contexte / problème

Le suivi actuel (`tools/g5kstat.sh`) est **pull, à la demande** : il faut lancer un scan SSH multi-site à chaque fois pour savoir ce qui tourne réellement. Conséquences déjà vécues cette nuit :
- des claims orphelins (workers morts sur des nœuds GPU disparus après préemption) sont restés invisibles pendant des heures avant d'être détectés manuellement ;
- l'état "sous-utilisé" de `paradoxe-27` (104 cœurs, load 3.0) n'a été repéré qu'en creusant à la main, pas signalé automatiquement ;
- chaque vérification coûte 10-100s de SSH séquentiel/retry, donc on vérifie moins souvent qu'on ne le devrait.

**Correction par rapport à la première version de cette spec** : elle proposait un agent Python maison par nœud, poussant des snapshots JSON. Vérification faite directement sur les nœuds (`ps aux`, `curl localhost:<port>/metrics`) : **tout ce dont on a besoin tourne déjà nativement sur chaque nœud Grid'5000**, sans rien installer. Pas de raison de réinventer un protocole de collecte -- juste besoin de le *lire*.

## Ce qui existe déjà (vérifié en direct ce soir)

| Besoin | Solution déjà présente | Confirmé comment |
|---|---|---|
| CPU / load par nœud, **agrégé côté API, sans SSH** | `prometheus-node-exporter` (port 9100 local) + **API Kwollect centrale** : `https://api.grid5000.fr/stable/sites/<site>/metrics?job_id=<id>&nodes=<node>` | `curl` direct sur `paradoxe-27` (job `4121204`) a renvoyé `prom_node_cpu_seconds_total` par cœur/mode, sans SSH vers le nœud lui-même -- juste vers le frontend/API du site. |
| GPU util % + VRAM utilisée/totale par GPU | `dcgm-exporter` (l'exporter Prometheus officiel NVIDIA), déjà lancé en root sur chaque nœud GPU, port 9400 local | `curl localhost:9400/metrics` sur `graffiti-3` a renvoyé `DCGM_FI_DEV_GPU_UTIL{gpu="0"...}=34`, `DCGM_FI_DEV_FB_USED`, etc. pour les 4 GPU. |
| Power / température | `bmc_node_power_watt`, `bmc_ambient_temp_celsius` | Déjà remontés dans Kwollect (confirmé sur `graffiti-3` via l'API, sans rien lancer). |
| État du pipeline projet (workers actifs, avancement des grilles, claims orphelins) | **Rien d'existant** -- spécifique à `tools/exp/` | N/A -- c'est la seule pièce réellement à construire. |

**Trou constaté** : Kwollect (l'agrégateur central) ne semble pas relayer les métriques `dcgm-exporter` (GPU) -- seulement CPU/power/temp. Donc le GPU nécessite un accès local port 9400 (via un SSH vers le nœud, ou une requête à l'API si Kwollect les relaie ailleurs -- à vérifier avec les gestionnaires du groupe `wide`, cf. contacts dans la skill `grid5000`).

## Architecture révisée : agrégateur en lecture seule, rien à déployer sur les nœuds

### 1. `tools/g5k_monitor.py` (remplace l'idée d'agent -- un seul script, côté frontend/local)

Pour chaque site avec un job actif (`oarstat -u jdomguia`) :
- **CPU/load** : une requête HTTP à l'API Kwollect (`/sites/<site>/metrics?job_id=...`), pas de SSH nécessaire au-delà de l'accès déjà configuré au site.
- **GPU** (si le job a des GPU) : un seul `ssh <node> curl -s localhost:9400/metrics` par nœud GPU (déjà nettement plus léger que le `nvidia-smi` + parsing actuel de `g5kstat.sh`, et donne des séries Prometheus standard, pas besoin de reparser du texte `nvidia-smi`).
- **Pipeline projet** (workers, claims, avancement des grilles) : reste ce que `g5kstat.sh -g deep` fait déjà (`pgrep`, lecture de `runs/*/state/*.json`, `runs/*/claims/`) -- la seule partie qui doit rester un scan direct, car rien d'externe ne le connaît.

### 2. Détection automatique des claims orphelins (le vrai gain sur l'incident de cette nuit)

Croiser, pour chaque grille active, la liste des `.claim` avec les `worker-id`/PID réellement vivants (via le scan pipeline ci-dessus) et le champ `cell-timeout-minutes` déjà présent dans `tools/exp/worker.py`. Un claim sans worker vivant correspondant, dont le state est `"running"` depuis plus que le timeout, est flaggé automatiquement -- exactement le problème découvert manuellement ce soir sur `i3_step2_bothArms` (13 cellules orphelines après la préemption GPU), mais détecté sans creuser à la main.

### 3. Intégration à `/g5kstat`

`tools/g5kstat.sh` garde son fonctionnement, mais route ses métriques CPU/GPU par défaut vers Kwollect + `dcgm-exporter` (rapide, standard) plutôt que `uptime`/`nvidia-smi` parsés à la main -- ne retombe sur le scan SSH direct actuel que si l'API est indisponible ou pour un site sans Kwollect. Le scan `pgrep`/état des grilles (partie pipeline projet) reste inchangé.

## Ce qui reste réellement à développer

**Uniquement** : le petit script d'agrégation/croisement (`tools/g5k_monitor.py`), qui interroge Kwollect + `dcgm-exporter` + l'état local des grilles et produit une vue unifiée avec détection d'anomalie (claim orphelin, nœud sous-utilisé). Pas de service à faire tourner en continu sur les nœuds, pas de nouveau format de données à faire adopter par le pipeline existant.

## Ce qui reste à trancher (pour l'agent research-optimization)

- Confirmer auprès des gestionnaires `wide` (contacts dans la skill `grid5000`) si les métriques `dcgm-exporter` peuvent être relayées dans Kwollect côté Nancy/Rennes -- éliminerait le dernier SSH nécessaire (GPU).
- Authentification à l'API Kwollect depuis l'extérieur du site (actuellement testé en `ssh`-ant d'abord vers le frontend puis `curl` en local -- vérifier si un accès direct depuis le poste local est possible sans ce détour).
- Seuil de "sous-utilisé" à flagger automatiquement (proposé : load < 20% de `nproc` pendant > 10 min pour CPU ; `DCGM_FI_DEV_GPU_UTIL` < 20% pendant > 10 min pour GPU) -- à valider empiriquement plutôt que deviné.
- Faut-il exposer ce script comme mode par défaut de `/g5kstat`, ou un nouveau flag (`-g kwollect`) le temps de valider la fiabilité par rapport au scan SSH direct actuel.

**[DÉCISION 2026-09-19]** Transmission de cette spec à experiment-manager pour implémentation de `tools/g5k_monitor.py` : agrégateur en lecture seule (Kwollect pour CPU/load, `dcgm-exporter` via SSH pour GPU, scan pipeline projet existant `pgrep`/`runs/*/state`/`runs/*/claims` inchangé), avec détection automatique des claims orphelins par croisement `.claim` vs worker-id/PID vivant et `cell-timeout-minutes` (cf. `tools/exp/worker.py`). Les trois points de la section précédente (relais dcgm-exporter dans Kwollect, accès API Kwollect direct depuis le poste local, seuil de sous-utilisation) restent ouverts, non tranchés par l'utilisateur -- experiment-manager doit les vérifier/valider empiriquement plutôt que trancher à sa place. Intégration finale dans `tools/g5kstat.sh` (fallback scan SSH direct si API indisponible) envisagée mais non décidée dans le détail.
