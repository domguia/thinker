#!/usr/bin/env python3
"""Agrégateur en lecture seule pour le suivi Grid'5000 -- voit ce que
`tools/g5kstat.sh` obtenait en SSH direct (CPU load, GPU util), mais via les
services déjà en place sur chaque site (pas de nouvel agent à déployer) :

- CPU / load par nœud     : API Kwollect (`prometheus-node-exporter` agrégé
                             côté site), une requête HTTP, pas de SSH au
                             nœud lui-même.
- GPU util % / VRAM       : `dcgm-exporter` (port 9400 local à chaque nœud
                             GPU) -- pas encore relayé par Kwollect au moment
                             de l'écriture (cf. dev_notes/ideas/
                             g5k_monitoring_agent_spec.md), donc un SSH+oarsh
                             reste nécessaire pour cette seule partie.
- Pipeline projet         : claims/état de grille sous `runs/<name>/` (sur
                             le NFS du site, lu depuis le frontend) --
                             rien d'externe ne le connaît, donc scan direct,
                             comme le fait déjà `tools/exp/status.py`.

Le vrai gain sur ce que faisait `g5kstat.sh -g deep` à la main : la
détection automatique des **claims orphelins** -- une cellule dont le
`.claim` existe, l'état dit `running` depuis plus que `cell-timeout-minutes`
(cf. `tools/exp/worker.py`), et dont le worker (host:pid du claim) n'est ni
sur un nœud actuellement assigné à un job actif, ni vivant sur ce nœud.
C'est exactement l'incident `i3_step2_bothArms` (13 cellules orphelines
après préemption GPU) détecté à la main cette nuit-là -- ce script doit le
signaler sans creuser manuellement.

Rien n'est décidé automatiquement : l'outil signale, il ne tue, ne relance,
ni ne modifie aucun claim (même politique que status.py). Read-only.

Usage :
    tools/g5k_monitor.py                       # tous les sites, jobs actifs de $G5K_USER
    tools/g5k_monitor.py -s "rennes nancy"      # restreindre les sites
    tools/g5k_monitor.py -j 4121204             # un seul job
    tools/g5k_monitor.py --no-gpu               # saute le SSH+oarsh dcgm (CPU/Kwollect seul)
    tools/g5k_monitor.py --runs-root runs       # racine des grilles à croiser pour les claims

Points non tranchés au moment de l'écriture (cf. spec) -- à valider en
conditions réelles avant de s'y fier aveuglément :
    - dcgm-exporter n'est peut-être pas relayé par Kwollect sur tous les
      sites -- le fallback SSH+oarsh est donc la voie par défaut pour le
      GPU, pas une voie de secours.
    - l'accès direct à l'API Kwollect depuis un poste hors G5K n'est pas
      confirmé -- ce script passe par un SSH vers le frontend du site
      (`<site>.grid5000.fr.g5k`) et un `curl` exécuté là-bas, comme
      `g5kstat.sh` le fait déjà pour tout le reste.
    - le seuil de "sous-utilisé" (proposé : load < 20% de nproc, ou
      GPU_UTIL < 20%, pendant > 10 min) n'est qu'une proposition -- à
      valider empiriquement (--underuse-load-pct, --underuse-minutes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
USER_G5K_DEFAULT = os.environ.get("G5K_USER", "jdomguia")
SITES_DEFAULT = "rennes lyon nancy lille nantes grenoble strasbourg toulouse"
STALE_S_DEFAULT = 600  # aligné sur tools/exp/status.py


# ---------------------------------------------------------------------------
# SSH -- même stratégie retry que g5kstat.sh (3 tentatives, 1s d'écart,
# vide sur échec total) : les hoquets SSH transitoires sont un pattern
# documenté sur ce projet, pas une exception.
# ---------------------------------------------------------------------------

def ssh_retry(target: str, cmd: str, attempts: int = 3, timeout: int = 20) -> str:
    for attempt in range(attempts):
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", target, cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.stdout.strip():
                return r.stdout
        except (subprocess.TimeoutExpired, OSError):
            pass
        if attempt < attempts - 1:
            time.sleep(1)
    return ""


# ---------------------------------------------------------------------------
# 1. Jobs actifs par site (même logique en deux temps que g5kstat.sh :
#    lister les IDs, puis résoudre chacun individuellement -- l'appel JSON
#    en vrac s'est montré incomplet sur les gros jobs de ce projet).
# ---------------------------------------------------------------------------

def frontend(site: str) -> str:
    return f"{site}.grid5000.fr.g5k"


def list_job_ids(site: str, user: str) -> list[str]:
    out = ssh_retry(frontend(site), f"oarstat -u {user}")
    if not out.strip():
        return []
    lines = out.strip().splitlines()[2:]  # 2 lignes d'en-tête
    return [l.split()[0] for l in lines if l.strip()]


def job_detail(site: str, jid: str) -> dict | None:
    out = ssh_retry(frontend(site), f"oarstat -j {jid} -f")
    if not out.strip():
        return None
    info: dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if " = " in line:
            k, v = line.split(" = ", 1)
            info[k.strip()] = v.strip()
    hosts_raw = info.get("assigned_hostnames", "")
    hosts = sorted({h.split(".")[0] for h in hosts_raw.split("+") if h})
    return {
        "job_id": jid,
        "site": site,
        "name": info.get("name", ""),
        "state": info.get("state", "?"),
        "hosts": hosts,
    }


def _site_jobs(site: str, user: str, only_job: str | None) -> list[dict]:
    job_ids = [only_job] if only_job else list_job_ids(site, user)
    out = []
    for jid in job_ids:
        d = job_detail(site, jid)
        if d is None:
            continue
        if only_job and d["state"] == "?":
            continue  # ce job n'est pas sur ce site
        out.append(d)
    return out


def active_jobs(sites: list[str], user: str, only_job: str | None) -> list[dict]:
    """Un SSH par site pour lister les jobs, en parallèle -- un balayage
    complet des 8 sites était sérialisé auparavant (~4-5s par site rien que
    pour un site sans job actif), constaté en pratique comme la première
    cause de timeout sur un scan multi-site sans `-s` (cf. validation
    croisée par experiment-manager, dev_notes/grid5000_usage.log.md)."""
    jobs = []
    with ThreadPoolExecutor(max_workers=min(len(sites), 8)) as ex:
        for result in ex.map(lambda s: _site_jobs(s, user, only_job), sites):
            jobs += result
    return jobs


# ---------------------------------------------------------------------------
# 2. CPU / load via Kwollect -- pas de SSH au nœud, seulement au frontend du
#    site (voie confirmée en direct : cf. spec, curl sur paradoxe-27 sans
#    toucher le nœud lui-même).
# ---------------------------------------------------------------------------

def kwollect_cpu(site: str, jid: str, host: str) -> dict | None:
    node_fqdn = f"{host}.{site}.grid5000.fr"
    url = (f"https://api.grid5000.fr/stable/sites/{site}/metrics"
           f"?job_id={jid}&nodes={node_fqdn}&metrics=prom_node_load1")
    # attempts/timeout réduits + `curl --max-time` explicite côté distant :
    # un scan multi-site parallèle ouvre plusieurs SSH simultanés vers le
    # même frontend, et le timeout par défaut de ssh_retry (3 x 20s = 63s)
    # suffit à lui seul à dépasser un budget de 60s si un seul appel cale
    # sous contention -- constaté en intermittent après la parallélisation
    # (cf. retour experiment-manager). Le frontend est déjà confirmé
    # joignable à ce stade (active_jobs() y a réussi), donc un appel qui ne
    # répond pas vite est un vrai signal, pas une raison de réessayer 3 fois.
    out = ssh_retry(frontend(site), f"curl -sf --max-time 8 '{url}'",
                     attempts=2, timeout=12)
    if not out.strip():
        return None
    try:
        samples = json.loads(out)
    except json.JSONDecodeError:
        return None
    if not samples:
        return None
    last = samples[-1]
    return {"load1": last.get("value"), "timestamp": last.get("timestamp")}


# ---------------------------------------------------------------------------
# 3. GPU via dcgm-exporter -- reste un SSH+oarsh (Kwollect ne le relaie pas
#    encore, cf. trou constaté dans la spec). Un seul aller-retour par nœud
#    GPU, texte Prometheus standard -- pas de reparsing `nvidia-smi`.
# ---------------------------------------------------------------------------

def dcgm_gpu(site: str, jid: str, host: str) -> list[dict]:
    node_fqdn = f"{host}.{site}.grid5000.fr"
    # Mêmes attempts/timeout resserrés + `curl --max-time` que kwollect_cpu
    # (voir son commentaire) -- l'appel traverse un hop supplémentaire
    # (oarsh), d'où un budget local un peu plus large.
    out = ssh_retry(
        frontend(site),
        f"OAR_JOB_ID={jid} oarsh {node_fqdn} 'curl -sf --max-time 8 localhost:9400/metrics' 2>/dev/null",
        attempts=2, timeout=15,
    )
    gpus: dict[str, dict] = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            metric, value = line.rsplit(" ", 1)
        except ValueError:
            continue
        if "{" not in metric:
            continue
        name, labels_raw = metric.split("{", 1)
        labels_raw = labels_raw.rstrip("}")
        labels = dict(
            kv.split("=", 1) for kv in labels_raw.split(",") if "=" in kv
        )
        gpu_idx = labels.get("gpu", "?").strip('"')
        entry = gpus.setdefault(gpu_idx, {"gpu": gpu_idx})
        if name == "DCGM_FI_DEV_GPU_UTIL":
            entry["util_pct"] = float(value)
        elif name == "DCGM_FI_DEV_FB_USED":
            entry["vram_used_mib"] = float(value)
        elif name == "DCGM_FI_DEV_FB_TOTAL":
            entry["vram_total_mib"] = float(value)
    return sorted(gpus.values(), key=lambda g: g["gpu"])


# ---------------------------------------------------------------------------
# 4. Pipeline projet -- inchangé dans son principe par rapport à
#    tools/exp/status.py : lire runs/<name>/{grid.jsonl,state/,claims/}.
#    **Corrigé après test réel** : `runs/` vit sur le home du frontend de
#    chaque site (`~/thinker/runs`, confirmé en SSH sur rennes), pas sur le
#    poste local d'où tourne ce script -- un scan `Path.glob()` local ne
#    voyait donc jamais aucune grille. Le scan doit passer par SSH comme le
#    reste, un script Python encodé en base64 exécuté sur le frontend pour
#    éviter tout problème de quoting entre les deux shells.
# ---------------------------------------------------------------------------

_REMOTE_PIPELINE_SCRIPT = """
import base64, glob, json, os
root = os.path.expanduser({runs_root!r})
out = {{}}
for grid_path in glob.glob(os.path.join(root, "*", "grid.jsonl")):
    name = os.path.basename(os.path.dirname(grid_path))
    cells = []
    with open(grid_path) as f:
        for l in f:
            l = l.strip()
            if l:
                cells.append(json.loads(l))
    claims = {{}}
    claims_dir = os.path.join(os.path.dirname(grid_path), "claims")
    if os.path.isdir(claims_dir):
        for fn in os.listdir(claims_dir):
            if fn.endswith(".claim"):
                try:
                    with open(os.path.join(claims_dir, fn)) as cf:
                        claims[fn[:-6]] = json.load(cf)
                except Exception:
                    pass
    states = {{}}
    state_dir = os.path.join(os.path.dirname(grid_path), "state")
    if os.path.isdir(state_dir):
        for fn in os.listdir(state_dir):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(state_dir, fn)) as sf:
                        states[fn[:-5]] = json.load(sf)
                except Exception:
                    pass
    out[name] = {{"cells": cells, "claims": claims, "states": states}}
print(json.dumps(out))
"""


def fetch_remote_pipeline(site: str, runs_root: str) -> dict:
    """Récupère toutes les grilles sous `runs_root` (chemin distant, ex.
    `~/thinker/runs`) en un seul aller-retour SSH -- encodage base64 pour ne
    pas se battre avec deux niveaux de shell quoting (ssh puis bash distant)."""
    script = _REMOTE_PIPELINE_SCRIPT.format(runs_root=runs_root)
    b64 = __import__("base64").b64encode(script.encode()).decode()
    out = ssh_retry(frontend(site), f"echo {b64} | base64 -d | python3 -", timeout=30)
    if not out.strip():
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def scan_orphan_claims(grid_name: str, cells: list[dict], claims: dict, states: dict,
                        active_hosts: set[str], stale_s: int) -> list[dict]:
    """Compare chaque `.claim` à l'état de la cellule et aux nœuds
    actuellement assignés à un job actif de l'utilisateur.

    Un claim est flaggé orphelin si son état n'est pas `done` et soit :
    - le nœud (`worker` du claim = `hostname:worker-id`) n'est plus sur
      aucun nœud actuellement actif -- certain, pas une hypothèse ;
    - ou l'état n'a pas bougé depuis plus de `stale_s` -- suspect, à
      vérifier (même politique que status.py : on signale, on ne tue rien).
    """
    now = time.time()
    cells_by_id = {c["run_id"]: c for c in cells}
    orphans = []

    for rid, claim in sorted(claims.items()):
        if rid not in cells_by_id:
            continue
        state = states.get(rid)
        status = (state or {}).get("status")
        if status == "done":
            continue  # terminée normalement, claim résiduel attendu

        worker = claim.get("worker", "")
        host = worker.split(":", 1)[0] if ":" in worker else worker
        # normalise en nom court : `worker` porte le FQDN local du nœud
        # (`socket.gethostname()` sur un nœud G5K, ex. "paradoxe-5.rennes.
        # grid5000.fr"), alors que `active_hosts` (issu de
        # `assigned_hostnames`) est en nom court -- comparer sans normaliser
        # marquait paradoxe-2/paradoxe-5/paradoxe-27 comme "disparus" alors
        # qu'ils étaient bel et bien actifs (faux positifs constatés en
        # test réel sur b1_assoc_recall_wideseed).
        host = host.split(".", 1)[0]
        claim_age = now - claim.get("ts", now)
        state_age = now - state.get("updated_at", now) if state else claim_age

        node_gone = bool(host) and host not in active_hosts
        stale = state_age > stale_s

        if node_gone or stale:
            orphans.append({
                "run_id": rid,
                "grid": grid_name,
                "worker": worker,
                "status": status,
                "claim_age_s": round(claim_age),
                "state_age_s": round(state_age) if state else None,
                "reason": "node_gone" if node_gone else "stale_running",
            })
    return orphans


# ---------------------------------------------------------------------------
# 5. Sous-utilisation -- seuils proposés dans la spec, pas encore validés
#    empiriquement : exposés en flags pour pouvoir les ajuster sans toucher
#    au code pendant la phase de validation.
# ---------------------------------------------------------------------------

def flag_underuse(node_row: dict, load_pct_threshold: float) -> str | None:
    load1 = node_row.get("load1")
    ncores = node_row.get("ncores")
    if load1 is not None and ncores:
        pct = 100.0 * float(load1) / ncores
        if pct < load_pct_threshold:
            return f"CPU sous-utilisé : load1={load1} sur {ncores} cœurs ({pct:.0f}%)"
    for gpu in node_row.get("gpus", []):
        util = gpu.get("util_pct")
        if util is not None and util < load_pct_threshold:
            return f"GPU{gpu['gpu']} sous-utilisé : util={util:.0f}%"
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-u", "--user", default=USER_G5K_DEFAULT)
    p.add_argument("-s", "--sites", default=SITES_DEFAULT)
    p.add_argument("-j", "--job", default=None, help="restreindre à un job")
    p.add_argument("--no-gpu", action="store_true",
                   help="saute le SSH+oarsh dcgm-exporter (CPU/Kwollect seul)")
    p.add_argument("--remote-runs-root", default="~/thinker/runs",
                   help="racine des grilles sur le HOME du frontend de chaque site "
                        "(runs/ vit sur le NFS distant, pas sur le poste local)")
    p.add_argument("--stale-seconds", type=int, default=STALE_S_DEFAULT)
    p.add_argument("--underuse-load-pct", type=float, default=20.0,
                   help="seuil de sous-utilisation CPU/GPU, en %% (proposition non validée)")
    p.add_argument("--json", action="store_true", help="sortie JSON brute plutôt que texte")
    args = p.parse_args()

    sites = args.sites.split()
    jobs = active_jobs(sites, args.user, args.job)

    if not jobs:
        if args.json:
            print(json.dumps({"jobs": [], "orphan_claims": []}))
        else:
            print(f"Aucun job actif pour {args.user} sur : {' '.join(sites)}")
        return

    active_hosts: set[str] = set()
    for j in jobs:
        active_hosts.update(j["hosts"])

    def _node_row(j: dict, host: str) -> dict:
        row = {"job_id": j["job_id"], "site": j["site"], "name": j["name"], "host": host}
        cpu = kwollect_cpu(j["site"], j["job_id"], host)
        if cpu:
            row["load1"] = cpu["load1"]
        if not args.no_gpu:
            gpus = dcgm_gpu(j["site"], j["job_id"], host)
            if gpus:
                row["gpus"] = gpus
        underuse = flag_underuse(row, args.underuse_load_pct)
        if underuse:
            row["underuse"] = underuse
        return row

    # Un nœud = jusqu'à deux SSH (Kwollect + dcgm) -- en parallèle, comme
    # active_jobs() : c'était le second gros contributeur au timeout
    # constaté sans `-s` (chaque nœud attendait son tour, en série).
    node_targets = [(j, host) for j in jobs if j["state"] == "Running" for host in j["hosts"]]
    node_rows = []
    if node_targets:
        # Plafonné à 6, pas au nombre total de nœuds : chaque appel ouvre un
        # SSH vers le frontend du site, et plusieurs nœuds d'un même site
        # partagent le même frontend -- trop de connexions simultanées vers
        # UN frontend peut le faire caler (contention constatée en
        # intermittent après une première parallélisation trop large, cf.
        # retour experiment-manager), même si chaque appel pris seul est rapide.
        with ThreadPoolExecutor(max_workers=min(len(node_targets), 6)) as ex:
            node_rows = list(ex.map(lambda t: _node_row(*t), node_targets))

    # Détection des orphelins désactivée si `--job` restreint la portée :
    # `active_hosts` ne couvrirait alors que les nœuds de CE job, pas de tous
    # les jobs actifs de l'utilisateur -- un claim appartenant à un nœud
    # réellement actif mais sous un AUTRE job serait signalé à tort
    # `node_gone` (faux positif constaté en test réel avec `-j 4121204` :
    # 28 claims de b1_assoc_recall_wideseed sur paradoxe-2/5, tous deux
    # actifs sous un job différent, non filtré ici).
    orphan_claims = []
    orphan_scope_limited = bool(args.job)
    if not orphan_scope_limited:
        sites_with_jobs = sorted({j["site"] for j in jobs})
        with ThreadPoolExecutor(max_workers=min(len(sites_with_jobs), 8)) as ex:
            pipelines = ex.map(
                lambda s: fetch_remote_pipeline(s, args.remote_runs_root), sites_with_jobs)
            for pipeline in pipelines:
                for grid_name, g in pipeline.items():
                    orphan_claims += scan_orphan_claims(
                        grid_name, g["cells"], g["claims"], g["states"],
                        active_hosts, stale_s=args.stale_seconds)

    if args.json:
        print(json.dumps({"jobs": jobs, "nodes": node_rows,
                          "orphan_claims": orphan_claims,
                          "orphan_scope_limited": orphan_scope_limited}, indent=2))
        return

    print(f"{len(jobs)} job(s) actif(s) pour {args.user} :")
    for row in node_rows:
        bits = [f"{row['site']}/{row['host']}", f"job {row['job_id']} ({row['name']})"]
        if "load1" in row:
            bits.append(f"load1={row['load1']}")
        for gpu in row.get("gpus", []):
            bits.append(f"gpu{gpu['gpu']} util={gpu.get('util_pct', '?')}%")
        print("  " + "  ".join(bits))
        if "underuse" in row:
            print(f"    SOUS-UTILISÉ : {row['underuse']}")

    if orphan_scope_limited:
        print("\nDétection de claims orphelins désactivée (-j restreint la portée : "
              "impossible de savoir quels nœuds sont actifs sous d'autres jobs -- "
              "relancer sans -j pour l'activer).")
    elif orphan_claims:
        print(f"\nCLAIMS ORPHELINS ({len(orphan_claims)}) -- à vérifier, rien n'a été touché :")
        for o in orphan_claims:
            print(f"  {o['grid']}  {o['run_id']}  worker={o['worker']}  "
                  f"status={o['status']}  raison={o['reason']}  "
                  f"claim_âgé_de={o['claim_age_s']}s")
    else:
        print("\nAucun claim orphelin détecté.")


if __name__ == "__main__":
    main()
