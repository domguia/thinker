#!/usr/bin/env python3
"""Worker de file : tire la cellule suivante de `grid.jsonl`, l'exécute, recommence.

Le patron vient de `dev_notes/compute_scheduling.md` §3. Ce qu'il achète :

- **la préemption ne coûte qu'une cellule par worker**, pas la grille ;
- **idempotence** : relancer la même commande après une préemption reprend le
  travail là où il en est (une cellule dont l'état est `done` est sautée) ;
- **hétérogénéité** : ajouter un nœud en cours de route suffit à consommer la
  même file ;
- **traçage PID** parent+enfant, pour un arrêt propre garanti.

Rien n'est décidé automatiquement : le worker n'abandonne, ne relance et ne
reclasse jamais un résultat. Une cellule échouée reste `failed` et visible.

Lancement type (Tier A, CPU, N workers sur un nœud) :

    for i in $(seq 0 $((NCORES-3))); do
      tools/exp/worker.py --grid runs/toy/grid.jsonl --worker-id $i &
    done; wait

Lancement type (Tier B/C, GPU, plusieurs workers par carte) :

    CUDA_VISIBLE_DEVICES=0 tools/exp/worker.py --grid runs/ia/grid.jsonl --worker-id 0 &
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def load_cells(grid: Path) -> list[dict]:
    with open(grid) as f:
        return [json.loads(l) for l in f if l.strip()]


def claim(claims_dir: Path, run_id: str, worker_tag: str) -> bool:
    """Réserve une cellule de façon atomique entre workers concurrents.

    `O_CREAT|O_EXCL` est atomique y compris sur NFS pour ce cas d'usage, et
    la granularité est grosse (une cellule = un run de plusieurs minutes),
    donc la contention n'est pas un sujet.
    """
    claims_dir.mkdir(parents=True, exist_ok=True)
    path = claims_dir / f"{run_id}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"worker": worker_tag, "pid": os.getpid(), "ts": time.time()}, f)
    return True


def cell_status(state_dir: Path, run_id: str) -> str | None:
    try:
        with open(state_dir / f"{run_id}.json") as f:
            return json.load(f).get("status")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_env(cpu_threads: int) -> dict:
    """Gabarit d'environnement — fixé une fois ici, jamais reconstruit par script.

    `OMP_NUM_THREADS`/`MKL_NUM_THREADS` à 1 par worker est le réglage qui
    évite la sursouscription N² sur les nœuds CPU (compute_scheduling.md §4) :
    pour des modèles à ~100 K paramètres, le parallélisme utile est entre les
    cellules, pas à l'intérieur d'une.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = str(cpu_threads)
    return env


def run_cell(cell: dict, state_dir: Path, log_dir: Path, env: dict,
             timeout_s: float | None, dry_run: bool) -> int:
    run_id = cell["run_id"]
    cmd = [sys.executable, "-u", str(REPO / cell["script"]),
           "--run_id", run_id, "--state_dir", str(state_dir)]
    for k, v in cell["config"].items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]

    if dry_run:
        print("DRY-RUN " + " ".join(cmd), flush=True)
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    print(f"[{time.strftime('%H:%M:%S')}] start {run_id} -> {log_path}", flush=True)

    with open(log_path, "a") as log:
        log.write(f"\n=== {time.ctime()} host={socket.gethostname()} "
                  f"cmd={' '.join(cmd)}\n")
        log.flush()
        # start_new_session : le run vit dans son propre groupe de processus,
        # donc un arrêt tue bien l'enfant et ses descendants, pas seulement
        # le wrapper (traçage PID parent+enfant du catalogue §3).
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=REPO, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            rc = -signal.SIGTERM

    print(f"[{time.strftime('%H:%M:%S')}] end   {run_id} rc={rc}", flush=True)
    return rc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, type=Path)
    p.add_argument("--worker-id", default="0")
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="threads BLAS par run (1 sur les nœuds CPU multi-workers)")
    p.add_argument("--cell-timeout-minutes", type=float, default=None,
                   help="borne dure par cellule, en plus des bornes internes du script")
    p.add_argument("--retry-failed", action="store_true",
                   help="reprendre aussi les cellules marquées failed/incomplete")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-frontend", action="store_true",
                   help="desactive le garde-fou frontend (voir ci-dessous) -- ne jamais utiliser "
                        "pour un vrai calcul, seulement pour un test delibere sur le frontend")
    args = p.parse_args()

    # Grid'5000 : le hostname court d'un frontend suit toujours le motif "f"+site
    # (frennes, fnancy, flyon, ...) alors qu'un noeud de calcul a son propre nom
    # de cluster (abacus18-1, paradoxe-5, graffiti-3, ...) -- jamais ce motif.
    # Erreur reelle rencontree deux fois cette nuit (Nancy puis Rennes) : lancer
    # des workers via `ssh <site>.grid5000.fr.g5k` au lieu de `ssh <noeud>...` --
    # le frontend accepte la commande puis tue les process au bout d'1-2 min
    # (limite cgroup probable), gaspillant du temps mural avant detection.
    _KNOWN_SITES = ("rennes", "nancy", "lyon", "grenoble", "toulouse", "lille",
                     "strasbourg", "nantes", "sophia")
    _hostname = socket.gethostname().split(".")[0]
    if not args.allow_frontend and _hostname in {f"f{s}" for s in _KNOWN_SITES}:
        print(f"REFUS : ce worker semble lance sur le FRONTEND ({_hostname}), pas un noeud de "
              f"calcul reserve -- relancer via 'ssh <noeud>.<site>.grid5000.fr.g5k' au lieu de "
              f"'ssh <site>.grid5000.fr.g5k'. Utiliser --allow-frontend pour forcer (deconseille).",
              file=sys.stderr)
        sys.exit(1)

    grid = args.grid.resolve()
    root = grid.parent
    state_dir, claims_dir, log_dir = root / "state", root / "claims", root / "logs"
    worker_tag = f"{socket.gethostname()}:{args.worker_id}"
    env = build_env(args.cpu_threads)
    timeout_s = args.cell_timeout_minutes * 60 if args.cell_timeout_minutes else None

    resumable = {"failed", "incomplete", None} if args.retry_failed else {None}
    done = skipped = 0

    # Recharge le fichier de grille à chaque passe plutôt qu'une fois au
    # démarrage : une grille étendue en cours de nuit (balayage LR élargi
    # après un optimum en bord de plage, par ex.) est vue sans relancer les
    # workers déjà en cours. Sûr par construction -- le claim atomique
    # empêche tout doublon, une passe qui ne trouve rien de neuf arrête le
    # worker exactement comme avant (grille épuisée = grille épuisée).
    while True:
        made_progress = False
        for cell in load_cells(grid):
            rid = cell["run_id"]
            status = cell_status(state_dir, rid)
            if status == "done" or status not in resumable:
                skipped += 1
                continue
            if args.retry_failed and status in ("failed", "incomplete"):
                # La revendication d'un essai précédent doit être levée pour rejouer.
                (claims_dir / f"{rid}.claim").unlink(missing_ok=True)
            if not claim(claims_dir, rid, worker_tag):
                skipped += 1
                continue
            run_cell(cell, state_dir, log_dir, env, timeout_s, args.dry_run)
            done += 1
            made_progress = True
        if not made_progress:
            break

    print(f"worker {worker_tag} terminé : {done} cellules exécutées, {skipped} sautées",
          flush=True)


if __name__ == "__main__":
    main()
