#!/usr/bin/env python3
"""Statut composite d'une grille : file + état des runs + télémétrie nœud.

Remplace les `oarstat` / `ps aux` / `nvidia-smi` faits à la main, et surtout
les fait **ensemble** : un run peut être `running` dans son fichier d'état
alors que son processus est mort et que son log ne bouge plus depuis 40 min.
C'est exactement le trou décrit en tête de
`dev_notes/experiment_tooling_requirements.md`.

L'outil signale, il ne tranche pas : une ligne SUSPECT reste une hypothèse à
vérifier, jamais un job tué automatiquement.

    tools/exp/status.py --grid runs/kb_chain_depth/grid.jsonl
    tools/exp/status.py --grid ... --telemetry   # ajoute nvidia-smi / load average
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

STALE_S = 600  # log figé au-delà → suspect, pas mort : à vérifier


def load_cells(grid: Path) -> list[dict]:
    with open(grid) as f:
        return [json.loads(l) for l in f if l.strip()]


def pid_alive(pid: int | None) -> bool | None:
    if not pid:
        return None
    return Path(f"/proc/{pid}").exists()


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}" if s >= 3600 else f"{s//60}m{s%60:02d}"


def telemetry() -> list[str]:
    """Deux branches de collecte, pas une : GPU *et* CPU (§1 de la spec)."""
    out = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                idx, util, used, total = [x.strip() for x in line.split(",")]
                out.append(f"  GPU {idx}: {util:>3}% util, {used}/{total} MiB VRAM")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out.append("  GPU : nvidia-smi indisponible")
    try:
        load = Path("/proc/loadavg").read_text().split()[:3]
        ncpu = len(list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")))
        out.append(f"  CPU : load {' '.join(load)} sur {ncpu} cœurs "
                   f"({'sursouscrit' if float(load[0]) > ncpu * 1.2 else 'ok'})")
    except Exception:
        pass
    return out


def oar_jobs() -> list[str]:
    try:
        r = subprocess.run(["oarstat", "-u"], capture_output=True, text=True, timeout=15)
        return r.stdout.strip().splitlines() if r.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, type=Path)
    p.add_argument("--telemetry", action="store_true", help="ajouter GPU/CPU/OAR du nœud courant")
    p.add_argument("--stale-seconds", type=int, default=STALE_S)
    args = p.parse_args()

    root = args.grid.resolve().parent
    state_dir, log_dir = root / "state", root / "logs"
    now = time.time()
    counts: dict[str, int] = {}
    rows, suspects = [], []

    for cell in load_cells(args.grid):
        rid = cell["run_id"]
        st = None
        try:
            st = json.loads((state_dir / f"{rid}.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        if st is None:
            counts["pending"] = counts.get("pending", 0) + 1
            rows.append(f"  {'pending':<10} {rid}")
            continue

        status = st.get("status", "?")
        counts[status] = counts.get(status, 0) + 1
        prog = st.get("last_progress") or {}
        step, max_steps = prog.get("step"), st.get("max_steps")
        age = now - st.get("updated_at", now)
        log_path = log_dir / f"{rid}.log"
        log_age = now - log_path.stat().st_mtime if log_path.exists() else None
        alive = pid_alive((st.get("meta") or {}).get("pid"))

        detail = f"step {step}/{max_steps}" if step else "—"
        eta = f"eta {fmt_dur(prog.get('eta_s'))}" if prog.get("eta_s") else ""
        rows.append(f"  {status:<10} {rid}  {detail:<16} {eta:<12} "
                    f"état il y a {fmt_dur(age)}")

        # Réconciliation : l'état dit 'running', la réalité dit peut-être non.
        if status == "running":
            why = []
            if alive is False:
                why.append("PID absent")
            if log_age is not None and log_age > args.stale_seconds:
                why.append(f"log figé depuis {fmt_dur(log_age)}")
            if age > args.stale_seconds:
                why.append(f"état figé depuis {fmt_dur(age)}")
            if why:
                suspects.append(f"  {rid} : {', '.join(why)}")

    total = sum(counts.values())
    print(f"Grille {args.grid}  —  {total} cellules")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    for r in rows:
        print(r)

    if suspects:
        print("\nSUSPECT — déclaré running mais la réalité ne suit pas "
              "(à vérifier, rien n'a été tué) :")
        for s in suspects:
            print(s)

    if args.telemetry:
        print("\nTélémétrie du nœud courant :")
        for line in telemetry():
            print(line)
        jobs = oar_jobs()
        if jobs:
            print("\nJobs OAR :")
            for j in jobs:
                print("  " + j)


if __name__ == "__main__":
    main()
