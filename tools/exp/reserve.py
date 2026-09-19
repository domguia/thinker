#!/usr/bin/env python3
"""Réservation OAR + résolution de l'assignation réelle.

Deux pièges déjà payés sur ce projet, traités ici une fois pour toutes :

1. **`-l gpu=2` peut atterrir sur 1 ou 2 nœuds** (friction log C2). Il faut
   lire l'assignation *réelle* avant de générer la moindre commande de
   lancement, jamais la supposer depuis la demande.
2. **Le nombre de cœurs varie beaucoup d'un nœud `paradoxe` à l'autre**
   (16 à 52 observés), donc le nombre de workers CPU à lancer ne peut pas
   être écrit en dur dans un script de lancement.

À lancer depuis une frontend Grid'5000.

    tools/exp/reserve.py --tier A --walltime 6 --name toy-grid       # CPU, queue normale
    tools/exp/reserve.py --tier B --gpus 4 --walltime 4 --name ia-grid  # GPU besteffort
    tools/exp/reserve.py --resolve 4091061     # assignation réelle d'un job existant
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Tiers de `dev_notes/compute_scheduling.md` §2. Le Tier A part en queue
# normale (donc sans préemption) : une grille CPU qui tourne tranquillement
# est plus productive qu'une grille GPU besteffort tuée trois fois.
TIERS = {
    "A": {"queue": "default", "besteffort": False,
          "desc": "toy_memory / petits modèles — CPU, queue normale, pas de préemption"},
    "B": {"queue": "default", "besteffort": True,
          "desc": "Indexed Attention (d_model=256) — GPU modeste, beaucoup de process"},
    "C": {"queue": "default", "besteffort": True,
          "desc": "Distillation / Teacher FP8 — vrai besoin VRAM (L40S/H100/A100)"},
}

# Ne jamais laisser un Tier A ou B occuper ces cartes : ce sont les seules
# qui font tourner le Teacher FP8, la seule ressource non substituable du projet.
RESERVED_FOR_TIER_C = ("abacus26", "abacus27")


def sh(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]} introuvable — à lancer depuis une frontend Grid'5000"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def resolve(job_id: str) -> dict:
    """Assignation réelle d'un job : nœuds, cœurs par nœud, GPU par nœud."""
    rc, out = sh(["oarstat", "-j", str(job_id), "-J"])
    info: dict = {"job_id": job_id}
    if rc == 0:
        try:
            data = json.loads(out)
            job = data.get(str(job_id), data)
            info["state"] = job.get("state")
            info["nodes"] = sorted({n.split(".")[0] for n in job.get("assigned_network_address", [])})
        except json.JSONDecodeError:
            pass
    if not info.get("nodes"):
        rc2, out2 = sh(["oarstat", "-j", str(job_id), "-p"])
        info["raw"] = out2 if rc2 == 0 else out

    per_node = {}
    for node in info.get("nodes", []):
        host = f"{node}"
        rc3, cores = sh(["oarsh", host, "nproc"], timeout=20)
        rc4, gpus = sh(["oarsh", host, "bash", "-lc",
                        "nvidia-smi --query-gpu=index,name --format=csv,noheader || true"],
                       timeout=20)
        per_node[node] = {
            "cores": int(cores) if rc3 == 0 and cores.isdigit() else None,
            "gpus": [g.strip() for g in gpus.splitlines() if g.strip()] if rc4 == 0 else [],
        }
    info["per_node"] = per_node
    return info


def print_launch_plan(info: dict) -> None:
    """Le nombre de workers vient de la mesure, jamais d'une valeur en dur."""
    print("\nPlan de lancement (à copier tel quel) :")
    for node, d in info.get("per_node", {}).items():
        if d["gpus"]:
            print(f"  # {node} — {len(d['gpus'])} GPU : "
                  f"{', '.join(d['gpus'])}")
            for i in range(len(d["gpus"])):
                print(f"  oarsh {node} \"cd \\$THINKER && CUDA_VISIBLE_DEVICES={i} "
                      f"tools/exp/worker.py --grid <grid.jsonl> --worker-id {node}-gpu{i}\" &")
        elif d["cores"]:
            # cœurs - 2 : on garde de la marge pour le système et l'I/O.
            n = max(1, d["cores"] - 2)
            print(f"  # {node} — {d['cores']} cœurs mesurés → {n} workers, "
                  f"OMP_NUM_THREADS=1 (fixé par le worker)")
            print(f"  oarsh {node} \"cd \\$THINKER && for i in \\$(seq 0 {n-1}); do "
                  f"tools/exp/worker.py --grid <grid.jsonl> --worker-id {node}-\\$i & done; wait\" &")
        else:
            print(f"  # {node} — assignation non résolue, vérifier à la main")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=sorted(TIERS), help="tier de charge (compute_scheduling.md §2)")
    p.add_argument("--gpus", type=int, default=0, help="GPU demandés (Tier B/C)")
    p.add_argument("--nodes", type=int, default=1, help="nœuds entiers demandés (Tier A)")
    p.add_argument("--walltime", type=float, default=4, help="heures")
    p.add_argument("--name", help="nom du job (obligatoire pour soumettre)")
    p.add_argument("--cluster", help="cluster visé (ex. paradoxe, abacus3)")
    p.add_argument("--resolve", metavar="JOB_ID", help="résoudre l'assignation d'un job existant")
    p.add_argument("--dry-run", action="store_true", help="afficher la commande sans soumettre")
    args = p.parse_args()

    if args.resolve:
        info = resolve(args.resolve)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print_launch_plan(info)
        return

    if not args.tier or not args.name:
        p.error("--tier et --name sont requis pour soumettre (ou utiliser --resolve)")

    tier = TIERS[args.tier]
    if args.tier in ("A", "B") and args.cluster in RESERVED_FOR_TIER_C:
        p.error(f"{args.cluster} est réservé au Tier C (Teacher FP8) — "
                f"y mettre un Tier {args.tier} bloque la seule ressource non "
                f"substituable du projet.")

    print(f"Tier {args.tier} : {tier['desc']}")

    resource = f"host={args.nodes}" if args.gpus == 0 else f"gpu={args.gpus}"
    cmd = ["oarsub", "-n", args.name,
           "-l", f"{resource},walltime={args.walltime}:00:00"]
    if tier["besteffort"]:
        cmd += ["-t", "besteffort", "-t", "idempotent"]
    if args.cluster:
        cmd += ["-p", f"cluster='{args.cluster}'"]
    cmd += ["--stdout", f"OAR.{args.name}.%jobid%.stdout",
            "--stderr", f"OAR.{args.name}.%jobid%.stderr",
            "--", "sleep", f"{int(args.walltime * 3600)}"]

    print("\nCommande :\n  " + " ".join(cmd))

    if args.gpus and not tier["besteffort"]:
        print("\nAVERTISSEMENT : le groupe `wide` n'a pas d'allocation GPU "
              "prioritaire à Rennes — un job GPU non-besteffort risque de ne "
              "jamais démarrer.")
    if tier["besteffort"]:
        print("\nRappel : besteffort = préemptible à tout moment (observé : ~25 min). "
              "La file de travail absorbe ça (une cellule perdue par worker), "
              "à condition de lancer via tools/exp/worker.py.")

    if args.dry_run:
        return

    print("\nVérifier d'abord les jobs existants :")
    rc, out = sh(["oarstat", "-u"])
    print(out if rc == 0 else f"  (oarstat indisponible : {out})")
    print("\nSoumission non automatique : copier la commande ci-dessus. "
          "Une fois le job démarré, résoudre l'assignation réelle avec :\n"
          f"  tools/exp/reserve.py --resolve <JOB_ID>")


if __name__ == "__main__":
    main()
