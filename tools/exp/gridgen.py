#!/usr/bin/env python3
"""Matérialise une grille d'expériences en fichier de file (`grid.jsonl`).

Une ligne = une cellule = un run complet. Généré une fois, consommé par
`worker.py`. C'est ce fichier qui remplace le registre de backlog dispersé
dans les `dev_notes/*_experiment_plan.md`.

Exemple
-------
    tools/exp/gridgen.py \
        --out runs/kb_chain_depth/grid.jsonl \
        --script learn/indexed_attention/train_kb_chain.py \
        --fixed max_steps=8000 max_time_minutes=45 \
        --sweep depth=2,3,4 seed=0,1,2 lr=3e-4,1e-3

Produit 18 cellules. Le `run_id` est un hash stable de (script, config) :
relancer `gridgen` sur la même spec régénère exactement les mêmes ids, donc
les cellules déjà faites restent reconnues comme faites.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"attendu clé=valeur, reçu: {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def coerce(v: str):
    """Garde les types lisibles dans le fichier de file, sans deviner trop."""
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def run_id_for(script: str, config: dict) -> str:
    """Empreinte stable (script, config) — base de l'anti-doublon.

    Le nom reste lisible : préfixe du script + hash court, pour qu'un
    `ls state/` soit interprétable sans outil.
    """
    blob = json.dumps({"script": script, "config": config}, sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:10]
    return f"{Path(script).stem}-{h}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="chemin du grid.jsonl à écrire")
    p.add_argument("--script", required=True, help="script d'entraînement (chemin depuis la racine du dépôt)")
    p.add_argument("--fixed", nargs="*", default=[], metavar="K=V",
                   help="hyperparamètres communs à toutes les cellules")
    p.add_argument("--sweep", nargs="*", default=[], metavar="K=V1,V2,...",
                   help="axes balayés (produit cartésien)")
    p.add_argument("--append", action="store_true",
                   help="ajouter à un grid.jsonl existant au lieu de l'écraser")
    args = p.parse_args()

    fixed = {k: coerce(v) for k, v in parse_kv(args.fixed).items()}
    sweep_raw = parse_kv(args.sweep)
    sweep = {k: [coerce(x) for x in v.split(",")] for k, v in sweep_raw.items()}

    # Garde-fou du piège rencontré deux fois : un run sans borne explicite
    # hérite d'un défaut (souvent --max_steps=50) et rend la cellule muette.
    for bound in ("max_steps", "max_time_minutes"):
        if bound not in fixed and bound not in sweep:
            p.error(f"--fixed {bound}=... est obligatoire : les deux bornes "
                    f"(max_steps ET max_time_minutes) doivent être co-spécifiées.")

    keys = list(sweep)
    combos = list(itertools.product(*(sweep[k] for k in keys))) if keys else [()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if args.append and out_path.exists():
        with open(out_path) as f:
            seen = {json.loads(l)["run_id"] for l in f if l.strip()}

    lines, dups = [], 0
    for combo in combos:
        config = {**fixed, **dict(zip(keys, combo))}
        rid = run_id_for(args.script, config)
        if rid in seen:
            dups += 1
            continue
        seen.add(rid)
        lines.append(json.dumps({"run_id": rid, "script": args.script, "config": config},
                                sort_keys=True))

    mode = "a" if args.append and out_path.exists() else "w"
    with open(out_path, mode) as f:
        for l in lines:
            f.write(l + "\n")

    print(f"{len(lines)} cellules écrites dans {out_path}"
          + (f" ({dups} doublons ignorés)" if dups else ""))
    if lines:
        print(f"état attendu dans : {out_path.parent / 'state'}/<run_id>.json")


if __name__ == "__main__":
    main()
