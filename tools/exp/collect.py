#!/usr/bin/env python3
"""Collecte des résultats d'une grille en tableau comparatif.

Lit les fichiers d'état `done` et produit une table : les axes qui varient
entre cellules en colonnes de gauche, les métriques à droite, **avec les
contrôles triviaux attachés**.

Le garde-fou F2 est appliqué ici aussi, pas seulement à l'écriture du script :
une accuracy sans `chance`/`margin`/`leak_check` s'affiche avec un marqueur
`(SANS CONTRÔLE)` au lieu d'un chiffre nu. Deux surinterprétations réelles de
ce projet sont parties de là.

    tools/exp/collect.py --grid runs/kb_chain_depth/grid.jsonl
    tools/exp/collect.py --grid ... --sort final_acc --csv resultats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.run_logging import ACCURACY_LIKE, REQUIRED_CONTROLS  # noqa: E402


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "—" if v is None else str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, type=Path)
    p.add_argument("--sort", help="métrique de tri (décroissant)")
    p.add_argument("--csv", type=Path, help="exporter aussi en CSV")
    p.add_argument("--include-unfinished", action="store_true",
                   help="inclure les cellules running/failed (métriques partielles)")
    args = p.parse_args()

    root = args.grid.resolve().parent
    state_dir = root / "state"
    cells = [json.loads(l) for l in open(args.grid) if l.strip()]

    records = []
    for cell in cells:
        try:
            st = json.loads((state_dir / f"{cell['run_id']}.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if st.get("status") != "done" and not args.include_unfinished:
            continue
        summary = st.get("summary") or {}
        controls = summary.get("controls") or {}
        records.append({
            "run_id": cell["run_id"],
            "status": st.get("status"),
            "config": cell["config"],
            "metrics": {k: v for k, v in summary.items()
                        if k not in ("controls", "run_id") and not isinstance(v, (dict, list))},
            "controls": controls,
        })

    if not records:
        print("Aucun résultat exploitable. "
              "Vérifier l'avancement avec : tools/exp/status.py --grid "
              f"{args.grid}")
        return

    # N'afficher que les axes qui varient : une colonne constante est du bruit.
    all_keys = sorted({k for r in records for k in r["config"]})
    axes = [k for k in all_keys
            if len({json.dumps(r["config"].get(k)) for r in records}) > 1] or all_keys
    metrics = sorted({k for r in records for k in r["metrics"]})

    if args.sort:
        records.sort(key=lambda r: (r["metrics"].get(args.sort) is None,
                                    -(r["metrics"].get(args.sort) or 0)))

    header = ["run_id", "status"] + axes + metrics
    rows = []
    for r in records:
        row = [r["run_id"], r["status"]] + [fmt(r["config"].get(k)) for k in axes]
        for m in metrics:
            val = fmt(r["metrics"].get(m))
            if ACCURACY_LIKE.search(m):
                missing = [c for c in REQUIRED_CONTROLS if c not in r["controls"]]
                if missing:
                    val += " (SANS CONTRÔLE)"
            row.append(val)
        rows.append(row)

    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
              for i, h in enumerate(header)]
    sep = "  "
    print(sep.join(h.ljust(w) for h, w in zip(header, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(str(c).ljust(w) for c, w in zip(r, widths)))

    # Les contrôles ne sont pas une colonne parmi d'autres : ils se lisent
    # collés à la métrique qu'ils qualifient.
    print("\nContrôles triviaux par cellule :")
    for r in records:
        c = r["controls"]
        if c:
            print(f"  {r['run_id']}: " + "  ".join(f"{k}={fmt(v)}" for k, v in sorted(c.items())))
        else:
            print(f"  {r['run_id']}: AUCUN — toute accuracy de cette ligne est "
                  f"ininterprétable telle quelle")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header + [f"control_{c}" for c in REQUIRED_CONTROLS])
            for rec, row in zip(records, rows):
                w.writerow(row + [rec["controls"].get(c) for c in REQUIRED_CONTROLS])
        print(f"\nCSV écrit : {args.csv}")


if __name__ == "__main__":
    main()
