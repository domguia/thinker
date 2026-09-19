"""Journalisation standard des runs d'expérience (Thinker).

Un seul point de vérité pour ce que tout script `learn/*/train_*.py` écrit :

- une ligne `progress: {json}` à `--log_every`, avec `eta_s` ;
- un fichier d'état `<state_dir>/<run_id>.json` réécrit **atomiquement** à
  chaque progression, pour que la courbe survive à un SIGKILL de walltime ou
  à une préemption besteffort (incident A5 du friction log) ;
- un résumé final `summary: {json}` + le même contenu dans le fichier d'état ;
- un refus d'émettre une métrique d'accuracy sans ses contrôles triviaux
  attachés (chance level, marge, leak_check) — incident F2.

Rien ici ne décide quoi que ce soit : l'objet écrit ce que le script lui
donne, et refuse seulement ce qui est structurellement ininterprétable.

Voir `.claude/skills/experiment-script-design/SKILL.md` pour la convention
d'écriture complète, et `dev_notes/experiment_tooling_requirements.md` pour
les besoins d'origine.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Une métrique dont le nom matche ceci ne peut pas être publiée seule :
# sans niveau de chance ni contrôle de fuite, elle n'est pas interprétable.
ACCURACY_LIKE = re.compile(
    r"(^|_)(acc|accuracy|exact_match|em|recall|precision|f1|hit_rate|success_rate)($|_)"
)

REQUIRED_CONTROLS = ("chance", "margin", "leak_check")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Écrit puis `os.replace` — un lecteur concurrent ne voit jamais un
    fichier à moitié écrit, y compris sur NFS (cf. compute_scheduling.md §5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class MissingControlsError(ValueError):
    """Une métrique d'accuracy a été publiée sans ses contrôles triviaux."""


class RunLogger:
    """Cycle de vie d'un run : `progress()` en boucle, puis `finish()`.

    Args:
        run_id: identifiant stable du run. C'est lui qui fait l'idempotence
            de la file de travail — deux lancements de la même cellule
            portent le même `run_id`.
        state_dir: répertoire où vit le fichier d'état (sur le storage
            partagé pour un job Grid'5000, jamais dans /tmp du nœud).
        config: les hyperparamètres complets du run (typiquement
            `vars(args)`) — l'empreinte anti-doublon et la comparaison
            diagnostic/grille en dépendent entièrement.
        max_steps / max_time_minutes: les deux bornes, co-spécifiées, qui
            servent à calculer l'ETA (piège du `--max_steps` par défaut).
        key_metric: la métrique affichée dans la ligne de progression.
    """

    def __init__(
        self,
        run_id: str,
        state_dir: str | os.PathLike,
        config: Mapping[str, Any],
        max_steps: int | None = None,
        max_time_minutes: float | None = None,
        key_metric: str = "loss",
        stream=None,
    ) -> None:
        self.run_id = run_id
        self.state_path = Path(state_dir) / f"{run_id}.json"
        self.config = dict(config)
        self.max_steps = max_steps
        self.max_time_minutes = max_time_minutes
        self.key_metric = key_metric
        self.stream = stream if stream is not None else sys.stdout

        self.start_time = time.time()
        self.last_progress: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self._finished = False

        self.meta = {
            "run_id": run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "oar_job_id": os.environ.get("OAR_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_sha": _git_sha(),
            "argv": sys.argv,
            "started_at": self.start_time,
        }
        self._write_state("running")

    # ------------------------------------------------------------------ état

    def _write_state(self, status: str, extra: Mapping[str, Any] | None = None) -> None:
        payload = {
            "status": status,
            "meta": self.meta,
            "config": self.config,
            "max_steps": self.max_steps,
            "max_time_minutes": self.max_time_minutes,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "last_progress": self.last_progress,
            "history": self.history,
            "updated_at": time.time(),
        }
        if extra:
            payload.update(extra)
        _atomic_write_json(self.state_path, payload)

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        # Une ligne préfixée + JSON : greppable par l'outil avec une seule
        # regex, non ambigüe à parser, et intacte après redirection dans un
        # fichier de log (contrairement à une barre `tqdm`).
        print(f"{kind}: {json.dumps(payload, sort_keys=True, default=str)}",
              file=self.stream, flush=True)

    # ------------------------------------------------------------- ETA

    def _eta_seconds(self, step: int) -> float | None:
        """Temps restant, borné par celle des deux limites qui mord en premier.

        Renvoie None tant qu'il n'y a pas de quoi extrapoler. L'ETA sert
        surtout à détecter un run qui rame anormalement : sur besteffort,
        c'est la préemption qui tue, pas le walltime.
        """
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return None
        etas = []
        if self.max_steps and step > 0:
            etas.append(elapsed * (self.max_steps - step) / step)
        if self.max_time_minutes:
            etas.append(max(0.0, self.max_time_minutes * 60 - elapsed))
        return round(min(etas), 1) if etas else None

    # -------------------------------------------------------- progression

    def progress(self, step: int, **metrics: Any) -> dict[str, Any]:
        """Émet une ligne de progression et persiste l'état.

        Appeler à `--log_every`. Les métriques passées ici sont libres (loss,
        lr, acc d'éval intermédiaire...) : les contrôles triviaux ne sont
        exigés qu'au résumé final, où la surinterprétation a lieu.
        """
        elapsed = time.time() - self.start_time
        rec = {
            "run_id": self.run_id,
            "ts": time.time(),
            "step": step,
            "max_steps": self.max_steps,
            "elapsed_s": round(elapsed, 1),
            "eta_s": self._eta_seconds(step),
            "steps_per_s": round(step / elapsed, 3) if elapsed > 0 else None,
            **metrics,
        }
        self.last_progress = rec
        self.history.append({k: v for k, v in rec.items() if k != "run_id"})
        self._emit("progress", rec)
        self._write_state("running")
        return rec

    # ------------------------------------------------------------- final

    def finish(
        self,
        summary: Mapping[str, Any],
        controls: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Émet le résumé final et marque le run `done`.

        `controls` doit contenir `chance`, `margin` et `leak_check` dès que
        `summary` porte une métrique d'accuracy — c'est le garde-fou F2,
        appliqué à l'écriture du script et pas seulement à l'affichage.
        """
        acc_keys = [k for k in summary if ACCURACY_LIKE.search(k)]
        if acc_keys:
            missing = [c for c in REQUIRED_CONTROLS if not controls or c not in controls]
            if missing:
                raise MissingControlsError(
                    f"métriques {acc_keys} publiées sans contrôles {missing}. "
                    f"Attendu: controls={{'chance': ..., 'margin': ..., 'leak_check': ...}}. "
                    f"Une accuracy sans niveau de chance ni contrôle de fuite n'est pas "
                    f"interprétable (voir friction log F2)."
                )

        payload = {
            "run_id": self.run_id,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "final_step": self.last_progress.get("step"),
            **dict(summary),
            "controls": dict(controls) if controls else None,
        }
        self._emit("summary", payload)
        self._write_state("done", {"summary": payload})
        self._finished = True
        return payload

    def fail(self, reason: str) -> None:
        """Marque le run échoué en conservant l'historique déjà persisté."""
        self._emit("failed", {"run_id": self.run_id, "reason": reason})
        self._write_state("failed", {"error": reason})
        self._finished = True

    # ------------------------------------------------- context manager

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._finished:
            self.fail(f"{exc_type.__name__}: {exc}")
        elif not self._finished:
            # Sortie sans finish() : ni done ni failed, l'outil doit le voir.
            self._write_state("incomplete")
        return False


def read_state(path: str | os.PathLike) -> dict[str, Any] | None:
    """Lecture tolérante d'un fichier d'état (None si absent ou en cours d'écriture)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def add_run_args(parser) -> None:
    """Ajoute les deux arguments que `tools/exp/worker.py` passe à tout script.

    À appeler sur le `argparse.ArgumentParser` de n'importe quel script
    d'expérience : c'est la seule chose qui le rend consommable par la file
    de travail, et ça reste sans effet sur un lancement à la main.
    """
    parser.add_argument("--run_id", default=None,
                        help="identité stable de la cellule (fournie par worker.py)")
    parser.add_argument("--state_dir", default="runs/adhoc/state",
                        help="répertoire des fichiers d'état des runs")


def logger_from_args(args, key_metric: str = "loss") -> RunLogger:
    """Construit un RunLogger depuis les `args` d'un script standard.

    Le `run_id` retombe sur une empreinte (script, config) quand il n'est pas
    fourni, pour qu'un lancement à la main produise quand même un état
    lisible par `tools/exp/status.py`.
    """
    import hashlib

    config = {k: v for k, v in vars(args).items() if k not in ("run_id", "state_dir")}
    run_id = getattr(args, "run_id", None)
    if not run_id:
        blob = json.dumps({"script": Path(sys.argv[0]).stem, "config": config},
                          sort_keys=True, default=str)
        run_id = f"{Path(sys.argv[0]).stem}-{hashlib.sha1(blob.encode()).hexdigest()[:10]}"
    return RunLogger(
        run_id=run_id,
        state_dir=args.state_dir,
        config=config,
        max_steps=getattr(args, "max_steps", None),
        max_time_minutes=getattr(args, "max_time_minutes", None),
        key_metric=key_metric,
    )
