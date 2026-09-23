#!/usr/bin/env python3
"""Extrait les prompts reellement tapes par l'utilisateur d'une session Claude Code.

Usage:
    python3 extract_user_prompts.py <session_id_ou_chemin.jsonl> [--all] [--out FICHIER]

Par defaut, cherche le .jsonl correspondant a <session_id> sous:
    ~/.claude/projects/*/<session_id>.jsonl
    ~/.claude-oly/projects/*/<session_id>.jsonl
(un chemin .jsonl direct est aussi accepte)

--all   garde aussi le bruit (commandes locales, hand-backs de sous-agents,
        notifications de taches, resumes de contexte). Par defaut ce bruit
        est filtre pour ne garder que les messages vraiment tapes par l'utilisateur.
--out   fichier de sortie markdown (defaut: stdout)
"""
import argparse
import json
import re
import sys
from pathlib import Path

CONFIG_DIRS = [
    Path.home() / ".claude",
    Path.home() / ".claude-oly",
]

NOISE_PREFIXES = (
    "<local-command-caveat>",
    "<command-name>",
    "<local-command-stdout>",
    "<task-notification>",
    "## Context Usage",
)


def find_session_file(session_id: str) -> Path:
    p = Path(session_id)
    if p.is_file():
        return p

    matches = []
    for base in CONFIG_DIRS:
        projects = base / "projects"
        if not projects.is_dir():
            continue
        matches.extend(projects.glob(f"*/{session_id}.jsonl"))

    if not matches:
        raise FileNotFoundError(
            f"Aucun fichier de session trouve pour '{session_id}' sous {', '.join(str(b) for b in CONFIG_DIRS)}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Plusieurs sessions correspondent, precise le chemin complet:\n"
            + "\n".join(str(m) for m in matches)
        )
    return matches[0]


def extract_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts).strip()
    return ""


def is_noise(text: str) -> bool:
    if text.startswith("<system-reminder>") and text.endswith("</system-reminder>"):
        return True
    if text.startswith(NOISE_PREFIXES):
        return True
    if text[:60].strip().startswith("Another Claude session sent a message"):
        return True
    return False


def strip_system_reminders(text: str) -> str:
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL).strip()


def extract_prompts(session_path: Path, keep_all: bool):
    prompts = []
    with session_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "user":
                continue
            text = extract_text(obj.get("message", {}))
            if not text:
                continue
            if not keep_all and is_noise(text):
                continue
            cleaned = strip_system_reminders(text) if not keep_all else text
            if not cleaned:
                continue
            prompts.append((obj.get("timestamp", "?"), cleaned))
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", help="UUID de session ou chemin vers un .jsonl")
    ap.add_argument("--all", action="store_true", help="garder aussi le bruit (commandes, hand-backs, notifications)")
    ap.add_argument("--out", help="fichier markdown de sortie (defaut: stdout)")
    args = ap.parse_args()

    session_path = find_session_file(args.session_id)
    prompts = extract_prompts(session_path, keep_all=args.all)

    header = f"# Prompts extraits - session {session_path.stem}\n\nSource: {session_path}\nTotal: {len(prompts)}\n"
    body_parts = [f"## Prompt {i} (ts: {ts})\n\n{text}\n" for i, (ts, text) in enumerate(prompts, 1)]
    output = header + "\n---\n\n" + "\n---\n\n".join(body_parts)

    if args.out:
        Path(args.out).write_text(output)
        print(f"OK: {len(prompts)} prompts ecrits dans {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
