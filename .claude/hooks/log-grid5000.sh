#!/usr/bin/env bash
# PostToolUse hook (matcher: Bash) — filet de sécurité mécanique.
# Journalise automatiquement toute commande liée à Grid'5000, sans jugement ni blocage.
set -u

input="$(cat)"
command="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)"

if [ -n "$command" ] && printf '%s' "$command" | grep -Eq '\b(oarsub|oardel|oarstat|kadeploy3?)\b|\bssh\b[^|]*\.g5k|\b(curl|wget|scp|rsync)\b[^|]*grid5000\.fr'; then
  mkdir -p "$(dirname "$0")/../../logs" 2>/dev/null
  log_file="$(cd "$(dirname "$0")/../.." && pwd)/logs/grid5000_raw.log"
  printf '%s\t%s\n' "$(date -Is)" "$command" >> "$log_file"
fi

exit 0
