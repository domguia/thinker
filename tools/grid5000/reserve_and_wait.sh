#!/usr/bin/env bash
# reserve_and_wait.sh -- submit a besteffort GPU/CPU job on a Grid'5000 site,
# resolve its real node assignment, and poll until Running/Error/timeout.
#
# Consolidates the ssh-g5k -> ssh <site> -> oarsub -> oarstat -f pattern that
# was being hand-typed repeatedly during multi-agent GPU searches (see
# agents/INFRA_AGENT.md, "Tooling" section, 2026-09-22).
#
# Usage:
#   tools/grid5000/reserve_and_wait.sh <site> <job_name> <walltime_HH:MM:SS> <resource_spec> [property] [poll_timeout_s]
#
# Examples:
#   tools/grid5000/reserve_and_wait.sh rennes my-job 3:00:00 "gpu=1" "cluster='abacus21'"
#   tools/grid5000/reserve_and_wait.sh nancy my-job 3:00:00 "gpu=1" "cluster='graffiti'"
#   tools/grid5000/reserve_and_wait.sh rennes cpu-job 1:30:00 "host=1"   # no property -> no -p flag
#
# `resource_spec` is whatever goes after -l (before ",walltime=..."), e.g.
# "gpu=1", "host=1", "host=1/gpu=1". `property` is the raw SQL WHERE clause
# body for -p (already quoted as needed, e.g. "cluster='abacus21'" or
# "(cluster='abacus21' OR cluster='abacus26')") -- pass "" to skip it.
#
# Prints, on success: RUNNING <job_id> <hostname>
# On failure/timeout: ERROR <job_id> <last_state> or TIMEOUT <job_id>
#
# Notes:
# - Always submits as besteffort+idempotent (the common case for this
#   project's ad-hoc GPU search). For a normal-queue / long-lived
#   reservation, use tools/exp/reserve.py instead -- this script is for
#   the "grab whatever's free/soonest across sites" search pattern.
# - Walltime's `sleep` placeholder command matches the walltime in seconds,
#   per the documented sleep/oarwalltime pitfall in the grid5000 skill.
# - Requires the `g5k` SSH alias (see grid5000 skill's recommended
#   ~/.ssh/config) to reach the bastion.

set -euo pipefail

SITE="${1:?site required (e.g. rennes)}"
NAME="${2:?job name required}"
WALLTIME="${3:?walltime required (HH:MM:SS)}"
RESOURCE="${4:?resource spec required (e.g. gpu=1)}"
PROPERTY="${5:-}"
POLL_TIMEOUT_S="${6:-120}"

# HH:MM:SS -> seconds, for the sleep placeholder.
IFS=: read -r h m s <<<"$WALLTIME"
WALLTIME_S=$((10#$h * 3600 + 10#$m * 60 + 10#$s))

PROP_FLAG=""
if [ -n "$PROPERTY" ]; then
  PROP_FLAG="-p \"$PROPERTY\""
fi

SUBMIT_OUT=$(ssh g5k "ssh $SITE.grid5000.fr 'oarsub -n $NAME -l $RESOURCE,walltime=$WALLTIME -t besteffort -t idempotent $PROP_FLAG \"sleep $WALLTIME_S\"'" 2>&1)
JOB_ID=$(echo "$SUBMIT_OUT" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2)

if [ -z "$JOB_ID" ]; then
  echo "SUBMIT_FAILED: $SUBMIT_OUT" >&2
  exit 1
fi

echo "SUBMITTED $JOB_ID" >&2

elapsed=0
while [ "$elapsed" -lt "$POLL_TIMEOUT_S" ]; do
  STATE_OUT=$(ssh g5k "ssh $SITE.grid5000.fr 'oarstat -f -j $JOB_ID 2>&1 | grep -E \"state|assigned_hostnames\"'" 2>/dev/null || true)
  if echo "$STATE_OUT" | grep -q "state = Running"; then
    HOST=$(echo "$STATE_OUT" | grep assigned_hostnames | sed 's/.*= //' | awk '{print $1}')
    echo "RUNNING $JOB_ID $HOST"
    exit 0
  fi
  if echo "$STATE_OUT" | grep -qiE "state = Error"; then
    echo "ERROR $JOB_ID Error"
    exit 2
  fi
  sleep 10
  elapsed=$((elapsed + 10))
done

echo "TIMEOUT $JOB_ID (still $(echo "$STATE_OUT" | grep state))"
exit 3
