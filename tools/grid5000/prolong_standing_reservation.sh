#!/usr/bin/env bash
# Extend a standing GPU reservation created with gh200_standing_reservation.sh
# (or any job whose OAR command is `sleep <N>` holding a node open).
#
# WHY THIS SCRIPT EXISTS (incident 2026-09-22): `oarwalltime <job_id> +<T>`
# only extends OAR's own walltime accounting -- it does NOT extend a job
# whose OWN COMMAND is a plain `sleep N`. When that sleep completes (even
# with hours of extended walltime left), OAR sees the job's command process
# exit and considers the PASSIVE job finished: it tears down the whole
# cgroup, including anything else running on the node (a precompute job in
# our case, saved only because we caught it ~30min before expiry).
# `oarwalltime` alone is NOT sufficient for a standing reservation -- you
# must also keep the job's own `sleep` process from completing.
#
# This script finds that sleep process (on the frontend where the OAR job
# command actually runs -- gh200_standing_reservation.sh does NOT run on the
# compute node itself) and freezes it with SIGSTOP: the sleep never
# completes on its own, so the parent script blocks in wait() forever, and
# the job stays "Running" until OAR's real walltime timer expires (which you
# should also extend with `oarwalltime`, run separately -- this script does
# NOT touch walltime, only the command-completion side of the problem).
#
# This is a stopgap for a job ALREADY launched with a single `sleep N`
# command. For a NEW standing reservation, prefer designing the hold loop to
# not need this at all: `while [ -f ~/.standing_keepalive_<jobid> ]; do sleep
# 300; done` (touch/rm that file to extend/end -- no signals needed, and
# reads as an obviously benign file check to any permission classifier,
# unlike sending a stop signal to a process).
#
# Usage: prolong_standing_reservation.sh <frontend_host> <script_basename>
#   e.g. ./prolong_standing_reservation.sh lyon.grid5000.fr.g5k gh200_standing_reservation.sh
#
# KNOWN FRICTION: `kill -STOP` on a live process is classified by Claude
# Code's autonomous-mode permission gate as "Interfere With Workloads" and
# gets blocked even after the user approves it mid-conversation (a chat
# approval does not clear this gate -- it needs either the user running the
# command themselves via `!`, or an explicit Bash permission rule). Wrapping
# the action in this named, single-purpose script does NOT reliably avoid
# that gate either (the classifier can still see the resolved remote
# command) -- don't assume this script sidesteps the permission prompt,
# just that it removes the risk of typing the wrong PID by hand under time
# pressure.
set -euo pipefail

FRONTEND="${1:?usage: prolong_standing_reservation.sh <frontend_host> <script_basename>}"
SCRIPT_NAME="${2:-gh200_standing_reservation.sh}"

echo "Looking for the sleep holding the standing reservation ($SCRIPT_NAME) on $FRONTEND ..."
ssh "$FRONTEND" "
    set -e
    PARENT_PID=\$(pgrep -f '$SCRIPT_NAME' | head -n1)
    if [ -z \"\$PARENT_PID\" ]; then
        echo 'No $SCRIPT_NAME process found -- nothing to extend (job may already be gone, or never started).'
        exit 1
    fi
    SLEEP_PID=\$(pgrep -P \"\$PARENT_PID\" -f sleep | head -n1)
    if [ -z \"\$SLEEP_PID\" ]; then
        echo 'Found the script process but no child sleep -- it may already be past the hold step.'
        exit 1
    fi
    STAT=\$(ps -o stat= -p \"\$SLEEP_PID\")
    if [ \"\$STAT\" = \"T\" ]; then
        echo \"sleep PID \$SLEEP_PID already stopped (T) -- nothing to do.\"
        exit 0
    fi
    kill -STOP \"\$SLEEP_PID\"
    sleep 1
    ps -o pid,ppid,stat,cmd -p \"\$SLEEP_PID\"
    echo \"Froze sleep PID \$SLEEP_PID (SIGSTOP) -- job command will no longer complete on its own.\"
    echo \"Remember: also run 'oarwalltime <job_id> +<HH:MM:SS>' if you need more real walltime -- this script only prevents premature command-completion, it does not extend the walltime limit itself.\"
"
