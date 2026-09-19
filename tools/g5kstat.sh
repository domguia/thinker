#!/bin/bash
# Automates what /g5kstat does by hand: list every active jdomguia job across
# all Grid'5000 sites, resolve its real assigned node(s), check ACTUAL
# utilization (nvidia-smi for GPU nodes, uptime/ps for CPU nodes -- never
# trust `oarstat`'s state alone, a job can show "Running" at 0% for a
# silently-dead launch), and print a summary at the chosen granularity.
#
# Read-only: never submits, kills, or modifies anything.
#
# Two-step design (list job IDs, then resolve each individually via
# `oarstat -j`) rather than one bulk `oarstat -J` per site: the bulk JSON
# call was found to intermittently return incomplete output on this
# project's larger jobs (big assigned_resources arrays) -- per-job queries
# are smaller and reliable.
#
# Usage: tools/g5kstat.sh [-u USER] [-s "site1 site2 ..."] [-j JOBID] [-g LEVEL] [-n N]
#
#   -g quick   job list + state + walltime remaining only, no ssh to any
#              compute node -- fastest, use for "is anything still running".
#   -g nodes   (default) + one aggregate utilization line per node
#              (nvidia-smi GPU%/VRAM, or CPU load average).
#   -g deep    + top N processes per node (ps aux --sort=-%cpu), catches a
#              job that's "Running"/low-aggregate-load because it's actually
#              stuck, not just idle -- most useful when `nodes` looks wrong.
#   -g kwollect  delegates to tools/g5k_monitor.py -- CPU load via the
#              Kwollect API (one HTTP call per node via the site frontend,
#              no oarsh needed) and GPU util/VRAM via dcgm-exporter (one
#              oarsh per GPU node, lighter than parsing nvidia-smi), plus
#              automatic orphan-claim detection against runs/*/claims on
#              each site's home (dev_notes/ideas/g5k_monitoring_agent_spec.md).
#              Not the default yet -- opt in with -g kwollect until its
#              reliability across sites/scenarios is validated further.
#   -j JOBID   restrict to one job (skips the site-wide listing step).
#   -n N       processes to show per node in `deep` mode (default 8).

set -uo pipefail

USER_G5K="jdomguia"
SITES="rennes lyon nancy lille nantes grenoble strasbourg toulouse"
GRANULARITY="nodes"
ONLY_JOB=""
TOP_N=8

while getopts "u:s:g:j:n:" opt; do
  case $opt in
    u) USER_G5K="$OPTARG" ;;
    s) SITES="$OPTARG" ;;
    g) GRANULARITY="$OPTARG" ;;
    j) ONLY_JOB="$OPTARG" ;;
    n) TOP_N="$OPTARG" ;;
  esac
done

case "$GRANULARITY" in
  quick|nodes|deep|kwollect) ;;
  *) echo "unknown -g '$GRANULARITY' (expected quick|nodes|deep|kwollect)" >&2; exit 2 ;;
esac

if [ "$GRANULARITY" = "kwollect" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  args=(-u "$USER_G5K" -s "$SITES")
  [ -n "$ONLY_JOB" ] && args+=(-j "$ONLY_JOB")
  exec python3 "$SCRIPT_DIR/g5k_monitor.py" "${args[@]}"
fi

now_epoch=$(date +%s)
rows=()

fmt_duration() {
  local s=$1
  if [ "$s" -le 0 ]; then echo "expired"; return; fi
  printf "%dh%02dm" $((s / 3600)) $(((s % 3600) / 60))
}

ssh_retry() {
  # ssh_retry <target> <remote-command> -- 3 attempts, 1s apart, empty
  # output on total failure. Covers this project's documented transient
  # SSH-hiccup pattern (a real job/site returning nothing once in a while).
  local target="$1" cmd="$2" out="" attempt
  for attempt in 1 2 3; do
    out=$(ssh -o ConnectTimeout=10 "$target" "$cmd" 2>/dev/null)
    [ -n "$out" ] && break
    sleep 1
  done
  echo "$out"
}

probe_node() {
  # Prints one or more lines describing $host's real utilization, at the
  # configured granularity. Never assumes GPU vs CPU -- tries nvidia-smi
  # first, falls back to CPU signals.
  local site="$1" jid="$2" host="$3"
  local node_fqdn="${host}.${site}.grid5000.fr"
  local util load procs

  util=$(ssh_retry "${site}.grid5000.fr.g5k" \
    "OAR_JOB_ID=$jid oarsh $node_fqdn 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null && echo __GPU_OK__'")

  if echo "$util" | grep -q '__GPU_OK__'; then
    echo "$util" | grep -v '__GPU_OK__' | awk -F', ' '{printf "gpu%s util=%s%% vram=%s/%sMiB  ", $1, $2, $3, $4}'
    echo
    if [ "$GRANULARITY" = "deep" ]; then
      procs=$(ssh_retry "${site}.grid5000.fr.g5k" \
        "OAR_JOB_ID=$jid oarsh $node_fqdn 'nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null'")
      [ -n "$procs" ] && echo "$procs" | sed 's/^/    gpu-proc: /'
    fi
  else
    load=$(ssh_retry "${site}.grid5000.fr.g5k" "OAR_JOB_ID=$jid oarsh $node_fqdn 'uptime'" | grep -oE 'load average:.*')
    echo "${load:-unreachable}"
    if [ "$GRANULARITY" = "deep" ]; then
      procs=$(ssh_retry "${site}.grid5000.fr.g5k" \
        "OAR_JOB_ID=$jid oarsh $node_fqdn 'ps aux --sort=-%cpu | head -n $((TOP_N + 1))'")
      [ -n "$procs" ] && echo "$procs" | tail -n +2 | awk '{printf "    %-8s %-5s %-5s %s\n", $2, $3"%cpu", $4"%mem", $11}'
    fi
  fi
}

for site in $SITES; do
  if [ -n "$ONLY_JOB" ]; then
    job_ids="$ONLY_JOB"
  else
    list=$(ssh_retry "${site}.grid5000.fr.g5k" "oarstat -u $USER_G5K")
    [ -z "$list" ] && continue
    job_ids=$(echo "$list" | awk 'NR>2 {print $1}')
    [ -z "$job_ids" ] && continue
  fi

  for jid in $job_ids; do
    detail=$(ssh_retry "${site}.grid5000.fr.g5k" "oarstat -j $jid -f")
    if [ -z "$detail" ]; then
      [ -n "$ONLY_JOB" ] && continue  # tried this site, job isn't here
      rows+=("$jid|?|$site|-|unreachable|-|-")
      continue
    fi

    name=$(echo "$detail" | grep -m1 '^\s*name = ' | sed 's/.*= //')
    state=$(echo "$detail" | grep -m1 '^\s*state = ' | sed 's/.*= //')
    walltime_raw=$(echo "$detail" | grep -m1 '^\s*walltime = ' | sed 's/.*= //')
    start_str=$(echo "$detail" | grep -m1 '^\s*start_time = ' | sed 's/.*= //')
    hosts=$(echo "$detail" | grep -m1 '^\s*assigned_hostnames = ' | sed 's/.*= //' \
      | tr '+' '\n' | sed 's/\..*//' | sort -u | tr '\n' ',' | sed 's/,$//')

    # start_time in plain `oarstat -f` text output is a human-readable
    # timestamp ("2026-09-19 14:54:30"), not a unix epoch -- parse it.
    # (Naive, no explicit TZ in the string -- `date -d` assumes local TZ on
    # both this computation and $now_epoch, so a systematic local/remote
    # offset cancels out; only matters if this machine's clock is just wrong.)
    start_epoch=""
    [ -n "$start_str" ] && start_epoch=$(date -d "$start_str" +%s 2>/dev/null)

    if [ "$state" != "Running" ] || [ -z "$start_epoch" ]; then
      rows+=("$jid|$name|$site|-|$state|-|-")
      continue
    fi

    IFS=':' read -r wh wm ws <<< "$walltime_raw"
    walltime_s=$(( (10#${wh:-0}) * 3600 + (10#${wm:-0}) * 60 + (10#${ws:-0}) ))
    remaining_h=$(fmt_duration $(( start_epoch + walltime_s - now_epoch )))

    if [ "$GRANULARITY" = "quick" ] || [ -z "$hosts" ]; then
      rows+=("$jid|$name|$site|${hosts:--}|$state|$remaining_h|-")
      continue
    fi

    IFS=',' read -ra host_arr <<< "$hosts"
    for host in "${host_arr[@]}"; do
      summary=$(probe_node "$site" "$jid" "$host")
      rows+=("$jid|$name|$site|$host|$state|$remaining_h|$summary")
    done
  done
done

if [ ${#rows[@]} -eq 0 ]; then
  echo "No active jobs for $USER_G5K on: $SITES"
  exit 0
fi

printf "%-10s %-22s %-10s %-16s %-10s %-9s %s\n" "JOB" "NAME" "SITE" "NODE" "STATE" "LEFT" "UTILIZATION"
for row in "${rows[@]}"; do
  # `util` (the last field) can contain embedded newlines in `deep` mode --
  # `read <<< "$row"` would silently stop at the first one and drop the
  # rest, so split with parameter expansion (whole-string, newline-safe)
  # instead of a line-oriented `read`.
  rest="$row"
  jid="${rest%%|*}";  rest="${rest#*|}"
  name="${rest%%|*}"; rest="${rest#*|}"
  site="${rest%%|*}"; rest="${rest#*|}"
  host="${rest%%|*}"; rest="${rest#*|}"
  state="${rest%%|*}"; rest="${rest#*|}"
  left="${rest%%|*}"; util="${rest#*|}"

  first_line="${util%%$'\n'*}"
  printf "%-10s %-22.22s %-10s %-16s %-10s %-9s %s\n" "$jid" "$name" "$site" "$host" "$state" "$left" "$first_line"
  if [ "$util" != "$first_line" ]; then
    echo "${util#*$'\n'}" | grep -v '^\s*$' || true
  fi
done
