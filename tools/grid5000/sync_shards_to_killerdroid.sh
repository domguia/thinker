#!/usr/bin/env bash
# Progressive shard rapatriement to killerdroid (Group Storage) while a
# precompute_teacher_targets.py job is still running -- keeps a site's home
# quota under control on long jobs instead of accumulating everything
# locally and hitting the hard quota mid-run (see dev_notes/grid5000_usage.
# log.md, 2026-09-21: 85.5GB/97.6GB hard limit hit on Lyon home while the
# openr1_math_full val job was writing shards).
#
# Discovered 2026-09-21: killerdroid's Group Storage mount
# (/srv/storage/killerdroid@storage3.rennes.grid5000.fr/...) is directly
# reachable from ANY site's frontend and compute nodes, not just Rennes --
# no need to hop through the Rennes frontend to write to it. A plain local
# `rsync`/`mv` between the two paths on the same host is enough and stays
# entirely inside the Grid'5000 network.
#
# Only touches FINALIZED shard files (shard_<start>_<end>.npz) -- never
# *.tmp.npz, which precompute_teacher_targets.py is still writing/renaming
# and would corrupt a copy taken mid-write. The final `merge_shards()` /
# `merge_partial.py` step reads the *local* shard dirs, so do NOT delete
# local shards while a merge for that same output is in flight (check the
# job's process is still doing per-example work, not already in its merge
# phase, before running with --delete-after-sync).
#
# Usage (run FROM a Grid'5000 frontend, e.g. via ssh from your own machine,
# never as a raw shell pipe on your local machine):
#   tools/grid5000/sync_shards_to_killerdroid.sh \
#     --shard-dirs "DIR1 DIR2" \
#     --dest /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/<subdir> \
#     [--interval 300] [--once] [--delete-after-sync]
#
# Typical background loop launch (from the site frontend, targeting a node
# whose home is NFS-shared with that frontend -- most sites; hydra/Lyon
# included, confirmed 2026-09-21):
#   nohup tools/grid5000/sync_shards_to_killerdroid.sh \
#     --shard-dirs "$HOME/thinker/data/distill/X.topk_shards $HOME/thinker/data/distill/X.hidden_shards" \
#     --dest /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/staging/X \
#     --interval 300 --delete-after-sync \
#     > ~/sync_shards.log 2>&1 &
#
# The merge (`merge_partial.py`, CPU-only) should still happen from Rennes
# once the site's job is done and all shards are rapatriated -- that part
# is unchanged, this script only handles keeping quota under control DURING
# a long run, it does not merge anything itself.

set -euo pipefail

# When launched via `sudo -u otheruser` from a root shell whose cwd is
# /root, GNU find's final fchdir-back-to-start fails with "Failed to
# restore initial working directory: /root: Permission denied" (the
# target user can't re-enter /root) -- this can abort the whole script
# under `set -e`. Move to a directory the target user can always access
# before doing anything else, regardless of how this script was invoked.
cd "${HOME:-/tmp}" 2>/dev/null || cd /tmp

INTERVAL=300
ONCE=0
DELETE_AFTER_SYNC=0
SHARD_DIRS=""
DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shard-dirs) SHARD_DIRS="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    --delete-after-sync) DELETE_AFTER_SYNC=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$SHARD_DIRS" || -z "$DEST" ]]; then
  echo "Usage: $0 --shard-dirs \"DIR1 DIR2\" --dest DEST_DIR [--interval SEC] [--once] [--delete-after-sync]" >&2
  exit 1
fi

mkdir -p "$DEST"

sync_once() {
  local total_synced=0
  for dir in $SHARD_DIRS; do
    [[ -d "$dir" ]] || continue
    local base
    base=$(basename "$dir")
    local out="$DEST/$base"
    mkdir -p "$out"

    # Only finalized shards (shard_<7digits>_<7digits>.npz), never *.tmp.npz.
    local files
    files=$(find "$dir" -maxdepth 1 -name 'shard_*.npz' ! -name '*.tmp.npz')
    [[ -z "$files" ]] && continue

    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      local fname
      fname=$(basename "$f")
      local src_size dst_size
      src_size=$(stat -c%s "$f")

      # Copy, verify size, only then optionally delete the source.
      cp "$f" "$out/$fname.copying"
      mv "$out/$fname.copying" "$out/$fname"
      dst_size=$(stat -c%s "$out/$fname")

      if [[ "$src_size" -eq "$dst_size" ]]; then
        total_synced=$((total_synced + 1))
        if [[ "$DELETE_AFTER_SYNC" -eq 1 ]]; then
          rm -f "$f"
        fi
      else
        echo "[sync_shards] SIZE MISMATCH on $fname ($src_size vs $dst_size), NOT deleting source" >&2
      fi
    done <<< "$files"
  done
  echo "[sync_shards] $(date -Iseconds) synced=$total_synced dest=$DEST"
}

if [[ "$ONCE" -eq 1 ]]; then
  sync_once
else
  while true; do
    sync_once
    sleep "$INTERVAL"
  done
fi
