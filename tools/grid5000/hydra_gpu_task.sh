#!/usr/bin/env bash
# Reusable template for running a GPU task on Grid'5000's hydra cluster (Lyon,
# NVIDIA GH200 Grace Hopper). Meant to be launched AS the `oarsub` job command
# itself (a PASSIVE job) from the Lyon frontend -- not run interactively.
#
# Why this exists: hydra needs a non-default kadeploy'd environment to get a
# working GPU, and its home is a separate, unshared NFS from every other
# Grid'5000 site -- any input data/checkpoint/code has to be streamed in from
# wherever it actually lives (this project keeps everything on Rennes) before
# a job can do useful work, and results streamed back out before the
# reservation ends. All of that plumbing is captured here once so it doesn't
# get re-derived (and re-debugged) from scratch every time.
#
# Usage: copy this file, fill in the CONFIG block below for the task at hand,
# upload it to the Lyon frontend, then:
#   oarsub -t exotic -t deploy -p "cluster='hydra'" -l walltime=<T> \
#     -n "<job-name>" "bash <your_copy>.sh"
#
# Background/lessons this encodes (see .claude/skills/grid5000/SKILL.md for
# the full story):
# - kadeploy3 -e ubuntugh2404-big -k takes ~9-10 min and gives root@<node> SSH
#   access; ubuntugh2404-big's Python needs `apt-get install python3.X-venv`
#   before `python3 -m venv` works, and `apt-get update` on that image
#   routinely errors on an unrelated broken third-party repo (tolerate it).
# - Plain `pip install torch` can still pull a wheel whose bundled CUDA
#   runtime is newer than the node's driver -- pin
#   `--index-url https://download.pytorch.org/whl/cu128` to match this image's
#   driver (570.124.06 / CUDA 12.8) rather than trusting the default.
# - SSH agent forwarding does NOT reach past the bastion, so root@<node>
#   cannot authenticate outward to other Grid'5000 hosts on its own -- any
#   cross-site transfer has to be a `ssh siteA "tar -cf - ..." | ssh
#   root@node "tar -xf - ..."` PIPE run from the frontend (which has real
#   credentials both ways), never staged on the frontend's own disk (its
#   /tmp is small, ~15GB seen on Lyon) and never attempted from inside a
#   root@node shell.
# - This script runs as a `sleep`-free PASSIVE oarsub command: it releases
#   the node itself as soon as it exits, success or failure, rather than
#   holding walltime hostage on an idle reservation.
#
# Lessons added 2026-09-21 (retrieval1_ecotaxe_a gap catch-up + openr1_math
# val test -- see dev_notes/grid5000_usage.log.md for the full story):
# - ALWAYS stream a large (~10GB+) Teacher checkpoint to the node's local
#   /tmp (via SOURCE_DIRS below) rather than pointing --model_dir at Group
#   Storage directly. Loading shard-by-shard over inter-site NFS took HOURS
#   (many small reads, high per-file latency); the same checkpoint loaded in
#   under a minute once copied local first. The one-time bulk copy is the
#   expensive part (~15 min for 29GB, fluctuating 17-227 MB/s depending on
#   storage3 contention) but is still far cheaper than direct NFS reads.
# - A reused venv from a PREVIOUS session on this cluster is not safe to
#   trust as-is: if the node was redeployed since, its system Python version
#   can have changed (seen: venv built for python3.13, redeployed image now
#   ships python3.12) and the venv's own python3 binary silently resolves to
#   the wrong interpreter -- `ModuleNotFoundError: No module named 'torch'`
#   even though the package is genuinely installed. Rebuilding the venv
#   fresh (this script always does) sidesteps this entirely.
# - `--attn_implementation flash_attention_2` works on this node WITHOUT a
#   manual flash-attn build: the `kernels` package (installed above) fetches
#   a precompiled `kernels-community/flash-attn2` kernel from the HF Hub on
#   first use. That first fetch (~1447 small files) can get rate-limited
#   (`HTTP 429`) without an `HF_TOKEN` -- the client retries with backoff
#   automatically, just let it; do NOT copy an HF token to another node/site
#   to work around this (blocked by this environment's own security policy
#   as credential exfiltration, and unnecessary -- retry always got through
#   within a few minutes in practice, or fall back to `--attn_implementation
#   sdpa` if flash-attn2 isn't worth the wait for a small job).
# - For a job with N examples smaller than the script's default
#   `--shard_size` (1000), NOTHING is checkpointed to disk until the very
#   end -- killing the process mid-run loses all progress. Pass an explicit
#   smaller `--shard_size` (e.g. 50) for anything you might need to
#   interrupt and resume (relaunching with the same --out_file auto-resumes
#   from the last complete shard).

set -uo pipefail

# ============================== CONFIG ======================================
# Fill these in per task. Nothing outside this block should need editing.

KADEPLOY_ENV="ubuntugh2404-big"                 # `kaenv3 -l | grep -i hydra` to confirm the current name

# Where the input data/checkpoint/code actually lives today (this project:
# Rennes). Reached directly (no .g5k suffix) since this script itself runs
# on a Grid'5000 frontend, already inside the internal network.
SOURCE_HOST="rennes.grid5000.fr"

# Each entry: "<source path on SOURCE_HOST>|<dest path under /tmp/task/ on the node>"
# A directory is streamed with `tar`; for a single file use SRC_IS_FILE=1 entries
# (see the transfer loop below) -- keep it simple and just list dirs here for
# the common case, add file-specific lines directly in the transfer step if needed.
SOURCE_DIRS=(
  # "/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/SomeCheckpoint|checkpoint"
  # "~/thinker|repo"
)

# The command to run once the venv + transfers are ready. Runs with CWD
# /tmp/task/repo and PYTHONPATH=. already set; use paths under /tmp/task/.
REMOTE_COMMAND='python3 learn/distill/precompute_teacher_targets.py --help'

# Extra Python packages beyond torch/torchvision (space-separated, one pip call).
# For a Qwen3.5-family FP8 Teacher (this project's KD Teacher) you need all of:
# kernels (finegrained-fp8 compute kernel + auto-fetched flash-attn2 from the
# Hub), flash-linear-attention (fast chunk_gated_delta_rule -- the model's
# Mamba-like SSM layers), causal-conv1d (fast causal_conv1d -- also SSM, needs
# python3.X-dev for Triton's JIT to compile at all, see Step 3). Without these
# three, the model still runs but 15-30x slower (reference PyTorch fallback).
EXTRA_PIP_PACKAGES="transformers accelerate safetensors numpy pillow \"kernels>=0.16.0,<0.17.0\" flash-linear-attention causal-conv1d"

# Where results should land back on Rennes Group Storage once REMOTE_COMMAND
# finishes (success or failure -- whatever landed in /tmp/task/out/ is synced).
RESULT_DEST="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/CHANGE_ME"

# =============================================================================

LOG=~/hydra_gpu_task_run.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date +%T)] $*"; }

log "=== hydra_gpu_task starting, OAR_JOB_ID=${OAR_JOB_ID:-unknown} ==="
NODE=$(head -n1 "$OAR_NODEFILE")
log "Target node: $NODE"

log "=== Step 1: kadeploy3 $KADEPLOY_ENV ==="
if ! kadeploy3 -e "$KADEPLOY_ENV" -f "$OAR_NODEFILE" -k; then
    log "MARKER: KADEPLOY_FAIL"; exit 1
fi
log "MARKER: KADEPLOY_OK"
sleep 15
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes root@${NODE}"
ok=0
for i in $(seq 1 20); do
    if $SSH 'echo ssh_ready' >/dev/null 2>&1; then ok=1; break; fi
    sleep 15
done
[ "$ok" -eq 1 ] || { log "MARKER: SSH_FAIL"; exit 1; }
log "MARKER: SSH_OK"

log "=== Step 2: stream inputs from $SOURCE_HOST straight onto the node's /tmp (no frontend disk staging) ==="
$SSH 'mkdir -p /tmp/task/out'
for entry in "${SOURCE_DIRS[@]:-}"; do
    [ -z "$entry" ] && continue
    src="${entry%%|*}"
    dst="${entry#*|}"
    log "streaming $src -> /tmp/task/$dst ..."
    $SSH "mkdir -p /tmp/task/$dst"
    ssh "$SOURCE_HOST" "tar -cf - --exclude='.cache/huggingface/download' -C \"$src\" ." \
      | $SSH "tar -xf - -C /tmp/task/$dst"
done
$SSH "du -sh /tmp/task/* 2>/dev/null || true"
log "MARKER: DATA_TRANSFER_OK"

log "=== Step 3: venv + torch/torchvision (cu128, pinned pair) + extra packages ==="
# Pinned to what's validated against THIS image's driver (570.124.06 / CUDA
# 12.8) as of 2026-09-21 -- torch 2.11.0+cu128, torchvision 0.26.0+cu128. A
# bare `pip install torch` (no version pin) can silently resolve to a newer
# release (seen: 2.14.0+cu13) whose bundled CUDA runtime exceeds the driver's
# cap -- `torch.cuda.is_available()` then comes back False with NO error, only
# a "driver too old" warning easy to miss in a wall of pip output. Re-verify
# these pins still match `nvidia-smi`'s CUDA Version column before reusing
# this on a future/redeployed hydra image; if the driver has moved on, bump
# both pins together (matching torch/torchvision release pair), never one
# alone. torchvision is needed only because `AutoProcessor.from_pretrained`
# probes for an image processor even for a text-only Teacher.
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
$SSH bash -s <<REMOTE
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
PYVER=\$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# python3.X-dev (Python.h) is required for flash-linear-attention/causal-conv1d's
# Triton JIT to compile CUDA extensions -- without it, Triton silently rolls
# back to a CPU fallback ("Triton is not supported on current platform") that
# is SLOWER than not having the optimized kernel at all. No error, just a
# warning easy to miss -- always install both venv and dev packages together.
apt-get install -y -qq "python\${PYVER}-venv" "python\${PYVER}-dev" 2>/dev/null || true
if python3 -m venv /root/tv 2>/dev/null; then
    source /root/tv/bin/activate
else
    pip() { command pip "\$@" --break-system-packages; }
fi
pip install --upgrade pip >/dev/null 2>&1 || true
# torch + torchvision MUST be installed together in one call, both pinned,
# both from the cu128 index -- installing torchvision separately afterwards
# (even with the same --index-url) can still pull a newer torch as its
# dependency and silently downgrade-breaks CUDA support (same failure mode as
# the unpinned case above, just triggered by a second package's resolver).
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
pip install $EXTRA_PIP_PACKAGES
python3 -c "import torch, torchvision; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
echo "MARKER: ENV_READY"
REMOTE
log "MARKER: ENV_STEP_DONE"

log "=== Step 4: run the task command ==="
$SSH bash -s <<REMOTE
set -euo pipefail
cd /tmp/task/repo 2>/dev/null || cd /tmp/task
if [ -f /root/tv/bin/activate ]; then source /root/tv/bin/activate; fi
export PYTHONPATH=.
$REMOTE_COMMAND
echo "MARKER: TASK_COMMAND_DONE"
REMOTE
task_rc=$?

log "=== Step 5: copy whatever landed in /tmp/task/out back to Rennes (success or partial) ==="
ssh "$SOURCE_HOST" "mkdir -p '$RESULT_DEST'"
$SSH "tar -cf - -C /tmp/task/out ." | ssh "$SOURCE_HOST" "tar -xf - -C '$RESULT_DEST'"
log "MARKER: RESULT_COPYBACK_OK"

if [ "$task_rc" -eq 0 ]; then
    log "MARKER: HYDRA_GPU_TASK_ALL_DONE"
else
    log "MARKER: TASK_COMMAND_FAILED (rc=$task_rc)"
fi
log "=== hydra_gpu_task finished ==="
