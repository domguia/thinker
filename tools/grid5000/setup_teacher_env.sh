#!/usr/bin/env bash
# Idempotent setup for the teacher311 micromamba env's optimized SSM kernels
# (flash-linear-attention + causal-conv1d), needed for any Qwen3.5-family
# Teacher forward pass (chunk_gated_delta_rule / causal_conv1d layers).
#
# Without these, transformers silently falls back to a pure-PyTorch reference
# implementation -- no error, just a "falling back to its reference PyTorch
# implementation... much slower" warning and ~15-30x slower throughput
# (observed: ~0.06-0.07 ex/s reference vs ~1.0-2.5 ex/s optimized). This exact
# install was done manually and undocumented-as-a-script at least twice
# (2026-09-21 on a Lyon/hydra node, 2026-09-22 on ecotaxe-1/Nantes) before
# this script existed -- run this instead of re-deriving the pip install by
# hand every time. See .claude/skills/grid5000/SKILL.md for the full story.
#
# Usage (run ON the target compute node, e.g. via oarsh):
#   bash tools/grid5000/setup_teacher_env.sh [micromamba_env_name]
# Default env name: teacher311
#
# Safe to run alongside an already-running GPU job on the same node --
# causal-conv1d's CUDA extension build is CPU/host-only compilation, no GPU
# memory used, and flash-linear-attention is a pure wheel (no compile at all).

set -euo pipefail

ENV_NAME="${1:-teacher311}"
MICROMAMBA="${HOME}/bin/micromamba"

if [ ! -x "$MICROMAMBA" ]; then
  echo "ERROR: $MICROMAMBA not found. See grid5000 skill: paths vary by site" \
       "(~/bin/micromamba is the uniform symlink, not ~/micromamba/micromamba)." >&2
  exit 1
fi

run() { "$MICROMAMBA" run -p "${HOME}/micromamba/envs/${ENV_NAME}" "$@"; }

echo "[setup_teacher_env] env=${ENV_NAME} on $(hostname)"

# --- flash-linear-attention: pure wheel, no compilation ---
if run python -c "import fla" 2>/dev/null; then
  echo "[setup_teacher_env] flash-linear-attention already installed, skipping"
else
  echo "[setup_teacher_env] installing flash-linear-attention..."
  run pip install flash-linear-attention
fi

# --- causal-conv1d: needs python3.X-dev for Triton to JIT-compile at all ---
# (without it, Triton fails with "fatal error: Python.h: No such file or
# directory" and silently rolls back to CPU -- no hard error, easy to miss)
PYVER=$(run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! run python -c "import sysconfig, os; assert os.path.exists(os.path.join(sysconfig.get_path('include'), 'Python.h'))" 2>/dev/null; then
  echo "[setup_teacher_env] Python.h missing for py${PYVER} -- installing python${PYVER}-dev"
  echo "[setup_teacher_env] (needs sudo; skip manually if this node has no apt/sudo access)"
  sudo -A apt-get install -y "python${PYVER}-dev" || \
    echo "[setup_teacher_env] WARNING: could not install python${PYVER}-dev -- causal_conv1d/Triton JIT may silently fall back to CPU"
fi

if run python -c "import causal_conv1d" 2>/dev/null; then
  echo "[setup_teacher_env] causal-conv1d already installed, skipping"
else
  echo "[setup_teacher_env] installing causal-conv1d (source build, ~10-15min)..."
  run pip install causal-conv1d --no-build-isolation
fi

echo "[setup_teacher_env] done. Verify with:"
echo "  ${MICROMAMBA} run -p ${HOME}/micromamba/envs/${ENV_NAME} python -c 'import fla, causal_conv1d; print(\"OK\")'"
