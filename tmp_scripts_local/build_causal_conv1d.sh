#!/usr/bin/env bash
set -e
export MAMBA_ROOT_PREFIX=~/micromamba
ENV_PREFIX=~/micromamba/envs/teacher311
cat > /tmp/_inner_build.sh <<'INNER'
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/teacher311/targets/x86_64-linux
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version
pip install causal-conv1d --no-build-isolation --no-cache-dir
INNER
~/micromamba/micromamba run -n teacher311 bash /tmp/_inner_build.sh
