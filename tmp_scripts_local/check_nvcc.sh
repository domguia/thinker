#!/usr/bin/env bash
export MAMBA_ROOT_PREFIX=~/micromamba
~/micromamba/micromamba run -n teacher311 which nvcc
~/micromamba/micromamba run -n teacher311 nvcc --version
