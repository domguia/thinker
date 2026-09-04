#!/usr/bin/env bash
# Phase 2: hold LR at $BEST_LR (found from phase 1), sweep kd_alpha and
# weight_decay one axis at a time.
set -euo pipefail
: "${BEST_LR:?set BEST_LR to the winning phase-1 value first}"

export MAMBA_ROOT_PREFIX=~/micromamba
cd ~/thinker

TRAIN=data/distill_cluster_run/reasoning/train_sample200.jsonl
TARGETS=data/distill_cluster_run/reasoning/train_sample200_topk32.npz
TOKENIZER=/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/Qwen3.8-27B-FP8
GROUP="hp-sweep-40Mcore-$(date +%Y%m%d-%H%M)"

run() {
  local name=$1; shift
  echo "=== $name ==="
  ~/micromamba/micromamba run -n legacygpu python learn/distill/train_sft.py \
    --train_file "$TRAIN" --tokenizer "$TOKENIZER" \
    --base_config gpt2 --n_layer 4 --n_embd 160 --n_head 4 \
    --block_size 512 --batch_size 2 --max_steps 200 --max_time_minutes 5 \
    --teacher_targets "$TARGETS" \
    --mup --mup_base_width 40 --lr "$BEST_LR" \
    --wandb --wandb_group "$GROUP" --run_name "$name" \
    --mlflow --mlflow_experiment "hp-sweep-40Mcore" \
    "$@" 2>&1 | tail -5
}

echo ">>> kd_alpha sweep (weight_decay=0.01 fixed)"
for a in 0.25 0.5 0.75; do
  run "alpha${a}" --kd_alpha "$a" --weight_decay 0.01
done

echo ">>> weight_decay sweep (kd_alpha=0.5 fixed)"
for wd in 0.0 0.01 0.1; do
  run "wd${wd}" --kd_alpha 0.5 --weight_decay "$wd"
done
