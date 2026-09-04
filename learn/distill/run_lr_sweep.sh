#!/usr/bin/env bash
# LR/kd_alpha/weight_decay sweep at the 40M-core tier, on the 200-example
# validation slice -- cheap enough to run every combination in minutes.
# Phase 1: LR grid (kd_alpha=0.5, weight_decay=0.01 fixed) to find the best LR.
# Phase 2 (run manually after inspecting phase 1): re-run with --best_lr set,
# sweep kd_alpha and weight_decay one axis at a time around it.
set -euo pipefail

export MAMBA_ROOT_PREFIX=~/micromamba
cd ~/thinker

TRAIN=data/distill_cluster_run/reasoning/train_sample200.jsonl
TARGETS=data/distill_cluster_run/reasoning/train_sample200_topk32.npz
TOKENIZER=/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/Qwen3.8-27B-FP8
GROUP="lr-sweep-40Mcore-$(date +%Y%m%d-%H%M)"

run() {
  local name=$1; shift
  echo "=== $name ==="
  ~/micromamba/micromamba run -n legacygpu python learn/distill/train_sft.py \
    --train_file "$TRAIN" --tokenizer "$TOKENIZER" \
    --base_config gpt2 --n_layer 4 --n_embd 160 --n_head 4 \
    --block_size 512 --batch_size 2 --max_steps 200 --max_time_minutes 5 \
    --teacher_targets "$TARGETS" \
    --mup --mup_base_width 40 \
    --wandb --wandb_group "$GROUP" --run_name "$name" \
    --mlflow --mlflow_experiment "lr-sweep-40Mcore" \
    "$@" 2>&1 | tail -5
}

echo ">>> Phase 1: LR grid (kd_alpha=0.5, weight_decay=0.01)"
for lr in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1; do
  run "lr${lr}" --lr "$lr" --kd_alpha 0.5 --weight_decay 0.01
done

echo ">>> Phase 1 done. Inspect best_loss per run above (or on W&B/MLflow), then run phase 2 manually:"
echo "    BEST_LR=<value> bash learn/distill/run_lr_sweep_phase2.sh"
