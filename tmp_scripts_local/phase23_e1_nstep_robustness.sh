#!/bin/bash
# E1 (WRITING_PLAN §4, paper adversarial-review priority, 2026-09-23): repeat the
# core n_step extrapolation result (phase13/14: fixed n_step=4 training gives a
# U-shaped extrapolation blowup at test time, n_step~U(1,8) training gives a
# near-flat curve, CE gap ~0.03 vs ~4.0) across 3 seeds per condition, to verify
# the gap is robust to seed variance rather than a single-run artifact. Same
# dataset/setup as phase13/14 (hotpotqa_full/train_repr10k_ab.jsonl, d_model=256,
# n_head=4, batch_size=64), eval at the wider n_step_test set requested:
# 1,2,4,8,12,16,24,32.
#
# Usage: bash phase23_e1_nstep_robustness.sh <mode: fixed|random> <seed>
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

MODE="$1"
SEED="$2"
if [ -z "$MODE" ] || [ -z "$SEED" ]; then
  echo "usage: $0 <fixed|random> <seed>"; exit 1
fi

if [ "$MODE" = "fixed" ]; then
  NSTEP_FLAG=""
  TAG="fixed_seed${SEED}"
else
  NSTEP_FLAG="--n_step_train_max 8"
  TAG="random_seed${SEED}"
fi

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 $NSTEP_FLAG --use_ff --bf16 --batch_size "${BATCH_SIZE:-16}" \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed $SEED --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --save_best_checkpoint_path checkpoints/e1_${TAG}_best.pt \
  > logs/e1_${TAG}_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/e1_${TAG}_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size "${BATCH_SIZE:-16}" --device cuda --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --out logs/e1_${TAG}_fullval.json \
  > logs/e1_${TAG}_fullval.log 2>&1

echo "E1_${TAG}_DONE"
