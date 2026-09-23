#!/bin/bash
# E8 (WRITING_PLAN §4, P1, filler): ablation of the training n_step distribution
# upper bound -- U(1,4) vs U(1,8) vs U(1,16) -- same retrieval setup as E1's
# random condition, single seed (filler priority, not seed-robustness testing).
# Usage: bash phase24_e8_nstep_distribution.sh <n_step_max: 4|8|16>
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

NMAX="$1"
if [ -z "$NMAX" ]; then
  echo "usage: $0 <n_step_max>"; exit 1
fi
TAG="e8_nmax${NMAX}"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --n_step_train_max "$NMAX" --use_ff --bf16 \
  --batch_size "${BATCH_SIZE:-16}" \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --save_best_checkpoint_path checkpoints/${TAG}_best.pt \
  > logs/${TAG}_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/${TAG}_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size "${BATCH_SIZE:-16}" --device cuda --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --out logs/${TAG}_fullval.json \
  > logs/${TAG}_fullval.log 2>&1

echo "${TAG}_DONE"
