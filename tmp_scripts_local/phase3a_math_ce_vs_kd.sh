#!/bin/bash
cd ~/thinker
export PYTHONPATH=.

# KD arm -- batch_size=8 (not 32): thinking sequences up to 1024 tokens OOM'd a 24GB
# GPU at batch_size=32 (see logs/math_small_kd_train.log, 2026-09-22), 8 matches the
# historical precedent for this same reasoning/math config (dev_notes/experiments/distillation.md).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data data/distill/openr1_math/train.jsonl \
  --val_data data/distill/openr1_math/val.jsonl \
  --teacher_targets data/distill/openr1_math/train_topk32_lfm2-1.2b.npz \
  --val_teacher_targets data/distill/openr1_math/val_topk32_lfm2-1.2b.npz \
  --kd_alpha 0.5 --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile \
  --batch_size 8 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --save_best_checkpoint_path checkpoints/math_small_kd_best.pt \
  > logs/math_small_kd_train.log 2>&1

# CE-only arm (same recipe, no teacher_targets/kd_alpha)
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data data/distill/openr1_math/train.jsonl \
  --val_data data/distill/openr1_math/val.jsonl \
  --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile \
  --batch_size 8 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --save_best_checkpoint_path checkpoints/math_small_ceonly_best.pt \
  > logs/math_small_ceonly_train.log 2>&1

echo PHASE3A_MATH_CE_VS_KD_DONE
