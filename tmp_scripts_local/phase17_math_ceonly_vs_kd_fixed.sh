#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Isolated ablation (supervisor-agent, 2026-09-22): clean CE-only vs KD comparison on
# openr1_math, FIXED n_step=4 (no --n_step_train_max) -- phase16 combined n_step-variable+KD
# in one run, confounding the two levers. This isolates the KD effect alone, matching
# phase3a's original CE-vs-KD methodology (dev_notes/experiments/prompt_response_pipeline.md,
# line ~539) but on qwen35/qwen_big (clean, post-<think>-fix data) instead of the old LFM2
# recipe, and on the larger 34279-example pool instead of 18000.
# Same OOM constraints as phase16 (qwen35 vocab 248320 x max_thinking_len 1024 makes
# topk_kd_loss's dense logsumexp expensive) -- batch_size=2 for train, batch_size=1 for eval
# (confirmed necessary even without gradients, see phase16's manual eval run 2026-09-22).
# KD arm uses whatever top-K coverage is available (topk_n1714, ~5% as of writing --
# data-agent asked to extend this in parallel, on a SEPARATE GPU from this run).
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/val/lastLayer_n3890.qwen_big.npz"

# --- Arm A: CE-only (no teacher_targets/kd_alpha) ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 1 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/math_ceonly_fixed_best.pt \
  > logs/math_ceonly_fixed_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/math_ceonly_fixed_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 1024 --max_answer_len 64 \
  --batch_size 1 --device cuda \
  --out logs/eval_thinker_math_ceonly_fixed_fullval.json \
  > logs/eval_thinker_math_ceonly_fixed_fullval.log 2>&1

# --- Arm B: KD top-K (kd_alpha=0.5), same fixed n_step, same recipe otherwise ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 1 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/math_kd_fixed_best.pt \
  > logs/math_kd_fixed_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/math_kd_fixed_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 1024 --max_answer_len 64 \
  --batch_size 1 --device cuda \
  --out logs/eval_thinker_math_kd_fixed_fullval.json \
  > logs/eval_thinker_math_kd_fixed_fullval.log 2>&1

echo PHASE17_MATH_CEONLY_VS_KD_FIXED_DONE
