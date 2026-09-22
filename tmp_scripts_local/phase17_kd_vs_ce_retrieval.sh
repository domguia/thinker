#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# supervisor-agent request (2026-09-22, via agent2): clean isolated KD-vs-CE-only comparison
# on retrieval, using the already-available top-K n=1500 batch (hotpotqa, post-<think>-fix).
# Unlike phase15 (KD + n_step-variable combined), this is FIXED n_step (no --n_step_train_max),
# same WSD+patience recipe, to answer directly: does KD (clean data) beat CE-only?
# Data paths and TEACHER_TARGETS directory-form (manifest-based, n=1500 subset) taken from
# phase15_nsteprand_cleankd.sh -- see that script's comments for the subset/IndexError gotcha.
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/val/thinkfix_n2000.qwen_big.npz"

COMMON_ARGS="--dataset_type retrieval --data $TRAIN_DATA --val_data $VAL_DATA \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --batch_size 16 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30"

# --- KD run ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  $COMMON_ARGS \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 \
  --save_best_checkpoint_path checkpoints/retrieval1_kdvsce_kd_best.pt \
  > logs/retrieval1_kdvsce_kd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_kdvsce_kd_best.pt \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_kdvsce_kd_fullval.json \
  > logs/eval_thinker_kdvsce_kd_fullval.log 2>&1

echo PHASE17_KD_RUN_DONE

# --- CE-only run (explicit deviation from KD default, requested by supervisor-agent for
# this comparison, see dev_notes CLAUDE.md training-methodology rule) ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  $COMMON_ARGS \
  --save_best_checkpoint_path checkpoints/retrieval1_kdvsce_ceonly_best.pt \
  > logs/retrieval1_kdvsce_ceonly_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_kdvsce_ceonly_best.pt \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_kdvsce_ceonly_fullval.json \
  > logs/eval_thinker_kdvsce_ceonly_fullval.log 2>&1

echo PHASE17_CEONLY_RUN_DONE
