#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Combined test (analyst-agent/user, 2026-09-22): n_step-variable (Uniform(1,8), confirmed to
# give near-flat extrapolation on CE-only -- see dev_notes/experiments/prompt_response_pipeline.md
# commit d3cc2dc) + clean (post-template-fix) top-K KD, on whatever first batch data-gen-agent
# makes available. Answers two questions in one run: (1) does clean KD actually beat CE-only now
# that the <think> contamination is fixed, (2) does the n_step-variable robustness gain survive
# under KD too. Same WSD+patience recipe as prior comparable runs.
#
# Updated 2026-09-22 from data-gen-agent: template fix confirmed (100%->0% <think> collapse on
# 200 examples), NaN bug root-caused (device_map="auto" multi-GPU sharding, fixed via --num_gpus 1).
# Clean text ready (train/val_thinkfix.jsonl, dataset_root hotpotqa_thinkfix/); top-K precompute
# (K=64, val n=2000 first) just (re)started after an infra hiccup -- still waiting on the .npz path.
TRAIN_DATA="data/distill/hotpotqa_thinkfix/train_thinkfix.jsonl"
VAL_DATA="data/distill/hotpotqa_thinkfix/val_thinkfix.jsonl"
TEACHER_TARGETS="__FILL_IN_CLEAN_TOPK_PATH__"                           # TODO: from data-gen-agent
VAL_TEACHER_TARGETS="__FILL_IN_CLEAN_VAL_TOPK_PATH__"                   # TODO: from data-gen-agent

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --kd_alpha 0.5 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --n_step_train_max 8 --use_ff --bf16 --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_nsteprand_cleankd_best.pt \
  > logs/retrieval1_nsteprand_cleankd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_nsteprand_cleankd_best.pt \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_nsteprand_cleankd_fullval.json \
  > logs/eval_thinker_nsteprand_cleankd_fullval.log 2>&1

echo PHASE15_NSTEPRAND_CLEANKD_DONE
