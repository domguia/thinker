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
# Clean text ready (train/val_thinkfix.jsonl, dataset_root hotpotqa_thinkfix/). Val top-K (K=64,
# n=2000) precompute finished 2026-09-22 ~13:20 (2000/2000, 1.28 ex/s, ~26min, after an NFS-hard-mount
# stall on ecotaxe-1 that was worked around by copying the Teacher checkpoint to local /tmp).
# Train top-K (9500 ex.) still in progress -- fill TEACHER_TARGETS once data-gen-agent provides it.
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl"
# New-format manifest-based store (directory, not a specific .npz): thinkfix_n1500 is a
# SUBSET of the 80999-row train.jsonl, not its first 1500 rows -- passing the bare .npz
# would hit the old-format path (offsets indexed by raw doc_id) and crash with an
# IndexError, exactly like the identical bug hit and fixed on phase16 (math) 2026-09-22.
# The directory form remaps doc_id -> local npz row via manifest.json's subset_file.
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/train"  # TODO: confirm once data-gen-agent's n=1500 batch is written
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/val/thinkfix_n2000.qwen_big.npz"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --n_step_train_max 8 --use_ff --bf16 --batch_size 16 \
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
  --batch_size 16 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_nsteprand_cleankd_fullval.json \
  > logs/eval_thinker_nsteprand_cleankd_fullval.log 2>&1

echo PHASE15_NSTEPRAND_CLEANKD_DONE
