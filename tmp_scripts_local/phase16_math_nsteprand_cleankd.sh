#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Parallel diagnostic (supervisor-agent/user, 2026-09-22): same combined recipe as phase15
# (n_step ~ Uniform(1,8) + top-K KD) but on openr1_math (reasoning), using top-K already
# precomputed against qwen_big (never affected by the <think>-collapse bug, confirmed on
# 100 examples -- this data is reasoning/math, not retrieval, so the empty <think> block
# issue in prepare_retrieval_data.py's CHATML_TEMPLATE never applied here).
# Runs while data-gen-agent prepares the retrieval train top-K, so GPU/time isn't idle.
# Teacher/Student must be same-family: qwen35 (Student) <-> qwen_big (Teacher), per
# .claude/skills/model-families/SKILL.md -- do NOT use --tokenizer lfm2 here (that pairing
# has no matching qwen_big-derived top-K).
# Train top-K currently a partial subset (topk_n1714, not the full 35011/18000) -- fine for
# a quick KD-vs-CE + n_step-variable comparison, not a final flagship number.
# batch_size=8 OOM'd on graffiti-11 (RTX 2080 Ti, 10.57GB, even at batch_size=4) AND on an
# idle A40 46GB (job 4127787, abacus22-2) -- topk_kd_loss's `student_logZ =
# logsumexp(student_logits, dim=-1)` materializes a DENSE (batch, seq, vocab) tensor despite
# only using the top-K indices, so memory scales with the full qwen35 vocab (248320, ~3.9x
# LFM2's 64400) x max_thinking_len (1024) regardless of K -- confirmed OOM even at 40GB+ in
# use before the crash. Dropped to batch_size=2 (proportionally ~4x smaller than phase3a's
# LFM2 batch_size=8, matching the vocab ratio).
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/val.jsonl"
# New-format manifest-based store (directory, not the .npz directly): topk_n1714 is a
# random SUBSET of the 34279-row train.jsonl, not its first 1714 rows -- passing the bare
# .npz would hit PromptResponseTeacherTargets' old-format path, which indexes `offsets` by
# the RAW doc_id (up to 34278) and crashes (IndexError, confirmed 2026-09-22 on graffiti-11).
# The directory form uses TeacherTopKStore, which remaps doc_id -> local npz row via
# manifest.json's subset_file (subsets/train/topk_n1714.indices.npy).
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/val/lastLayer_n3890.qwen_big.npz"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --n_step_train_max 8 --use_ff --bf16 \
  --batch_size 2 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/math_nsteprand_cleankd_best.pt \
  > logs/math_nsteprand_cleankd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/math_nsteprand_cleankd_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 1024 --max_answer_len 64 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_math_nsteprand_cleankd_fullval.json \
  > logs/eval_thinker_math_nsteprand_cleankd_fullval.log 2>&1

echo PHASE16_MATH_NSTEPRAND_CLEANKD_DONE
