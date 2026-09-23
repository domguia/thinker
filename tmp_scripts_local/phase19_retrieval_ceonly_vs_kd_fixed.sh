#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Isolated ablation (supervisor-agent, 2026-09-22): clean CE-only vs KD comparison on
# retrieval (hotpotqa), FIXED n_step=4 (no --n_step_train_max), same methodology as
# phase17 (math). Uses thinkfix_p46250 (46250/80999, ~57% coverage) -- much denser than
# the n1500 (~1.9%) used in phase15's combined n_step-variable+KD run, which is why this
# is a separate, cleaner comparison rather than reusing that result.
# MANDATORY per supervisor-agent (2026-09-22, post wikitext-collapse discovery): do NOT
# conclude from answer_ce alone -- run the qualitative check (now wired for reasoning,
# already existed for retrieval) on both checkpoints before reporting a result as solid.
# Retrieval sequences are short (block_size=16 + max_answer_len=32) -- batch_size=16
# confirmed OOM-free on a 2080Ti-class GPU in phase15, no special memory handling needed
# here unlike math/wikitext's dense full-vocab KD loss on long sequences.
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/val/thinkfix_n2000.qwen_big.npz"

# --- Arm A: CE-only ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 16 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval_ceonly_fixed_best.pt \
  > logs/retrieval_ceonly_fixed_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval_ceonly_fixed_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type retrieval \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda \
  --out logs/eval_thinker_retrieval_ceonly_fixed_fullval.json \
  > logs/eval_thinker_retrieval_ceonly_fixed_fullval.log 2>&1

# --- Arm B: KD top-K (kd_alpha=0.5), same fixed n_step ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 16 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval_kd_fixed_best.pt \
  > logs/retrieval_kd_fixed_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval_kd_fixed_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type retrieval \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda \
  --out logs/eval_thinker_retrieval_kd_fixed_fullval.json \
  > logs/eval_thinker_retrieval_kd_fixed_fullval.log 2>&1

echo PHASE19_RETRIEVAL_CEONLY_VS_KD_FIXED_DONE
