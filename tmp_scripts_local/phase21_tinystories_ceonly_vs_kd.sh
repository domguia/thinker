#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# CE-vs-KD isolated comparison on tinystories (2026-09-23, self-directed continuation
# of the wikitext/retrieval/math methodology per supervisor-agent's standing directive
# to verify every dataset qualitatively before drawing conclusions). agent2 already ran
# KD-alone on tinystories (collapse diagnostic, no CE-only baseline for comparison) --
# this completes the last of the 4 datasets (wiki/retrieval/math/tinystories) mentioned
# in the user's original priority list with the clean isolated ablation. Top-K coverage
# is FULL (4503/4503 train, 497/497 val, qwen_big).
#
# Same reformat convention as phase18 (make_prompt_response_from_realtext.py, 96-token
# prompt + 48-token answer span). "thinking" stays empty (no reasoning trace in this data).
TS_ROOT="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/tinystories"
TRAIN_PR="$TS_ROOT/train_pr.jsonl"
VAL_PR="$TS_ROOT/val_pr.jsonl"

if [ ! -f "$TRAIN_PR" ]; then
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/distill/make_prompt_response_from_realtext.py \
    --in_file "$TS_ROOT/train.jsonl" --out_file "$TRAIN_PR" --tokenizer Qwen/Qwen3.5-0.8B \
    --prompt_tokens 96 --answer_tokens 48
fi
if [ ! -f "$VAL_PR" ]; then
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/distill/make_prompt_response_from_realtext.py \
    --in_file "$TS_ROOT/val.jsonl" --out_file "$VAL_PR" --tokenizer Qwen/Qwen3.5-0.8B \
    --prompt_tokens 96 --answer_tokens 48
fi

TEACHER_TARGETS="$TS_ROOT/topk/train"
VAL_TEACHER_TARGETS="$TS_ROOT/topk/val/lastLayer_n497.qwen_big.npz"

# --- Arm A: CE-only ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_PR" \
  --val_data "$VAL_PR" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 8 --max_thinking_len 8 --max_answer_len 48 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/tinystories_ceonly_best.pt \
  > logs/tinystories_ceonly_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/tinystories_ceonly_best.pt \
  --val_data "$VAL_PR" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 8 --max_answer_len 48 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_tinystories_ceonly_fullval.json \
  > logs/eval_thinker_tinystories_ceonly_fullval.log 2>&1

# --- Arm B: KD top-K (kd_alpha=0.5) ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_PR" \
  --val_data "$VAL_PR" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 8 --max_thinking_len 8 --max_answer_len 48 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/tinystories_kd_best.pt \
  > logs/tinystories_kd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/tinystories_kd_best.pt \
  --val_data "$VAL_PR" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 8 --max_answer_len 48 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_tinystories_kd_fullval.json \
  > logs/eval_thinker_tinystories_kd_fullval.log 2>&1

echo PHASE21_TINYSTORIES_CEONLY_VS_KD_DONE
