#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# CE-vs-KD isolated comparison on wikitext (supervisor-agent, 2026-09-22): never done before
# on this pipeline -- agent2 only ran KD-alone on tinystories (collapse diagnostic, no CE
# baseline) and wikitext was untouched. Top-K coverage here is FULL (4497/4497 train,
# 503/503 val, qwen_big, never affected by the retrieval <think>-collapse bug since this is
# plain-text LM data with no chat template at all).
#
# Reformat step (agent2's convention, 2026-09-21/22, same script/defaults used for
# tinystories): make_prompt_response_from_realtext.py splits each raw document into a
# 96-token "problem" (prompt) + 48-token "answer" span, both verbatim substrings of the
# original "text" -- keeps doc_id (=row index) aligned with the already-precomputed top-K
# store, so no re-precompute needed. "thinking" stays empty for every example (no natural
# reasoning trace in this data) -- known benign side effect: aggregate "thinking" CE/loss
# fields show NaN (0/0 on an all-masked stream), not a real corruption, see phase16's note
# and agent2's original tinystories entry in this journal (2026-09-22).
WIKI_ROOT="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/wikitext"
TRAIN_PR="$WIKI_ROOT/train_pr.jsonl"
VAL_PR="$WIKI_ROOT/val_pr.jsonl"

if [ ! -f "$TRAIN_PR" ]; then
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/distill/make_prompt_response_from_realtext.py \
    --in_file "$WIKI_ROOT/train.jsonl" --out_file "$TRAIN_PR" --tokenizer Qwen/Qwen3.5-0.8B \
    --prompt_tokens 96 --answer_tokens 48
fi
if [ ! -f "$VAL_PR" ]; then
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/distill/make_prompt_response_from_realtext.py \
    --in_file "$WIKI_ROOT/val.jsonl" --out_file "$VAL_PR" --tokenizer Qwen/Qwen3.5-0.8B \
    --prompt_tokens 96 --answer_tokens 48
fi

TEACHER_TARGETS="$WIKI_ROOT/topk/train"
VAL_TEACHER_TARGETS="$WIKI_ROOT/topk/val/lastLayer_n503.qwen_big.npz"

# --- Arm A: CE-only ---
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_PR" \
  --val_data "$VAL_PR" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 8 --max_thinking_len 8 --max_answer_len 48 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/wikitext_ceonly_best.pt \
  > logs/wikitext_ceonly_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/wikitext_ceonly_best.pt \
  --val_data "$VAL_PR" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 8 --max_answer_len 48 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_wikitext_ceonly_fullval.json \
  > logs/eval_thinker_wikitext_ceonly_fullval.log 2>&1

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
  --save_best_checkpoint_path checkpoints/wikitext_kd_best.pt \
  > logs/wikitext_kd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/wikitext_kd_best.pt \
  --val_data "$VAL_PR" \
  --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff \
  --max_thinking_len 8 --max_answer_len 48 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_wikitext_kd_fullval.json \
  > logs/eval_thinker_wikitext_kd_fullval.log 2>&1

echo PHASE18_WIKITEXT_CEONLY_VS_KD_DONE
