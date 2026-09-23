#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
mkdir -p dev_notes/qualitative

for tag in kd ceonly; do
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare_reasoning.py \
    --checkpoint checkpoints/math_small_${tag}_best.pt \
    --val_data data/distill/openr1_math/val.jsonl \
    --n_samples 30 --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff \
    --n_ctx 256 --max_thinking_len 1024 --max_answer_len 64 --device cuda \
    --out dev_notes/qualitative/math_${tag}_greedy.md \
    > logs/generate_qualitative_math_${tag}_greedy.log 2>&1
done
echo PHASE3A_QUALITATIVE_DONE
