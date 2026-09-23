#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

for tag in kd ceonly; do
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
    --checkpoint checkpoints/math_small_${tag}_best.pt \
    --val_data data/distill/openr1_math/val.jsonl \
    --dataset_type reasoning --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff \
    --max_thinking_len 1024 --max_answer_len 64 \
    --batch_size 8 --device cuda --out logs/eval_thinker_math_${tag}_fullval.json \
    > logs/eval_thinker_math_${tag}_fullval.log 2>&1
done
echo PHASE3A_FULLVAL_DONE
