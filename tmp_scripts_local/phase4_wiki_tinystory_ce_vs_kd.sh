#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# general_realtext already has top-K precomputed (2700 ex., LFM2 family) -- fast first pass,
# no precompute wait, uses idle GPU capacity right now rather than waiting on the new
# storage-tree format for the bigger 9k wikitext_sample5k+tinystories_sample5k combo.

# KD arm
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data data/distill/general_realtext/train.jsonl \
  --val_data data/distill/general_realtext/val.jsonl \
  --teacher_targets data/distill/general_realtext/train_topk32.npz \
  --val_teacher_targets data/distill/general_realtext/val_topk32.npz \
  --kd_alpha 0.5 --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile \
  --batch_size 16 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --save_best_checkpoint_path checkpoints/wiki_tinystory_realtext_kd_best.pt \
  > logs/wiki_tinystory_realtext_kd_train.log 2>&1

# CE-only arm
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data data/distill/general_realtext/train.jsonl \
  --val_data data/distill/general_realtext/val.jsonl \
  --tokenizer lfm2 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile \
  --batch_size 16 --max_thinking_len 1024 --max_answer_len 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --save_best_checkpoint_path checkpoints/wiki_tinystory_realtext_ceonly_best.pt \
  > logs/wiki_tinystory_realtext_ceonly_train.log 2>&1

echo PHASE4_WIKI_TINYSTORY_DONE
