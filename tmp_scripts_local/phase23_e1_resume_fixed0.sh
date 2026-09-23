#!/bin/bash
# Resume E1 fixed_seed0 after preemption (graffiti-3 killed at step 2360/6000,
# best checkpoint @step2250 val_answer=7.9144 preserved on home NFS).
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --batch_size 16 \
  --init_from_checkpoint checkpoints/e1_fixed_seed0_best.pt \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --save_best_checkpoint_path checkpoints/e1_fixed_seed0_best.pt \
  > logs/e1_fixed_seed0_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/e1_fixed_seed0_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda --extrapolate_n_steps 1,2,4,8,12,16,24,32 \
  --out logs/e1_fixed_seed0_fullval.json \
  > logs/e1_fixed_seed0_fullval.log 2>&1

echo E1_FIXED_SEED0_RESUME_DONE
