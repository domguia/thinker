#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --batch_size 8 \
  --lr 1e-4 --seed 0 --num_workers 2 --max_steps 10 --val_every 10 --val_batches 2 \
  --scheduled_sampling_p 0.3 --scheduled_sampling_warmup_steps 5 \
  --save_checkpoint_path checkpoints/smoke_ss_test.pt \
  2>&1
echo SMOKE_SS_DONE
