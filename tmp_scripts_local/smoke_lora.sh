#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --batch_size 8 \
  --lr 1e-4 --seed 0 --num_workers 2 --max_steps 10 --val_every 10 --val_batches 2 \
  --answer_head_init data/distill/qwen35_0.8b_head_init_d256.npz --freeze_answer_head \
  --answer_head_lora_rank 16 \
  --save_checkpoint_path checkpoints/smoke_lora_test.pt \
  2>&1
echo SMOKE_LORA_DONE
