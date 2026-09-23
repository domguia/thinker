#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# baseline (scheduled_sampling_p=0, already have retrieval1_ceonly_best.pt from earlier --
# retrain here anyway for a same-job apples-to-apples qualitative comparison) vs
# scheduled_sampling_p=0.25, same WSD+patience recipe, CE-only (isolates the exposure-bias
# question from KD/template contamination).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --scheduled_sampling_p 0.25 --scheduled_sampling_warmup_steps 500 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_ss025_ceonly_best.pt \
  > logs/retrieval1_ss025_ceonly_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_ss025_ceonly_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --out logs/eval_thinker_ss025_ceonly_fullval.json \
  > logs/eval_thinker_ss025_ceonly_fullval.log 2>&1

echo PHASE10_SCHEDULED_SAMPLING_DONE
