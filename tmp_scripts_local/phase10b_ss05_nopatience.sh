#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Stronger, more decisive scheduled-sampling test: p=0.5 (vs 0.25), faster warmup (300 vs 500),
# and NO early-stopping (--patience omitted) so training runs the full 6000 steps regardless of
# a CE plateau -- the first attempt (p=0.25) only got ~2000 real scheduled-sampling steps before
# patience=6 stopped it at the CE optimum, which may not be enough exposure for the mechanism to
# show an effect on GENERATION quality specifically (a different objective than CE).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --scheduled_sampling_p 0.5 --scheduled_sampling_warmup_steps 300 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_ss05_nopatience_best.pt \
  --save_checkpoint_path checkpoints/retrieval1_ss05_nopatience_final.pt \
  > logs/retrieval1_ss05_nopatience_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_ss05_nopatience_final.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --out logs/eval_thinker_ss05_nopatience_fullval.json \
  > logs/eval_thinker_ss05_nopatience_fullval.log 2>&1

echo PHASE10B_SS05_NOPATIENCE_DONE
