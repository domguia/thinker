#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# n_step ~ Uniform(1,8) per training batch (--n_step_train_max 8), CE-only, dataset AB 9500
# (modest budget, per analyst-agent). No --compile: torch.compile would specialize/recompile
# per distinct n_step value seen (up to 8 graphs), not worth the overhead for this diagnostic
# run. --extrapolate_n_steps at the end directly compares against the fixed-n_step=4 U-shape
# curve already obtained (dev_notes/experiments/prompt_response_pipeline.md).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --n_step_train_max 8 --use_ff --bf16 --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_ceonly_nsteprand_best.pt \
  > logs/retrieval1_ceonly_nsteprand_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_ceonly_nsteprand_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_ceonly_nsteprand_fullval.json \
  > logs/eval_thinker_ceonly_nsteprand_fullval.log 2>&1

echo PHASE14_NSTEPRAND_DONE
