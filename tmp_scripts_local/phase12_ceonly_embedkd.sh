#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Clean isolation test (analyst-agent/user, 2026-09-22): CE-only + embedding-KD, NO top-K KD
# (--teacher_targets/--kd_alpha omitted) -- isolates the embedding-anchor signal from the
# <think>-contaminated top-K, and doesn't depend on data-gen-agent's pending template fix
# (extract_teacher_head_lite.py reads the Teacher's raw embed_tokens/lm_head weight tensors
# directly via safetensors, no forward pass/prompt/template involved -- confirmed independent
# of the malformed-prompt issue). Same WSD+patience recipe as the CE-only baseline (7.6315,
# degenerate generation) for direct comparison.
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --embed_teacher_target data/distill/qwen_big_head_init_d256.npz --embed_kd_weight 0.1 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_ceonly_embedkd_best.pt \
  > logs/retrieval1_ceonly_embedkd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_ceonly_embedkd_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --out logs/eval_thinker_ceonly_embedkd_fullval.json \
  > logs/eval_thinker_ceonly_embedkd_fullval.log 2>&1

echo PHASE12_CEONLY_EMBEDKD_DONE
