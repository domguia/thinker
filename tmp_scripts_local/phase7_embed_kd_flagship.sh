#!/bin/bash
cd ~/thinker
export PYTHONPATH=.

# KD + embedding-KD at flagship scale (AB dataset, WSD+patience) -- same recipe as
# retrieval1_reprkd_wsd2_best.pt / retrieval1_ab_topkonly_best.pt, plus
# --embed_teacher_target/--embed_kd_weight 0.1 (only tested at small synthetic
# scale before, 2026-09-2x journal entry -- this is the first flagship-scale run).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --teacher_targets /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/retrieval1_backup/train_repr10k_ab.npz \
  --val_teacher_targets data/distill/hotpotqa_full/val_topk32.npz \
  --kd_alpha 0.5 --embed_teacher_target data/distill/qwen_big_head_init_d256.npz --embed_kd_weight 0.1 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 64 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --save_best_checkpoint_path checkpoints/retrieval1_embedkd_wsd_best.pt \
  > logs/retrieval1_embedkd_wsd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_embedkd_wsd_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 64 --device cuda --out logs/eval_thinker_embedkd_wsd_fullval.json \
  > logs/eval_thinker_embedkd_wsd_fullval.log 2>&1

# Qualitative comparison: KD-pur (retrieval1_ab_topkonly_best.pt) vs KD+embedding (this run),
# non-greedy sampling (same temperature as the collapse diagnostic) since greedy is known to
# collapse on this task (see dev_notes/experiments/prompt_response_pipeline.md).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  --checkpoint checkpoints/retrieval1_ab_topkonly_best.pt \
  --checkpoint2 checkpoints/retrieval1_embedkd_wsd_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --n_samples 30 --tokenizer qwen35 \
  --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 --device cuda \
  --temperature 0.8 --top_p 0.9 --seed 0 \
  --out dev_notes/qualitative/thinker_kdpur_vs_embedkd_sampled_t08.md \
  > logs/generate_qualitative_kdpur_vs_embedkd.log 2>&1

echo PHASE7_EMBED_KD_DONE
