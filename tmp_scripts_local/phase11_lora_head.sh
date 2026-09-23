#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Frozen answer head (Teacher-init, Qwen3.5-0.8B -- the better-performing init from the earlier
# binary ablation, 8.0489 vs 8.2926 for the 27B init) + LoRA rank=32 adapter, same WSD+patience
# recipe as the binary ablation runs, batch_size=32 (VRAM headroom, same as the earlier frozen-
# head runs). Tests whether partial trainable capacity recovers some of the gap vs the fully-
# frozen binary result (8.0489) and the unfrozen baseline (7.8643).
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --teacher_targets /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/retrieval1_backup/train_repr10k_ab.npz \
  --val_teacher_targets data/distill/hotpotqa_full/val_topk32.npz \
  --repr_teacher_hidden /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/retrieval1_backup/train_repr10k_ab._hidden \
  --repr_teacher_layer 64 --repr_kd_weight 0.05 --repr_kd_warmup_steps 100 --kd_alpha 0.5 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 32 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --answer_head_init data/distill/qwen35_0.8b_head_init_d256.npz --freeze_answer_head \
  --answer_head_lora_rank 32 \
  --save_best_checkpoint_path checkpoints/retrieval1_frozenhead_08b_lora32_best.pt \
  > logs/retrieval1_frozenhead_08b_lora32_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_frozenhead_08b_lora32_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --answer_head_lora_rank 32 \
  --batch_size 32 --device cuda --out logs/eval_thinker_frozenhead_08b_lora32_fullval.json \
  > logs/eval_thinker_frozenhead_08b_lora32_fullval.log 2>&1

echo PHASE11_LORA_HEAD_DONE
