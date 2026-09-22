#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# supervisor-agent request (2026-09-22): re-test KD top-K + embedding-KD combined
# (--embed_teacher_target/--embed_kd_weight, previously neutral/slightly negative on
# contaminated data, phase7) on the clean post-<think>-fix top-K n=1500 data
# (hotpotqa_thinkfix). Same WSD+patience recipe as phase17's KD baseline, fixed n_step=4,
# batch_size=16 to match, embed_kd_weight=0.1 (same value as phase7).
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/val/thinkfix_n2000.qwen_big.npz"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big --kd_alpha 0.5 \
  --embed_teacher_target data/distill/qwen_big_head_init_d256.npz --embed_kd_weight 0.1 \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --batch_size 8 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 180 \
  --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval1_embedkd_cleankd_best.pt \
  > logs/retrieval1_embedkd_cleankd_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval1_embedkd_cleankd_best.pt \
  --val_data "$VAL_DATA" \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 8 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
  --out logs/eval_thinker_embedkd_cleankd_fullval.json \
  > logs/eval_thinker_embedkd_cleankd_fullval.log 2>&1

echo PHASE19_EMBEDKD_DONE
