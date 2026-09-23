#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Scaling test (user via supervisor-agent, 2026-09-23): does a bigger Thinker core
# change the severity of the teacher-forced calibration collapse (agent2's diagnostic,
# ~0.4-0.6% argmax accuracy at d_model=256, structural-cause hypothesis after Baseline C)?
# d_model 256 -> 512 (2x width, n_head proportional 4->8), same KD recipe/data as phase20
# (retrieval, full top-K coverage), n_step=4 fixed, same methodology. 768 held in reserve
# if this signal is ambiguous (approved by supervisor-agent to judge/chain autonomously).
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/topk/val/thinkfix_n2000.qwen_big.npz"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type retrieval --data "$TRAIN_DATA" \
  --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" \
  --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big \
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 512 --n_head 8 --n_step 4 --use_ff --bf16 \
  --batch_size 8 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval_kd_scale_d512_best.pt \
  > logs/retrieval_kd_scale_d512_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval_kd_scale_d512_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type retrieval \
  --tokenizer qwen35 --d_model 512 --n_head 8 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 8 --device cuda \
  --out logs/eval_thinker_retrieval_kd_scale_d512_fullval.json \
  > logs/eval_thinker_retrieval_kd_scale_d512_fullval.log 2>&1

echo PHASE22_RETRIEVAL_KD_SCALE_D512_DONE
