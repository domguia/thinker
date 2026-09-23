#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
# Resume after besteffort preemption (infra-agent, 2026-09-23, job 4131713 -> 4132033):
# original run reached ~step 3640/6000 (val_answer=4.2248 checkpoint saved) before being
# killed. No real optimizer/step-count resume mechanism yet (see dev_notes/ideas/
# checkpoint_resume_signal.md) -- --init_from_checkpoint reloads WEIGHTS ONLY, fresh
# optimizer/LR schedule. Budget halved (max_steps 3000, lr_decay_steps proportionally
# reduced) since the weights are already well into training, to avoid redoing ~6000
# steps of compute on top of what's already baked into the checkpoint.
TRAIN_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/train.jsonl"
VAL_DATA="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/val.jsonl"
TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/train"
VAL_TEACHER_TARGETS="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/openr1_math/topk/val/lastLayer_n3890.qwen_big.npz"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
  --dataset_type reasoning --data "$TRAIN_DATA" --val_data "$VAL_DATA" \
  --teacher_targets "$TEACHER_TARGETS" --val_teacher_targets "$VAL_TEACHER_TARGETS" \
  --teacher_name qwen_big --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 1 --max_thinking_len 1024 --max_answer_len 64 \
  --init_from_checkpoint checkpoints/math_kd_fixed_best.pt \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 375 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 3000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/math_kd_fixed_best.pt \
  > logs/math_kd_fixed_resume_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/math_kd_fixed_best.pt --val_data "$VAL_DATA" --dataset_type reasoning \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --max_thinking_len 1024 --max_answer_len 64 \
  --batch_size 1 --device cuda \
  --out logs/eval_thinker_math_kd_fixed_fullval.json \
  > logs/eval_thinker_math_kd_fixed_fullval.log 2>&1

echo PHASE17C_MATH_KD_RESUME_DONE
