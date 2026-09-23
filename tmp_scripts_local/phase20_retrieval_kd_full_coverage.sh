#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Suite de phase19 (supervisor-agent, 2026-09-22/23): "continuer les comparaisons
# CE-vs-KD sur retrieval a mesure que de nouveaux paliers top-K arrivent". Le tier
# thinkfix_full (couverture 100%, vs 57% de thinkfix_p46250 utilise en phase19) est
# arrive cette nuit (data-agent, 2026-09-23 03:26). TeacherTopKStore fusionne tous
# les subsets du manifest sous le meme dossier -- meme pointeur de repertoire que
# phase19, la couverture full est prise automatiquement des que le fichier existe.
# n_step=4 fixe, meme methodologie que phase17/18/19. CE-only deja connu (phase19,
# 7.0325) -- on ne relance QUE l'arm KD ici avec la couverture complete.
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
  --kd_alpha 0.5 --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 \
  --batch_size 16 \
  --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
  --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
  --qualitative_eval_at_end --qualitative_eval_n_samples 30 \
  --save_best_checkpoint_path checkpoints/retrieval_kd_fullcov_best.pt \
  > logs/retrieval_kd_fullcov_train.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
  --checkpoint checkpoints/retrieval_kd_fullcov_best.pt \
  --val_data "$VAL_DATA" \
  --dataset_type retrieval \
  --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
  --batch_size 16 --device cuda \
  --out logs/eval_thinker_retrieval_kd_fullcov_fullval.json \
  > logs/eval_thinker_retrieval_kd_fullcov_fullval.log 2>&1

echo PHASE20_RETRIEVAL_KD_FULLCOV_DONE
