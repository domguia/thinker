#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
for ndocs in 5 10 20; do
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/train_prompt_response.py \
    --dataset_type retrieval --data data/distill/hotpotqa_full/train_repr10k_ab.jsonl \
    --val_data data/distill/hotpotqa_full/val.jsonl \
    --teacher_targets /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/retrieval1_backup/train_repr10k_ab.npz \
    --val_teacher_targets data/distill/hotpotqa_full/val_topk32.npz \
    --repr_teacher_hidden /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/retrieval1_backup/train_repr10k_ab._hidden \
    --repr_teacher_layer 64 --repr_kd_weight 0.05 --repr_kd_warmup_steps 100 --kd_alpha 0.5 \
    --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --bf16 --compile --batch_size 32 \
    --n_docs_max "${ndocs}" \
    --lr 1e-4 --lr_decay_to 1e-5 --lr_stable_frac 0.125 --lr_decay_steps 750 --patience 6 \
    --seed 0 --num_workers 4 --max_steps 6000 --val_every 250 --max_time_minutes 150 \
    --save_best_checkpoint_path "checkpoints/retrieval1_kbsize_ndocs${ndocs}_best.pt" \
    > "logs/retrieval1_kbsize_ndocs${ndocs}_train.log" 2>&1
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
    --checkpoint "checkpoints/retrieval1_kbsize_ndocs${ndocs}_best.pt" \
    --val_data data/distill/hotpotqa_full/val.jsonl \
    --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max "${ndocs}" \
    --batch_size 32 --device cuda --out "logs/eval_thinker_kbsize_ndocs${ndocs}_fullval.json" \
    > "logs/eval_thinker_kbsize_ndocs${ndocs}_fullval.log" 2>&1
done
echo PHASE2_KB_SWEEP_DONE
