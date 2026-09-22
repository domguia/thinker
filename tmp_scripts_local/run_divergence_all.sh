#!/bin/bash
cd ~/thinker
export PYTHONPATH=.

declare -A CKPTS=(
  [kd]="retrieval1_kdvsce_kd_best.pt"
  [ceonly]="retrieval1_kdvsce_ceonly_best.pt"
  [lora]="retrieval1_frozenhead_lora32_cleankd_best.pt"
  [embedkd]="retrieval1_embedkd_cleankd_best.pt"
)

for tag in kd ceonly lora embedkd; do
  ckpt="${CKPTS[$tag]}"
  lora_flag=""
  if [ "$tag" = "lora" ]; then
    lora_flag="--answer_head_lora_rank 32"
  fi
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/diagnose_generation_divergence.py \
    --checkpoint "checkpoints/$ckpt" \
    --val_data /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl \
    --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
    $lora_flag \
    --n_samples 25 --device cuda \
    --out "logs/divergence_${tag}.json" \
    > "logs/divergence_${tag}.log" 2>&1
done
echo ALL_DIVERGENCE_DONE
