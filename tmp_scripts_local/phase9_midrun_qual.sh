#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
mkdir -p dev_notes/qualitative

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  --checkpoint checkpoints/retrieval1_81k_ceonly_best.pt \
  --val_data data/distill/hotpotqa_full/val.jsonl \
  --n_samples 30 --ref_model qwen35 --tokenizer qwen35 \
  --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 --device cuda \
  --out dev_notes/qualitative/thinker_81k_ceonly_midrun_greedy.md \
  > logs/generate_qualitative_81k_midrun.log 2>&1

echo PHASE9_MIDRUN_QUAL_DONE
