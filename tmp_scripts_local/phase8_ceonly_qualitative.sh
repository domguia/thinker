#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null
mkdir -p dev_notes/qualitative

# Causal control (user + analyst-agent, 2026-09-22): does the CE-only checkpoint (never
# exposed to the Teacher's contaminated <think> KD signal) avoid the generation collapse?
# Explicit deviation from the project's KD-default policy, flagged as a diagnostic control,
# not a checkpoint to compare in production (see CLAUDE.md).
COMMON="--checkpoint checkpoints/retrieval1_ceonly_best.pt --val_data data/distill/hotpotqa_full/val.jsonl \
  --n_samples 30 --ref_model qwen35 --tokenizer qwen35 \
  --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 --device cuda"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  $COMMON \
  --out dev_notes/qualitative/thinker_ceonly_vs_qwen35_greedy.md \
  > logs/generate_qualitative_ceonly_greedy.log 2>&1

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  $COMMON --temperature 0.8 --top_p 0.9 --seed 0 \
  --out dev_notes/qualitative/thinker_ceonly_vs_qwen35_sampled_t08.md \
  > logs/generate_qualitative_ceonly_sampled.log 2>&1

echo PHASE8_CEONLY_QUALITATIVE_DONE
