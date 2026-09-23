#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
COMMON="--checkpoint checkpoints/retrieval1_reprkd_wsd2_best.pt --val_data data/distill/hotpotqa_full/val.jsonl \
  --n_samples 30 --ref_model qwen35 --tokenizer qwen35 \
  --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 --device cuda"

# (1) non-greedy sampling, temperature=0.8, top_p=0.9, same fixed 30 examples
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  $COMMON --temperature 0.8 --top_p 0.9 --seed 0 \
  --out dev_notes/qualitative/thinker_vs_qwen35_08b_wsd2_sampled_t08.md \
  > logs/generate_qualitative_sampled_t08.log 2>&1

# (2) greedy + first-step top-10 logits inspection (diagnostic, confident-wrong vs flat)
~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/generate_qualitative_compare.py \
  $COMMON --log_first_step_topk 10 \
  --out dev_notes/qualitative/thinker_vs_qwen35_08b_wsd2_logitdiag.md \
  > logs/generate_qualitative_logitdiag.log 2>&1

echo PHASE6B_SAMPLING_DIAG_DONE
