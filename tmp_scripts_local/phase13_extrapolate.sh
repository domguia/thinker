#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Core project thesis test (spec -1, plan): the recurrent register-update weights are
# SHARED across all n_step iterations -- does the model generalize to n_step_test != the
# training n_step (4) at pure inference time, no retraining? Tests both the flagship KD
# checkpoint and the 81k CE-only checkpoint (best CE numbers available so far).
for tag in wsd2 81k_ceonly; do
  case $tag in
    wsd2) ckpt=checkpoints/retrieval1_reprkd_wsd2_best.pt ;;
    81k_ceonly) ckpt=checkpoints/retrieval1_81k_ceonly_best.pt ;;
  esac
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/eval_thinker_full_val.py \
    --checkpoint $ckpt \
    --val_data data/distill/hotpotqa_full/val.jsonl \
    --tokenizer qwen35 --d_model 256 --n_head 4 --n_step 4 --use_ff --block_size 16 --n_docs_max 10 \
    --batch_size 64 --device cuda --extrapolate_n_steps 1,2,4,6,8,12,16 \
    --out logs/extrapolate_${tag}.json \
    > logs/extrapolate_${tag}.log 2>&1
done
echo PHASE13_EXTRAPOLATE_DONE
