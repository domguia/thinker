#!/bin/bash
cd ~/thinker
export PYTHONPATH=.
find ~/.cache/mamba/proc/ -size 0 -delete 2>/dev/null

# Baseline C (supervisor-agent/user, 2026-09-23): does the exposure-bias-independent
# collapse (argmax teacher-forced ~0% correct, converging to a small recurrent token set --
# agent2's diagnosis, confirmed identical across two independent Thinker CE-only checkpoints,
# math/wikitext/retrieval) come from Thinker's recurrent core (register/OutputStream
# mechanism), or is it a general small-model/vocab/data-scale artifact? Test: train a
# CLASSIC DENSE transformer (train_sft.py, no recurrence, standard causal LM) on the SAME
# wikitext data (top-K coverage FULL, 4497/4497), same tokenizer (qwen35/qwen_big vocab),
# roughly comparable width (n_embd=256, n_head=4, matching Thinker's d_model=256/n_head=4)
# -- if THIS also collapses to the same kind of degenerate mode, the cause is general to
# this small-model/vocab/data regime, not specific to Thinker's mechanism.
# CE-only for now (fastest path to an answer) -- add a KD arm after if useful.
WIKI_ROOT="/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/wikitext"

~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/distill/train_sft.py \
  --train_file "$WIKI_ROOT/train.jsonl" \
  --val_file "$WIKI_ROOT/val.jsonl" \
  --tokenizer qwen35 --base_config gpt2 \
  --n_layer 6 --n_embd 256 --n_head 4 --block_size 256 \
  --batch_size 4 --bf16 \
  --lr 1e-4 --max_steps 6000 --max_time_minutes 150 \
  --val_every 250 \
  --save_dir checkpoints/baselineC_wikitext \
  --seed 0 \
  > logs/baselineC_wikitext_ceonly_train.log 2>&1

echo BASELINEC_WIKITEXT_DENSE_TRANSFORMER_DONE
