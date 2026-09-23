#!/bin/bash
cd ~/thinker
export PYTHONPATH=.

# Isole la récurrence : même checkpoint KD (entraîné n_step=4), n_step d'inférence
# varié 1 vs 4, mesure calibration teacher-forcée uniquement (pas de retraining).
for ns in 1 4; do
  ~/bin/micromamba run -p ~/micromamba/envs/teacher311 python learn/indexed_attention/diagnose_generation_divergence.py \
    --checkpoint checkpoints/retrieval1_kdvsce_kd_best.pt \
    --val_data /srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa/val.jsonl \
    --tokenizer qwen35 --d_model 256 --n_head 4 --n_step $ns --use_ff --block_size 16 --n_docs_max 10 \
    --n_samples 25 --device cuda \
    --out "logs/divergence_nstep${ns}.json" \
    > "logs/divergence_nstep${ns}.log" 2>&1
done
echo NSTEP_CHECK_DONE
