# LLM Compressor Experiments

This log tracks experiments for the "LLM as Data Compressor" project. 
The goal is to quantify the relationship between prompt length, text length, and compression efficiency.

## Metrics Definition
- **Prompt Bits**: `n_prompt_tokens * log2(vocab_size)`
- **Correction Bits**: Entropy of the ranks needed to fix the LLM's predictions.
- **Compression Ratio**: `Original Bits / (Prompt Bits + Correction Bits)`

## Results Log

| Date | Model | Target Len (tokens) | Prompt Len (tokens) | Steps | Accuracy (top-1) | Compression Ratio | Notes |
|------|-------|---------------------|---------------------|-------|------------------|-------------------|-------|
| 2024-05-22 | GPT-2 | 20 | 5 | 10 | 30% | 2.19x | Initial demo run. |

## Observations
- Soft-to-discrete projection often causes a drop in accuracy compared to the raw latent state, increasing correction bits.
- Larger models generally have a better "prior", potentially reducing correction bits.
- Very short texts are harder to compress because the fixed overhead of the prompt is high.
| 2026-04-05 | GPT-2 | 13 | 10 | 30 | 61.5% | 1.13x | Text: 'The Sun is the star ...' |
| 2026-04-05 | GPT-2 | 13 | 10 | 30 | 61.5% | 1.13x | Text: 'The Sun is the star ...' |
| 2026-04-05 | GPT-2 | 53 | 10 | 30 | 39.6% | 2.52x | Text: 'The Sun is the star ...' |
