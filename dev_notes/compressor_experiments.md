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
| 2026-04-05 | gpt2 | 9 | 5 | 50 | 55.6% | 1.48x | Exp: wiki_length_study, Run: short_p5 |
| 2026-04-05 | gpt2 | 9 | 10 | 50 | 33.3% | 0.78x | Exp: wiki_length_study, Run: short_p10 |
| 2026-04-05 | gpt2 | 9 | 20 | 50 | 33.3% | 0.42x | Exp: wiki_length_study, Run: short_p20 |
| 2026-04-05 | gpt2 | 31 | 5 | 50 | 38.7% | 2.80x | Exp: wiki_length_study, Run: medium_p5 |
| 2026-04-05 | gpt2 | 31 | 10 | 50 | 41.9% | 1.95x | Exp: wiki_length_study, Run: medium_p10 |
| 2026-04-05 | gpt2 | 9 | 10 | 30 | 33.3% | 0.78x | Exp: resume_study, Run: short |
| 2026-04-05 | gpt2 | 31 | 10 | 30 | 48.4% | 2.00x | Exp: resume_study, Run: medium |
| 2026-04-05 | gpt2 | 51 | 10 | 30 | 52.9% | 2.65x | Exp: resume_study, Run: long |
