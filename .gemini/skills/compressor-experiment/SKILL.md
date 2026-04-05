---
name: compressor-experiment
description: Run and log experiments for the LLM as Data Compressor project. Use this skill to optimize prompts for specific texts, calculate compression ratios, and maintain structured logs of results.
---

# Compressor Experiment Skill

## Overview
This skill automates the process of finding an optimal prompt (via soft-prompt optimization followed by discrete projection) to compress a target text. It handles model loading, optimization, evaluation of ranks, and structured logging to both a global log and detailed experiment directories.

## Workflow

1. **Define the Task**: Identify the text to compress and the model to use (default is `gpt2`).
2. **Execute Experiment**: Run the bundled script with appropriate parameters.
3. **Analyze Results**: Review the summary in `dev_notes/compressor_experiments.md` or detailed artifacts in `logs/<exp_name>/`.

## Usage

Run an experiment using the following command structure:

```bash
export PYTHONPATH=$PYTHONPATH:.
conda run -n thinker python .agents/skills/compressor-experiment/scripts/run_and_log.py \
    --text "Your text here" \
    --model "gpt2" \
    --n_prompt 10 \
    --n_steps 100 \
    --exp_name "my-experiment" \
    --run_id "1"
```

### Parameters
- `--text`: The target text to compress.
- `--model`: Hugging Face model identifier (e.g., `gpt2`, `distilgpt2`).
- `--n_prompt`: Number of tokens in the prompt to optimize.
- `--n_steps`: Number of gradient descent steps for soft-prompt optimization.
- `--exp_name`: Directory name under `logs/` to store artifacts.
- `--run_id`: Sub-directory/identifier for the specific run.

## Metrics Explained
- **Compression Ratio**: Ratio of original text bits to (prompt bits + correction bits).
- **Accuracy**: Percentage of tokens correctly predicted as top-1 by the LLM using the optimized prompt.
- **Correction Bits**: Estimated information required to "fix" the LLM's predictions using token ranks.

## Best Practices
- **Text Length**: For very short texts, use fewer prompt tokens to avoid overhead.
- **Optimization**: If accuracy is low, increase `n_steps` or adjust `n_prompt`.
- **Reproducibility**: Always use descriptive `exp_name` and increment `run_id` for related trials.
