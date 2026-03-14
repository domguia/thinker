# thinker-autoresearch

This is an automated experimental setup to let an LLM do its own research and optimize the `ToyThinker` model on the toy reasoning tasks.

## Setup

To set up a new experiment in this environment (e.g., Google Colab):
1. **Agree on a run tag**: Propose a tag based on the experiment objective or date (e.g. \`colab-run1\`).
2. **Review the context**: Read these files for full context:
   - `README.md` & `ideas-draft.md` — Project context and thoughts.
   - `model_utils.py` & `numbers_data.py` — The core components (Attention layers, SwiGLU, RMSNorm) and Curriculum Dataset generation. **You can modify these if necessary but focus mostly on `train.py` and `toy_model.py`.**
   - `toy_model.py` — The core architecture (`ToyThinker`), which handles latent iterations and routing. **Modifying this is highly encouraged.**
   - `train.py` — The training loop, dataset initialization, optimizer, and time budget. **Modifying this is highly encouraged.**
3. **Initialize `results.tsv`**: Create a `results.tsv` with just the header row.

## Experimentation Loop

The core concept is that you run `python train.py` and it will automatically execute for a **fixed time budget of 15 minutes** (wall clock training time limit). Feel free to edit the `max_time_minutes` inside `train.py` if 15 minutes is too long for the active Colab session (e.g. reducing it to 5 minutes for faster iterations).

**Your Goal:** Achieve the **lowest `best_loss`** (and highest accuracy) within the fixed time budget.

**What you CAN do:**
- Iterate on `train.py`, `toy_model.py`, and `model_utils.py`.
- Change architectures, optimizers (AdamW vs Muon), learning rates, dataset distributions, curriculum triggers, width, height, latent size, etc.

**Output format:**
At the end of the `train.py` run, a block will be printed:
```text
---
best_loss:        0.997900
training_seconds: 300.1
total_seconds:    305.9
num_steps:        953
num_params_M:     5.30
```
Use `grep "^best_loss:" run.log` to extract your metric.

## Logging

When an experiment is done, log it to `results.tsv` (tab-separated):

```tsv
commit	best_loss	status	description
```
* status: `keep`, `discard`, or `crash`

If the `best_loss` decreases compared to the baseline, you **keep** the changes. If it's worse or equal, you **revert** to the previous working state using git or by undoing edits.

## Colab Workflow

If running in Colab via an autonomous agent (like Gemini Advanced or Claude):
1. Hack the code.
2. Run `!python train.py > run.log 2>&1`
3. Check the results: `!grep "^best_loss:" run.log` or `!tail -n 20 run.log` (if crashed).
4. Log to `results.tsv`.
5. Repeat indefinitely.

Never stop unless the human interrupts.
