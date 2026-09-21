# Thinker - Project Instructions

## Training methodology default: KD, not pure CE

**Default to knowledge distillation (KD from the Teacher) for every training run in this project.** Pure cross-entropy (no `--teacher_targets`/`--kd_alpha`) is only acceptable when (a) the user explicitly asks for a CE-only run, or (b) there is genuinely no way to get Teacher targets in time/at all for that data (and even then, flag it explicitly as a deviation, don't default to it silently).

**Why:** the project's whole training recipe (flagship, general, reasoning) runs on KD, and KD itself likely acts as a regularizer — training a comparison/control checkpoint in pure CE introduces an extra confound (different training regime) on top of whatever the experiment is actually trying to isolate, and makes it non-comparable to every other checkpoint in the project. Time is also short before the ICLR deadline, so there's little room for CE-only runs that would need redoing in KD anyway.

**How to apply:** when designing or dispatching any new training run (including small/synthetic diagnostic datasets, e.g. causal-control checkpoints), precompute Teacher targets first (top-K logits + hidden states together, see storage/precompute policy) and train with the same KD recipe as the flagship (`--teacher_targets`, `--kd_alpha 0.5`, same architecture flags) unless there is a specific, stated reason not to.

## repr-KD hidden-layer capture: always the last layer

**Every `precompute_teacher_targets.py --hidden_layers` run must include the last layer**, by passing the literal `last` (alone, or combined with an intermediate index, e.g. `--hidden_layers last,8`) — never a hardcoded layer number chosen because it "looks like" the last layer for one specific Teacher. `parse_hidden_layers` (2026-09-21) supports mixing `last` into the comma-separated list for exactly this reason.

**Why:** the final hidden layer is the most broadly reusable embedding — usable for repr-KD, downstream probing, or anything else — almost regardless of what the model was trained for. A hardcoded index (e.g. `16`) is only "the last layer" by coincidence for a Teacher with exactly that many layers (this happened once, undetected: LFM2-1.2B has `num_hidden_layers=16`, so `--hidden_layers 16` WAS already the last layer, but would silently stop being so for any other Teacher depth). Which intermediate layer(s) to add alongside it is a judgment call (see the project's own prior discussion of what different depths tend to encode) — but the last layer itself is not optional.

## Model design references

- `dev_notes/reference_big_llm_architecture_comparison.md` — full text of Sebastian Raschka's "The Big LLM Architecture Comparison" (source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison). Covers architectural details (attention variants, MoE, normalization placement, positional encoding, vocab size trade-offs, etc.) for DeepSeek V3/R1, OLMo 2, Gemma 3/4, Mistral Small 3.1/3, Llama 4, Qwen3, SmolLM3, Kimi K2, GPT-OSS, GLM-4.5/5, Qwen3-Next, MiniMax-M2, Kimi Linear, Olmo 3, Nemotron 3 Nano/Super, and more. Consult it when choosing or comparing base model architectures for experiments.
