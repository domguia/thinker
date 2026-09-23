# Base model candidates: small vocab / small lm_head + reasoning + size ladder

Context: for the indexed-attention / associative-recall experiments, we want a base
model whose vocabulary (and thus embedding + lm_head) is small relative to total
parameter count, so the compute and analysis are not dominated by the embedding
layer, while still being competitive at reasoning (math/logic) and — ideally —
available as a family spanning small to large sizes with a stable tokenizer, so we
can prototype cheap and scale the same code up later.

Findings below were verified directly against each model's `config.json` on
Hugging Face (vocab_size, hidden_size, num_hidden_layers) unless noted otherwise.
Compiled 2026-09-16.

## 1. First filter: small vocab / small lm_head (2025-2026 models, few-billion range)

| Model | Vocab | Size | Architecture | Verdict |
|---|---|---|---|---|
| **Zamba2-2.7B** (Zyphra, Feb 2025) | 32,000 | 2.7B | hybrid Mamba2 + a single shared attention block reused across layers | ❌ no reasoning benchmark reported at all (no GSM8K/MATH in the technical report or model card) |
| **LFM2** (Liquid AI, Jul 2025) | 65,536 | 350M / 700M / 1.2B / 2.6B | hybrid short-conv + GQA attention | ✅ kept |
| Falcon3-Mamba-7B (TII, Dec 2024) | 65,536 | 7B | pure Mamba, **no attention layers at all** | ❌ unusable if the mechanism must patch attention |
| **MiniCPM4 / 4.1** (OpenBMB, Jun 2025) | 73,448 | 0.5B or 8B only | dense, standard transformer (sparse attention InfLLM v2) | ✅ kept, but no mid-range size |
| Granite 4.0-H-Micro (IBM) | 100,352 | 3B | hybrid Mamba + attention | noted, not pursued further |
| MiniCPM5-2B (OpenBMB, Sep 2026) | 130,560 | 1B / 2B | dense, but vocab **inherited from the Qwen family** | ❌ contradicts the "OpenBMB = small vocab" assumption |
| Nemotron Nano 2/3 (NVIDIA) | 131,072 | 9B / 12B / 30B-A3B+ | hybrid Mamba-Transformer | ❌ too large, out of budget |
| Qwen 2.5/3 | 151,936 | wide range | dense / MoE | ❌ vocab too large (the original complaint that started this search) |
| **Qwen 3.5** (corrected 2026-09-21, was wrongly lumped with 2.5/3 above) | **248,320** (verified in `config.json`, constant across sizes: 0.8B and 9B checked) | 0.8B → 397B-A17B (MoE) | dense / MoE | ❌ even larger than Qwen 2.5/3 — coincidentally almost identical to this project's own Teacher-aligned vocab (248,077, see `experiments/distillation.md`'s muP tied-head runs) |
| Gemma 3 | 262,144 | 270M → 27B | dense | ❌ worst case: ~63% of the 270M model's parameters are in the embedding table |

## 2. Second filter: + reasoning performance

| Model | Vocab | Size | GSM8K | MATH500 | AIME24/25 |
|---|---|---|---|---|---|
| **LFM2.5-1.2B-Thinking** | 65,536 | 1.2B | 85.60 | 87.96 | —/31.73 |
| **MiniCPM4.1-8B** (reasoning mode) | 73,448 | 8B | 94.01–94.16 | 95.60–97.40 | 80.83–83.33 / 72.08–73.33 |
| MiniCPM4-8B (base, non-reasoning) | 73,448 | 8B | 91.51 | 78.60 | — |
| LFM2-2.6B (base) | 65,536 | 2.6B | 82.41 | — | — |
| MiniCPM4-0.5B | 73,448 | 0.5B | 52.08 (loses to Qwen3-0.6B's 61.71) | 29.60 | — |
| Zamba2-2.7B | 32,000 | 2.7B | *not reported* | *not reported* | — |

Added from Sebastian Raschka's "The Big LLM Architecture Comparison"
(see `reference_big_llm_architecture_comparison.md`):

- **Olmo-3-7B-Think** (Allen AI, Nov 2025) — vocab **100,278**, dense only (MHA +
  sliding window, no MoE/Mamba), AIME25 **70.7%**, matches Qwen3-8B on MATH, leads
  on HumanEvalPlus. Weights + training data + training code are **fully open**.

## 3. Third filter: + a size ladder (small → large) on one stable tokenizer

| Family | Available sizes | Vocab stability | Reasoning at the top end |
|---|---|---|---|
| **OLMo 2/3** (Allen AI) | 1B → 7B → 13B → 32B | **100,352 identical across every size** (1B through 32B; verified in each config.json) | Olmo-3-Think (7B/32B): AIME25 70.7% |
| LFM2 (v1) | 350M / 700M / 1.2B / 2.6B | 65,536 constant... | ...but **jumps to 128,000** starting at LFM2.5-2.6B / LFM2.5-8B-A1B |
| Ministral 3 (Mistral, Dec 2025) | 3B / 8B / 14B, reasoning variants promised | 131,072 (Tekken tokenizer) | ❌ vocab too large, same issue as Qwen |
| SmolLM3 | 3B only | 128,256 (Llama 3.2 tokenizer) | ❌ vocab too large, no size range |
| MiniCPM4 | 0.5B or 8B only | 73,448 constant | ❌ no mid-range size, not really a "ladder" |

## Models seen (Raschka article) but ruled out immediately — out of the "few-billion" budget

DeepSeek V3/R1 (671B), Llama 4 (400B), Kimi K2 (1T), GPT-OSS (20B/120B), GLM-4.5/5
(355B/744B), Qwen3-Next, MiniMax-M2, Kimi Linear, DeepSeek V3.2, Mistral 3 Large
(673B), Nemotron 3 Super/Ultra (120B/500B), Arcee Trinity Large, Gemma 4 (31B) —
all flagship-scale, too large for this use case.

## Final consolidated recommendation

- **To prototype cheaply and scale up without changing architecture or
  tokenizer**: **OLMo 2/3**. Start on OLMo-2-1B, port the same code to
  7B/13B/32B, finish on Olmo-3-Think for reasoning quality. The only family here
  guaranteeing an identical vocab from 1B to 32B, and fully open (weights + data +
  training code) — valuable for thesis reproducibility.
- **Best reasoning-per-vocab-size trade-off if a size ladder isn't needed**:
  **LFM2.5-1.2B-Thinking** (65,536 vocab, MATH500 87.96).
- **Best absolute reasoning at a still-reduced vocab, if 8B is acceptable**:
  **MiniCPM4.1-8B**.

## Ordre d'expérimentation retenu (2026-09-16)

1. **LFM2** (LFM2.5-1.2B-Thinking en priorité)
2. **OLMo 2/3** (ladder 1B → 32B, vocab stable)
3. **Qwen** (2.5/3, vocab 151,936) — réintroduit ici en tant que **point de
   comparaison**, malgré l'exclusion du filtre vocab en section 1 (le problème
   d'origine qui a motivé cette recherche). Passé en dernier, après avoir établi
   des résultats sur les deux familles à petit vocab. **Qwen 3.5 exclu du choix
   d'alias `qwen`** (vocab 248,320, vérifié 2026-09-21 -- voir correction en
   section 1 ci-dessus ; ne pas confondre avec Qwen 2.5/3 malgré le nom proche).

Alias de tokenizer/modèle utilisables via `--tokenizer` dans les scripts de
préparation de données et d'entraînement (résolus par
`core/model_families.py`, vérifiés directement sur Hugging Face le
2026-09-16) :

| Alias | ID Hugging Face | vocab_size (tokenizer) |
|---|---|---|
| `lfm2` (défaut) | `LiquidAI/LFM2-350M` (plus petite variante ; tokenizer identique à LFM2.5-1.2B-Thinking) | 64,400 |
| `olmo` | `allenai/OLMo-2-0425-1B` (plus petite variante de la famille OLMo-2) | 100,278 |
| `qwen` | `Qwen/Qwen3-0.6B` | 151,643 |

Tout autre nom (ex. `gpt2`, ou un chemin HF complet `org/repo`) est renvoyé
inchangé -- rétrocompatible avec l'usage existant de `--tokenizer`.

**Portée actuelle (2026-09-16) : uniquement le choix du tokenizer.** `Thinker`
(indexed-attention) et le modèle SFT de `distill/train_sft.py` restent
entraînés from scratch -- seul l'embedding/lm_head change de taille selon le
tokenizer choisi. Aucun poids pré-entraîné de LFM2/OLMo/Qwen n'est chargé ;
la "connaissance générale" de ces familles n'est donc pas encore héritée par
nos modèles (voir discussion connaissance générale / WikiText vs TinyStories
plus haut). Patcher l'attention d'un modèle pré-entraîné, ou l'utiliser comme
baseline externe telle quelle, sont des extensions futures non implémentées
ici.
