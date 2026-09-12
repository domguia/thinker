# Distillation onboarding — data pipeline

Context: before applying distillation to the custom Thinker architecture, we're
building practical expertise on standard distillation, starting with **data
selection & preparation**. Full background/rationale: `raw/Distill-getting-start.md`.

The student model will be trained from a **randomly initialized** checkpoint
(not fine-tuned from pretrained weights), so this is pretraining-scale data
work, not a small SFT fine-tune. Three dataset categories cover the three
aspects of the Thinker architecture: **reasoning**, **general** (language
modeling), and **retrieval** (aligned with the external KV/KB memory).
ARC-AGI was considered but dropped for now (eval-only later, via RE-ARC
synthetic augmentation if we revisit it).

## Status

| Script | Category | Source(s) | Local smoke-test |
|---|---|---|---|
| `prepare_reasoning_data.py` | Reasoning | `open-r1/OpenR1-Math-220k` (`default` config) | ✅ validated (gpt2 tokenizer, 20 examples) |
| `prepare_general_data.py` | General | `Salesforce/wikitext` (`wikitext-103-raw-v1`) + `roneneldan/TinyStories` | ⏳ written, not yet run to completion locally (blocked by poor local bandwidth/latency, not a code issue) |
| `prepare_retrieval_data.py` | Retrieval | `hotpotqa/hotpot_qa` (`distractor` config) | ⏳ written, not yet run to completion locally |
| `train_sft.py` | — (training loop) | reads any of the above JSONL outputs | ✅ validated (tiny random-init GPT-2 arch, 3.4M params, loss 10.86→9.52 over 30 steps on CPU) |

All three `prepare_*` scripts share the same design: stream examples from the
Hub (no full parquet download needed for small samples), format into ChatML
(`<|im_start|>user...assistant...<|im_end|>`) or raw text, tokenize to filter
by length, write `train.jsonl`/`val.jsonl` with a `text` field. `train_sft.py`
is dataset-agnostic — it just reads `text` from whichever JSONL it's pointed at.

**Tokenizer**: defaults to `Qwen/Qwen3-0.6B`. All Qwen3 sizes (0.6B → 235B-A22B)
share the exact same tokenizer (vocab ~151,936), so this choice doesn't lock us
into a model size — only the tokenizer files are downloaded here (~10-15 MB),
not any model weights.

## Sizing recap (no data downloaded to produce these — HF metadata API only)

Formula for offline Top-K=32 logit precomputation storage (int32 indices +
fp16 values + fp16 renorm scalar): **≈ 32×(4+2)+2 ≈ 194 bytes/token**.

| Dataset (config) | Download (parquet) | Tokens (full dataset, rough) | Top-32 logits storage |
|---|---|---|---|
| OpenR1-Math-220k (`default`) | 2.15 GB | ~150–250M (usable trace text) | ~30–50 GB |
| Salesforce/wikitext (`wikitext-103-raw-v1`) | 315 MB | ~130M | ~25 GB |
| roneneldan/TinyStories (`default`) | 1.0 GB | ~505M | ~98 GB |
| hotpot_qa (`distractor`, train) | 332 MB | ~140M | ~27 GB |

**Key finding**: Top-K=32 storage is ~90x the raw text size. Fine for these
volumes (tens of GB total), but would be infeasible (TBs) at a real
pretraining-scale "general" corpus like fineweb-edu (10B+ tokens) — that
category would need on-the-fly Teacher inference instead of precomputed
logits, or a much smaller general slice. Not a blocker now since we're doing
Sequence-level SFT first (no logits yet); revisit when we add logit KD.

fineweb-edu itself was tried as the "general" source and dropped for now —
resolving its file listing over a slow/high-latency link stalled repeatedly
(confirmed via cache inspection that this was a latency problem, not an
accidental large download: only ~100 KB had transferred after 10+ minutes).
WikiText-103 + TinyStories are much lighter (~1.3 GB combined) and can be
swapped back to fineweb-edu later once run from the cluster (fast network).

## Running at scale on Grid'5000 (manual — not run automatically by the assistant)

Per the `grid5000` skill's staged workflow, this is a **CPU-only, data-prep**
step — don't burn a GPU reservation on it.

```bash
# from a Grid'5000 frontend, if no CPU reservation is already up:
oarsub -I -n "distill-dataprep" -l host=1,walltime=2:00:00 -q default

# once on the node: lightweight env, no conda needed
python3 -m venv ~/venvs/distill-dataprep
source ~/venvs/distill-dataprep/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets

# sync the repo (from your local machine, or git clone if the repo is pushed)
# rsync -avz /home/djm/Workspaces/research/thinker/ <node>.grid5000.fr:~/thinker/
cd ~/thinker

# work on fast local scratch, copy results back to /home before walltime ends
mkdir -p /tmp/distill_data
```

Example commands (raise `--n_samples` well above each dataset's row count to
pull everything via streaming — the loop just stops when the stream ends):

```bash
# Reasoning (~93.7k rows in the default config)
python learn/distill/prepare_reasoning_data.py \
  --n_samples 100000 --max_length 4096 \
  --out_dir /tmp/distill_data/reasoning

# General (start with a bounded subset per source; raise later if needed)
python learn/distill/prepare_general_data.py \
  --n_samples 20000 --max_length 2048 \
  --out_dir /tmp/distill_data/general

# Retrieval (~90.4k rows in the distractor train split)
python learn/distill/prepare_retrieval_data.py \
  --n_samples 100000 --max_length 4096 \
  --out_dir /tmp/distill_data/retrieval

# copy results back to persistent storage before walltime expires
cp -r /tmp/distill_data ~/thinker/data/distill_cluster_run
```

Then release the CPU node (`oardel <job_id>`) as soon as data prep is done —
don't hold a reservation idle.

## Teacher download & inference benchmark

`download_teacher.py` and `bench_teacher.py` prepare the next phase (logit
KD with a real Teacher) without running anything automatically -- you launch
these yourself on Grid'5000.

Candidate Teacher: **`Qwen/Qwen3.8-27B-FP8`** (~30.9 GB, needs ≥~31 GB free
VRAM). Model-specific details (architecture, VRAM/quantization notes,
reasoning-effort behavior relevant to `bench_teacher.py`) live in
[`qwen3.8-27b-notes.md`](./qwen3.8-27b-notes.md); the concrete experiment
matrix and GPU-specific instructions to validate it on Grid'5000 are in
[`qwen3.8-benchmark-plan.md`](./qwen3.8-benchmark-plan.md).

Staged workflow (matches the `grid5000` skill's CPU-download / GPU-compute
split):

```bash
# 1. Download on the existing CPU reservation (paradoxe-7, Rennes) -- no GPU needed
python learn/distill/download_teacher.py \
  --repo_id Qwen/Qwen3.8-27B-FP8 --local_dir /tmp/teachers/Qwen3.8-27B-FP8
# then copy the snapshot to persistent storage (home or Group Storage) before
# the CPU job's walltime ends, since /tmp is wiped at job end.

# 2. GPU reservation for the benchmark -- pick a site, see availability note below.
oarsub -I -n "distill-teacher-bench" -l gpu=1,walltime=2:00:00 -p "gpu_model = 'H100'"
python learn/distill/bench_teacher.py \
  --model_dir /path/to/Qwen3.8-27B-FP8 --top_k 32 --out_file teacher_bench_results.json
```

GPU site availability for this Teacher's ~31 GB VRAM requirement (checked
2026-09-03, changes constantly) is in
[`qwen3.8-27b-notes.md`](./qwen3.8-27b-notes.md#gpu-site-choice-checked-2026-09-03).

`bench_teacher.py` reports tokens/sec generation throughput on a few sample
prompts, and measures actual Top-K=32 logit storage bytes/token to check
against the ~194 bytes/token formula above.

## Teacher target precomputation (Top-K logits + optional hidden states)

`precompute_teacher_targets.py` turns a prepared JSONL (any of the three
categories) into a Teacher-targets dataset for offline KD: one forward pass
per example through the Teacher, storing:
- per-token Top-K logit indices/values plus a residual log-sum-exp scalar
  (the aggregated mass of every non-Top-K token) so the full softmax
  denominator is still exactly reconstructable for the KL loss.
- optionally, per-token hidden states for one or more layers
  (`--hidden_layers none|last|all|<comma-separated indices>`, decided at
  inference time) for feature-level distillation -- meant for small samples
  only, exact training use still open, this just makes extraction available
  to experiment with. Much heavier than Top-K logits (one full hidden_dim
  vector per token per layer vs. K values).

Output is one compressed `.npz` per input file (flat `(total_tokens, ...)`
arrays + an `offsets` array marking example boundaries, shared across logits
and any extracted hidden-state layers).

Validated locally (`gpt2` fallback path in `load_model_and_tokenizer`):
- Top-K only, K=8, 3 tiny examples: measured storage exactly matched the
  formula (50.0 bytes/token, i.e. `K*6+2`), and the residual reconstructs
  the softmax denominator correctly (spot-checked: `logsumexp(topk_values)
  + exp(residual)` recovers the same log-denominator used to produce it).
- `--hidden_layers last` and `--hidden_layers all`: correct shapes
  (`(num_tokens, 768)` per layer for gpt2, 13 layers for `all` = 12 blocks +
  the embedding output at index 0).

```bash
# Top-K logits only
python learn/distill/precompute_teacher_targets.py \
  --input_file /tmp/distill_data/reasoning/train.jsonl \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --top_k 32 --out_file /tmp/distill_data/reasoning/train_topk32.npz

# + last hidden layer, on a small sample
python learn/distill/precompute_teacher_targets.py \
  --input_file /tmp/distill_data/reasoning/sample.jsonl \
  --model_dir /path/to/Qwen3.8-27B-FP8 \
  --top_k 32 --hidden_layers last \
  --out_file /tmp/distill_data/reasoning/sample_targets.npz
```

No batching yet (one example at a time) -- run `bench_teacher.py` first to
get real per-example latency on the Teacher, and only add batching if that
throughput turns out to be a bottleneck for the full dataset size.

## Student size vs. training duration (to decide the target model size)

Since Top-K logits are precomputed offline, student training compute is the
same as a from-scratch pretraining run of the student alone (only the loss
changes: KL instead of plain CE) -- so the standard `FLOPs ≈ 6 × N × D`
approximation applies (N = params, D = training tokens). Distillation
typically needs fewer tokens than pure CE pretraining thanks to the Teacher's
denser signal, so **D = 10 × N** is used below as a moderate assumption
(vs. the Chinchilla-standard 20×N) -- not rigorously established from the
literature reviewed so far, treat as a starting point. Duration scales
linearly with D: double every number below for a 20×N (Chinchilla-like)
budget, or halve for a more aggressive 5×N bet.

**Updated 2026-09-05 with real measured throughput** (superseding the FLOP/MFU-assumption table below): `EXP-003/004/005` give one real single-GPU wall-clock data point per tier, using the actual Teacher-vocab-aligned total param count (core + tied head) and the real `train_sft.py` KD loop (`--batch_size 4 --block_size 512`, no `torch.compile`, no multi-GPU):

| Student size (core) | Total params (core+head) | GPU (single) | Measured throughput | D = 10×N (tokens) | Wall-clock @ D |
|---|---|---|---|---|---|
| 40M | 41.1M | L40S (`abacus26`) | 10,807 tok/s (409,600 tok / 37.9s) | 411M | ~10.6 h |
| 150M | 406.2M | A100 40GB (`abacus21`) | 4,491 tok/s (409,600 tok / 91.2s) | 4.06B | ~251 h (10.5d) ❌ |
| 500M | 810.8M | L40S (`abacus26`) | 5,300 tok/s (409,600 tok / 77.3s) | 8.11B | ~425 h (17.7d) ❌ |

❌ = exceeds Grid'5000's ~1-week single-reservation limit (would need checkpoint/resume across multiple besteffort reservations, now implemented — see `--checkpoint_every`/`--resume_from`).

**Why these numbers are much higher than the old table below**: backing out
achieved TFLOPS from these measurements (`6 × N × tok/s`) gives roughly
1.5% MFU at 40M-core, 3.5% at 150M-core (A100), 14% at 500M-core (L40S) —
far below the 15-35% assumed in the old table, because `--batch_size 4
--block_size 512` (2048 tokens/step) is tiny: too small to keep the GPU
compute-bound, especially at low param counts where per-step Python/kernel-
launch overhead dominates. MFU clearly improves with model size here (more
work per launched kernel), so these single-point measurements shouldn't be
read as a fixed hardware ceiling — a larger batch size (untested so far)
would very likely close much of this gap, and is worth trying before
committing to a long real training reservation.

**Not measured yet, deliberately not estimated**: 1B/3B-core rows, and every
GPU type other than the two above (P100, V100, A40, H100, multi-GPU
configurations). The old table's numbers for those cells came from a fixed
MFU assumption now shown to be wrong by ~2-20×, and `train_sft.py` has no
multi-GPU/distributed support yet, so a "×N GPUs" column would need that
implemented (and benchmarked) before it means anything. Given the trend of
increasing MFU with model size, 1B/3B-core would likely fare better than a
naive linear extrapolation from the 40M-core point — but that's a guess,
not a measurement. Get a real batch-size sweep and a real 1B-core data point
before trusting any duration estimate at that scale.

### GPU-dependent estimate for the 500M-core credibility target (2026-09-12)

Explicit estimate requested for the 500M-core tier (810.8M total, the tier
chosen above to match Qwen3.5-0.8B's footprint), broken down **by GPU
model** since throughput depends heavily on it. Two scenarios, since the
only real data point we have (`EXP-005`, L40S) is a deliberately
unoptimized config (`--batch_size 4 --block_size 512`, no `torch.compile`,
14% MFU):

**Peak bf16/fp16 Tensor Core FLOPS (dense, no sparsity — official NVIDIA
datasheets)**: L40S 362 TFLOPS, A100 (40/80GB) 312 TFLOPS, H100 (SXM) 989
TFLOPS, H200 989 TFLOPS (same GH100 compute die as H100 — its advantage is
memory bandwidth/capacity, not raw FLOPS, which mostly helps by allowing a
larger batch size before running out of VRAM, not by raising the FLOPS
ceiling itself).

| GPU | Scénario | Débit estimé | Wall-clock pour D=8.11B tokens (10×N) |
|---|---|---|---|
| L40S | **Mesuré** (`EXP-005`, batch=4/block=512, 14% MFU) | 5,300 tok/s (réel) | ~425 h (17.7j) ❌ |
| L40S | Optimiste (batch/config tunés, ~30% MFU — atteignable sur Ada avec FlashAttention) | ~22,300 tok/s (estimé) | ~101 h (4.2j) |
| A100 | Extrapolé au même config non-optimisé (mise à l'échelle par le ratio de FLOPS crête, **non mesuré à cette taille**) | ~4,570 tok/s (extrapolé) | ~493 h (20.6j) ❌ |
| A100 | Optimiste (~35% MFU, FlashAttention2, plage bien établie en littérature pour ce type d'entraînement) | ~22,400 tok/s (estimé) | ~101 h (4.2j) |
| H100 (SXM) | Extrapolé au même config non-optimisé | ~14,470 tok/s (extrapolé) | ~156 h (6.5j) ⚠️ proche de la limite |
| H100 (SXM) | Optimiste (~40% MFU — dans la plage 40-55% citée pour du pré-entraînement dense bien réglé sur H100) | ~81,300 tok/s (estimé) | **~28 h (1.15j)** |
| H200 | Optimiste (même calcul crête que H100 ; la VRAM/bande passante en plus permettrait probablement un batch encore plus gros, donc au moins aussi bon) | ≥ ~81,300 tok/s (estimé) | ≤ ~28 h |

Pour le budget de tokens : `D=10×N` (8.11B, colonne ci-dessus) est
l'hypothèse centrale déjà retenue ; borne basse `D=5×N` (4.05B tokens, ~moitié
des heures ci-dessus) et borne haute Chinchilla `D=20×N` (16.2B tokens,
~double) si l'hypothèse centrale s'avère trop optimiste sur la qualité
atteinte.

**Réserves importantes sur ces chiffres** :
- Seule la ligne L40S "mesuré" est une vraie mesure — toutes les autres
  lignes (A100/H100/H200 à cette taille, et les scénarios "optimiste" pour
  toutes les GPU) sont des **extrapolations**, pas des mesures. Le projet a
  déjà appris cette leçon une fois (l'ancienne table basée sur des
  hypothèses de MFU fixes s'est révélée fausse d'un facteur 2-20× une fois
  mesurée réellement) — donc traiter ce tableau comme une fourchette de
  planification, pas une garantie.
- L'extrapolation A100/H100 "non-optimisé" suppose que le MFU à
  configuration fixe (batch=4) se comporte proportionnellement au ratio de
  FLOPS crête entre GPU — une simplification : la bande passante mémoire et
  le coût de lancement des kernels par étape ne suivent pas forcément le
  même ratio que le calcul brut, donc ces cellules sont les moins fiables du
  tableau.
- Le scénario "optimiste" est plausible (la littérature 2024-2025 rapporte
  couramment 40-55% MFU sur H100 pour du pré-entraînement dense bien réglé
  avec FlashAttention2/3 et un batch suffisant) mais **n'a jamais été testé
  sur ce pipeline** — `train_sft.py` n'a ni `torch.compile` ni support
  multi-GPU actuellement, et le batch=4 actuel est probablement bien en
  dessous de ce qu'il faudrait pour s'approcher de ce régime.

**Action recommandée avant de réserver un long créneau** : un balayage de
`--batch_size` (et éventuellement `torch.compile`) au palier 500M-core sur
un nœud libre (L40S ou A100), quelques heures seulement — remplace toutes
les extrapolations ci-dessus par un vrai point de mesure, et dira
directement si l'estimation "optimiste" (~1-4 jours selon le GPU, largement
dans la limite d'1 semaine de Grid'5000) est atteignable ou si le
"mesuré/non-optimisé" (~3-20 jours, nécessitant checkpoint/resume déjà
implémenté) est la réalité à laquelle se préparer.

**Mise à jour critique (experiment-manager, 2026-09-13) — le point mesuré `EXP-005` n'était pas comparable au peak bf16 cité** :
`train_sft.py` crée le modèle en **fp32 complet** (`AutoModelForCausalLM.from_config` sans dtype précisé, aucun `autocast`/`GradScaler` dans le fichier) — donc les 362/312/989 TFLOPS "peak bf16 Tensor Core" utilisés partout ci-dessus comme dénominateur du MFU ne sont **pas** le bon plafond pour un run fp32 (débit très inférieur, TF32 automatique sur Ampere+ mis à part, mais encore ~4-8× sous le peak bf16 selon la génération). Le MFU mesuré jusqu'ici (1,5-14%) compare donc un débit fp32 réel à un plafond bf16 théorique — **pas une vraie mesure de marge de progression**.

Confirmé en re-testant le sweep batch_size au palier 500M-core sur L40S 44 Go, en fp32 : `batch=4` → 3 238 tok/s (vs. 5 300 tok/s dans `EXP-005` — écart probablement dû à une contention GPU partagée, pas à réinterpréter comme une régression), `batch=8` **OOM** (tente d'allouer 3,79 Go alors qu'il ne reste que 2,15 Go libres, `batch=4` utilise déjà ~40/44 Go), `batch=16` OOM direct. **La marge de scaling du batch_size est quasi nulle en l'état** — le vrai levier n'est pas le batch size mais :
1. **bf16/mixed-precision** (`torch.autocast`) — divise la mémoire par ~2 (débloquant un vrai scaling du batch) **et** utilise réellement les Tensor Cores (rendant le MFU mesuré comparable au peak cité pour la première fois). En cours d'implémentation.
2. **La tête de vocabulaire Teacher (248k tokens) domine la mémoire** — le tenseur de logits `(batch, block_size, vocab)` à `block_size=512` est déjà énorme en fp32, avant même de compter le calcul de la loss KD dessus. Indépendant de (1), cumulable — si (1) seul ne suffit pas, la piste littérature est un calcul de cross-entropy/KL **fusionné par chunks** qui ne matérialise jamais le tenseur de logits complet (ex. Liger-Kernel, "Cut Your Losses", NeurIPS 2024/2025) plutôt que de réduire le vocabulaire lui-même (qui casse l'alignement avec le Teacher).

**Conséquence pour toutes les estimations ci-dessus** : le tableau GPU-dépendant et les scénarios "mesuré" vs. "optimiste" restent la bonne structure de raisonnement, mais **le point "mesuré" L40S doit être refait en bf16** avant d'être considéré comme fiable — le nombre fp32 actuel (5 300 ou 3 238 tok/s selon le run) sous-estime probablement le vrai débit atteignable d'un facteur significatif, dans une direction inconnue tant que le point bf16 n'existe pas. Ne pas figer de décision de réservation longue sur les chiffres fp32 actuels.

<details>
<summary>Old table (2026-09-03, FLOP/MFU-assumption based — superseded above, kept for history)</summary>

MFU (achieved vs. peak FLOPs) assumptions, given no FlashAttention on
Pascal/Volta and an unoptimized training loop: P100/V100 ~15-20%, A100/A40
~30%, H100 ~35%.

| Student size (core) | P100×2 | V100×4 | A100×3 (`abacus21`) | A40×2 | H100×4 (`abacus27`) |
|---|---|---|---|---|---|
| 40M | ~15.7 min | ~1 min | ~21 s | ~1.1 min | ~4 s |
| 150M | ~59 h (2.5d) | ~3.75 h | ~1.3 h | ~4.2 h | ~16 min |
| 500M | ~655 h (27d) ❌ | ~42 h (1.7d) | ~14.8 h | ~46 h (1.9d) | ~3 h |
| 1B | impractical | ~167 h (7d) ⚠️ | ~59 h (2.5d) | ~185 h (7.7d) ❌ | ~12 h |
| 3B | impractical | impractical | ~22.3 d ❌ | impractical | ~4.5 d |

❌ = exceeds Grid'5000's ~1-week single-reservation limit. ⚠️ = right at it.

This table also used `N` = the *total* size implied by the old (misleading,
gpt2-vocab) numbers, not the real Teacher-vocab-aligned totals — both errors
(wrong N, and an MFU assumption ~2-20× too optimistic) compounded to make
every cell here far too fast.

</details>

**Sizing methodology (settled 2026-09-04): tiers are core size, not total
size.** Once the student's vocabulary is aligned with the Teacher's
(`Qwen3.8-27B-FP8`'s tokenizer, **248,077** tokens — not the ~151,936 figure
that's correct for plain Qwen3 *text* models; this checkpoint is a larger-
vocab Qwen3.5 VLM), the embedding/lm_head table becomes large enough to
distort what a tier label means. What we actually care about evaluating is
the **core** (the transformer blocks + positional embedding — the part that
does the actual reasoning/computation and that scaling decisions are really
about), not the vocab-sized table, whose size is basically fixed once the
tokenizer is chosen and has nothing to do with model capacity in the sense
we're scaling. So: **tier labels below refer to core params.** The
embedding/lm_head ("head") size is reported alongside per run (see
`train_sft.py`'s startup line, e.g. "406.2M params total = 152.2M core +
254.0M head") because it's the *majority* of the model at small core sizes
(dominant enough to be misleading if ignored) but becomes proportionally
negligible as core size grows (a fixed ~254M-token table, tied, is ~62% of a
406M total but would be under 10% of a 3B-core model) — not worth a
per-tier breakdown once it stops being the majority.

**muP and the tied-head compromise**: canonical muP (Yang et al., *Tensor
Programs V*) requires untying the LM head from the input embedding so each
can get its own init/scaling rule (see `learning_journal.md`'s "Weight
tying" entry for why). With this project's large Teacher-aligned
vocabulary, untying would **double** the head table, making it dominate
even more at small-to-mid core sizes — directly working against the point
of tracking core size separately. Deliberate compromise: `train_sft.py`'s
`--mup` keeps the head **tied** by default (pass `--mup_untie_head` for
canonical muP instead), still applies muP's init/LR scaling to the core and
the logit/`width_mult` rescaling at the loss, but skips the readout
zero-init/untie step. Not canonical muP — a documented, deliberate
deviation.

**Decision (2026-09-03, methodology settled 2026-09-04)**: validated the
pipeline at the **40M-core** tier (`n_layer=4, n_embd=160, n_head=4`; ~1.4M
core + ~39.7M tied head with the Teacher's vocab, ~41.1M total measured),
then the **150M-core** tier (`n_layer=12, n_embd=1024, n_head=16`; 152.2M
core + 254.0M tied head, 406.2M total measured, `logs/EXP-004-*`) — the LR
found for muP's base width (40) transferred unchanged (no retuning) across
a 25.6× width jump, converging cleanly (best combined loss 0.29 vs. the
40M-core run's 0.67, on the same 200-example validation slice), then the
**500M-core** tier (`n_layer=25, n_embd=1280, n_head=16`; 493.2M core +
317.5M tied head, 810.8M total measured, `logs/EXP-005-*`) — same LR reused
again unchanged, 80× width jump from the tuning base, still no instability
(best combined loss 0.38 on the same 200-example slice). Old "50M"
tier dropped (too close to the corrected 40M figure to be a distinct step).
Next: continue up through the remaining core tiers (1B, 3B core) once
there's a reason to move past pipeline validation into a real training
budget.

## Techniques to reduce training cost / facilitate learning (2026-09-13)

Question raised: rather than only tuning batch size/MFU on the same fixed
recipe, can we combine techniques from the "low-resource-compute
distillation" literature to genuinely shrink the token/compute budget
needed, so the first "credibility" checkpoint at a relevant scale
(500M-core, §above) is reached faster? Literature review below, ranked by
expected value vs. implementation cost for **this specific project**.

### 1. Initialize the tied embedding/head from the Teacher's own table (cheap, high value, do first)

The tied embedding+lm_head is already the **majority** of total params at
small-to-mid core sizes (up to ~62% at 150M-core, per the sizing section
above) — currently random-initialized like everything else. Since our
tokenizer *is* the Teacher's own (248,077-token vocab, chosen specifically
for alignment), the Teacher's own embedding matrix is a free, directly
compatible source of a much better starting point than random init for a
huge fraction of our parameters. Concretely: project the Teacher's
`d_model`-wide embedding rows down to our student's (smaller) embedding
dimension (a random projection or a truncated SVD/PCA over the Teacher's
embedding matrix, computed once, offline, on CPU) instead of initializing
that table from scratch. This doesn't require the student and Teacher to
share an architecture — it only reuses the token representation, which is
architecture-agnostic. Cheap (one offline matrix operation, no GPU
training needed to produce it) and directly reduces the *effective* work
the embedding/head must learn from data.

### 2. Structured pruning + distillation of the *reasoning core* itself — does NOT apply here, flagged explicitly

The highest-profile "facilitation" technique in the literature is NVIDIA's
**Minitron** (pruning + light distillation retraining, up to **40x fewer
tokens** than training from scratch) and **Sheared LLaMA** — both derive
the student's weights *structurally* from the teacher's own weight
matrices (depth or width pruning), then use a comparatively short
distillation pass to recover accuracy. **This only works when student and
teacher share the same architecture family** (the student literally is a
subset of the teacher's layers/matmuls). Our Thinker core (recurrent
weight-shared loop, no FF, external hierarchical KV memory) is a
deliberately different architecture from the Teacher's own dense/hybrid
gated-DeltaNet stack — there is no substructure of the Teacher's weights
that *is* a Thinker. So the core 40x-fewer-tokens result **does not
transfer to our reasoning core**, only to the embedding/head (point 1
above, which shares no architectural assumption). Worth stating explicitly
so this isn't chased as a false hope for the core itself.

### 3. Data quality over quantity — Teacher-generated "textbook-quality" + on-policy data (high value, moderate cost)

Two literature threads converge on the same actionable idea:
- **Phi / "Textbooks Are All You Need"** (Gunasekar et al. 2023): phi-1
  (1.3B) reached strong code-benchmark performance (50.6% HumanEval)
  training on **~7B tokens** (a mix of filtered web + LLM-synthesized
  "textbook-quality" exercises), ~100x less data than typical models of
  similar capability at the time — the reported driver is data quality,
  not scale.
- **On-policy distillation (GKD, Agarwal et al. 2024; MiniLLM, Gu et al.
  2023)**: training the student on its own (or the Teacher's) generated
  sequences with Teacher feedback, rather than only static-corpus
  logit-matching, gives a reported **1.7-2.1x sample-efficiency gain**
  over standard KD across several task types.
- **Combined, concrete plan for this project**: since the Teacher
  (27B) is available locally, use it to *generate* task-aligned synthetic
  training data — reasoning traces and retrieval/multi-hop examples
  specifically shaped like the capabilities Thinker's architecture targets
  (cf. `data/kb_chain_retrieval.py`-style structure, but real text) —
  instead of relying only on generic scraped corpora (wikitext,
  TinyStories) that don't specifically exercise the KB-retrieval mechanism
  at all. This is both "textbook-quality" (curated/synthesized, not raw
  web noise) and "on-policy" (the exact model whose knowledge we're
  distilling produced it), and directly reduces the token budget needed
  versus a generic corpus by concentrating supervision on what the
  architecture actually needs to learn.

### 4. Sequence-length curriculum (cheap, proven precedent, low risk)

"Dataset Decomposition" (Pouransari et al., NeurIPS 2024): a variable
sequence-length curriculum (short sequences first) reports **up to 6x
faster training to target accuracy**, and instability is empirically
correlated with long sequences especially early in training. Directly
combinable with the curriculum-learning practice already validated
elsewhere in this project (`n_facts`, `n_hops`) — same idea, applied to
`--block_size` instead of task difficulty: start `train_sft.py` at a
short `block_size`, increase once the model is past its early unstable
phase, rather than fixed length throughout. Cheap to implement, no new
data or architecture change needed.

### Priority for this project, given limited time

1. **Embedding/head init from the Teacher's own table** — do this first,
   it's a pure offline preprocessing step, no training-loop change, and
   the head is disproportionately expensive to learn from scratch at our
   scale.
2. **Teacher-generated task-aligned synthetic data** (reasoning +
   retrieval) — moderate effort (need a generation pass through the
   Teacher, already have `bench_teacher.py`/`precompute_teacher_targets.py`
   plumbing to build on) but likely the single highest-leverage change to
   the actual token budget `D`.
3. **Sequence-length curriculum on `block_size`** — cheap add to the
   training loop, low risk, do alongside the batch-size sweep already
   requested from experiment-manager.
4. Do **not** pursue core-architecture weight transfer/pruning from the
   Teacher (point 2 above) — confirmed non-applicable given the
   architectural mismatch, not worth the implementation time.

## Reference target: same-family smaller models (2026-09-12)

The project's actual end goal is to demonstrate **reliable** distillation of
the Teacher (`Qwen/Qwen3.8-27B-FP8`) into our custom Thinker architecture —
"reliable" meaning a credible, honestly-reported quality bar, not
necessarily matching the Teacher's own performance. Question raised: at what
student scale is a comparison actually meaningful, and are same-family
smaller released models a valid reference?

**Confirmed via HF (`huggingface.co/collections/Qwen/qwen35`,
`Qwen/Qwen3.5-0.8B`)**: within the Qwen3.8 line itself there is no small
sibling — 27B is the smallest dense-ish option (`qwen3.8-27b-notes.md`
line 140-144), the next ones jump to 360 GB / 4.9 TB. The **previous**
generation, Qwen3.5, does have a real small-dense ladder: 0.8B, 2B, 4B, 9B,
27B, then MoE variants (35B-A3B, 122B-A10B, 397B-A17B) — same lineage,
same general hybrid-attention design philosophy (gated DeltaNet + regular
attention, sometimes + MoE), just one generation back. **Using Qwen3.5's
small siblings (starting with the 0.8B) as reference points is credible and
is in fact the best available same-family comparison**, precisely because
Qwen3.8 skips straight past this size range.

**Caveats to keep in mind when actually comparing**:
- `Qwen3.5-0.8B` is a **VLM** (`Qwen3_5ForConditionalGeneration`,
  image-text-to-text) — for a fair comparison against our text-only Thinker,
  only use its text benchmarks (MMLU-Pro 29.7%, C-Eval 46.4%, IFEval 52.1%),
  ignore vision scores (MMMU, MMBench).
- One generation back means the training recipe/data are not identical to
  what produced our 27B Teacher — treat its scores as a **quality bar**
  (what's achievable at that param count from a broadly comparable
  lineage/design), not as a hard target or a budget estimate.
- Our own "core vs. head" accounting (see sizing methodology above) makes
  raw total-param comparisons slightly apples-to-oranges: our tied,
  Teacher-vocab-aligned head (248,077 tokens) is a much larger fraction of
  total params at small core sizes than a typical ~150k-vocab model's own
  embedding table would be. Report core size alongside total size when
  comparing, don't just match raw total params.

**On budget** (the actual question asked: what did it cost to produce these
smaller family siblings, as a reference for our own budget?): Qwen's own
technical report for the prior Qwen3 generation (arXiv 2505.09388) confirms
their smaller models (0.6B/1.7B/4B/8B/14B/30B-A3B) are **not** trained from
scratch at Chinchilla-scale — they're produced via a "Strong-to-Weak
Distillation" pipeline (off-policy + on-policy phases) from the flagship,
explicitly to cut the compute needed versus training each size from scratch.
This is a real precedent for the same bet this project is already making
(`D = 10×N` instead of Chinchilla's `20×N`, see the sizing table above) —
it doesn't give us their exact token counts, but it does mean their smaller
models' training cost is not directly comparable to their own pretraining
cost either, so it's not usable as a literal budget number regardless of
generation match.

**Correction (2026-09-13) — a single "total params" comparison against
Qwen3.5-0.8B is not the right unit, three separate axes matter:**

1. **Trainable parameter footprint** (storage/capacity) — embedding+head
   (tied to the Teacher's vocab) + the core register/attention projections
   (shared once, reused every loop iteration) + the tiny shared compressor
   (`LevelCompressor`'s pooling query + `intrablock_pos`, block_size×d_model,
   negligible) + output streams. **The KB itself adds no dedicated
   parameter block** — unlike the original `ToyThinker`'s `static_mem`
   (a `nn.Embedding` sized to the knowledge base), `IndexedThinker` embeds
   KB leaves through the *same* shared `self.embed` table used for
   input/query tokens (`core/indexed_thinker_model.py:105,157,160`) — so
   there is no separate "KB parameter count" to add on top of embedding+core,
   and no double-counting risk either way.
2. **Effective compute depth** (FLOPs per forward pass) — this is where a
   flat total-param comparison misleads. Our core is the **same weights**
   applied `N_step` times (looped/recurrent), not `N_step` independent
   layers — so the parameter budget doesn't grow with depth the way a
   standard stacked transformer's does, only the compute (and activation
   memory) does. The correct **per-unit** comparison for the core is
   therefore **one Qwen layer's parameter budget, not the whole model's** —
   exactly analogous to how our shared core block is applied once per
   iteration, the way one Qwen layer's weights are applied once per layer
   in its stack. Pulled the exact numbers from `Qwen/Qwen3.5-0.8B`'s
   `config.json`: `vocab_size=248,320` (nearly identical to our own
   Teacher-aligned 248,077 — same tokenizer lineage), `hidden_size=1024`,
   `tie_word_embeddings=true` → embedding/head ≈ 248,320×1024 ≈ **254.3M**
   (~28-32% of the ~0.8-0.9B total, depending which public figure is used),
   leaving **~546-646M core over 24 layers ≈ ~23-27M/layer average**
   (rough average only — the 24 layers are heterogeneous: 18 gated-DeltaNet
   linear-attention layers interleaved with 6 full-attention layers every
   4th layer, no MoE here so at least no expert-count asymmetry to further
   complicate it).
   - **Practical implication**: if our core is sized closer to
     "~1 Qwen layer" (~25M) rather than "~1 Qwen model" (~550-650M, which is
     roughly where the already-measured 500M-core tier actually sits — much
     closer to *the whole Qwen stack* than to *one Qwen layer*), and
     `N_step` is chosen per the ~2-4×(target reasoning depth) heuristic
     already in `dev_notes/indexed_attention_experiment_plan.md`
     (Phase 1quater), the **total trainable parameter footprint could land
     far below Qwen3.5-0.8B's ~0.8-0.9B** (roughly embedding ~254M matched
     to Qwen's own `hidden_size=1024` + core ~25M + negligible extras ≈
     ~280-300M total) while remaining **FLOPs-comparable in the forward
     pass** to Qwen's 24-layer stack — a substantially smaller stored-weight
     footprint for a comparable compute depth, a genuine efficiency
     argument *for* the shared-core thesis (§-1's whole point: capacity
     doesn't need to scale with depth if it lives in reusable weights +
     external KV, not per-layer FF).
   - **Important caveat, not to over-claim**: FLOPs-equivalence is **not**
     the same as capability-equivalence. A small core looped 48-96 times
     applies the *same* learned function repeatedly (bounded per-step
     transformation diversity), whereas 24 *distinct* Qwen layers each learn
     a *different* transformation — matching compute doesn't guarantee
     matching what the compute can express. This is exactly the open
     empirical question the Universal Transformer / Looped Transformer
     literature flags (cited in the `indexed_attention_experiment_plan.md`
     N_step section) and exactly what Phase 1quater's `N_step` sweep is
     designed to test — treat the smaller-footprint framing above as a
     hoped-for outcome to verify, not an assumption to design around yet.
3. **KB inference/activation memory** — real, but **not a parameter
   count at all**: building the hierarchy over the KB's leaves produces
   activations (per-level compressed K/V) that must be held during
   `attend()`, scaling with the number of facts × d_model × node count —
   this is architecturally closer to a **context-length / KV-cache memory
   cost** in a standard transformer than to "model size." Track and report
   it separately (tokens/facts held × bytes, like a KV-cache sizing
   calculation) rather than folding it into a parameter-count comparison.

**Resulting decision — target scale for the first "credibility" checkpoint**:
keep the already-validated **500M-core tier** (810.8M total measured,
`EXP-005`) as the concrete, already-working checkpoint to reach first (no
need to invent a new tier just for this) — its total footprint still lands
close to Qwen3.5-0.8B's, which remains useful for a *quality-bar*
comparison (axis 1 above). But **do not read that size match as evidence
we need a ~500M core to be "as capable" as Qwen3.5-0.8B** — per axis 2,
the more relevant comparison for the core specifically is against one
Qwen layer (~25M), with `N_step` doing the work depth normally would; a
follow-up, smaller-core + larger-`N_step` configuration is worth testing
once Phase 1quater has real N_step-sweep data, as a genuinely more
parameter-efficient way to reach a comparable point. Keep the small-scale
Indexed-Attention-mechanism experiments (`dev_notes/indexed_attention_*`)
separate from this quality-bar comparison either way: the former validate
the *mechanism* at deliberately tiny/controlled scale (and carry the
transfer caveats documented there), the latter is about a *credible
end-to-end result* at a scale chosen to match a real external reference —
don't conflate the two when deciding "is our current scale big enough".

## Next steps

- Validate `prepare_general_data.py` / `prepare_retrieval_data.py` end-to-end
  (blocked locally by network; the cluster run above will also validate them).
- Once real data is in hand: extend `train_sft.py` (or a new script) with
  Teacher logit KD (KL loss on Top-K=32) as described in
  `raw/Distill-getting-start.md`.
- Decide target student model size to size the total token budget precisely
  (Chinchilla-style ~20 tokens/param baseline, likely reducible thanks to
  denser distillation supervision — see literature pointers in
  `raw/Distill-getting-start.md`).
