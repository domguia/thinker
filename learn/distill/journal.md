# Distill project journal

Freeform notes on the distillation work (`learn/distill/`) that don't fit the
other project journals: not Grid'5000 infra mechanics (`dev_notes/grid5000_usage.log.md`),
not comprehension/learning notes (`dev_notes/learning_journal.md`), and not
formal experiment results (`dev_notes/experiment.log.md`). This is for
decisions, findings, and context specific to the distillation pipeline and
methodology that are worth remembering later but don't belong in any of those.

## 2026-09-04

- Root-caused a long-standing bug: `precompute_teacher_targets.py`/`bench_teacher.py`
  forced `--dtype bfloat16` by default, which dequantizes the FP8 Teacher
  checkpoint to ~55.6GB before it even touches VRAM. On GPUs with <56GB VRAM
  this silently triggers CPU offload and a ~50-100x throughput collapse.
  Fixed by defaulting to `--dtype auto` (HF's `torch_dtype="auto"`, keeps the
  checkpoint's native stored dtype).
- Learned while testing the fix on an A100 80GB (`ecotaxe-2`, Nantes): A100
  has no native FP8 tensor cores (only H100+ does), so the FP8-quantized
  weights still get dequantized to bf16 at compute time regardless of the
  `--dtype` flag — this is a hardware limit, not something `--dtype auto`
  can avoid. It only matters in practice when the dequantized size doesn't
  fit in VRAM; on an 80GB card the ~55GB bf16 footprint still fits fine, so
  no CPU offload / no throughput collapse there either way.
- Started the FP8-vs-bf16 agreement evaluation (comparing to reference
  ~96%/98.9% top-token agreement numbers in `qwen3.8-27b-notes.md`) on
  `ecotaxe-2`, but the besteffort job got preempted after ~52 min (still
  stuck at 0/1184 shards loaded — NFS home read of a 29GB checkpoint was
  unexpectedly slow) by a higher-priority "night" reservation that occupies
  both `ecotaxe` nodes until ~02:46. A full Grid'5000-wide scan at that point
  found zero free Ampere+/Hopper GPU nodes anywhere.
- Pivoted to `abacus3` (Rennes), a free besteffort node with 4x RTX A5000
  (Ampere, sm_86 — compatible with the default torch env, no need for the
  `legacygpu` env used for V100/P100). Model checkpoints aren't present on
  Rennes home yet (only downloaded to Nantes so far) — need to transfer or
  re-download before this can proceed.

## 2026-09-05

- **Critical finding**: the `Qwen/Qwen3.8-27B-FP8` checkpoint has been
  loading with corrupted weights this whole time — `transformers`
  (5.17.0.dev0) never applies the FP8 `weight_scale_inv` dequantization
  scale for this checkpoint's quant scheme, silently discarding those
  tensors as "unexpected" and loading the raw FP8 byte patterns
  reinterpreted as bf16. Full writeup, evidence, and next steps in
  `qwen3.8-27b-notes.md`'s new "CRITICAL" section — this affects every past
  KD run (`EXP-003/004/005`) that used FP8-precomputed Teacher targets.
  Practical fix going forward: use the bf16 checkpoint (`Qwen/Qwen3.8-27B`,
  verified correct — confident, structured logits) for any new Teacher
  target precompute, until the FP8 loading gap is actually fixed.
- Also hit and fixed, in order, while chasing this down: (1) a real
  multi-GPU sharding bug on 2x L40S producing all-zero bf16 logits (worked
  around by using a single 80GB GPU, since the bf16 model needs ~55.6GB and
  doesn't fit on one 48GB L40S — ruled out as the cause of the FP8-vs-bf16
  mismatch once the same 0% agreement showed up even single-GPU on both
  sides); (2) besteffort preemption killing two separate jobs mid-run
  (`ecotaxe` nodes at Nantes, `abacus3` at Rennes) — Grid'5000-wide GPU
  availability can go from "found a free Ampere+ node" to zero free nodes
  anywhere within an hour; (3) a `pip install torchvision` in the Nantes
  `teacher311` env silently pulling in a mismatched torch build — false
  alarm in the end (the broken-looking `cuda.is_available()=False` was from
  checking on the frontend node, which has no GPU, not a real regression);
  (4) missing `pillow`/`torchvision` in the Nantes `teacher311` env causing
  `AutoProcessor.from_pretrained` to fail (Rennes' copy of the env had them,
  Nantes' didn't — per-site envs can drift).
