"""Compare two precomputed Top-K target files for the SAME model at two
different precisions/quantization levels (e.g. FP8 vs bf16), to sanity-check
our agreement evaluation methodology against published reference numbers
(see qwen3.8-27b-notes.md's quantization-agreement table: ~96% top-token
agreement at 4-bit, ~98.9% at 8-bit, vs. full bf16 precision).

Unlike eval_agreement.py (student vs. Teacher, needs a trained checkpoint),
this compares two Top-K .npz outputs directly -- both come from the SAME
model/vocab, so token indices are already aligned and comparable without
running any forward pass here.

Only reports index-based agreement metrics (top-1, top-5), not a KL
divergence: with only Top-K indices/values on both sides (no full-vocab
logits), a token that's in one file's Top-K but not the other's has no
recoverable per-token probability on the side missing it -- only its Top-K
set's aggregate "everything else" bucket total, which isn't a valid
per-token estimate. Computing a real KL here would need at least one side's
full logits (see topk_kd_loss in train_sft.py for that case, where the
student's full logits ARE available).

Requires the two .npz files to have been precomputed on the EXACT SAME
input file (same tokenization, same offsets) via precompute_teacher_targets.py.

Example:
    python learn/distill/compare_teacher_precision.py \
      --reference reasoning/val_sample40_bf16_topk32.npz \
      --test reasoning/val_sample40_fp8_topk32.npz
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Top-K .npz treated as the ground truth (e.g. bf16)")
    parser.add_argument("--test", required=True, help="Top-K .npz being compared against the reference (e.g. FP8)")
    args = parser.parse_args()

    ref = np.load(args.reference)
    test = np.load(args.test)

    if not np.array_equal(ref["offsets"], test["offsets"]):
        raise ValueError("offsets differ -- the two files weren't precomputed on the same input/tokenization")

    ref_indices = ref["indices"]  # (N, K), sorted descending by value (torch.topk)
    test_indices = test["indices"]
    ref_top1 = ref_indices[:, 0]
    test_top1 = test_indices[:, 0]

    top1_agreement = (ref_top1 == test_top1).mean()
    # Softer metric: does the reference's top prediction still show up anywhere
    # in the test model's Top-K, even if it's no longer rank 1?
    top1_in_test_topk = np.array([ref_top1[i] in test_indices[i] for i in range(len(ref_top1))]).mean()

    print(f"Positions compared: {len(ref_top1)} (K={ref_indices.shape[1]})")
    print(f"top1_agreement:        {top1_agreement:.4f}")
    print(f"ref_top1_in_test_topk: {top1_in_test_topk:.4f}")
    print("Reference points (qwen3.8-27b-notes.md): ~0.96 (4-bit), ~0.989 (8-bit) top-token agreement vs. bf16")


if __name__ == "__main__":
    main()
