"""Chunked CE+KD loss (spec dev_notes/indexed_attention_spec.md §13.3, "levier
memoire" -- distinct from a real FLOPs-reduction lever like sampled softmax or
ALBERT-style factorization, both still [OUVERT]).

Motivation (distillation.md, 2026-09-13 500M-core MFU/batch-size sweep): the
batch-size ceiling was IDENTICAL between fp32 and bf16 (6 on an L40S) --
bf16's autocast sped up compute but did not raise the memory ceiling, because
the bottleneck is the Teacher-aligned vocab (248,077 tokens): materializing
the full (batch, block_size, vocab) logits tensor for CE's cross-entropy and
KD's topk_kd_loss (both need a logsumexp over the full vocab) dominates
memory regardless of the model's own dtype. This is the "chunked-loss idea
(a la Liger-Kernel/'Cut Your Losses')" flagged there as the most direct lever
and never implemented until now.

What this buys, precisely: NOT fewer FLOPs (the head matmul d_model->vocab
still runs in full for every token) -- only lower PEAK MEMORY, by never
holding more than one (chunk_size, vocab) logits tensor at a time instead of
the full (N, vocab) tensor, trading it for extra recompute via
torch.utils.checkpoint (each chunk's forward is discarded after producing its
loss, then redone during backward). This is exactly why row/token-chunking
(not vocab-chunking) is correct and simple here: cross-entropy and
topk_kd_loss are per-token quantities (softmax/logsumexp normalizes each
token's own row independently) -- unlike online-softmax attention, no
cross-chunk accumulator is needed, so chunks are fully independent and the
result (loss value and every gradient) is identical to the unchunked call up
to floating-point summation-order noise. Verified in
tests/test_chunked_loss.py.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce_kd(hidden_chunk, head, labels_chunk, teacher_indices_chunk, teacher_values_chunk,
                  teacher_residual_chunk, teacher_mask_chunk, has_kd):
    """One row-chunk's CE-sum and KD-sum (SUM reduction, not mean -- the
    caller divides by the true total valid-token count pooled across every
    chunk, so the final value matches a single unchunked mean exactly).

    Meant to run under torch.utils.checkpoint: only `hidden_chunk` (small,
    d_model-wide) needs to be kept for backward -- the (chunk_size, vocab)
    logits tensor this function computes internally is recomputed during
    backward rather than retained, which is the actual memory saving.
    """
    logits_chunk = head(hidden_chunk)  # (chunk, vocab) -- only ever this size at once
    ce_sum = F.cross_entropy(logits_chunk, labels_chunk, ignore_index=-100, reduction="sum")
    if not has_kd:
        return ce_sum, logits_chunk.new_zeros(())

    from learn.distill.train_sft import topk_kd_loss  # deferred: avoids a module-load cycle

    n_valid = teacher_mask_chunk.sum().clamp(min=1)
    kd_mean = topk_kd_loss(
        logits_chunk.unsqueeze(0), teacher_indices_chunk.unsqueeze(0), teacher_values_chunk.unsqueeze(0),
        teacher_residual_chunk.unsqueeze(0), teacher_mask_chunk.unsqueeze(0),
    )
    return ce_sum, kd_mean * n_valid


def chunked_ce_kd_loss(hidden, head, labels, chunk_size=2048, teacher_indices=None,
                        teacher_values=None, teacher_residual=None, teacher_mask=None,
                        use_checkpoint=True):
    """Drop-in replacement for `F.cross_entropy(head(hidden), labels, ...)`
    plus (optionally) `topk_kd_loss(head(hidden), ...)`, computed without ever
    materializing `head(hidden)` for more than `chunk_size` rows at once.

    hidden: (N, d_model), already flattened across every batch/sequence
    dimension by the caller. head: any callable (chunk, d_model) -> (chunk,
    vocab) -- nn.Linear, HF's model.lm_head, or Thinker's OutputStream.head
    all satisfy this. labels: (N,) with -100 at ignored positions. teacher_*:
    all four required together (or all left None for CE-only), each (N, K)
    or (N,) exactly as topk_kd_loss expects, flattened the same way as
    hidden/labels.

    Returns (ce, kd) as the same two scalar means the unchunked
    F.cross_entropy/topk_kd_loss calls would give -- this function changes
    only peak memory (and, with use_checkpoint=True, adds one extra forward
    pass per chunk during backward), never the loss value or its gradient
    beyond floating-point summation-order noise (verified in
    tests/test_chunked_loss.py against the unchunked call).

    chunk_size <= 0 (or >= N) is a single whole-tensor chunk -- same code
    path as chunking, useful to confirm numerical equivalence and as the
    natural "disabled" state.
    """
    n = hidden.shape[0]
    has_kd = teacher_indices is not None
    if chunk_size is None or chunk_size <= 0:
        chunk_size = n

    total_ce_sum = hidden.new_zeros(())
    total_kd_sum = hidden.new_zeros(())
    total_ce_valid = hidden.new_zeros((), dtype=torch.long)
    total_kd_valid = hidden.new_zeros((), dtype=torch.long)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        h_chunk, l_chunk = hidden[start:end], labels[start:end]
        if has_kd:
            ti, tv, tr, tm = (teacher_indices[start:end], teacher_values[start:end],
                              teacher_residual[start:end], teacher_mask[start:end])
        else:
            ti = tv = tr = tm = None

        if use_checkpoint:
            ce_sum, kd_sum = checkpoint(_chunk_ce_kd, h_chunk, head, l_chunk, ti, tv, tr, tm,
                                         has_kd, use_reentrant=False)
        else:
            ce_sum, kd_sum = _chunk_ce_kd(h_chunk, head, l_chunk, ti, tv, tr, tm, has_kd)

        total_ce_sum = total_ce_sum + ce_sum
        total_kd_sum = total_kd_sum + kd_sum
        total_ce_valid = total_ce_valid + (l_chunk != -100).sum()
        if has_kd:
            total_kd_valid = total_kd_valid + tm.sum()

    ce = total_ce_sum / total_ce_valid.clamp(min=1)
    kd = total_kd_sum / total_kd_valid.clamp(min=1) if has_kd else hidden.new_zeros(())
    return ce, kd
