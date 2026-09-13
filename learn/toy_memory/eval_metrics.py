"""
Chance levels and trivial-predictor controls for the toy-model copy/cumsum
memory tasks (dev_notes/toy_memory_experiment_plan.md).

WHY THIS EXISTS: `scripts/train.py`'s only accuracy metric is a per-token
mean (`(targets == preds).float().mean()`, its own comment reads "Sequence
exact match or token average accuracy? Let's use token average"). On a
sequence task this alone is not a safe reference -- some positions are
predictable without any computation (e.g. the first output position of a
cumsum task equals the first input token, no summation needed; a copy task
is entirely predictable this way). Same lesson, same failure mode, as
`learn/indexed_attention/eval_metrics.py` (which this mirrors): a raw
accuracy number is not interpretable without its trivial-predictor controls
printed alongside it.

Two trivial predictors, measured empirically (not assumed):
  - `copy_input`: predict target = input, no computation at all. This IS
    the exact solver for the copy task (task=None) -- reported as a sanity
    check there, not a shortcut. For cumsum it is a genuine no-computation
    baseline any real result must beat.
  - `most_common_token`: the empirical per-position mode of the target,
    estimated on a disjoint reference sample and scored on a fresh batch.
    For these tasks (x drawn i.i.d. uniform over vocab_size, y a fixed
    deterministic function of x) this should sit near 1/vocab_size -- kept
    as a measured check on that assumption rather than an assumed constant.
"""

import math

import torch


def capacity_budget(read_step: int, n_latent: int, d_model: int,
                    seq_len: int, vocab_size: int) -> dict:
    """`read_step` does not vary one thing -- it varies both (a) how many
    compute steps directly attend to `x`, AND (b) the total FIFO write
    bandwidth available to consolidate `x`'s content before it disappears
    from direct view (thinker-5b, 2026-09-13, tracing
    `core/toy_model.py:183-208`: `memory = x` before the loop makes `x`
    visible unconditionally at i=0, then again at every i<=read_step, i.e.
    `read_step + 1` compute steps total; one `n_latent`-sized latent is
    appended to the FIFO per compute step). Below the information content
    of `x`, failure is an information-theoretic impossibility, not a verdict
    on whether the memory mechanism works -- exactly the confound the
    Indexed Attention contre-expertise had to defuse for the compressor bug.
    This computes both sides of that comparison so every report is
    self-interpreting instead of silently assuming the non-constraining
    regime.

    Primary flag is a VECTOR-count comparison (`write_budget_vectors <
    seq_len`), not bits-vs-dims (thinker-5b, 2026-09-13): a float dimension
    carries arbitrarily many bits in principle, so `budget_dims < input_bits`
    is off by a "bits per float dimension" factor and stays green in
    regimes that are obviously constraining -- e.g. `n_latent=2, d_model=64,
    read_step=0, seq_len=32, vocab_size=16` gives `budget_dims=128 ==
    input_bits=128`, no flag, despite 2 latent vectors plainly being unable
    to hold the identity of 32 tokens. The real constraint is architectural,
    not information-theoretic: an attention-based compressor cannot route
    an arbitrary number of tokens into fewer vector SLOTS regardless of how
    many float dimensions each slot has. `budget_dims`/`input_bits` are kept
    as secondary context (an absolute information floor, still worth
    knowing) but must never be the flag a reader trusts.
    """
    write_budget_vectors = (read_step + 1) * n_latent
    budget_dims = write_budget_vectors * d_model
    input_bits = seq_len * math.log2(vocab_size)
    return {
        "write_budget_vectors": write_budget_vectors,
        "budget_dims": budget_dims,
        "input_bits": input_bits,
        "capacity_constraining": write_budget_vectors < seq_len,
    }


def token_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-token mean accuracy -- the metric `scripts/train.py` already
    reports, kept for continuity with prior numbers in experiment.log.md."""
    return (preds == targets).float().mean().item()


def exact_match_rate(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Whole-sequence exact match: 1.0 only if every position is correct.
    The metric that actually matters for "did the model solve the episode",
    as opposed to token_accuracy which a model can inflate by nailing the
    easy positions and never getting the hard (memory-dependent) ones."""
    return (preds == targets).all(dim=-1).float().mean().item()


def per_position_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(T,) accuracy at each output position, averaged over the batch. The
    read_step sweep (Exp. 1) is expected to show this DEGRADE towards later
    positions specifically if memory (not re-reading the input) is what
    carries information forward -- a flat profile across positions despite
    a low read_step would itself be an interesting/surprising result."""
    return (preds == targets).float().mean(dim=0)


def copy_input_baseline(inputs: torch.Tensor, targets: torch.Tensor) -> dict:
    """Accuracy of predicting target = input, no computation. Requires
    inputs.shape == targets.shape (true for copy/cumsum, both seq_len -> seq_len)."""
    assert inputs.shape == targets.shape, (
        f"copy_input_baseline expects matching shapes, got inputs={tuple(inputs.shape)} "
        f"targets={tuple(targets.shape)}"
    )
    return {
        "token_acc": token_accuracy(inputs, targets),
        "exact_match": exact_match_rate(inputs, targets),
    }


def most_common_token_baseline(target_sampler, vocab_size: int, seq_len: int,
                                ref_batch: int = 4096, eval_batch: int = 4096,
                                seed: int = 12345) -> dict:
    """Empirical per-position mode of the target, estimated on one batch
    (`ref_batch` episodes, seeded) and scored on a disjointly-seeded fresh
    batch (`eval_batch` episodes) -- never scored on the same sample used to
    estimate the mode, or this would silently overfit to sampling noise.

    `target_sampler(n, seed)` must return a (n, seq_len) target tensor for
    `n` fresh episodes (the caller wires this to whichever dataset/task is
    under test); this module has no dataset dependency of its own.
    """
    ref_targets = target_sampler(ref_batch, seed)
    mode_per_position = torch.mode(ref_targets, dim=0).values  # (seq_len,)

    eval_targets = target_sampler(eval_batch, seed + 1)
    preds = mode_per_position.unsqueeze(0).expand(eval_batch, seq_len)
    return {
        "token_acc": token_accuracy(preds, eval_targets),
        "exact_match": exact_match_rate(preds, eval_targets),
        "vocab_chance": 1.0 / vocab_size,
    }


def format_report(task: str, token_acc: float, exact_acc: float,
                  copy_baseline: dict, mode_baseline: dict,
                  pos_acc: torch.Tensor = None, budget: dict = None) -> str:
    """One block, printed by every toy-memory training/eval run -- mirrors
    `learn/indexed_attention/eval_metrics.py`'s discipline of never quoting
    an accuracy number without its trivial-predictor controls next to it.
    """
    is_copy_task = task in (None, "copy")
    copy_label = "EXACT SOLVER for the copy task (sanity check, not a shortcut)" \
        if is_copy_task else "no-computation shortcut -- must be beaten for a real result"
    margin_note = "" if is_copy_task else (
        f"margin_over_copy_input: {token_acc - copy_baseline['token_acc']:+.4f} (token)  "
        f"{exact_acc - copy_baseline['exact_match']:+.4f} (exact)\n"
    )
    pos_line = ""
    if pos_acc is not None:
        first, last = pos_acc[0].item(), pos_acc[-1].item()
        pos_line = (
            f"per_position_acc:     first={first:.4f}  last={last:.4f}  "
            f"min={pos_acc.min().item():.4f}  (full vector available, not printed)\n"
        )
    budget_line = ""
    if budget is not None:
        flag = "YES -- a cliff here is an info-theoretic bound, not a mechanism verdict" \
            if budget["capacity_constraining"] else "no"
        budget_line = (
            f"capacity_budget:      write_budget={budget['write_budget_vectors']} vectors "
            f"({budget['budget_dims']} dims) vs. input={budget['input_bits']:.1f} bits  "
            f"capacity_constraining: {flag}\n"
        )
    return (
        f"--- chance-level report (learn/toy_memory/eval_metrics.py, task={task}) ---\n"
        f"token_acc (per-token, legacy metric): {token_acc:.4f}\n"
        f"exact_match_acc (whole sequence):     {exact_acc:.4f}\n"
        f"{pos_line}"
        f"{budget_line}"
        f"copy_input_baseline:  token={copy_baseline['token_acc']:.4f}  "
        f"exact={copy_baseline['exact_match']:.4f}   ({copy_label})\n"
        f"most_common_token:    token={mode_baseline['token_acc']:.4f}  "
        f"exact={mode_baseline['exact_match']:.4f}  vocab_chance={mode_baseline['vocab_chance']:.4f}\n"
        f"{margin_note}"
        "---------------------------------------------------------------------"
    )
