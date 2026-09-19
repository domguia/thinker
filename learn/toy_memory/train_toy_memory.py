"""
Exp. 0 + Exp. 1 (dev_notes/toy_memory_experiment_plan.md): does the toy
Thinker's "medium-term memory built on the fly" actually carry information,
or does the model simply re-read the input at every step?

`core/toy_model.py::ToyThinker.forward` decides per-step whether the input
`x` is included in the attended memory: `memory = latents if i >= read_step
else [x] + latents`. Both existing runners (`scripts/train.py:77`,
`scripts/th1nker_runner.py:1044`) hard-code `read_step = n_step - 1`, so `x`
was re-readable at every compute step except the very last (output) one --
every prior result obtained with these scripts is compatible with "the
latent is a scratchpad, x is never actually memorized", never disproving it.
This script makes `read_step` an explicit, fixed, sweepable argument instead,
and reports exact-match/per-position accuracy against trivial baselines
(`eval_metrics.py`) instead of the bare per-token accuracy `scripts/train.py`
used, which cannot distinguish "solved the task" from "nailed the easy
positions".

**CRITICAL FIX (2026-09-13, found by model-design after a suspicious 96/96
exact_match=1.0000 grid result)**: `ToyThinker.forward`'s `is_output_ar=True`
path builds its output query as `out_query = embd_out_pos(pos) +
embd_vocab(target)` -- the target sequence embedded UNSHIFTED, i.e. position
i's query already contains an embedding of `target[i]`, the very token being
scored at that position. `attn_compute` is a pre-norm RESIDUAL stack
(core/layers.py::CustomFlexDecoderLayer, `x = res + ...` at every sub-layer),
so this embedding survives to the output, and the tied output head
(`F.linear(output, embd_vocab.weight)`) reads it straight back off --
independent of `memory`/`x`/`read_step` entirely. Verified empirically: a
model trained this way still predicts `targets` perfectly even when `inputs`
is swapped for a completely unrelated random sequence (see the mismatch test
this module's `evaluate_with_leak_check` mirrors). This is why 96/96 cells
converged to exact 1.0000 with zero variance across every read_step, both
scales, both tasks -- **the metric measured whether the model can read back
an embedding handed directly to it, not whether it solved the task from
memory.** `outs[4]` (the `is_full_ar` causal stream) has the same defect,
since `target` is fed unshifted there too.

**Fix**: shift the query by one position with a reserved BOS id (standard
teacher-forcing convention -- position i's query carries `target[i-1]`,
never `target[i]`), so the residual stream can no longer leak the current
label. `ToyThinker` is built with `vocab_size + 1` (the extra row is the BOS
id, never a valid label). `evaluate()` now also runs a permanent mismatch
check (`leak_check` in `eval_metrics.py`) every time -- feeds unrelated
`inputs` with the same `targets` and asserts accuracy stays near chance, so
this bug (or a regression of it) can never again silently pass as "the model
solved the task."

Task order per the plan: copy first (pure transport, no computation -- if
this breaks under a low read_step, everything downstream will too), then
cumsum (the smallest task where re-reading vs. memorizing the running sum
actually matters), then add/subtract (below).

**add/subtract (wired 2026-09-14, reviving the user's original base-16
arithmetic proposal from dev_notes/experiment.log.md's 18 Dec 2023 entry --
never fully pursued at the time precisely because large numbers need
carry-holding short-term memory, which is exactly what this project now
needs a task to discriminate)**: `x[:, :L]` and `x[:, L:]` (L = seq_len // 2)
are two L-digit base-`vocab_size` numbers, LSB-first; the target is their
sum/difference, LSB-first, ripple carry/borrow, in the first L output
positions (positions L: are a deterministic 0 filler so the output keeps
shape (batch, seq_len) -- see per_position_accuracy for the real/filler
split, never read exact_match alone here without checking it).

Why this is the right memory-discriminating task, unlike
`data/kb_chain_retrieval.py` (see dev_notes/indexed_attention_spec.md Sec
9.1): that task's flaw was that every hop's fact is a static token already
sitting in the KB, so a single unified softmax can jump straight to whatever
hop is currently needed -- no temporal memory required regardless of how
good the mechanism is. Here, the carry/borrow at digit i is NOT present
anywhere in the input; it can only be produced by having actually processed
digits 0..i-1 first. There is no KB shortcut to it. A model that gets late
digits right without ever holding a running carry across compute steps
would be a genuine surprise, not an artifact of task design -- this avoids
both traps identified in Sec 9.1 (Markovian-shortcut-via-static-facts, and
single-softmax-solves-it-in-one-shot).

Deliberately does NOT reuse `data.numbers.NumbersCopyDataset`: its
`__iter__` carries a `progressive_copy`-curriculum's mutable class-level
state (`target_len`, `challenge_factor`, `acc_history`) that this experiment
doesn't want, and offers no seed control for a reproducible disjointly-seeded
held-out batch (the same methodology already used throughout
`learn/indexed_attention/*`). The copy/cumsum generating process is a few
lines, reproduced directly below with an explicit `torch.Generator`.
"""

import argparse
import time

import torch
import torch.nn as nn

from core.toy_model import ToyThinker, all_losses_compute
from core.run_logging import add_run_args, logger_from_args
from learn.toy_memory.eval_metrics import (
    token_accuracy, exact_match_rate, per_position_accuracy,
    copy_input_baseline, most_common_token_baseline, format_report, capacity_budget,
)

TASKS = ("copy", "cumsum", "add", "subtract")


def _ripple_add_or_subtract(a: torch.Tensor, b: torch.Tensor, vocab_size: int,
                            subtract: bool) -> torch.Tensor:
    """LSB-first ripple carry (add) / borrow (subtract), digit by digit. The
    carry/borrow at position i is produced only by processing positions
    0..i-1 first -- it is never a token present in the input, unlike a
    kb_chain_retrieval fact (see module docstring's add/subtract section for
    why that distinction is what makes this task actually discriminate
    memory use, not just re-computable from a static lookup)."""
    L = a.shape[1]
    carry = torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    out = torch.zeros_like(a)
    for i in range(L):
        if subtract:
            s = a[:, i] - b[:, i] - carry
            borrow = (s < 0).to(a.dtype)
            out[:, i] = s + borrow * vocab_size
            carry = borrow
        else:
            s = a[:, i] + b[:, i] + carry
            out[:, i] = s % vocab_size
            carry = s // vocab_size
    return out


def sample_batch(batch: int, seq_len: int, vocab_size: int, task: str,
                 device, generator: torch.Generator = None):
    x = torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
    if task == "copy":
        y = x.clone()
    elif task == "cumsum":
        y = torch.cumsum(x, dim=1) % vocab_size
    elif task in ("add", "subtract"):
        assert seq_len % 2 == 0, (
            f"task={task!r} requires an even seq_len (split evenly between operand A and B digits), "
            f"got seq_len={seq_len}"
        )
        L = seq_len // 2
        a, b = x[:, :L], x[:, L:]
        result = _ripple_add_or_subtract(a, b, vocab_size, subtract=(task == "subtract"))
        y = torch.zeros_like(x)
        y[:, :L] = result
        # y[:, L:] stays a deterministic 0 filler so target keeps shape (batch, seq_len),
        # matching output_len == input_len elsewhere in this script -- per_position_accuracy
        # exposes the real (first L)/filler (last L) split, never collapse to one number
        # without checking it for this task.
    else:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return x.to(device), y.to(device)


def shift_targets(targets: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Standard teacher-forcing shift: position i's query input becomes
    target[i-1] (position 0 gets the reserved BOS id), so the residual
    stream can never carry the CURRENT position's own label -- see the
    module docstring's CRITICAL FIX note for why the unshifted version leaks."""
    bos_col = torch.full((targets.shape[0], 1), bos_id, dtype=targets.dtype, device=targets.device)
    return torch.cat([bos_col, targets[:, :-1]], dim=1)


def forward_and_predict(model, inputs, targets, n_latent, n_step, read_step, n_memory, bos_id):
    """One forward pass + argmax predictions from the (shifted-query) logits
    stream. `targets` is used unshifted for the returned predictions'
    ground truth; the query fed to the model is shift_targets(targets,
    bos_id) so the model can never read its own label off the residual
    stream (see module docstring)."""
    shifted = shift_targets(targets, bos_id)
    outs = model(inputs, shifted, n_latent, n_step, read_step, n_memory=n_memory,
                 is_full_ar=False, is_output_ar=True, output_step=1)
    logits = outs[1][:, -1, :, :]  # (B, T, vocab) at the last compute step
    preds = torch.argmax(logits, dim=2)
    return outs, preds


def evaluate(model, args, device, read_step: int, bos_id: int, seq_len: int = None,
            n_eval: int = 8, seed: int = 999) -> dict:
    """Held-out pass (disjoint seed from training) at a given `read_step`
    and `seq_len` (defaults to `args.seq_len` when not curriculum-ing over
    it). Averages over `n_eval` batches for a less noisy read. Also runs the
    permanent mismatch/leak check (eval_metrics.leak_check): feeds
    unrelated `inputs` alongside the same `targets` and confirms accuracy
    drops to near chance -- catches a regression of the residual-leak bug
    this module was fixed for, rather than trusting the shift silently."""
    seq_len = seq_len if seq_len is not None else args.seq_len
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    mismatch_gen = torch.Generator().manual_seed(seed + 54321)
    tok_accs, exact_accs, pos_accs, leak_accs = [], [], [], []
    with torch.no_grad():
        for _ in range(n_eval):
            inputs, targets = sample_batch(args.batch_size, seq_len, args.vocab_size,
                                           args.task, device, generator=gen)
            _, preds = forward_and_predict(model, inputs, targets, args.n_latent,
                                           args.n_step, read_step, args.n_memory, bos_id)
            tok_accs.append(token_accuracy(preds, targets))
            exact_accs.append(exact_match_rate(preds, targets))
            pos_accs.append(per_position_accuracy(preds, targets))

            mismatched_inputs, _ = sample_batch(args.batch_size, seq_len, args.vocab_size,
                                                args.task, device, generator=mismatch_gen)
            _, leak_preds = forward_and_predict(model, mismatched_inputs, targets, args.n_latent,
                                                args.n_step, read_step, args.n_memory, bos_id)
            leak_accs.append(token_accuracy(leak_preds, targets))
    model.train()
    return {
        "token_acc": sum(tok_accs) / len(tok_accs),
        "exact_match": sum(exact_accs) / len(exact_accs),
        "pos_acc": torch.stack(pos_accs).mean(dim=0),
        "leak_token_acc": sum(leak_accs) / len(leak_accs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--vocab_size", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--n_latent", type=int, default=8)
    p.add_argument("--n_step", type=int, default=6,
                   help="fixed reasoning-step count (Exp. 1 sweeps read_step at fixed n_step, not the reverse)")
    p.add_argument("--read_step", type=int, default=None,
                   help="explicit, no hidden default -- the value this run TRAINS at (matched train/test "
                        "condition). 0..n_step. read_step=n_step reproduces the old always-visible-input "
                        "behavior (the control upper bound); read_step=0 means x is visible only at the very "
                        "first compute step, never again. Ignored if --read_step_curriculum is set; required "
                        "otherwise.")
    p.add_argument("--read_step_curriculum", default=None,
                   help="thinker-5b (2026-09-13): direct-training grids at a fixed read_step conflate "
                        "'the mechanism can't do this' with 'the optimization landscape from a random init "
                        "doesn't lead there' -- exactly the confound already resolved twice on this project "
                        "(ToyThinker copy curriculum, Dec 2023; Indexed Attention n_facts=64 curriculum, "
                        "Phase -1/0). Comma-separated DECREASING read_step stages, e.g. '6,5,4,3,2,1,0' -- "
                        "starts at the first (highest/easiest) stage, promotes to the next when held-out "
                        "exact_match crosses --curriculum_promote_acc. Mirrors "
                        "learn/indexed_attention/train_kb_chain.py's --hop_curriculum (there increasing, "
                        "here decreasing since read_step=n_step is the easy/control end).")
    p.add_argument("--seq_len_curriculum", default=None,
                   help="experiment-manager (2026-09-13): a from-scratch run at a large seq_len (e.g. 32) "
                        "converges too slowly to be affordable per grid cell -- even the read_step=n_step "
                        "control took 2700+ steps and was still climbing (8.7%%->48.6%% exact_match) when a "
                        "30-min budget ran out. Same fix as --read_step_curriculum, applied to task length "
                        "instead: comma-separated INCREASING seq_len stages, e.g. '8,16,24,32'. read_step "
                        "stays FIXED (--read_step, not --read_step_curriculum -- the two curricula are not "
                        "combined in this version) while seq_len ramps up on the same promotion rule. The "
                        "model is built once at max(stages) (embeddings sized for it); earlier stages just "
                        "use a shorter prefix, no padding needed.")
    p.add_argument("--curriculum_promote_acc", type=float, default=0.9)
    p.add_argument("--curriculum_min_steps", type=int, default=500)
    p.add_argument("--final_stage_min_steps", type=int, default=1500,
                   help="experiment-manager (2026-09-13, generalized same day after the cumsum LR sweep): "
                        "with a single shared --max_time_minutes/--max_steps budget across all curriculum "
                        "stages, a run whose early stages converge slowly can exhaust the whole budget before "
                        "a LATER stage gets a fair shot. First found at the last stage (16 vs 24, purely a "
                        "budget artifact, not an eval bug -- final_stage_seq_len/final_stage_read_step always "
                        "labeled the reached stage correctly). Then found to apply to INTERMEDIATE stages too: "
                        "cumsum at read_step=6 converges cleanly in ~4000 direct steps at seq_len=8 alone (LR "
                        "ruled out -- the default lr=1e-3 works fine standalone), yet the curriculum run at the "
                        "same read_step/lr stalled at seq_len=16, never promoting further -- stage 1 alone "
                        "likely ate most of the shared budget, leaving too little for stage 2. Generalized fix: "
                        "EVERY curriculum stage (not just the last) is now guaranteed at least this many steps "
                        "before --max_time_minutes/--max_steps can end the run, extending past them if needed "
                        "(a message is printed when this happens). Applies per-stage, so a 4-stage curriculum "
                        "can take up to ~4x this floor in the worst case -- budget accordingly, or lower this "
                        "for a cheaper/noisier read.")
    p.add_argument("--eval_read_step", type=int, default=None,
                   help="Exp. 1's extrapolation condition: additionally evaluate (never train) at this "
                        "alternate read_step, alongside the matched/current read_step above, at every eval "
                        "checkpoint. Independent of --read_step_curriculum -- can be combined with either mode.")
    p.add_argument("--n_memory", type=int, default=10000, help="FIFO cap on latents kept in memory (Exp. 2 axis, fixed here)")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--d_hid", type=int, default=128)
    p.add_argument("--nlayers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="target/peak LR, reached at the end of warmup (or from step 0 if --lr_warmup_steps=0)")
    p.add_argument("--lr_warmup_steps", type=int, default=0,
                   help="experiment-manager (2026-09-14): cumsum seq_len=32 direct (no curriculum) at "
                        "read_step=6 stays mixed/unstable across the WHOLE lr in {1e-3, 1.5e-3, 3e-3} sweep -- "
                        "lowering lr alone did not stabilize it (worst case improved from total collapse to "
                        "partial progress, but no lr gave 3/3 clean seeds). Next untried lead per the plan: "
                        "linear LR warmup FROM --lr_warmup_init TO --lr over this many steps, then held constant "
                        "-- distinct from lowering the peak lr itself. 0 (default) disables warmup, exactly "
                        "reproducing the old behavior (constant --lr from step 0).")
    p.add_argument("--lr_warmup_init", type=float, default=None,
                   help="LR at step 0 when --lr_warmup_steps > 0 (linearly ramped up to --lr). Defaults to "
                        "--lr / 10 if not set. Ignored when --lr_warmup_steps=0.")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_args(p)
    args = p.parse_args()
    logger = logger_from_args(args)

    if args.read_step_curriculum:
        stages = [int(x) for x in args.read_step_curriculum.split(",")]
        assert stages == sorted(stages, reverse=True) and len(set(stages)) == len(stages), (
            "--read_step_curriculum stages must be strictly decreasing (start easy/high, end hard/low)"
        )
    else:
        assert args.read_step is not None, "--read_step is required unless --read_step_curriculum is set"
        stages = [args.read_step]
    assert all(0 <= s <= args.n_step for s in stages), "every read_step stage must be in [0, n_step]"

    if args.seq_len_curriculum:
        assert not args.read_step_curriculum, (
            "--seq_len_curriculum and --read_step_curriculum are not combined in this version -- "
            "use a fixed --read_step with --seq_len_curriculum"
        )
        seq_stages = [int(x) for x in args.seq_len_curriculum.split(",")]
        assert seq_stages == sorted(seq_stages) and len(set(seq_stages)) == len(seq_stages), (
            "--seq_len_curriculum stages must be strictly increasing (start short/easy, end long/hard)"
        )
        assert seq_stages[-1] == args.seq_len, (
            f"--seq_len_curriculum's last stage ({seq_stages[-1]}) should equal --seq_len ({args.seq_len}) "
            f"-- it's the target scale, kept as the single source of truth for reporting/capacity_budget"
        )
    else:
        seq_stages = [args.seq_len]
    max_seq_len = max(seq_stages)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    bos_id = args.vocab_size  # reserved row, never a valid label (see shift_targets)
    model = ToyThinker(
        vocab_size=args.vocab_size + 1, max_latent=max(args.n_latent, 16),
        max_input_len=max_seq_len, max_output_len=max_seq_len,
        d_model=args.d_model, nhead=args.nhead, d_hid=args.d_hid, nlayers=args.nlayers,
        n_probe=1, dropout=0.0,  # all_losses_compute calls compute_probe_loss() unconditionally,
                                  # which crashes on probes=None -- n_probe=0 is not a supported no-op here.
    ).to(device)
    stage_idx = 0
    current_read_step = stages[stage_idx]
    seq_stage_idx = 0
    current_seq_len = seq_stages[seq_stage_idx]

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"task={args.task} n_step={args.n_step} read_step_stages={stages} seq_len_stages={seq_stages} "
          f"eval_read_step={args.eval_read_step} params={n_params/1e3:.1f}K", flush=True)

    def print_budget(read_step, seq_len, label):
        # thinker-5b (2026-09-13): read_step varies TWO things at once -- how
        # many compute steps directly see x, AND the total FIFO write
        # bandwidth available to consolidate x's content before it
        # disappears. Print the budget so a cliff at low read_step is never
        # misread as a mechanism verdict when it's actually a capacity bound.
        b = capacity_budget(read_step, args.n_latent, args.d_model, seq_len, args.vocab_size)
        print(f"capacity_budget ({label} read_step={read_step}, seq_len={seq_len}): "
              f"write_budget={b['write_budget_vectors']} vectors ({b['budget_dims']} dims) "
              f"vs. input={b['input_bits']:.1f} bits  "
              f"capacity_constraining={b['capacity_constraining']}", flush=True)
        if b["capacity_constraining"]:
            print("WARNING: this read_step is in the capacity-constraining regime -- "
                  "a failure here is an info-theoretic/architectural bound, not a memory-mechanism "
                  "verdict. Main sweep should use n_latent >= seq_len (dev_notes/toy_memory_experiment_plan.md).",
                  flush=True)
        return b

    budget = print_budget(current_read_step, current_seq_len, "matched")
    if args.eval_read_step is not None:
        print_budget(args.eval_read_step, current_seq_len, "extrapolation")

    lr_warmup_init = args.lr_warmup_init if args.lr_warmup_init is not None else args.lr / 10
    if args.lr_warmup_steps > 0:
        print(f"lr_warmup: {lr_warmup_init:.2e} -> {args.lr:.2e} over {args.lr_warmup_steps} steps, "
              f"then held constant at {args.lr:.2e}", flush=True)

    def lr_at(step: int) -> float:
        # experiment-manager (2026-09-14): cumsum seq_len=32 stayed mixed/unstable across the WHOLE
        # peak-lr sweep {1e-3, 1.5e-3, 3e-3} -- distinct lead from lowering the peak lr itself.
        if args.lr_warmup_steps <= 0 or step >= args.lr_warmup_steps:
            return args.lr
        return lr_warmup_init + (args.lr - lr_warmup_init) * (step / args.lr_warmup_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=1e-2)

    # Trivial baselines (learn/toy_memory/eval_metrics.py) -- depend on
    # seq_len, so recomputed whenever a --seq_len_curriculum promotion
    # changes it (see the promotion block below), never left stale from an
    # earlier, shorter stage. Reported alongside every accuracy number,
    # never cited without them.
    def compute_baselines(seq_len):
        def target_sampler(n, seed):
            gen = torch.Generator().manual_seed(seed)
            _, y = sample_batch(n, seq_len, args.vocab_size, args.task, "cpu", generator=gen)
            return y

        probe_x, probe_y = sample_batch(4096, seq_len, args.vocab_size, args.task, "cpu",
                                        generator=torch.Generator().manual_seed(54321))
        cb = copy_input_baseline(probe_x, probe_y)
        mb = most_common_token_baseline(target_sampler, args.vocab_size, seq_len, seed=11111)
        print(f"trivial baselines (seq_len={seq_len}) -- copy_input: token={cb['token_acc']:.4f} "
              f"exact={cb['exact_match']:.4f}  |  most_common_token: "
              f"token={mb['token_acc']:.4f} exact={mb['exact_match']:.4f} "
              f"vocab_chance={mb['vocab_chance']:.4f}", flush=True)
        return cb, mb

    copy_baseline, mode_baseline = compute_baselines(current_seq_len)

    start_time = time.time()
    max_time_seconds = args.max_time_minutes * 60
    best_exact = 0.0
    stage_start_step = 0  # step at which the CURRENT (read_step, seq_len) combo began -- reset on every promotion
    model.train()

    step = 0
    while True:
        elapsed = time.time() - start_time
        stage_dwell = step - stage_start_step
        # Guarantee --final_stage_min_steps at EVERY curriculum stage, not just the last --
        # experiment-manager found cumsum (read_step=6) converges cleanly in ~4000 direct
        # steps at seq_len=8 alone (ruling out LR), yet the curriculum run at the same
        # read_step/lr stalled at seq_len=16 -- stage 1 alone likely ate most of the shared
        # budget, leaving too little for stage 2. Extends past --max_time_minutes/--max_steps
        # if needed -- see --final_stage_min_steps' help text. Only meaningful with an actual
        # curriculum (>1 stage) -- with a single fixed read_step/seq_len, the user's own
        # --max_steps/--max_time_minutes is already the intended budget, not a floor to override.
        curriculum_active = len(stages) > 1 or len(seq_stages) > 1
        protecting_stage = curriculum_active and stage_dwell < args.final_stage_min_steps
        if step >= args.max_steps and not protecting_stage:
            extended_note = f" (extended past the original --max_steps={args.max_steps} to protect the " \
                            f"current stage's dwell)" if step > args.max_steps else ""
            print(f"Step budget of {step} reached. Stopping.{extended_note}", flush=True)
            break
        if elapsed > max_time_seconds:
            if protecting_stage:
                if stage_dwell == 0:
                    print(f"Time budget of {args.max_time_minutes} minutes reached, but extending to "
                          f"guarantee --final_stage_min_steps={args.final_stage_min_steps} at the current "
                          f"stage (read_step={current_read_step}, seq_len={current_seq_len}).", flush=True)
            else:
                print(f"Time budget of {args.max_time_minutes} minutes reached. Stopping.", flush=True)
                break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        inputs, targets = sample_batch(args.batch_size, current_seq_len, args.vocab_size,
                                       args.task, device)
        outs, preds = forward_and_predict(model, inputs, targets, args.n_latent,
                                          args.n_step, current_read_step, args.n_memory, bos_id)
        loss, _ = all_losses_compute(outs, targets, target_emb=None, last_step_only=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if step % args.eval_every == 0:
            matched = evaluate(model, args, device, read_step=current_read_step, bos_id=bos_id,
                               seq_len=current_seq_len)
            # model-design (2026-09-14): compare against mode_baseline['token_acc'], not the flat
            # vocab_chance -- add/subtract has a deterministic 0-filler half that ANY predictor
            # (leaking or not) gets right for free, so mode_baseline['token_acc'] (which already
            # measures that trivial floor) is the correct near-chance reference; vocab_chance alone
            # made every add/subtract run print a false-positive LEAK SUSPECTED.
            leak_ref = mode_baseline["token_acc"]
            leak_flag = "" if matched["leak_token_acc"] < leak_ref + 0.15 else \
                " *** LEAK SUSPECTED (mismatch accuracy far above chance) ***"
            print(f"step={step:6d} lr={lr_at(step):.2e} stage_read_step={current_read_step} stage_seq_len={current_seq_len} "
                  f"elapsed={elapsed/60:.2f}m loss={loss.item():.4f} "
                  f"[matched read_step={current_read_step}] "
                  f"token_acc={matched['token_acc']:.4f} exact_match={matched['exact_match']:.4f} "
                  f"leak_check={matched['leak_token_acc']:.4f}{leak_flag}", flush=True)
            logger.progress(step, loss=loss.item(), lr=lr_at(step),
                            token_acc=matched["token_acc"],
                            exact_match=matched["exact_match"],
                            leak_check=matched["leak_token_acc"],
                            stage_read_step=current_read_step,
                            stage_seq_len=current_seq_len)

            if args.eval_read_step is not None:
                extrap = evaluate(model, args, device, read_step=args.eval_read_step, bos_id=bos_id,
                                  seq_len=current_seq_len)
                print(f"           [extrapolation read_step={args.eval_read_step}] "
                      f"token_acc={extrap['token_acc']:.4f} exact_match={extrap['exact_match']:.4f} "
                      f"leak_check={extrap['leak_token_acc']:.4f}", flush=True)

            if matched["exact_match"] > best_exact:
                best_exact = matched["exact_match"]

            # thinker-5b (2026-09-13) / experiment-manager (seq_len variant,
            # same day): promote only on a real signal, same rule as
            # train_kb_chain.py's --hop_curriculum -- both a minimum dwell
            # time at this stage AND a held-out accuracy threshold, never
            # time alone (promotes through noise) or accuracy alone (could
            # promote off a single lucky batch before the eval average has
            # stabilized). Exactly one of the two curricula is active in a
            # given run (asserted above), so only one branch below ever fires.
            if (stage_idx < len(stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and matched["exact_match"] >= args.curriculum_promote_acc):
                stage_idx += 1
                stage_start_step = step
                current_read_step = stages[stage_idx]
                budget = print_budget(current_read_step, current_seq_len, "matched")
                print(f"step={step:6d} CURRICULUM PROMOTE -> read_step={current_read_step}", flush=True)

            if (seq_stage_idx < len(seq_stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and matched["exact_match"] >= args.curriculum_promote_acc):
                seq_stage_idx += 1
                stage_start_step = step
                current_seq_len = seq_stages[seq_stage_idx]
                budget = print_budget(current_read_step, current_seq_len, "matched")
                copy_baseline, mode_baseline = compute_baselines(current_seq_len)
                print(f"step={step:6d} CURRICULUM PROMOTE -> seq_len={current_seq_len}", flush=True)

        step += 1

    final = evaluate(model, args, device, read_step=current_read_step, bos_id=bos_id,
                     seq_len=current_seq_len, n_eval=16)
    print("\n---", flush=True)
    print(format_report(args.task, final["token_acc"], final["exact_match"],
                        copy_baseline, mode_baseline, pos_acc=final["pos_acc"], budget=budget), flush=True)
    leak_ref = mode_baseline["token_acc"]  # see the periodic-eval leak_flag comment above for why
    leak_verdict = "OK (near chance)" if final["leak_token_acc"] < leak_ref + 0.15 else \
        "*** LEAK SUSPECTED -- do not trust final_exact_match above ***"
    print(f"leak_check (mismatched-input token_acc, should sit near mode_baseline={leak_ref:.4f} "
          f"[vocab_chance={mode_baseline['vocab_chance']:.4f}]): "
          f"{final['leak_token_acc']:.4f}  -> {leak_verdict}", flush=True)
    print(f"read_step_stages:      {stages}", flush=True)
    print(f"final_stage_read_step: {current_read_step} (stage {stage_idx + 1}/{len(stages)})", flush=True)
    print(f"seq_len_stages:        {seq_stages}", flush=True)
    print(f"final_stage_seq_len:   {current_seq_len} (stage {seq_stage_idx + 1}/{len(seq_stages)})", flush=True)
    print(f"best_exact_match:  {best_exact:.4f}", flush=True)
    print(f"final_exact_match: {final['exact_match']:.4f}", flush=True)
    print(f"training_seconds:  {elapsed:.1f}", flush=True)
    print(f"num_steps:         {step}", flush=True)

    logger.finish(
        summary={"final_exact_match": final["exact_match"],
                 "final_token_acc": final["token_acc"],
                 "best_exact_match": best_exact,
                 "final_stage_read_step": current_read_step,
                 "final_stage_seq_len": current_seq_len,
                 "num_steps": step, "training_seconds": elapsed},
        # leak_ref, pas vocab_chance : add/subtract a une moitié déterministe
        # que n'importe quel prédicteur obtient gratuitement, et vocab_chance
        # seul faisait crier au LEAK sur tous les runs (voir le commentaire de
        # leak_flag dans la boucle d'éval).
        controls={"chance": leak_ref,
                  "margin": final["token_acc"] - leak_ref,
                  "leak_check": final["leak_token_acc"]},
    )

    if args.eval_read_step is not None:
        extrap_final = evaluate(model, args, device, read_step=args.eval_read_step, bos_id=bos_id,
                                seq_len=current_seq_len, n_eval=16)
        print(f"final_extrapolation_read_step={args.eval_read_step}: "
              f"token_acc={extrap_final['token_acc']:.4f} exact_match={extrap_final['exact_match']:.4f} "
              f"leak_check={extrap_final['leak_token_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
