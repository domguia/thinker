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

Task order per the plan: copy first (pure transport, no computation -- if
this breaks under a low read_step, everything downstream will too), then
cumsum (the smallest task where re-reading vs. memorizing the running sum
actually matters), addition-base16 last (reuses `NumbersComputeDataset`
separately, not wired here).

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
from learn.toy_memory.eval_metrics import (
    token_accuracy, exact_match_rate, per_position_accuracy,
    copy_input_baseline, most_common_token_baseline, format_report, capacity_budget,
)

TASKS = ("copy", "cumsum")


def sample_batch(batch: int, seq_len: int, vocab_size: int, task: str,
                 device, generator: torch.Generator = None):
    x = torch.randint(0, vocab_size, (batch, seq_len), generator=generator)
    if task == "copy":
        y = x.clone()
    elif task == "cumsum":
        y = torch.cumsum(x, dim=1) % vocab_size
    else:
        raise ValueError(f"unknown task {task!r}, expected one of {TASKS}")
    return x.to(device), y.to(device)


def forward_and_predict(model, inputs, targets, n_latent, n_step, read_step, n_memory):
    """One forward pass + argmax predictions from the non-AR (additive
    teacher-forced query) logits stream, matching the metric already used by
    `scripts/train.py` (`outs[1]`, not the causal `outs[4]` stream) for
    continuity with prior numbers in experiment.log.md."""
    outs = model(inputs, targets, n_latent, n_step, read_step, n_memory=n_memory,
                 is_full_ar=True, is_output_ar=True, output_step=1)
    logits = outs[1][:, -1, :, :]  # (B, T, vocab) at the last compute step
    preds = torch.argmax(logits, dim=2)
    return outs, preds


def evaluate(model, args, device, read_step: int, n_eval: int = 8, seed: int = 999) -> dict:
    """Held-out pass (disjoint seed from training) at a given `read_step`.
    Averages over `n_eval` batches for a less noisy read."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tok_accs, exact_accs, pos_accs = [], [], []
    with torch.no_grad():
        for _ in range(n_eval):
            inputs, targets = sample_batch(args.batch_size, args.seq_len, args.vocab_size,
                                           args.task, device, generator=gen)
            _, preds = forward_and_predict(model, inputs, targets, args.n_latent,
                                           args.n_step, read_step, args.n_memory)
            tok_accs.append(token_accuracy(preds, targets))
            exact_accs.append(exact_match_rate(preds, targets))
            pos_accs.append(per_position_accuracy(preds, targets))
    model.train()
    return {
        "token_acc": sum(tok_accs) / len(tok_accs),
        "exact_match": sum(exact_accs) / len(exact_accs),
        "pos_acc": torch.stack(pos_accs).mean(dim=0),
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
    p.add_argument("--curriculum_promote_acc", type=float, default=0.9)
    p.add_argument("--curriculum_min_steps", type=int, default=500)
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.read_step_curriculum:
        stages = [int(x) for x in args.read_step_curriculum.split(",")]
        assert stages == sorted(stages, reverse=True) and len(set(stages)) == len(stages), (
            "--read_step_curriculum stages must be strictly decreasing (start easy/high, end hard/low)"
        )
    else:
        assert args.read_step is not None, "--read_step is required unless --read_step_curriculum is set"
        stages = [args.read_step]
    assert all(0 <= s <= args.n_step for s in stages), "every read_step stage must be in [0, n_step]"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = ToyThinker(
        vocab_size=args.vocab_size, max_latent=max(args.n_latent, 16),
        max_input_len=args.seq_len, max_output_len=args.seq_len,
        d_model=args.d_model, nhead=args.nhead, d_hid=args.d_hid, nlayers=args.nlayers,
        n_probe=1, dropout=0.0,  # all_losses_compute calls compute_probe_loss() unconditionally,
                                  # which crashes on probes=None -- n_probe=0 is not a supported no-op here.
    ).to(device)
    stage_idx = 0
    current_read_step = stages[stage_idx]

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"task={args.task} n_step={args.n_step} read_step_stages={stages} "
          f"eval_read_step={args.eval_read_step} params={n_params/1e3:.1f}K", flush=True)

    def print_budget(read_step, label):
        # thinker-5b (2026-09-13): read_step varies TWO things at once -- how
        # many compute steps directly see x, AND the total FIFO write
        # bandwidth available to consolidate x's content before it
        # disappears. Print the budget so a cliff at low read_step is never
        # misread as a mechanism verdict when it's actually a capacity bound.
        b = capacity_budget(read_step, args.n_latent, args.d_model, args.seq_len, args.vocab_size)
        print(f"capacity_budget ({label} read_step={read_step}): "
              f"write_budget={b['write_budget_vectors']} vectors ({b['budget_dims']} dims) "
              f"vs. input={b['input_bits']:.1f} bits  "
              f"capacity_constraining={b['capacity_constraining']}", flush=True)
        if b["capacity_constraining"]:
            print("WARNING: this read_step is in the capacity-constraining regime -- "
                  "a failure here is an info-theoretic/architectural bound, not a memory-mechanism "
                  "verdict. Main sweep should use n_latent >= seq_len (dev_notes/toy_memory_experiment_plan.md).",
                  flush=True)
        return b

    budget = print_budget(current_read_step, "matched")
    if args.eval_read_step is not None:
        print_budget(args.eval_read_step, "extrapolation")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Trivial baselines, measured once up front against a disjointly-seeded
    # reference/eval split (learn/toy_memory/eval_metrics.py) -- reported
    # alongside every accuracy number below, never cited without them.
    def target_sampler(n, seed):
        gen = torch.Generator().manual_seed(seed)
        _, y = sample_batch(n, args.seq_len, args.vocab_size, args.task, "cpu", generator=gen)
        return y

    probe_x, probe_y = sample_batch(4096, args.seq_len, args.vocab_size, args.task, "cpu",
                                    generator=torch.Generator().manual_seed(54321))
    copy_baseline = copy_input_baseline(probe_x, probe_y)
    mode_baseline = most_common_token_baseline(target_sampler, args.vocab_size, args.seq_len,
                                               seed=11111)
    print(f"trivial baselines -- copy_input: token={copy_baseline['token_acc']:.4f} "
          f"exact={copy_baseline['exact_match']:.4f}  |  most_common_token: "
          f"token={mode_baseline['token_acc']:.4f} exact={mode_baseline['exact_match']:.4f} "
          f"vocab_chance={mode_baseline['vocab_chance']:.4f}", flush=True)

    start_time = time.time()
    max_time_seconds = args.max_time_minutes * 60
    best_exact = 0.0
    stage_start_step = 0
    model.train()

    for step in range(args.max_steps):
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            print(f"Time budget of {args.max_time_minutes} minutes reached. Stopping.", flush=True)
            break

        inputs, targets = sample_batch(args.batch_size, args.seq_len, args.vocab_size,
                                       args.task, device)
        outs, preds = forward_and_predict(model, inputs, targets, args.n_latent,
                                          args.n_step, current_read_step, args.n_memory)
        loss, _ = all_losses_compute(outs, targets, target_emb=None, last_step_only=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if step % args.eval_every == 0:
            matched = evaluate(model, args, device, read_step=current_read_step)
            print(f"step={step:6d} stage_read_step={current_read_step} elapsed={elapsed/60:.2f}m "
                  f"loss={loss.item():.4f} [matched read_step={current_read_step}] "
                  f"token_acc={matched['token_acc']:.4f} exact_match={matched['exact_match']:.4f}", flush=True)

            if args.eval_read_step is not None:
                extrap = evaluate(model, args, device, read_step=args.eval_read_step)
                print(f"           [extrapolation read_step={args.eval_read_step}] "
                      f"token_acc={extrap['token_acc']:.4f} exact_match={extrap['exact_match']:.4f}", flush=True)

            if matched["exact_match"] > best_exact:
                best_exact = matched["exact_match"]

            # thinker-5b (2026-09-13): promote only on a real signal, same
            # rule as train_kb_chain.py's --hop_curriculum -- both a minimum
            # dwell time at this stage AND a held-out accuracy threshold,
            # never time alone (which would promote through noise) or
            # accuracy alone (which could promote off a single lucky batch
            # before the eval average has stabilized).
            if (stage_idx < len(stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and matched["exact_match"] >= args.curriculum_promote_acc):
                stage_idx += 1
                stage_start_step = step
                current_read_step = stages[stage_idx]
                budget = print_budget(current_read_step, "matched")
                print(f"step={step:6d} CURRICULUM PROMOTE -> read_step={current_read_step}", flush=True)

    final = evaluate(model, args, device, read_step=current_read_step, n_eval=16)
    print("\n---", flush=True)
    print(format_report(args.task, final["token_acc"], final["exact_match"],
                        copy_baseline, mode_baseline, pos_acc=final["pos_acc"], budget=budget), flush=True)
    print(f"read_step_stages:      {stages}", flush=True)
    print(f"final_stage_read_step: {current_read_step} (stage {stage_idx + 1}/{len(stages)})", flush=True)
    print(f"best_exact_match:  {best_exact:.4f}", flush=True)
    print(f"final_exact_match: {final['exact_match']:.4f}", flush=True)
    print(f"training_seconds:  {elapsed:.1f}", flush=True)
    print(f"num_steps:         {step + 1}", flush=True)

    if args.eval_read_step is not None:
        extrap_final = evaluate(model, args, device, read_step=args.eval_read_step, n_eval=16)
        print(f"final_extrapolation_read_step={args.eval_read_step}: "
              f"token_acc={extrap_final['token_acc']:.4f} exact_match={extrap_final['exact_match']:.4f}", flush=True)


if __name__ == "__main__":
    main()
