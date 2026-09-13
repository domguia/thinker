"""
Phase 2 / Phase 1quater training loop: multi-hop chain retrieval
(data/kb_chain_retrieval.py::KBChainDataset) -- the direct test of the
project's central thesis (iterative extraction+processing beats a single
pass) unlike data/kb_retrieval.py's single-lookup task where N_step > 1-2 has
nothing to add.

Per thinker-e9 (sister session)'s finding: with no FF in the main loop, each
iteration only does ONE of {retrieve, process} rather than both fused like a
standard transformer layer -- expect needing roughly 2-4x N_step relative to
n_hops before judging a plateau real, and re-sweep LR at every N_step tested
(the stable-LR window found at N_step=2-3 is not guaranteed to hold at higher
N_step, same lesson as Phase -1's d_model sweep).

Curriculum support mirrors train_kb_retrieval.py: `--hop_curriculum "1,2,3"`
increases n_hops (with n_distractors fixed) while keeping the leaf-sequence
shape constant at max_facts = last_stage + n_distractors.

`--n_step_random_max`: when set, samples n_step ~ Uniform(1, N) per batch
instead of a fixed value (Universal/Looped-Transformer-style randomized-depth
training), to test extrapolation to N_step_test > N used at training time --
per thinker-e9's recommendation, run alongside (not instead of) a fixed-N_step
run for comparison.
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data.kb_chain_retrieval import KBChainDataset
from core.indexed_thinker_model import Thinker
from learn.indexed_attention.eval_metrics import (
    prediction_stats, trivial_baselines, format_report,
)


def build_model(args, total_vocab_size, device):
    return Thinker(
        vocab_size=total_vocab_size,
        d_model=args.d_model,
        n_register=args.n_register,
        block_size=args.block_size,
        depth=args.depth,
        n_slots=args.n_slots,
        n_head=args.n_head,
        sm_cap=args.sm_cap,
        use_ff=args.use_ff,
        ff_hidden_mult=args.ff_hidden_mult,
        detach_sm_keys=args.detach_sm_keys,
        level_dropout_p=args.level_dropout_p,
        decouple_kv=not args.shared_kv_pooling,
    ).to(device)


def sample_n_step(args):
    if args.n_step_random_max is not None:
        return int(torch.randint(1, args.n_step_random_max + 1, (1,)).item())
    return args.n_step


def evaluate(model, ds, args, device, n_step, n_batches=10, with_stats=False):
    """Returns held-out accuracy; with_stats=True also returns
    `pred_in_kb_rate` -- the fraction of predictions landing on ANY value
    present in the episode's KB. That rate is what makes the accuracy
    interpretable: near 1.0, the model is choosing among the episode's
    n_facts values, so the meaningful reference is 1/n_facts, NOT the
    uniform-over-vocabulary rate (see learn/indexed_attention/eval_metrics.py
    for why this matters -- it invalidated earlier readings of the multi-hop
    plateau)."""
    model.eval()
    correct, in_kb, total = 0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
            kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
            kb_mask = kb_mask.to(device)
            query_tokens, labels = query_tokens.to(device), labels.to(device)
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=n_step, kb_leaf_mask=kb_mask)
            preds = streams["answer"][:, 0, :].argmax(dim=-1)
            st = prediction_stats(preds.cpu(), labels.cpu(), kb_tokens.cpu(), ds.n_facts)
            correct += st["correct"]
            in_kb += st["in_kb"]
            total += st["total"]
    model.train()
    acc = correct / total
    return (acc, in_kb / total) if with_stats else acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_hops", type=int, default=2, help="chain length (ignored if --hop_curriculum is set)")
    parser.add_argument("--hop_curriculum", default=None, help="comma-separated increasing n_hops stages, e.g. '1,2,3'")
    parser.add_argument("--curriculum_promote_acc", type=float, default=0.9)
    parser.add_argument("--curriculum_min_steps", type=int, default=500)
    parser.add_argument("--n_distractors", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_register", type=int, default=1)
    parser.add_argument("--n_slots", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_step", type=int, default=4, help="fixed core loop iterations (ignored if --n_step_random_max set)")
    parser.add_argument("--n_step_random_max", type=int, default=None,
                         help="if set, sample n_step ~ Uniform(1, N) per training batch (randomized-depth training)")
    parser.add_argument("--n_step_eval", type=int, default=None,
                         help="n_step used for eval/extrapolation probe; defaults to --n_step (or --n_step_random_max if that's set)")
    parser.add_argument("--use_ff", action="store_true")
    parser.add_argument("--ff_hidden_mult", type=int, default=4)
    parser.add_argument("--detach_sm_keys", action="store_true")
    parser.add_argument("--level_dropout_p", type=float, default=0.0)
    parser.add_argument("--shared_kv_pooling", action="store_true",
                         help="ABLATION: restore the pre-2026-09-13 LevelCompressor that pools "
                              "parent K and parent V with the SAME softmax. That variant cannot "
                              "represent a key->value association and was the root cause of the "
                              "n_hops>=2 plateau -- use it only to reproduce the old behavior.")
    parser.add_argument("--sm_cap", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_time_minutes", type=float, default=15.0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--extrapolate_n_steps", default=None,
                         help="comma-separated N_step_test values > training N_step to probe in-memory "
                              "after training finishes, no checkpoint needed (e.g. '8,16,24')")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.hop_curriculum:
        stages = [int(x) for x in args.hop_curriculum.split(",")]
        assert stages == sorted(stages)
    else:
        stages = [args.n_hops]
    max_facts = stages[-1] + args.n_distractors

    if args.depth == 0:
        args.block_size = max_facts * 4
    else:
        expected_leaves = args.block_size ** args.depth
        assert expected_leaves == max_facts * 4, (
            f"max_facts={max_facts} (-> {max_facts * 4} leaves) doesn't match "
            f"block_size={args.block_size} ** depth={args.depth} = {expected_leaves}"
        )

    def make_datasets(n_hops):
        ds = KBChainDataset(n_hops=n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                             max_facts=max_facts, seed=args.seed)
        eval_ds = KBChainDataset(n_hops=n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                                  max_facts=max_facts, seed=args.seed + 10_000)
        return ds, eval_ds

    stage_idx = 0
    ds, eval_ds = make_datasets(stages[stage_idx])
    model = build_model(args, ds.total_vocab_size, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_step_eval = args.n_step_eval or args.n_step_random_max or args.n_step
    num_params = sum(p.numel() for p in model.parameters())
    print(f"depth={args.depth} block_size={args.block_size} max_facts={max_facts} n_distractors={args.n_distractors} "
          f"hop_stages={stages} n_step={args.n_step} n_step_random_max={args.n_step_random_max} "
          f"d_model={args.d_model} num_params={num_params / 1e6:.3f}M device={device}")

    best_loss_t = None
    start = time.time()
    step = 0
    stage_start_step = 0
    while True:
        kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
        kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
        kb_mask = kb_mask.to(device)
        query_tokens, labels = query_tokens.to(device), labels.to(device)

        n_step = sample_n_step(args)
        opt.zero_grad()
        _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=n_step, kb_leaf_mask=kb_mask)
        logits = streams["answer"][:, 0, :]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        with torch.no_grad():
            best_loss_t = loss.detach() if best_loss_t is None else torch.minimum(best_loss_t, loss.detach())
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - start
            print(f"step {step:5d} stage_n_hops={stages[stage_idx]} n_step={n_step} loss {loss.item():.4f} elapsed {elapsed:.1f}s")

        if step % args.eval_every == 0:
            acc = evaluate(model, eval_ds, args, device, n_step=n_step_eval)
            print(f"step {step:5d} stage_n_hops={stages[stage_idx]} eval_acc(held_out,n_step={n_step_eval}) {acc:.4f}")

            if (stage_idx < len(stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and acc >= args.curriculum_promote_acc):
                stage_idx += 1
                stage_start_step = step
                ds, eval_ds = make_datasets(stages[stage_idx])
                print(f"step {step:5d} CURRICULUM PROMOTE -> n_hops={stages[stage_idx]}")

        elapsed_min = (time.time() - start) / 60
        if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
            break

    final_acc, pred_in_kb_rate = evaluate(model, eval_ds, args, device, n_step=n_step_eval,
                                          n_batches=20, with_stats=True)
    baselines = trivial_baselines(eval_ds, batch_size=args.batch_size)
    elapsed = time.time() - start

    print("---")
    print(f"best_loss:        {best_loss_t.item():.6f}")
    print(f"final_acc:        {final_acc:.6f}")
    print(f"final_stage_n_hops: {stages[stage_idx]}")
    print(f"n_step_eval:      {n_step_eval}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"seed:             {args.seed}")
    print(f"depth:            {args.depth}")
    print(f"decouple_kv:      {not args.shared_kv_pooling}")
    print(f"pred_in_kb_rate:  {pred_in_kb_rate:.6f}")
    print(f"conditional_chance: {baselines['conditional_chance']:.6f}")
    print(format_report(final_acc, pred_in_kb_rate, baselines, n_hops=stages[stage_idx]))

    if args.extrapolate_n_steps:
        # In-memory extrapolation probe (thinker-e9's suggestion): no
        # checkpoint needed, just re-evaluate the already-trained model at
        # N_step_test > training N_step on a fresh, disjointly-seeded KB.
        probe_ds = KBChainDataset(n_hops=stages[stage_idx], n_distractors=args.n_distractors,
                                   vocab_size=args.vocab_size, max_facts=max_facts, seed=args.seed + 20_000)
        print("--- extrapolation probe (N_step_test > training N_step) ---")
        for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
            acc = evaluate(model, probe_ds, args, device, n_step=n_step_test, n_batches=20)
            print(f"n_step_test={n_step_test:3d} acc={acc:.4f}")


if __name__ == "__main__":
    main()
