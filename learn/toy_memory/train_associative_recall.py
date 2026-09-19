"""
Exp. 7 (dev_notes/toy_memory_experiment_plan.md Sec 5bis): the first toy task
designed so that a NULL `n_memory` result is actually falsifiable evidence
against multi-slot memory, not just a task that happened not to need it.

**Why Exp. 2/6 (`copy`/`cumsum`/`add`/`subtract`, `n_memory` sweep) couldn't
settle this, even with capacity_budget ruled out**: all four tasks only
ever require a SINGLE scalar/vector sufficient statistic to be carried
forward at each step (the running sum for cumsum, the running carry bit for
add/subtract -- at most 1 bit regardless of how many digits, since a base-N
addition's carry is always in {0, 1}). `n_memory=1` (plain recurrence) is
therefore PROVABLY sufficient for these tasks no matter how good the
mechanism is -- scaling `seq_len` further would not fix this, the required
state never grows past 1 scalar. This was found only after the `add` result
came back clean-but-uninformative (`experiment-manager`, 2026-09-14): a real
design mistake, corrected here rather than chasing more scale on a task that
structurally cannot answer the question.

**The second, independent trap** (`ToyThinker`'s existing `read_step`,
`core/toy_model.py`): `x` is always attended to WHOLLY or not at all -- there
is no way to reveal only PART of it at a given step. So even a task that
needs K genuinely distinct facts (e.g. "K key-value pairs, then a query")
is still solvable in one shot if the facts and the query are ever visible
TOGETHER (even for a single step): a single cross-attention call can jump
straight to the matching pair, no memory needed at all -- the same
single-softmax trap already identified for `kb_chain_retrieval.py`
(indexed_attention_spec.md Sec 9.1), just phrased differently.

**Fix, `core/toy_model.py::ToyThinker.forward`'s new `x_reveal_mask`
argument**: per-(step, position) visibility instead of a single global
`read_step` threshold. This task uses it to reveal exactly one (key, value)
pair per step, in order, each disappearing forever once the next pair
arrives -- then reveals ONLY the query key at the final step, by which time
every fact has already left `x` and can only be answered from what got
written into the FIFO `latents` memory along the way.

**Task**: `n_facts` (K) distinct keys (sampled without replacement so a
query is never ambiguous) paired with iid random values, presented one pair
per compute step; a query key (matching one of the K keys, uniformly) is
revealed at the final step; target is the paired value -- a single token,
not a sequence, so there is no teacher-forcing/target-embedding channel for
the label to leak through at all (unlike copy/cumsum/add, no analogue of the
`is_output_ar` leak bug -- `target=1` (an int, just "how many output
positions") in the `ToyThinker.forward` call, never the label Tensor itself,
so there is no embedding of the true label for the residual stream to leak).

**Falsifiable capacity prediction** (`capacity_ceiling` below, distinct from
and sharper than `eval_metrics.capacity_budget`'s bits/dims check): with a
FIFO of size `n_memory` and K pairs written before the query, the pair at
original index `i` survives to query time iff `i >= K - n_memory` (else it
was evicted, oldest-first, before the query ever arrives) -- an outcome
determined purely by the write schedule, before any question of whether the
mechanism itself works. Since the queried index is uniform over `{0..K-1}`,
a PERFECT mechanism should score exactly `min(1, n_memory/K)` (from genuine
retrieval) plus `(1 - min(1, n_memory/K)) / vocab_size` (chance on the
evicted fraction, no information left to retrieve it). A measured curve that
tracks this prediction as `n_memory` varies (at fixed K) would be the
clearest possible confirmation that memory genuinely holds `n_memory`
DISTINCT facts simultaneously, not just the latest one -- the result Exp. 2
was designed to produce but structurally couldn't.
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F

from core.toy_model import ToyThinker
from core.run_logging import add_run_args, logger_from_args
from learn.toy_memory.eval_metrics import most_common_token_baseline


def build_reveal_mask(n_facts: int, n_step: int, T: int) -> torch.Tensor:
    """(n_step, T) bool -- step i in [0, n_facts) reveals ONLY positions
    (2i, 2i+1) (the i-th key/value pair); step n_facts reveals ONLY position
    2*n_facts (the query key); any step beyond that (extra delay) reveals
    nothing -- pure recurrence on the already-written FIFO memory, a margin
    that makes the test harder (further from the write step), not easier."""
    mask = torch.zeros(n_step, T, dtype=torch.bool)
    for i in range(n_facts):
        mask[i, 2 * i] = True
        mask[i, 2 * i + 1] = True
    mask[n_facts, 2 * n_facts] = True
    return mask


def sample_batch(batch: int, n_facts: int, vocab_size: int, device,
                 generator: torch.Generator = None):
    assert vocab_size >= n_facts, "need at least n_facts distinct tokens available for the keys"
    # vectorized without-replacement sampling: argsort of iid random scores per row
    # gives a uniformly random permutation of [0, vocab_size) per batch row.
    key_perm = torch.argsort(torch.rand(batch, vocab_size, generator=generator), dim=1)
    keys = key_perm[:, :n_facts]                                            # (B, K), distinct per row
    values = torch.randint(0, vocab_size, (batch, n_facts), generator=generator)  # (B, K), iid, may repeat/collide with keys
    query_idx = torch.randint(0, n_facts, (batch,), generator=generator)     # (B,) which fact is queried
    query_key = keys.gather(1, query_idx.unsqueeze(1)).squeeze(1)           # (B,)
    target = values.gather(1, query_idx.unsqueeze(1)).squeeze(1)           # (B,)

    T = 2 * n_facts + 1
    x = torch.empty(batch, T, dtype=torch.long)
    x[:, 0:2 * n_facts:2] = keys
    x[:, 1:2 * n_facts:2] = values
    x[:, 2 * n_facts] = query_key
    return x.to(device), target.to(device), query_idx


def capacity_ceiling(n_facts: int, n_memory: int, vocab_size: int) -> dict:
    """See module docstring's 'Falsifiable capacity prediction'."""
    retain_frac = min(1.0, n_memory / n_facts)
    predicted_acc = retain_frac + (1 - retain_frac) / vocab_size
    return {"retain_frac": retain_frac, "predicted_acc": predicted_acc}


def latent_capacity_note(n_latent: int, n_facts: int) -> str:
    """experiment-manager's full n_memory sweep (2026-09-14, 60/60 cells) found NO
    degradation as n_memory shrinks -- n_memory=1 seed3 hit acc_by_idx=[1,1,1,1],
    impossible if the FIFO were the only channel. Root cause: `latent`
    (`core/toy_model.py`'s recurrent state, updated every step via
    `attn_compute(latent, memory, ...)`) is a SEPARATE, n_memory-INDEPENDENT
    channel -- it persists/accumulates through its own recurrent update
    regardless of what the FIFO holds, exactly like R in the main Thinker
    (disable_sm's mechanistic story: "querying the SM re-consults a function of
    what R already contains"). It also has `n_latent` internal vector slots of
    its own (self-attention among its own positions) -- same vector-SLOT
    accounting as `capacity_budget` (bits-vs-dims is the wrong unit, see that
    docstring), so `n_latent >= n_facts` alone can already be enough to hold
    every fact regardless of n_memory. A `n_memory` sweep is only informative
    once this alternate channel is constrained below what the task needs --
    hence the WARNING below, printed for every run of this script from now on
    so this confound is never silently reintroduced."""
    if n_latent >= n_facts:
        return (f"WARNING: n_latent={n_latent} >= n_facts={n_facts} -- the recurrent `latent` state "
                f"alone may have enough internal vector slots to hold every fact regardless of "
                f"n_memory, making any n_memory sweep run here uninformative about EXTERNAL "
                f"multi-slot memory specifically. Use n_latent < n_facts (e.g. n_latent=1) to force "
                f"reliance on the FIFO before trusting a capacity_ceiling comparison.")
    return f"n_latent={n_latent} < n_facts={n_facts} -- latent alone cannot hold every fact, n_memory sweep is meaningful here."


def recency_echo_predicted_acc(n_facts: int, vocab_size: int) -> float:
    """experiment-manager (2026-09-14): the LR sweep's plateau (~27.5-31.1%
    on 8/9 cells, all landing in the same narrow band regardless of a 10x LR
    range -- not the usual per-LR-window signature) looks exactly like a
    degenerate shortcut: always output "whatever value was written LAST"
    (position 2*n_facts-1, still fully in the FIFO even at n_memory=1),
    ignoring the query entirely. This is correct whenever the query happens
    to ask about the last-written fact (prob 1/n_facts, since query_idx is
    uniform) and at chance otherwise -- 1/4 + (3/4)/32 = 27.34% at
    n_facts=4, vocab_size=32, matching the observed plateau almost exactly.
    A trivial-predictor CONTROL (like copy_input_baseline for copy/cumsum),
    not itself evidence the model does this -- see `evaluate()`'s
    `acc_excl_last`/`recency_match_excl_last` (an unconfounded version of
    this check, after the first attempt turned out not to discriminate
    anything, see that docstring) for the real diagnostic."""
    p_match = 1.0 / n_facts
    return p_match + (1 - p_match) / vocab_size


def evaluate(model, args, device, n_step: int, reveal_mask: torch.Tensor,
            n_eval: int = 8, seed: int = 999) -> dict:
    """`model_matches_recency` (first version, 2026-09-14) was CONFOUNDED --
    experiment-manager caught it: it stayed ~25-30% in BOTH the plateau
    cells and the one 97.6% success, because a genuinely-retrieving model
    ALSO agrees with the recency-echo prediction whenever the query happens
    to target the last-written fact (prob 1/n_facts, by construction --
    "correct" and "echoes recency" coincide there regardless of mechanism).
    It measured a floor every model must hit, not whether a model relies on
    the shortcut. Fixed by CONDITIONING on the fact this coincidence cannot
    occur -- `query_idx != n_facts-1` (the ~75% of episodes where the
    correct answer and the recency-echo answer are DIFFERENT tokens, since
    keys/values are independent draws): only there does an above-chance
    `acc_excl_last`/high `recency_match_excl_last` unambiguously mean
    something (genuine retrieval / the shortcut, respectively)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    accs, retained_accs, evicted_accs = [], [], []
    acc_excl_last_l, recency_match_excl_last_l = [], []
    # model-design (2026-09-14): experiment-manager's unconfounded acc_excl_last landed at
    # 26.6-32.8% on all 8 "plateau" cells -- clearly above chance (3.1%) but far below the
    # 98% the one escaped seed reaches, i.e. genuine but PARTIAL retrieval, not a pure
    # shortcut. ~1/3 (the fraction of the excl_last subset made of the SECOND-to-last fact,
    # query_idx=n_facts-2, at n_facts=4) matches this band closely enough to be worth
    # checking directly rather than guessing: per-query_idx accuracy, to see whether the
    # "partial" solution is actually "reliably recalls the last ~2 writes, nothing older"
    # (a small effective window) rather than a diffuse partial signal across all indices.
    acc_by_idx_correct = torch.zeros(args.n_facts)
    acc_by_idx_count = torch.zeros(args.n_facts)
    with torch.no_grad():
        for _ in range(n_eval):
            x, target, query_idx = sample_batch(args.batch_size, args.n_facts, args.vocab_size,
                                                device, generator=gen)
            outs = model(x, target=1, n_latent=args.n_latent, n_step=n_step,
                        n_memory=args.n_memory, is_full_ar=False, is_output_ar=False,
                        x_reveal_mask=reveal_mask.to(device),
                        latent_reset_steps={args.n_facts} if args.latent_reset_at_query else None)
            logits = outs[1][:, -1, -1, :]  # (B, vocab) -- single kept step, single output position
            preds = torch.argmax(logits, dim=1)
            correct = (preds == target)
            accs.append(correct.float().mean().item())

            retained = query_idx.to(device) >= (args.n_facts - args.n_memory)
            if retained.any():
                retained_accs.append(correct[retained].float().mean().item())
            if (~retained).any():
                evicted_accs.append(correct[~retained].float().mean().item())

            # excludes the ~1/n_facts of episodes where the query targets the LAST-written
            # fact -- there, "correct" and "echoes recency" are the same token by
            # construction, so agreement there is uninformative (see docstring above).
            recency_pred = x[:, 2 * args.n_facts - 1]
            not_last = query_idx.to(device) != (args.n_facts - 1)
            if not_last.any():
                acc_excl_last_l.append(correct[not_last].float().mean().item())
                recency_match_excl_last_l.append((preds[not_last] == recency_pred[not_last]).float().mean().item())

            qidx_cpu = query_idx.cpu()
            correct_cpu = correct.cpu()
            for idx in range(args.n_facts):
                m = qidx_cpu == idx
                acc_by_idx_correct[idx] += correct_cpu[m].sum()
                acc_by_idx_count[idx] += m.sum()
    model.train()
    acc_by_idx = (acc_by_idx_correct / acc_by_idx_count.clamp(min=1)).tolist()
    return {
        "acc": sum(accs) / len(accs),
        "retained_acc": sum(retained_accs) / len(retained_accs) if retained_accs else float("nan"),
        "evicted_acc": sum(evicted_accs) / len(evicted_accs) if evicted_accs else float("nan"),
        "acc_excl_last": sum(acc_excl_last_l) / len(acc_excl_last_l) if acc_excl_last_l else float("nan"),
        "recency_match_excl_last": sum(recency_match_excl_last_l) / len(recency_match_excl_last_l)
            if recency_match_excl_last_l else float("nan"),
        "acc_by_idx": acc_by_idx,  # [idx 0 (oldest write) .. idx n_facts-1 (most recent write)]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_facts", type=int, default=4, help="K distinct (key, value) pairs -- Exp.7's main axis, sweep alongside n_memory. "
                        "Ignored if --n_facts_curriculum is set; required otherwise.")
    p.add_argument("--n_facts_curriculum", default=None,
                   help="experiment-manager (2026-09-14): after fixing the x_reveal_mask step-0 bug "
                        "(core/toy_model.py, commit 317545f), a 9-cell LR sweep at n_facts=4 found 0/9 "
                        "seeds escaping to the genuine solution (vs. 4/15 before the fix, when a shortcut "
                        "was available) -- consistent with this project's repeated 'the solution is rare "
                        "in the loss landscape, hard to find from a random init' signature "
                        "(experiment.log.md, 18 Dec 2023; ToyThinker copy curriculum; Indexed Attention "
                        "n_facts=64 curriculum), now possibly WORSE since the easy shortcut is gone. Same "
                        "fix as elsewhere on this project: comma-separated INCREASING n_facts stages (e.g. "
                        "'2,3,4'), promotes to the next when held-out acc_excl_last crosses "
                        "--curriculum_promote_acc (acc_excl_last, not raw acc, since acc alone can be "
                        "inflated by the recency-echo coincidence -- see that metric's docstring). No "
                        "model resizing needed between stages (ToyThinker's embedding tables don't depend "
                        "on n_facts/T), just recomputes n_step/T/reveal_mask/baselines per stage.")
    p.add_argument("--n_memory", type=int, default=10000, help="FIFO cap on latents kept in memory -- the axis this task is designed to discriminate")
    p.add_argument("--extra_delay", type=int, default=0,
                   help="extra pure-recurrence compute steps after the query is revealed, before output -- "
                        "makes the test HARDER (further from the write step), never easier; 0 = output immediately at the query step")
    p.add_argument("--latent_reset_at_query", action="store_true",
                   help="experiment-manager (2026-09-14): n_memory sweep showed ZERO degradation even at "
                        "n_latent=1 (n_memory=1 seed0 hit acc_by_idx=[.96,.98,.96,.96] on all 4 indices) -- "
                        "`latent`'s own step-to-step recurrent carry is an n_memory-INDEPENDENT channel, "
                        "regardless of n_latent's width (see core/toy_model.py forward()'s "
                        "latent_reset_steps docstring). This flag severs that carry right before the query "
                        "step (core/toy_model.py's new latent_reset_steps={n_facts}), forcing the FIFO to be "
                        "the only channel left -- WITHOUT this flag, any n_memory sweep here is NOT a valid "
                        "test of external memory capacity, only a measure of what latent's own recurrence "
                        "can superpose on its own (default False reproduces the exact prior, confounded, "
                        "behavior for comparison).")
    p.add_argument("--vocab_size", type=int, default=32, help="must be >= n_facts (distinct keys drawn without replacement)")
    p.add_argument("--n_latent", type=int, default=8)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--d_hid", type=int, default=128)
    p.add_argument("--nlayers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--curriculum_promote_acc", type=float, default=0.9)
    p.add_argument("--curriculum_min_steps", type=int, default=500)
    p.add_argument("--final_stage_min_steps", type=int, default=1500,
                   help="guarantees at least this many steps at EVERY curriculum stage before "
                        "--max_time_minutes/--max_steps can end the run (same fix as "
                        "train_toy_memory.py's flag of the same name, after that script's cumsum "
                        "curriculum got starved at an intermediate stage by a shared budget).")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_args(p)
    args = p.parse_args()
    logger = logger_from_args(args)

    if args.n_facts_curriculum:
        n_facts_stages = [int(v) for v in args.n_facts_curriculum.split(",")]
        assert n_facts_stages == sorted(n_facts_stages) and len(set(n_facts_stages)) == len(n_facts_stages), (
            "--n_facts_curriculum stages must be strictly increasing (start easy/small, end hard/large)"
        )
        assert n_facts_stages[-1] == args.n_facts or args.n_facts == 4, (
            f"--n_facts_curriculum's last stage ({n_facts_stages[-1]}) should match --n_facts "
            f"({args.n_facts}) if --n_facts was set explicitly -- kept as the single source of truth"
        )
    else:
        n_facts_stages = [args.n_facts]

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # model-design (2026-09-14): ToyThinker's embedding tables (embd_latent/embd_out_pos) size
    # off n_latent/max_output_len only, never off T/n_step -- so, unlike train_toy_memory.py's
    # seq_len_curriculum, no "build for max, use a prefix at early stages" trick is needed here;
    # the SAME model instance works unchanged as n_facts (hence T/n_step) grows across stages.
    model = ToyThinker(
        vocab_size=args.vocab_size, max_latent=max(args.n_latent, 16),
        max_input_len=2 * max(n_facts_stages) + 1, max_output_len=1,
        d_model=args.d_model, nhead=args.nhead, d_hid=args.d_hid, nlayers=args.nlayers,
        n_probe=0, dropout=0.0,
    ).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"n_facts_stages={n_facts_stages} n_memory={args.n_memory} vocab_size={args.vocab_size} "
          f"params={n_params/1e3:.1f}K", flush=True)

    def setup_stage(n_facts):
        """Recomputes every n_facts-dependent quantity for a new curriculum stage (or the
        single fixed stage when no curriculum is used) -- mirrors train_toy_memory.py's
        compute_baselines(seq_len), called once up front and again on every promotion."""
        args.n_facts = n_facts  # evaluate()/sample_batch read args.n_facts directly
        n_step = n_facts + 1 + args.extra_delay
        T = 2 * n_facts + 1
        reveal_mask = build_reveal_mask(n_facts, n_step, T)
        ceiling = capacity_ceiling(n_facts, args.n_memory, args.vocab_size)
        print(f"stage n_facts={n_facts}: n_step={n_step} T={T}", flush=True)
        print(f"capacity_ceiling: retain_frac={ceiling['retain_frac']:.4f}  "
              f"predicted_acc={ceiling['predicted_acc']:.4f}  "
              f"(perfect mechanism should land here -- see module docstring)", flush=True)
        print(latent_capacity_note(args.n_latent, n_facts), flush=True)
        recency_baseline_acc = recency_echo_predicted_acc(n_facts, args.vocab_size)
        print(f"recency_echo_baseline: predicted_acc_if_always_echoes_last_value={recency_baseline_acc:.4f}  "
              f"(trivial-shortcut control -- see recency_echo_predicted_acc docstring; the real diagnostic "
              f"is acc_excl_last/recency_match_excl_last below, not whether acc lands near this number)", flush=True)

        def target_sampler(n, seed):
            gen = torch.Generator().manual_seed(seed)
            _, t, _ = sample_batch(n, n_facts, args.vocab_size, "cpu", generator=gen)
            return t.unsqueeze(1)

        mode_baseline = most_common_token_baseline(target_sampler, args.vocab_size, 1, seed=11111)
        print(f"trivial baseline -- most_common_value: acc={mode_baseline['token_acc']:.4f}  "
              f"vocab_chance={mode_baseline['vocab_chance']:.4f}", flush=True)
        return n_step, T, reveal_mask, ceiling, recency_baseline_acc, mode_baseline

    stage_idx = 0
    n_step, T, reveal_mask, ceiling, recency_baseline_acc, mode_baseline = setup_stage(n_facts_stages[stage_idx])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    start_time = time.time()
    max_time_seconds = args.max_time_minutes * 60
    best_acc = 0.0
    stage_start_step = 0  # step at which the CURRENT n_facts stage began -- reset on every promotion
    model.train()

    step = 0
    while True:
        elapsed = time.time() - start_time
        stage_dwell = step - stage_start_step
        # Guarantee --final_stage_min_steps at EVERY curriculum stage, not just the last --
        # same fix, same reason, as train_toy_memory.py's flag of the same name.
        curriculum_active = len(n_facts_stages) > 1
        protecting_stage = curriculum_active and stage_dwell < args.final_stage_min_steps
        if step >= args.max_steps and not protecting_stage:
            print(f"Step budget of {step} reached. Stopping.", flush=True)
            break
        if elapsed > max_time_seconds:
            if protecting_stage:
                if stage_dwell == 0:
                    print(f"Time budget of {args.max_time_minutes} minutes reached, but extending to "
                          f"guarantee --final_stage_min_steps={args.final_stage_min_steps} at the current "
                          f"stage (n_facts={args.n_facts}).", flush=True)
            else:
                print(f"Time budget of {args.max_time_minutes} minutes reached. Stopping.", flush=True)
                break

        x, target, _ = sample_batch(args.batch_size, args.n_facts, args.vocab_size, device)
        outs = model(x, target=1, n_latent=args.n_latent, n_step=n_step,
                    n_memory=args.n_memory, is_full_ar=False, is_output_ar=False,
                    x_reveal_mask=reveal_mask.to(device),
                    latent_reset_steps={args.n_facts} if args.latent_reset_at_query else None)
        logits = outs[1][:, -1, -1, :]  # (B, vocab)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if step % args.eval_every == 0:
            ev = evaluate(model, args, device, n_step, reveal_mask)
            logger.progress(step, loss=loss.item(), acc=ev["acc"],
                            retained_acc=ev["retained_acc"],
                            acc_excl_last=ev.get("acc_excl_last"),
                            stage_n_facts=args.n_facts)
            print(f"step={step:6d} stage_n_facts={args.n_facts} elapsed={elapsed/60:.2f}m loss={loss.item():.4f} "
                  f"acc={ev['acc']:.4f} retained_acc={ev['retained_acc']:.4f} "
                  f"evicted_acc={ev['evicted_acc']:.4f} (predicted_acc={ceiling['predicted_acc']:.4f}) "
                  f"acc_excl_last={ev['acc_excl_last']:.4f} "
                  f"recency_match_excl_last={ev['recency_match_excl_last']:.4f} "
                  f"acc_by_idx={['%.3f' % a for a in ev['acc_by_idx']]}", flush=True)
            if ev["acc"] > best_acc:
                best_acc = ev["acc"]

            # Promote on acc_excl_last, not raw acc -- raw acc is inflated by the ~1/n_facts
            # recency-echo coincidence regardless of mechanism (see that metric's docstring),
            # so it would let a pure shortcut satisfy the promotion criterion.
            # At n_facts=1 there is structurally no "excl_last" episode (the query always
            # targets the only fact ever written), so acc_excl_last is nan and `nan >= threshold`
            # is always False -- a curriculum starting at n_facts=1 could never promote past
            # stage 0 without this fallback. acc alone is safe here specifically because the
            # shortcut/genuine-retrieval distinction acc_excl_last exists to make doesn't apply
            # when there is only one fact to begin with.
            promote_metric = ev["acc"] if math.isnan(ev["acc_excl_last"]) else ev["acc_excl_last"]
            if (stage_idx < len(n_facts_stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and promote_metric >= args.curriculum_promote_acc):
                stage_idx += 1
                stage_start_step = step
                best_acc = 0.0
                n_step, T, reveal_mask, ceiling, recency_baseline_acc, mode_baseline = setup_stage(n_facts_stages[stage_idx])
                print(f"step={step:6d} CURRICULUM PROMOTE -> n_facts={n_facts_stages[stage_idx]}", flush=True)

        step += 1

    final = evaluate(model, args, device, n_step, reveal_mask, n_eval=16)
    print("\n---", flush=True)
    print(f"n_facts_stages:        {n_facts_stages}", flush=True)
    print(f"final_stage_n_facts:   {args.n_facts} (stage {stage_idx + 1}/{len(n_facts_stages)})", flush=True)
    print(f"final_acc:            {final['acc']:.4f}", flush=True)
    print(f"final_retained_acc:   {final['retained_acc']:.4f}  (should be near 1.0 if the mechanism works at all)", flush=True)
    print(f"final_evicted_acc:    {final['evicted_acc']:.4f}  (should be near vocab_chance={mode_baseline['vocab_chance']:.4f} -- info genuinely gone, not a mechanism failure)", flush=True)
    print(f"predicted_acc:        {ceiling['predicted_acc']:.4f}  (perfect-mechanism reference, capacity_ceiling)", flush=True)
    print(f"acc_excl_last:        {final['acc_excl_last']:.4f}  "
          f"(accuracy on the ~{1 - 1/args.n_facts:.0%} of episodes where the query does NOT target the "
          f"last-written fact -- near vocab_chance={1/args.vocab_size:.4f} means the overall acc above is "
          f"coming almost entirely from the shortcut, not genuine retrieval; well above chance means real "
          f"retrieval is happening on at least part of the buffer)", flush=True)
    print(f"recency_match_excl_last: {final['recency_match_excl_last']:.4f}  "
          f"(on that same subset, fraction of predictions equal to 'echo the last-written value' -- near "
          f"1.0 alongside a chance-level acc_excl_last confirms the shortcut; near chance is consistent "
          f"with genuine query-conditioned retrieval instead)", flush=True)
    print(f"acc_by_idx (index 0=oldest write .. {args.n_facts-1}=most recent write): "
          f"{['%.4f' % a for a in final['acc_by_idx']]}  "
          f"(a small effective window -- near-perfect on the last 1-2 indices, near chance on older ones "
          f"-- would explain a partial-but-above-chance acc_excl_last as 'recalls the last few writes, not "
          f"further back', distinct from either a pure recency shortcut or full n_memory-wide retrieval)", flush=True)
    print(f"most_common_value baseline: {mode_baseline['token_acc']:.4f}", flush=True)
    print(f"best_acc:             {best_acc:.4f}", flush=True)
    print(f"training_seconds:     {elapsed:.1f}", flush=True)
    print(f"num_steps:            {step}", flush=True)

    logger.finish(
        summary={"final_acc": final["acc"],
                 "final_retained_acc": final["retained_acc"],
                 "final_evicted_acc": final["evicted_acc"],
                 "acc_excl_last": final["acc_excl_last"],
                 "recency_match_excl_last": final["recency_match_excl_last"],
                 "predicted_acc": ceiling["predicted_acc"],
                 "best_acc": best_acc, "final_stage_n_facts": args.n_facts,
                 "num_steps": step, "training_seconds": elapsed},
        # Ici le raccourci de récence est le vrai risque de surinterprétation,
        # pas une fuite d'entrée : acc_excl_last au niveau du hasard avec un
        # recency_match_excl_last proche de 1.0 signe le raccourci. C'est donc
        # lui qui sert de leak_check.
        controls={"chance": mode_baseline["vocab_chance"],
                  "margin": final["acc_excl_last"] - mode_baseline["vocab_chance"],
                  "leak_check": final["recency_match_excl_last"]},
    )


if __name__ == "__main__":
    main()
