"""X1 M3: looped dense transformer (single shared GPT2 block, reused n_step
times -- same weight-tying mechanism as learn/distill/train_looped_dense.py,
E3) on the T1/T3 synthetic tasks -- gate G2 (X1_DISPATCH.md §3): must reach
>=95% EM in-distribution before it's trusted as the "looped dense" reference
point for comparison against M1 Thinker (G3).

Reuses learn/x1/tasks.py's place-value position scheme and
learn/x1/train_dense.py's data collation / greedy-decode / exact-match-eval
machinery (including the attention_mask fix from gate G1, commit 2aaea79) --
only the model construction and the per-step n_step curriculum are new here.

Training curriculum: n_step ~ U(1, n_step_train_max) per step (X1_DISPATCH.md
§2 "recette E8/E13"), transformer.h temporarily sliced to the first n_step
entries (all aliases of the same block) for that forward pass, then restored
before the optimizer step / checkpointing so a saved checkpoint always has
the full n_layer=n_step_train_max structure.

Eval (in- and out-of-distribution EM) is done at a FIXED n_step_test (default:
n_step_train_max, i.e. full budget) via --n_step_test -- X1's actual OOD
question is "does more n_step_test help on larger problems", but that sweep
belongs to the full grid (G2 itself only needs ONE n_step_test value, high
enough to give the model its best shot, to pass or fail the gate)."""
from __future__ import annotations

import argparse
import random
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from learn.x1.tasks import TASKS, PAD_ID, VOCAB_SIZE, TOK2ID, EOS
from learn.x1.train_dense import collate, exact_match_eval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--train_size_range", required=True, help="e.g. 32,32")
    ap.add_argument("--test_size_range", required=True, help="e.g. 64,512")
    ap.add_argument("--position_offset_max", type=int, default=0)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_positions", type=int, default=1088)
    ap.add_argument("--n_step_train_max", type=int, default=8, help="ModuleList aliased to this many "
                     "entries; also the upper bound of the per-step random.randint(1, n) curriculum")
    ap.add_argument("--n_step_test", type=int, default=None, help="default: n_step_train_max")
    ap.add_argument("--n_step_test_sweep", default=None, help="comma-separated n_step_test values "
                     "(may exceed n_step_train_max, see eval_em's dynamic aliasing) evaluated at the end "
                     "using the same trained checkpoint -- e.g. 1,2,4,8,12,16,24,32")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--max_time_minutes", type=float, default=60.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    n_step_test = args.n_step_test or args.n_step_train_max

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    gen_fn = TASKS[args.task]
    train_lo, train_hi = map(int, args.train_size_range.split(","))
    test_lo, test_hi = map(int, args.test_size_range.split(","))
    eos_id = TOK2ID[EOS]

    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = args.n_step_train_max
    config.n_embd = args.n_embd
    config.n_head = args.n_head
    config.vocab_size = VOCAB_SIZE
    config.n_positions = args.n_positions
    config.n_ctx = args.n_positions
    model = AutoModelForCausalLM.from_config(config).to(device)

    # Weight-tie every block to block 0 (same mechanism as E3's
    # train_looped_dense.py) -- model.parameters() dedupes by object identity.
    base_block = model.transformer.h[0]
    for i in range(1, args.n_step_train_max):
        model.transformer.h[i] = base_block
    full_h = model.transformer.h

    core_params = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "lm_head" not in n)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model (M3 looped dense): {n_params / 1e6:.2f}M params ({core_params / 1e6:.2f}M in the 1 shared "
          f"block), n_step_train_max={args.n_step_train_max} n_embd={args.n_embd}", flush=True)

    def eval_em(examples, n_step: int) -> float:
        # dynamic aliasing (not full_h[:n_step]): lets n_step_test EXCEED
        # n_step_train_max (the actual H2 extrapolation question) -- GPT2Model.forward
        # iterates over self.h directly (not config.n_layer), so any length works.
        model.transformer.h = torch.nn.ModuleList([base_block for _ in range(n_step)])
        em = exact_match_eval(model, examples, device, eos_id)
        model.transformer.h = full_h
        return em

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    step = 0
    best_id_em = 0.0
    while step < args.max_steps and (time.time() - start) / 60 < args.max_time_minutes:
        batch_examples = gen_fn(args.batch_size, (train_lo, train_hi), seed=args.seed * 1_000_003 + step,
                                 position_offset_max=args.position_offset_max)
        batch = collate(batch_examples, device)
        n_step = random.randint(1, args.n_step_train_max)
        model.transformer.h = full_h[:n_step]
        # use_cache=False required: aliased ModuleList entries (weight-tied
        # blocks) share layer_idx=0, so a single forward call that loops over
        # them with caching enabled appends mismatched-length KV cache
        # entries within the SAME call -- confirmed via the identical crash
        # signature as E3 (learn/distill/train_looped_dense.py).
        out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                     attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        model.transformer.h = full_h
        step += 1
        if step % 20 == 0:
            print(f"step={step} loss={out.loss.item():.4f} n_step={n_step} elapsed={(time.time()-start)/60:.2f}m",
                  flush=True)
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_000 + step,
                                      position_offset_max=0)
                id_em = eval_em(id_examples, n_step_test)
            print(f"step={step} IN-DIST EM={id_em:.4f} (n_step_test={n_step_test})", flush=True)
            model.train()
            if id_em > best_id_em:
                best_id_em = id_em
                if args.save_dir:
                    import os
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pt")

    model.eval()
    with torch.no_grad():
        id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_001, position_offset_max=0)
        id_em = eval_em(id_examples, n_step_test)
        ood_examples = gen_fn(args.n_eval, (test_lo, test_hi), seed=999_002, position_offset_max=0)
        ood_em = eval_em(ood_examples, n_step_test)
    print(f"FINAL in-distribution EM={id_em:.4f} ({args.train_size_range}) n_step_test={n_step_test}")
    print(f"FINAL OOD EM={ood_em:.4f} ({args.test_size_range}) n_step_test={n_step_test}")
    print(f"best_in_dist_em_during_training={best_id_em:.4f}")

    if args.n_step_test_sweep:
        sweep_values = [int(v) for v in args.n_step_test_sweep.split(",")]
        for ns in sweep_values:
            id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_001, position_offset_max=0)
            id_sweep = eval_em(id_examples, ns)
            ood_examples = gen_fn(args.n_eval, (test_lo, test_hi), seed=999_002, position_offset_max=0)
            ood_sweep = eval_em(ood_examples, ns)
            print(f"SWEEP n_step_test={ns} in-dist EM={id_sweep:.4f} OOD EM={ood_sweep:.4f}")


if __name__ == "__main__":
    main()
