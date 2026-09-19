"""
I7 real-scale rerun, corrected: the toy CPU checkpoint used earlier tonight
had training margin (N_step - n_hops) = 14, making "no drift to N_step=24"
almost trivial (retracted, see experiment.log.md). This version uses
Etape 4's real hardened-scale checkpoints (n_hops=4, d_model=256,
--save_checkpoint_path added tonight) trained at N_step in {5,6,8,16} --
i.e. training margins of {1,2,4,12} -- so the margin itself is the swept
variable, directly testing whether a LOW-margin model (margin=1, almost no
slack) still holds its answer well past its own trained N_step, or whether
that's only true when a large margin was baked in.

model(...) takes n_step directly (core/indexed_thinker_model.py's Thinker),
so no manual unroll is needed for the eval-time generalization sweep -- only
the read-step curve (I7b) needs to read out accuracy at every step of one
long run, which does need a manual unroll matching evaluate()'s call shape.
"""
from __future__ import annotations

import argparse
import json

import torch

from data.kb_chain_retrieval import KBChainDataset
from learn.indexed_attention.train_kb_chain import build_model

CKPT_DIR = "checkpoints/etape4"
# run_id -> (n_step_trained, seed), from runs/itemA_etape4_checkpoints/grid.jsonl
CHECKPOINTS = {
    "train_kb_chain-53032a3d96": (5, 0), "train_kb_chain-47bb9c2a65": (5, 1), "train_kb_chain-38ca85c38f": (5, 2),
    "train_kb_chain-cb6d3477c5": (6, 0), "train_kb_chain-cb31098b3e": (6, 1), "train_kb_chain-4d46227607": (6, 2),
    "train_kb_chain-8b2a97ac80": (8, 0), "train_kb_chain-6ba7eeac84": (8, 1), "train_kb_chain-f484380d01": (8, 2),
    "train_kb_chain-66e7186acb": (16, 0), "train_kb_chain-effb075d67": (16, 1), "train_kb_chain-33f8de12a7": (16, 2),
}

N_HOPS = 4
N_DISTRACTORS = 4
VOCAB_SIZE = 64
D_MODEL = 256
DEPTH = 2
BLOCK_SIZE = 4


class Args:
    """Minimal stand-in for argparse.Namespace, matching build_model()'s reads."""
    d_model = D_MODEL
    n_register = 1
    block_size = BLOCK_SIZE
    depth = DEPTH
    n_slots = 1
    n_head = 1
    sm_cap = None
    disable_sm = False
    use_ff = False
    ff_hidden_mult = 4
    detach_sm_keys = False
    level_dropout_p = 0.0
    shared_kv_pooling = False
    pool_n_head = 1
    k_dim = None


@torch.no_grad()
def eval_at_n_step(model, ds, n_step: int, n_batches: int, batch_size: int) -> float:
    correct, total = 0, 0
    for _ in range(n_batches):
        kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(batch_size)
        _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=n_step, kb_leaf_mask=kb_mask)
        preds = streams["answer"][:, 0, :].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, choices=list(CHECKPOINTS))
    ap.add_argument("--n_eval_batches", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_step_trained, seed = CHECKPOINTS[args.run_id]
    margin = n_step_trained - N_HOPS
    max_facts = N_HOPS + N_DISTRACTORS

    torch.manual_seed(1)
    ds_args = Args()
    ds = KBChainDataset(n_hops=N_HOPS, n_distractors=N_DISTRACTORS, vocab_size=VOCAB_SIZE,
                         max_facts=max_facts, seed=seed + 10_000)  # held-out eval seed, per make_datasets()
    model = build_model(ds_args, ds.total_vocab_size, torch.device("cpu"))
    model.load_state_dict(torch.load(f"{CKPT_DIR}/{args.run_id}.pt", map_location="cpu"))
    model.eval()

    print(f"=== run_id={args.run_id} trained N_step={n_step_trained} margin={margin} (n_hops={N_HOPS}) ===")
    test_points = sorted(set([2, 4, 6, 8, 12, 16, 20, 24, n_step_trained,
                               n_step_trained + 4, n_step_trained + 8, n_step_trained * 2]))
    results = {}
    for n_step_test in test_points:
        if n_step_test < 1:
            continue
        acc = eval_at_n_step(model, ds, n_step_test, args.n_eval_batches, args.batch_size)
        flag = " <-- trained N_step" if n_step_test == n_step_trained else ""
        print(f"N_step_test={n_step_test:3d}  acc={acc:.4f}{flag}")
        results[n_step_test] = acc

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"run_id": args.run_id, "n_step_trained": n_step_trained, "seed": seed,
                       "margin": margin, "results": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
