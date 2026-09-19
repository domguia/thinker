"""
Evaluation harness for a trained Thinker real-text checkpoint (produced by
train_real_text.py's --save_checkpoint_path), directly testing the
architecture's own hypotheses rather than just reporting a loss number:

1. **Memory manipulation (spec Sec.-1's central claim)**: does the model
   actually USE the KB/long-range memory read (o_kb via
   HierarchicalMemory.attend), or could it get the same held-out loss
   relying on the local/recent context alone? Tested by toggling
   `model.disable_kb` (a plain attribute read in Thinker.forward -- no
   submodule surgery needed, see core/indexed_thinker_model.py) between two
   evaluation passes on the SAME checkpoint and comparing held-out loss.
   If disabling the KB barely hurts, the model isn't manipulating long-range
   memory the way the architecture is meant to -- a real negative finding,
   not a training bug.

2. **Step extrapolation ("thinking longer")**: reuses train_real_text.py's
   own evaluate_at_nstep() at several n_step_test values, on held-out data.

Requires --val_data (a genuine held-out split, see train_real_text.py's
--val_data docstring) -- refuses to run on the training file alone, since
an ablation/extrapolation reading on memorized data would be uninformative
about the architecture, not just imprecise.
"""
from __future__ import annotations

import argparse
import json

import torch

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.real_text_windows import RealTextWindowDataset
from learn.indexed_attention.train_real_text import evaluate_at_nstep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to a .pt saved by --save_checkpoint_path")
    ap.add_argument("--val_data", required=True, help="held-out JSONL, same format as train_real_text.py's --data")
    ap.add_argument("--tokenizer", default="lfm2")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_ctx", type=int, default=256)
    ap.add_argument("--t_local", type=int, default=32)
    ap.add_argument("--t_tgt", type=int, default=32)
    ap.add_argument("--n_lanes", type=int, default=8)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--n_slots", type=int, default=1)
    ap.add_argument("--n_step", type=int, default=6, help="training n_step, used as the extrapolation reference point")
    ap.add_argument("--extrapolate_n_steps", default="1,2,4,8,12,16",
                     help="comma-separated n_step_test values, should include --n_step itself for reference")
    ap.add_argument("--n_eval_batches", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    device = torch.device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    vocab_size = len(tok)

    val_ds = RealTextWindowDataset(args.val_data, tok, n_ctx=args.n_ctx, t_local=args.t_local,
                                    t_tgt=args.t_tgt, pad_id=pad_id)
    print(f"loaded held-out val: {len(val_ds.docs)} docs, {len(val_ds.windows)} windows", flush=True)

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        max_target_len=args.t_tgt, stream_sequence={"answer": True},
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"loaded checkpoint {args.checkpoint}", flush=True)

    results = {}

    # 1. Memory ablation: disable_kb is a plain bool read in Thinker.forward()
    # (core/indexed_thinker_model.py) -- togglable post-hoc on an already
    # loaded model, no reconstruction needed.
    model.disable_kb = False
    loss_with_kb = evaluate_at_nstep(model, val_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                      device, args.n_step, n_eval_batches=args.n_eval_batches)
    model.disable_kb = True
    loss_without_kb = evaluate_at_nstep(model, val_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                         device, args.n_step, n_eval_batches=args.n_eval_batches)
    model.disable_kb = False
    kb_gap = loss_without_kb - loss_with_kb
    print(f"\n[memory ablation] loss_with_kb={loss_with_kb:.4f}  loss_without_kb={loss_without_kb:.4f}  "
          f"gap={kb_gap:.4f} ({'KB genuinely used' if kb_gap > 0.02 else 'KB barely used -- check before trusting the mechanism'})",
          flush=True)
    results["memory_ablation"] = {"loss_with_kb": loss_with_kb, "loss_without_kb": loss_without_kb, "gap": kb_gap}

    # 2. Step extrapolation.
    print("\n[step extrapolation] (held-out)", flush=True)
    extrap = {}
    for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
        loss = evaluate_at_nstep(model, val_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                  device, n_step_test, n_eval_batches=args.n_eval_batches)
        ppl = torch.exp(torch.tensor(loss)).item()
        extrap[n_step_test] = loss
        marker = " <- training n_step" if n_step_test == args.n_step else ""
        print(f"  n_step_test={n_step_test:3d} loss={loss:.4f} ppl={ppl:.2f}{marker}", flush=True)
    results["extrapolation"] = extrap

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
