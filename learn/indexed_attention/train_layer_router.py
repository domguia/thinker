"""
Phase 12 / S3 curriculum, third step in the etape-1 disambiguation sequence
(dev_notes/indexed_attention_experiment_plan.md, S3 "etape 1"):

  1. diagnose_layer_source_attribution.py -- raw, UNTRAINED dot product
     against gate_proj-derived FFN keys: no signal (mass ~ chance).
  2. diagnose_layer_linear_probe.py -- is layer identity linearly decodable
     from x_l AT ALL? Yes, near-perfectly (test_acc=0.9995), and the
     normalization ablation confirmed it survives L2-normalization (not a
     ||x_l|| growing-with-depth artifact, cf. A1) -- a genuine DIRECTIONAL
     signal exists.
  3. THIS script: does a genuine retrieval MECHANISM (a trained query
     projection feeding a softmax over the region bank, exactly the shape
     HierarchicalMemory.attend() uses) actually learn to exploit that
     directional signal to route to the correct memory region? Step 2
     answered a classification question ("can any linear map decode
     layer id"); this answers the actually load-bearing question for
     regime 4 ("can a retrieval-shaped map route correctly").

Trains a single nn.Linear query projection Q (matching HierarchicalMemory's
q_proj shape/role) to MAXIMIZE the same "mass on the true region" quantity
etape 1 measured (not a generic classification loss) -- the two should
coincide when the projection is expressive enough, but optimizing the
retrieval-shaped objective directly is the honest test for regime 4's
actual use case, not a proxy for it.

CPU-only, tiny (one Linear layer, held-out split, a few hundred gradient
steps) -- same cost order as the linear probe.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn.indexed_attention.diagnose_layer_linear_probe import collect_layer_activations
from learn.indexed_attention.diagnose_layer_source_attribution import load_texts

MODEL_ID = "allenai/OLMo-2-0425-1B"


def mass_on_true_region(scores: torch.Tensor, region_of: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    scores: (N, n_regions*d_ff), region_of: (n_regions*d_ff,) region id per key,
    y: (N,) true region id per query. Returns (N,) softmax mass on each
    query's true region.
    """
    probs = torch.softmax(scores, dim=-1)
    n_regions = int(region_of.max().item()) + 1
    # (N, n_regions) mass per region, via a one-hot matmul (vectorized, no python loop)
    one_hot = nn.functional.one_hot(region_of, num_classes=n_regions).float()  # (n_keys, n_regions)
    mass_per_region = probs @ one_hot  # (N, n_regions)
    return mass_per_region.gather(1, y.unsqueeze(1)).squeeze(1), mass_per_region


def train_router(X: torch.Tensor, y: torch.Tensor, K_all: torch.Tensor, region_of: torch.Tensor,
                  d_model: int, scale: float, test_frac: float = 0.3, epochs: int = 300,
                  lr: float = 1e-2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, y_train = X[train_idx].float(), y[train_idx]
    X_test, y_test = X[test_idx].float(), y[test_idx]

    q_proj = nn.Linear(d_model, d_model, bias=False)
    opt = torch.optim.Adam(q_proj.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        scores = (q_proj(X_train) @ K_all.T) * scale
        mass_true, _ = mass_on_true_region(scores, region_of, y_train)
        loss = -(mass_true.clamp_min(1e-8).log()).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores_tr = (q_proj(X_train) @ K_all.T) * scale
        mass_true_tr, mass_per_region_tr = mass_on_true_region(scores_tr, region_of, y_train)
        top1_tr = (mass_per_region_tr.argmax(-1) == y_train).float().mean().item()

        scores_te = (q_proj(X_test) @ K_all.T) * scale
        mass_true_te, mass_per_region_te = mass_on_true_region(scores_te, region_of, y_test)
        top1_te = (mass_per_region_te.argmax(-1) == y_test).float().mean().item()

    n_regions = int(region_of.max().item()) + 1
    return {
        "train_mass_on_true_region_mean": mass_true_tr.mean().item(),
        "test_mass_on_true_region_mean": mass_true_te.mean().item(),
        "train_top1_region_acc": top1_tr,
        "test_top1_region_acc": top1_te,
        "chance_floor_mass": 1.0 / n_regions,
        "chance_floor_top1": 1.0 / n_regions,
        "n_train": len(train_idx), "n_test": len(test_idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_docs", type=int, default=64)
    ap.add_argument("--tokens_per_shard", type=int, default=256)
    ap.add_argument("--layers", default="0,8,15")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/layer_router_train.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)
    cfg = model.config
    d_model = cfg.hidden_size
    head_dim = d_model // cfg.num_attention_heads
    scale = 1.0 / math.sqrt(head_dim)
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0, {n_layers})"

    d_ff = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    K_regions = [model.model.layers[l].mlp.gate_proj.weight.detach().float() for l in layers]
    K_all = torch.cat(K_regions, dim=0)  # (n_regions*d_ff, d_model)
    region_of = torch.repeat_interleave(torch.arange(len(layers)), d_ff)

    texts = load_texts(args.text, n_needed=args.n_docs)
    print(f"pooling activations over {len(texts)} shards x {args.tokens_per_shard} tokens ...", flush=True)
    activations = collect_layer_activations(model, tok, texts, layers, args.tokens_per_shard)
    for l in layers:
        print(f"  layer {l}: {activations[l].shape[0]} tokens collected", flush=True)

    X = torch.cat([activations[l] for l in layers], dim=0)
    y = torch.cat([torch.full((activations[l].shape[0],), i, dtype=torch.long) for i, l in enumerate(layers)])

    result = train_router(X, y, K_all, region_of, d_model, scale, seed=args.seed, epochs=args.epochs, lr=args.lr)
    result["layers"] = layers
    print(f"\n[router] train_mass={result['train_mass_on_true_region_mean']:.4f}  "
          f"test_mass={result['test_mass_on_true_region_mean']:.4f}  "
          f"train_top1={result['train_top1_region_acc']:.4f}  test_top1={result['test_top1_region_acc']:.4f}  "
          f"chance={result['chance_floor_mass']:.4f}  n_train={result['n_train']} n_test={result['n_test']}",
          flush=True)
    verdict = ("test_top1_region_acc clears chance by a wide margin -> a TRAINED query projection "
               "genuinely routes to the correct FFN-derived memory region via softmax attention "
               "(not just classifiable, actually retrieval-usable): regime 4 is a validated "
               "mechanism, not just a theoretical possibility"
               if result["test_top1_region_acc"] > result["chance_floor_top1"] + 0.3 else
               "test_top1_region_acc stays near chance despite the linear probe's near-perfect "
               "classification accuracy -> the directional signal exists but a single linear "
               "projection into a softmax-over-huge-key-bank doesn't recover it as cleanly as a "
               "direct classifier does (e.g. drowned by the same mass-vs-peak dynamics A1 found) "
               "-- regime 4 needs more than a bare linear q_proj, or regime 3 remains the safer bet")
    print(f"[verdict] {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**result, "verdict": verdict}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
