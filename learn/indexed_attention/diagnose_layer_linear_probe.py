"""
Phase 12 / S3 curriculum, follow-up to diagnose_layer_source_attribution.py's
negative result (dev_notes/indexed_attention_experiment_plan.md, S3 "etape
1"): a real query's raw dot product against gate_proj-derived FFN keys does
NOT retrieve its own host layer's region above chance (mass ~0.33 everywhere,
identical to the shuffled-label control) -- consistent with, and largely
predicted by, A1/A2's earlier finding that FFN key sets across layers sit at
the random-vector cosine floor (quasi-orthogonal, no directional alignment).

That result leaves two readings open, and they call for opposite next steps:
  (a) x_l genuinely carries no separable "which depth am I" information --
      an EXPLICIT external signal (regime 3's step embedding) is required,
      not just helpful;
  (b) the information IS present in x_l but not accessible via a raw,
      untrained dot product against gate_proj rows specifically -- a learned
      (even simple, linear) mapping could recover it, which would keep
      regime 4 (content-addressed, no loop-weight signal) alive as long as
      SOME learned projection is added on the query or key side.

This script distinguishes (a) from (b) the cheap way: a linear probe
(single nn.Linear + softmax, no hidden layer) trained to predict "which of
the N candidate layers did this token's x_l come from" directly from x_l.
If even a linear, small-data probe clears chance by a wide margin on held-out
tokens, (b) is right and a learned per-layer signal is worth pursuing before
falling back to the more expensive regime 3 (explicit step embedding in the
loop). If the probe stays near chance too, (a) is the better-supported
reading and regime 3/D become the more promising next moves.

Cheap by design: d_model=2048, 3-15 classes, a few thousand tokens, a linear
model -- trains in seconds on CPU once activations are extracted. No GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn.indexed_attention.diagnose_layer_source_attribution import load_texts

MODEL_ID = "allenai/OLMo-2-0425-1B"


@torch.no_grad()
def collect_layer_activations(model, tok, texts: list[str], layers: list[int],
                               tokens_per_shard: int = 256) -> dict:
    """
    Returns {layer_idx: (n_tok, d_model) tensor of real x_l activations},
    pooled across several short text shards so more than one forward's worth
    of tokens can be collected (a single 512-token pass alone would give too
    few examples per class for even a linear probe's held-out split).
    """
    per_layer = {l: [] for l in layers}
    shard = []
    for t in texts:
        shard.append(t)
        ids = tok("\n\n".join(shard), return_tensors="pt", truncation=True, max_length=tokens_per_shard)
        out = model(**ids, output_hidden_states=True)
        for l in layers:
            per_layer[l].append(out.hidden_states[l][0].clone())
        shard = []
    return {l: torch.cat(v, dim=0) for l, v in per_layer.items()}


def train_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int, d_model: int,
                 test_frac: float = 0.3, epochs: int = 200, lr: float = 1e-2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, y_train = X[train_idx].float(), y[train_idx]
    X_test, y_test = X[test_idx].float(), y[test_idx]

    probe = nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(X_train), y_train)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (probe(X_train).argmax(-1) == y_train).float().mean().item()
        test_acc = (probe(X_test).argmax(-1) == y_test).float().mean().item()
        final_loss = nn.functional.cross_entropy(probe(X_train), y_train).item()

    return {
        "train_acc": train_acc, "test_acc": test_acc, "final_train_loss": final_loss,
        "chance_floor": 1.0 / n_classes, "n_train": len(train_idx), "n_test": len(test_idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_docs", type=int, default=64, help="number of short text shards to pool tokens from")
    ap.add_argument("--tokens_per_shard", type=int, default=256)
    ap.add_argument("--layers", default="0,8,15")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/layer_linear_probe.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0, {n_layers})"

    texts = load_texts(args.text, n_needed=args.n_docs)
    print(f"pooling activations over {len(texts)} shards x {args.tokens_per_shard} tokens ...", flush=True)
    activations = collect_layer_activations(model, tok, texts, layers, args.tokens_per_shard)
    for l in layers:
        print(f"  layer {l}: {activations[l].shape[0]} tokens collected", flush=True)

    X = torch.cat([activations[l] for l in layers], dim=0)
    y = torch.cat([torch.full((activations[l].shape[0],), i, dtype=torch.long) for i, l in enumerate(layers)])

    result = train_probe(X, y, n_classes=len(layers), d_model=d_model, seed=args.seed)
    result["layers"] = layers
    print(f"\n[probe] train_acc={result['train_acc']:.4f}  test_acc={result['test_acc']:.4f}  "
          f"chance_floor={result['chance_floor']:.4f}  n_train={result['n_train']} n_test={result['n_test']}",
          flush=True)
    verdict = ("test_acc clears chance by a wide margin -> reading (b): information IS present, "
               "learnable-tag regime worth pursuing"
               if result["test_acc"] > result["chance_floor"] + 0.15 else
               "test_acc near chance -> reading (a): no separable depth signal in x_l, "
               "an explicit step signal (regime 3) is likely required")
    print(f"[verdict] {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**result, "verdict": verdict}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
