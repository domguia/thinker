"""
Phase 12 -- S0: exact FFN -> attention conversion, zero training, zero
approximation (dev_notes/indexed_attention_experiment_plan.md, Piste B / S0).

For a plain ReLU/GELU FFN, `FFN(x) = phi(x @ W_in) @ W_out` is *already*
`phi(Q K^T) V` with `Q=x`, `K=W_in^T`, `V=W_out` -- exact, no gating needed.
OLMo-2-1B uses SwiGLU (`down_proj(silu(gate_proj(x)) * up_proj(x))`), which
is the flagged "piege d'implementation principal de S0": a single elementwise
nonlinearity on ONE score isn't enough, since silu(x@gate_i) alone throws
away the (x@up_i) factor that also depends on x. The exact form needs TWO
per-neuron scores combined multiplicatively (spec's "gated linear
attention"):

    weight_i(x) = silu(x . gate_i) * (x . up_i)
    FFN(x)      = sum_i weight_i(x) * down_i  =  weight(x) @ V

with K_gate = W_gate (d_ff, d_model), K_up = W_up (d_ff, d_model),
V = W_down^T (d_ff, d_model) -- down_proj.weight has shape (d_model, d_ff)
in the HF Linear convention, so down_proj(y) = y @ down_proj.weight.T, i.e.
V = down_proj.weight.T (row i = down_proj's i-th INPUT direction = the
value vector for neuron i). This is NOT routed through
HierarchicalMemory.attend()'s softmax (no softmax anywhere in an exact FFN
reconstruction) -- it's a direct bilinear-gated recomputation, checked
bit-for-bit (up to fp precision) against the model's own `layer.mlp(x)`.

Pass/fail gate (per the plan's own S0 decision table): perplexity/output
identical to the numeric epsilon -> harness correct, proceed to S1. Nonzero
gap -> conversion bug (most likely the gating), do not proceed.

Consequence for branch (i) of S1 (dev_notes/indexed_attention_experiment_plan.md,
"deux lectures separees ... noyau relu/sigmoid cote FFN"): now that the
exact SwiGLU kernel is available and cheap (this module), there is no reason
to fall back to a generic relu/sigmoid placeholder kernel for THIS host
model's KB read in S1/S2 -- use `gated_swiglu_weights` below as branch (i)'s
actual FFN-side kernel, not an approximation of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn.indexed_attention.diagnose_layer_source_attribution import load_texts

MODEL_ID = "allenai/OLMo-2-0425-1B"


def extract_ffn_kv(layer) -> dict:
    """Static K/V for one OLMo-2 SwiGLU layer -- see module docstring for shapes."""
    mlp = layer.mlp
    return {
        "K_gate": mlp.gate_proj.weight.detach(),  # (d_ff, d_model)
        "K_up": mlp.up_proj.weight.detach(),       # (d_ff, d_model)
        "V": mlp.down_proj.weight.detach().T.contiguous(),  # (d_ff, d_model)
    }


def gated_swiglu_weights(x: torch.Tensor, K_gate: torch.Tensor, K_up: torch.Tensor) -> torch.Tensor:
    """
    weight_i(x) = silu(x . gate_i) * (x . up_i) -- Q=x, no extra projection,
    per-neuron gated attention weight, exact (not approximate).
    x: (..., d_model). Returns (..., d_ff).
    """
    scores_gate = x @ K_gate.T
    scores_up = x @ K_up.T
    return F.silu(scores_gate) * scores_up


def reconstruct_ffn(x: torch.Tensor, kv: dict) -> torch.Tensor:
    """Exact FFN(x) via the gated-attention form -- should equal layer.mlp(x) bit-for-bit."""
    weights = gated_swiglu_weights(x, kv["K_gate"], kv["K_up"])  # (..., d_ff)
    return weights @ kv["V"]  # (..., d_model)


@torch.no_grad()
def check_layer(model, x_l: torch.Tensor, layer_idx: int) -> dict:
    layer = model.model.layers[layer_idx]
    kv = extract_ffn_kv(layer)
    real = layer.mlp(x_l)
    recon = reconstruct_ffn(x_l, kv)
    abs_err = (real - recon).abs()
    rel_err = abs_err / real.abs().clamp_min(1e-8)
    return {
        "layer": layer_idx,
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "max_rel_err": rel_err.max().item(),
        "mean_rel_err": rel_err.mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_tokens", type=int, default=512)
    ap.add_argument("--layers", default="0,4,8,11,15", help="comma-separated OLMo layer indices to check")
    ap.add_argument("--rel_err_threshold", type=float, default=1e-4,
                     help="pass/fail gate: max_rel_err must stay below this (fp32 numerics)")
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/s0_conversion_check.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0, {n_layers})"

    texts = load_texts(args.text)
    ids = tok("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=args.n_tokens)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hidden = out.hidden_states  # hidden[l] = input to layer l (x_l)

    results = {}
    all_pass = True
    for l in layers:
        r = check_layer(model, hidden[l][0], l)
        results[l] = r
        passed = r["max_rel_err"] < args.rel_err_threshold
        all_pass = all_pass and passed
        print(f"[layer {l:2d}] max_abs_err={r['max_abs_err']:.3e}  mean_abs_err={r['mean_abs_err']:.3e}  "
              f"max_rel_err={r['max_rel_err']:.3e}  mean_rel_err={r['mean_rel_err']:.3e}  "
              f"{'PASS' if passed else 'FAIL'} (threshold={args.rel_err_threshold:.0e})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    verdict = "PASS -- harness correct, proceed to S1" if all_pass else "FAIL -- conversion bug, do not proceed to S1"
    print(f"\nsummary: {json.dumps({'layers': layers, 'threshold': args.rel_err_threshold, 'all_pass': all_pass})}")
    print(f"S0 verdict: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
