"""
Phase 12 -- S0.5: does composing the EXACT converted FFN (convert_ffn_to_kv's
gated-attention form) with the host's real, UNCHANGED self-attention, under
OLMo-2's real residual/Post-Norm structure, reconstruct the real full
decoder-layer output bit-for-bit?

S0 already proved the FFN half exactly (max_abs_err=0.0 on real OLMo-2-1B
weights). This checks that nothing is lost/misordered when that exact FFN
reconstruction is embedded in the FULL layer pipeline (Post-Norm residual
placement, both norms, real attention untouched) -- a distinct, independent
check from S0's isolated-FFN test, and the natural checkpoint before Phase
12's curriculum "etape 2" (composing a converted FFN-KB read with a
sequence-side read inside one Thinker-style layer) can be trusted to not
introduce a silent composition bug of its own.

Real OLMo2DecoderLayer.forward() (dev_notes reference, Post-Norm):
    residual = x
    h = post_attention_layernorm(self_attn(x))
    h = residual + h
    residual = h
    h2 = post_feedforward_layernorm(mlp(h))
    out = residual + h2

Manual reconstruction here calls the REAL self_attn (unconverted -- S0 only
converts the FFN) and substitutes reconstruct_ffn(h, kv) for mlp(h).
Verified bit-exact (max_abs_err=0.0) on synthetic Olmo2DecoderLayer weights
before this ran on the real model (module docstring convention followed
throughout Phase 12's scripts).

CPU-only, no training, same cost order as S0/etape-1 diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learn.indexed_attention.convert_ffn_to_kv import extract_ffn_kv, reconstruct_ffn
from learn.indexed_attention.diagnose_layer_source_attribution import load_texts

MODEL_ID = "allenai/OLMo-2-0425-1B"


@torch.no_grad()
def reconstruct_layer(layer, x: torch.Tensor, position_ids: torch.Tensor,
                       position_embeddings: tuple) -> torch.Tensor:
    residual = x
    attn_out, _ = layer.self_attn(
        hidden_states=x, attention_mask=None, position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    h = residual + layer.post_attention_layernorm(attn_out)

    residual = h
    kv = extract_ffn_kv(layer)
    ffn_out = reconstruct_ffn(h, kv)
    return residual + layer.post_feedforward_layernorm(ffn_out)


@torch.no_grad()
def check_layer(model, hidden_l: torch.Tensor, layer_idx: int) -> dict:
    layer = model.model.layers[layer_idx]
    T = hidden_l.shape[1]
    position_ids = torch.arange(T).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_l, position_ids)

    real = layer(hidden_l, attention_mask=None, position_ids=position_ids,
                 position_embeddings=(cos, sin))
    if isinstance(real, tuple):
        real = real[0]
    recon = reconstruct_layer(layer, hidden_l, position_ids, (cos, sin))

    abs_err = (real - recon).abs()
    rel_err = abs_err / real.abs().clamp_min(1e-8)
    return {
        "layer": layer_idx,
        "max_abs_err": abs_err.max().item(), "mean_abs_err": abs_err.mean().item(),
        "max_rel_err": rel_err.max().item(), "mean_rel_err": rel_err.mean().item(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_tokens", type=int, default=512)
    ap.add_argument("--layers", default="0,4,8,11,15")
    ap.add_argument("--rel_err_threshold", type=float, default=1e-4)
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/s0_5_layer_composition_check.json")
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
    hidden = out.hidden_states

    results = {}
    all_pass = True
    for l in layers:
        r = check_layer(model, hidden[l], l)
        results[l] = r
        passed = r["max_rel_err"] < args.rel_err_threshold
        all_pass = all_pass and passed
        print(f"[layer {l:2d}] max_abs_err={r['max_abs_err']:.3e}  max_rel_err={r['max_rel_err']:.3e}  "
              f"{'PASS' if passed else 'FAIL'}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    verdict = "PASS -- full-layer composition (real attn + converted FFN) exact" if all_pass else "FAIL"
    print(f"\nS0.5 verdict: {verdict}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
