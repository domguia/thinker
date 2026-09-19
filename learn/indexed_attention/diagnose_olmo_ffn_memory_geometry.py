"""
Phase 12 S0-pre-check: two free (no training, no GPU) linear-algebra measures
on OLMo-2-1B's actual pretrained weights, requested by model-design to decide
S1's branch (i)/(ii) and S3's feasibility before any conversion code is written.

A1 -- FFN/attention key-space overlap, per layer.
    OLMo uses SwiGLU: FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down.
    W_gate carries the *selection* (which neurons fire), so it is the default
    "FFN keys" analog to W_in in the plain two-matrix case (spec's
    FFN(x) = sigma(x W_in) W_out <-> phi(Q K^T) V correspondence) -- W_up is
    reported too since it's cheap and the choice is argued, not proven.
    For each layer: principal angles between span(W_gate^T) (d_ff key
    vectors in d_model space) and span(W_K^T) (attention key vectors),
    energy of W_gate^T projected onto span(W_K), and the cosine-similarity
    distribution between the two key sets.

A2 -- cross-layer redundancy of the FFN memories.
    Stack all 16 layers' W_gate^T (16 * d_ff keys) and report: singular
    value spectrum + effective rank at 90%/99% energy thresholds; cosine
    similarity of per-layer key sets to each other (mean pairwise, and
    adjacent-vs-distant layers specifically, to check Geva's low/high
    stratification prediction).

CPU only. Downloads/caches the HF checkpoint on first run (~2GB, fp32
loaded as fp32 then cast to fp64 for the SVD/angle math -- numerically
important at d_ff=8192, d_model=2048 scale, fp32 SVD residuals are not
negligible here).
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "allenai/OLMo-2-0425-1B"


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """cos(principal angles) between span(A) and span(B), A:(n,d) B:(m,d)."""
    Qa, _ = torch.linalg.qr(A.T)  # (d, rank_a) orthonormal basis of span(A)
    Qb, _ = torch.linalg.qr(B.T)
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return s.clamp(-1, 1)  # cos(theta_i), descending


def effective_rank(S: torch.Tensor, energy: float) -> int:
    cum = torch.cumsum(S ** 2, dim=0)
    total = cum[-1]
    return int((cum < energy * total).sum().item()) + 1


def layer_a1(w_gate: torch.Tensor, w_up: torch.Tensor, w_k: torch.Tensor) -> dict:
    out = {}
    for name, w_ffn in (("W_gate", w_gate), ("W_up", w_up)):
        cos_angles = principal_angles(w_ffn, w_k)  # against attention keys
        # energy of w_ffn projected onto span(w_k)
        Qk, _ = torch.linalg.qr(w_k.T)
        proj = w_ffn @ Qk  # (d_ff, rank_k)
        energy_frac = (proj.pow(2).sum() / w_ffn.pow(2).sum()).item()
        # pairwise cosine sim distribution (subsample for cost: 512 x 512)
        idx_f = torch.randperm(w_ffn.shape[0])[:512]
        idx_k = torch.randperm(w_k.shape[0])[:min(512, w_k.shape[0])]
        cs = torch.nn.functional.cosine_similarity(
            w_ffn[idx_f].unsqueeze(1), w_k[idx_k].unsqueeze(0), dim=-1)
        out[name] = {
            "top10_principal_angle_cos": cos_angles[:10].tolist(),
            "n_angles_above_0.5": int((cos_angles > 0.5).sum().item()),
            "energy_fraction_on_attn_key_subspace": energy_frac,
            "cosine_sim_mean": cs.mean().item(),
            "cosine_sim_abs_mean": cs.abs().mean().item(),
            "cosine_sim_p90_abs": cs.abs().flatten().kthvalue(
                int(0.9 * cs.numel())).values.item(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/result.json")
    ap.add_argument("--dtype", default="float64")
    args = ap.parse_args()

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    dtype = getattr(torch, args.dtype)

    layers = model.model.layers
    n_layers = len(layers)
    d_model = model.config.hidden_size
    d_ff = model.config.intermediate_size
    print(f"n_layers={n_layers} d_model={d_model} d_ff={d_ff}", flush=True)

    a1_per_layer = []
    gate_keys_all = []
    for li, layer in enumerate(layers):
        w_gate = layer.mlp.gate_proj.weight.detach().to(dtype)  # (d_ff, d_model)
        w_up = layer.mlp.up_proj.weight.detach().to(dtype)
        w_k = layer.self_attn.k_proj.weight.detach().to(dtype)  # (d_kv, d_model)
        res = layer_a1(w_gate, w_up, w_k)
        res["layer"] = li
        a1_per_layer.append(res)
        gate_keys_all.append(w_gate)
        print(f"[A1] layer {li:2d}  W_gate energy_on_attn_key_subspace="
              f"{res['W_gate']['energy_fraction_on_attn_key_subspace']:.4f}  "
              f"cos_abs_mean={res['W_gate']['cosine_sim_abs_mean']:.4f}", flush=True)

    print("\n[A2] stacking all layers' W_gate keys ...", flush=True)
    K_all = torch.cat(gate_keys_all, dim=0)  # (n_layers*d_ff, d_model)
    S = torch.linalg.svdvals(K_all)
    rank90 = effective_rank(S, 0.90)
    rank99 = effective_rank(S, 0.99)
    print(f"[A2] total keys={K_all.shape[0]} d_model={d_model} "
          f"effective_rank(90%)={rank90} effective_rank(99%)={rank99}", flush=True)

    # inter-layer cosine similarity of key SETS: mean |cos| between a
    # subsample of layer i's keys and layer j's keys, adjacent vs distant.
    subsample = [gk[torch.randperm(gk.shape[0])[:256]] for gk in gate_keys_all]
    inter = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(n_layers):
            cs = torch.nn.functional.cosine_similarity(
                subsample[i].unsqueeze(1), subsample[j].unsqueeze(0), dim=-1)
            inter[i, j] = cs.abs().mean()
    adjacent = torch.tensor([inter[i, i + 1].item() for i in range(n_layers - 1)])
    distant = torch.tensor([inter[i, j].item() for i in range(n_layers)
                             for j in range(n_layers) if abs(i - j) >= n_layers // 2])
    print(f"[A2] adjacent-layer mean|cos|={adjacent.mean().item():.4f}  "
          f"distant-layer mean|cos|={distant.mean().item():.4f}", flush=True)

    result = {
        "model": MODEL_ID, "n_layers": n_layers, "d_model": d_model, "d_ff": d_ff,
        "A1_per_layer": a1_per_layer,
        "A2": {
            "total_keys": K_all.shape[0],
            "effective_rank_90pct": rank90,
            "effective_rank_99pct": rank99,
            "singular_values_top20": S[:20].tolist(),
            "adjacent_layer_mean_abs_cos": adjacent.mean().item(),
            "distant_layer_mean_abs_cos": distant.mean().item(),
            "inter_layer_abs_cos_matrix": inter.tolist(),
        },
    }
    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
