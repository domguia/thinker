"""
Corrected follow-up to diagnose_olmo_ffn_memory_geometry.py (A1/A2), after
model-design/ff2attn caught that BOTH original metrics were uninformative:

A1 v1 bug: mean pairwise cosine similarity between individual W_gate/W_K
row vectors doesn't distinguish "incompatible key spaces" from "same
ambient space, different rotation" -- two arbitrary orthonormal bases of
the SAME R^d also average to the random-vector cosine floor. Since W_K is
full rank (span = R^d_model entirely, per v1's own finding), the real
question isn't subspace membership at all -- it's SCORE SCALE/GEOMETRY:
given a real query from the residual stream, are q.k_ffn and q.k_attn
comparable in magnitude? If not, a unified softmax is dominated by one
source regardless of relevance (spec Sec.3.2's "mass vs. peak" risk).

A2 v1 bug: rank of 131072 vectors in R^2048 is bounded by 2048 by
construction -- "not fully redundant" is guaranteed, uninformative about
S3. The real question is per-key SUBSTITUABILITY across layers: does key
i in layer L have a near-duplicate in layer L'? That's a nearest-neighbor
tail statistic, not a global rank or a mean cosine (both would hide a
decisive few-thousand-near-duplicates signal inside 131k mostly-unrelated
keys).

CPU only. Reuses a real-text sample already in this repo (no new data dep).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"


@torch.no_grad()
def a1_score_geometry(model, tok, text_path: str, n_tokens: int, layers_to_check: list[int]) -> dict:
    """Real q.k score distributions, attention-side vs FFN-side, per layer."""
    texts = []
    for line in open(text_path):
        if line.strip():
            texts.append(json.loads(line)["text"])
        if len(texts) >= 8:
            break
    ids = tok("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=n_tokens)
    out = model(**ids, output_hidden_states=True)
    hidden = out.hidden_states  # tuple, hidden[l] = input to layer l
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    scale = 1.0 / math.sqrt(head_dim)

    per_layer = {}
    for li in layers_to_check:
        layer = model.model.layers[li]
        x = hidden[li][0].double()  # (T, d_model) -- real residual-stream activations
        T = x.shape[0]
        q = layer.self_attn.q_proj(x.float()).double()          # (T, d_model)
        k_attn = layer.self_attn.k_proj(x.float()).double()      # (T, d_model) real per-token keys
        k_ffn = layer.mlp.gate_proj.weight.detach().double()     # (d_ff, d_model) static "keys"

        # per-head reshape for a fair scaled-dot-product comparison
        qh = q.view(T, n_heads, head_dim)
        kh_attn = k_attn.view(T, n_heads, head_dim)
        # scores per head, then flatten heads for the pooled distribution
        scores_attn = torch.einsum("thd,shd->ths", qh, kh_attn) * scale  # (T,H,T)
        # FFN has no natural head split (d_ff doesn't factor by n_heads) -- treat gate_proj
        # as a single d_model-wide key set, score with the FULL query vector (not per-head),
        # same scale factor applied for a like-for-like magnitude comparison.
        scores_ffn = (q @ k_ffn.T) * scale  # (T, d_ff)

        k_attn_norms = k_attn.norm(dim=-1)
        k_ffn_norms = k_ffn.norm(dim=-1)

        # unified-softmax mass share: concatenate [attn keys ; ffn keys] per query,
        # one softmax, report mean fraction of mass landing on the FFN side.
        sc_attn_flat = scores_attn.mean(dim=1)  # (T,T) average over heads for this pooled check
        cat = torch.cat([sc_attn_flat, scores_ffn], dim=-1)  # (T, T+d_ff)
        probs = torch.softmax(cat, dim=-1)
        mass_ffn = probs[:, T:].sum(dim=-1)  # (T,) fraction of softmax mass on FFN keys

        per_layer[li] = {
            "k_attn_norm_mean": k_attn_norms.mean().item(), "k_attn_norm_std": k_attn_norms.std().item(),
            "k_ffn_norm_mean": k_ffn_norms.mean().item(), "k_ffn_norm_std": k_ffn_norms.std().item(),
            "score_attn_mean": scores_attn.mean().item(), "score_attn_std": scores_attn.std().item(),
            "score_attn_p99": scores_attn.flatten().kthvalue(int(0.99 * scores_attn.numel())).values.item(),
            "score_attn_max": scores_attn.max().item(),
            "score_ffn_mean": scores_ffn.mean().item(), "score_ffn_std": scores_ffn.std().item(),
            "score_ffn_p99": scores_ffn.flatten().kthvalue(int(0.99 * scores_ffn.numel())).values.item(),
            "score_ffn_max": scores_ffn.max().item(),
            "unified_softmax_mass_on_ffn_mean": mass_ffn.mean().item(),
            "unified_softmax_mass_on_ffn_median": mass_ffn.median().item(),
            "n_ffn_keys_in_softmax": k_ffn.shape[0], "n_attn_keys_in_softmax": T,
        }
        print(f"[A1v2] layer {li:2d}  ||k_attn||={k_attn_norms.mean().item():.2f}  "
              f"||k_ffn||={k_ffn_norms.mean().item():.2f}  "
              f"score_attn(mean/p99/max)={scores_attn.mean().item():.2f}/"
              f"{per_layer[li]['score_attn_p99']:.2f}/{scores_attn.max().item():.2f}  "
              f"score_ffn(mean/p99/max)={scores_ffn.mean().item():.2f}/"
              f"{per_layer[li]['score_ffn_p99']:.2f}/{scores_ffn.max().item():.2f}  "
              f"mass_on_ffn(mean)={mass_ffn.mean().item():.6f}  "
              f"(n_attn_keys={T} vs n_ffn_keys={k_ffn.shape[0]})", flush=True)
    return per_layer


@torch.no_grad()
def a2_nn_substitutability(model, layer_pairs: list[tuple[int, int]]) -> dict:
    """Nearest-neighbor max-cosine tail between FFN key sets of two layers."""
    layers = model.model.layers
    keys_cache = {}

    def keys_of(li):
        if li not in keys_cache:
            keys_cache[li] = layers[li].mlp.gate_proj.weight.detach().double()
        return keys_cache[li]

    out = {}
    for (i, j) in layer_pairs:
        Ki, Kj = keys_of(i), keys_of(j)
        Ki_n = Ki / Ki.norm(dim=-1, keepdim=True)
        Kj_n = Kj / Kj.norm(dim=-1, keepdim=True)
        cos = Ki_n @ Kj_n.T  # (d_ff_i, d_ff_j)
        if i == j:
            cos.fill_diagonal_(-2.0)  # exclude self-match for the intra-layer control
        max_cos, _ = cos.max(dim=-1)
        frac_above = {t: (max_cos > t).float().mean().item() for t in (0.5, 0.7, 0.9)}
        out[f"{i}-{j}"] = {
            "mean_max_cos": max_cos.mean().item(),
            "median_max_cos": max_cos.median().item(),
            "p99_max_cos": max_cos.kthvalue(int(0.99 * max_cos.numel())).values.item(),
            "frac_above_0.5": frac_above[0.5], "frac_above_0.7": frac_above[0.7],
            "frac_above_0.9": frac_above[0.9],
        }
        print(f"[A2v2] layers {i:2d}-{j:2d}  mean_max_cos={out[f'{i}-{j}']['mean_max_cos']:.3f}  "
              f"frac>0.5={frac_above[0.5]:.4f} frac>0.7={frac_above[0.7]:.4f} "
              f"frac>0.9={frac_above[0.9]:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/result_v2.json")
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_tokens", type=int, default=512)
    args = ap.parse_args()

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)

    a1 = a1_score_geometry(model, tok, args.text, args.n_tokens,
                            layers_to_check=[0, 4, 8, 11, 15])

    pairs = [(i, i + 1) for i in range(n_layers - 1)]          # adjacent
    pairs += [(0, 15), (0, 8), (7, 15), (3, 12)]                # distant
    pairs += [(i, i) for i in (0, 8, 15)]                       # intra-layer control
    a2 = a2_nn_substitutability(model, pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"A1": a1, "A2": a2}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
