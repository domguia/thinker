"""
Phase 12 / S3 curriculum (dev_notes/indexed_attention_experiment_plan.md,
section "Resolution de la question ouverte S3 (signal d'etape)", etape 1 --
step B): does a real query naturally retrieve its OWN host layer's converted
FFN memory when several layers' FFN-derived (K, V) are combined into ONE
union memory, with NO added per-layer tag (regime 5's premise, tested before
paying for a trained embedding/tag, regime 3/4)?

Mechanism: for each candidate OLMo-2-1B layer l, treat mlp.gate_proj.weight
as that layer's static FFN "keys" (same convention already used by
diagnose_olmo_ffn_memory_geometry_v2.py's A1/A2 -- consistent with this
project's prior art, not a new choice). Concatenate 3 layers' key sets into
one region-labeled bank, query with each layer's OWN real per-token queries
(self_attn.q_proj(x_l), same stand-in already used by A1 since Thinker's own
register/query mechanism isn't wired to a pretrained host yet), one unified
softmax, and measure how much attention mass lands on the correct region vs
a chance-level control (same total bank, region labels randomly shuffled --
NOT "how much would a random query get", since group sizes are already equal
and would trivially float to 1/3 -- the real question this control answers
is "does the ACTUAL W_gate-layer-origin grouping carry a signal beyond the
group sizes", i.e. permuting the grouping itself, per this project's
chance-level-control convention, cf. I4/A2's own use of trivial controls).

No training anywhere -- frozen pretrained weights, no gradient. CPU-only,
same order of cost as diagnose_olmo_ffn_memory_geometry_v2.py (~2 min on
paradoxe-27 for 16 layers; this touches only 3).

This is deliberately NOT routed through HierarchicalMemory.build_static +
.attend(): attend() applies its own (freshly-initialized, untrained)
self.q_proj to the query, which would inject a random rotation unrelated to
OLMo's real query geometry. S0's own equivalence is Q=x (identity query, no
extra projection) -- using the host's own real q_proj(x) here is the
appropriate stand-in for "a real query", but it must NOT be composed with a
second, untrained linear layer on top. build_static's job (injecting frozen
K/V bypassing the compressor/k_proj/v_proj) stays useful for S0's own exact
reconstruction check and for later curriculum stages once Thinker's own
register produces the query -- not needed for this specific attribution
measurement, which is why the scoring is done directly here (same pattern
already used by diagnose_olmo_ffn_memory_geometry_v2.py's a1_score_geometry).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"

# Smoke-test fallback text, used only when --text points at a file that
# doesn't exist locally (the real corpus, data/distill/general_realtext/,
# lives on the Grid5000 NFS where the other Phase 12 diagnostics run, not
# necessarily on every machine this script is written/reviewed on).
_FALLBACK_TEXTS = [
    "The city council voted to approve the new budget after months of debate.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "She packed her bags quickly, unsure of when she would return home again.",
    "The stock market fell sharply after the central bank raised interest rates.",
    "A large glacier calved into the sea, sending waves toward the nearby village.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and fresh yeast.",
    "Researchers published a study showing that the treatment reduced symptoms significantly.",
    "The old library was renovated last year, adding a new wing for rare manuscripts.",
]


@torch.no_grad()
def load_texts(text_path: str, n_needed: int = 8) -> list[str]:
    p = Path(text_path)
    if not p.exists():
        print(f"WARNING: {text_path} not found, using {len(_FALLBACK_TEXTS)} "
              f"built-in fallback sentences instead (smoke-test data, not the "
              f"real corpus).", flush=True)
        return _FALLBACK_TEXTS
    texts = []
    for line in open(p):
        if line.strip():
            texts.append(json.loads(line)["text"])
        if len(texts) >= n_needed:
            break
    return texts


@torch.no_grad()
def region_attribution(model, tok, texts: list[str], layers: list[int], n_tokens: int, seed: int = 0,
                        query_mode: str = "identity") -> dict:
    """
    query_mode:
      - "identity" (default): Q = x_l, no extra projection -- matches S0's exact
        conversion convention (FFN(x) = phi(x @ W_in^T) @ W_out, Q=x). This is
        the principled test of whether the raw FFN-key GEOMETRY discriminates
        by layer at all, independent of any attention-specific machinery.
      - "q_proj": Q = layer.self_attn.q_proj(x_l), the host's own REAL trained
        query projection -- same stand-in already used by A1/A2 (no Thinker
        register exists yet to query with). Caveat found by a synthetic sanity
        check before running this on real weights: q_proj is a generic linear
        map never trained to interact with gate_proj via a dot product -- a
        RANDOM projection of this kind destroys any pre-existing geometric
        alignment between x and K_ffn, so a null result under "q_proj" alone
        would NOT show "no natural signal exists", only "not through this
        particular unrelated map". Keep both readings; "identity" is the one
        that actually answers etape 1's question, "q_proj" is a secondary,
        weaker-guarantee cross-check (same convention A1 already used).
    """
    ids = tok("\n\n".join(texts), return_tensors="pt", truncation=True, max_length=n_tokens)
    out = model(**ids, output_hidden_states=True)
    hidden = out.hidden_states  # hidden[l] = input to layer l (x_l)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    scale = 1.0 / math.sqrt(head_dim)

    d_ff = model.model.layers[0].mlp.gate_proj.weight.shape[0]
    n_regions = len(layers)

    # region l's static FFN "keys" -- same convention as A1/A2 (gate_proj rows).
    K_regions = [model.model.layers[l].mlp.gate_proj.weight.detach().double() for l in layers]
    K_all = torch.cat(K_regions, dim=0)  # (n_regions * d_ff, d_model)
    region_of = torch.repeat_interleave(torch.arange(n_regions), d_ff)  # (n_regions*d_ff,)

    g = torch.Generator().manual_seed(seed)
    shuffled_region_of = region_of[torch.randperm(region_of.shape[0], generator=g)]

    results = {}
    for true_idx, l in enumerate(layers):
        x = hidden[l][0].double()  # (T, d_model), real residual-stream activations at layer l
        if query_mode == "identity":
            q = x
        elif query_mode == "q_proj":
            q = model.model.layers[l].self_attn.q_proj(x.float()).double()
        else:
            raise ValueError(f"unknown query_mode: {query_mode}")
        # per-head scores against the FULL region bank (no natural head split for
        # FFN keys, same simplification as A1v2 -- scored with the full query vector)
        scores = (q @ K_all.T) * scale  # (T, n_regions*d_ff)
        probs = torch.softmax(scores, dim=-1)

        # real grouping: mass landing on region == true_idx (this layer's own FFN)
        mass_true = probs[:, region_of == true_idx].sum(dim=-1)  # (T,)
        mass_per_region = {r: probs[:, region_of == r].sum(dim=-1).mean().item() for r in range(n_regions)}

        # chance-level control: same bank, same total size, region labels shuffled --
        # isolates whether the TRUE W_gate-layer grouping carries signal, not just
        # "3 equal-size groups sum to ~1/3 trivially".
        mass_true_shuffled = probs[:, shuffled_region_of == true_idx].sum(dim=-1)

        results[l] = {
            "true_idx": true_idx,
            "mass_true_region_mean": mass_true.mean().item(),
            "mass_true_region_median": mass_true.median().item(),
            "mass_per_region_mean": mass_per_region,
            "mass_chance_control_mean": mass_true_shuffled.mean().item(),
            "chance_floor_uniform": 1.0 / n_regions,
        }
        print(f"[layer {l:2d}, query_mode={query_mode}] mass_on_own_region(mean/median)="
              f"{mass_true.mean().item():.4f}/{mass_true.median().item():.4f}  "
              f"mass_per_region={ {k: round(v,4) for k,v in mass_per_region.items()} }  "
              f"chance_control(shuffled_labels)={mass_true_shuffled.mean().item():.4f}  "
              f"uniform_floor={1.0/n_regions:.4f}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--n_tokens", type=int, default=512)
    ap.add_argument("--layers", default="0,8,15", help="comma-separated OLMo layer indices, low/mid/high stratification")
    ap.add_argument("--query_mode", default="both", choices=["identity", "q_proj", "both"],
                     help="'identity' (Q=x, the principled S0-consistent test), 'q_proj' (host's real "
                          "trained query projection, weaker-guarantee secondary cross-check, A1's convention), "
                          "or 'both' (default, run and report both)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/olmo_ffn_geometry/layer_source_attribution.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)
    assert all(0 <= l < n_layers for l in layers), f"layers must be in [0, {n_layers})"

    texts = load_texts(args.text)
    modes = ["identity", "q_proj"] if args.query_mode == "both" else [args.query_mode]
    all_results = {}
    for mode in modes:
        all_results[mode] = region_attribution(model, tok, texts, layers, args.n_tokens, seed=args.seed,
                                                 query_mode=mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsummary: {json.dumps({'layers': layers, 'n_regions': len(layers), 'query_modes': modes})}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
