"""
Lightweight variant of extract_teacher_embed_init.py: projects a Teacher's
input embedding and output head (hidden_size -> d_model) WITHOUT loading the
full model (no GPU needed, no ~50GB+ VRAM requirement) -- reads only the two
weight tensors (embed_tokens.weight, lm_head.weight) directly from their
safetensors shard via the model's own model.safetensors.index.json, using
safetensors.safe_open so only those two tensors are ever materialized.

Same projection math as extract_teacher_embed_init.py's project() (SVD by
default -- keeps the d_model directions of highest variance in the Teacher's
own embedding space), same output format (.npz with "embed_init"/"head_init"
keys, consumed by train_prompt_response.py's --answer_head_init and
core/indexed_thinker_model.py's stream_head_init/embed_init).

Use this instead of extract_teacher_embed_init.py whenever only the
embedding/head projection is needed (not a full Teacher forward pass) --
CPU-only, a few GB of RAM, no GPU reservation required.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open


def project(weight: torch.Tensor, d_model: int, method: str, seed: int) -> torch.Tensor:
    """weight: (vocab, hidden_size) -> (vocab, d_model). Same as extract_teacher_embed_init.py."""
    hidden_size = weight.shape[1]
    if method == "random":
        g = torch.Generator().manual_seed(seed)
        P = torch.empty(hidden_size, d_model, dtype=weight.dtype)
        torch.nn.init.orthogonal_(P, generator=g)
        return weight @ P
    if method == "svd":
        U, S, _ = torch.pca_lowrank(weight.float(), q=d_model)
        return (U * S).to(weight.dtype)
    raise ValueError(f"unknown --method {method!r}")


def load_tensor(model_dir: str, key: str) -> torch.Tensor:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard = json.load(f)["weight_map"][key]
        shard_path = os.path.join(model_dir, shard)
    else:
        shard_path = single_path
    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True, help="local HF checkpoint directory (not a repo id)")
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--method", default="svd", choices=["random", "svd"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed_key", default="model.language_model.embed_tokens.weight",
                     help="exact key from model.safetensors.index.json for the input embedding")
    ap.add_argument("--head_key", default="lm_head.weight",
                     help="exact key from model.safetensors.index.json for the output head; pass "
                          "the same value as --embed_key if the checkpoint has tied embeddings "
                          "(tie_word_embeddings=true, no separate lm_head.weight)")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    print(f"loading {args.embed_key} from {args.model_dir} ...", flush=True)
    input_embed = load_tensor(args.model_dir, args.embed_key)
    print(f"embed shape={tuple(input_embed.shape)} dtype={input_embed.dtype}", flush=True)

    print(f"projecting input embedding ({args.method}) -> d_model={args.d_model} ...", flush=True)
    embed_init = project(input_embed, args.d_model, args.method, args.seed).float().numpy()
    save_kwargs = {"embed_init": embed_init, "vocab_size": input_embed.shape[0],
                   "hidden_size": input_embed.shape[1], "d_model": args.d_model, "method": args.method}
    del input_embed

    tied = args.head_key == args.embed_key
    print(f"tied_word_embeddings={tied}", flush=True)
    if tied:
        # downstream consumers (train_prompt_response.py's --answer_head_init) always read "head_init" --
        # for a tied-embedding checkpoint the head IS the embedding, so reuse the same projected matrix
        # rather than omitting the key.
        save_kwargs["head_init"] = embed_init
    else:
        print(f"loading {args.head_key} ...", flush=True)
        head_weight = load_tensor(args.model_dir, args.head_key)
        print(f"head shape={tuple(head_weight.shape)} dtype={head_weight.dtype}", flush=True)
        print(f"projecting output head ({args.method}) -> d_model={args.d_model} ...", flush=True)
        save_kwargs["head_init"] = project(head_weight, args.d_model, args.method, args.seed).float().numpy()

    np.savez_compressed(args.out_file, **save_kwargs)
    print(f"wrote {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
