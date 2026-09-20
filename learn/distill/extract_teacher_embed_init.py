"""Project a Teacher's input embedding (and output head, if untied) down to
`d_model` for Thinker's `embed_init`/`stream_embed_init`/`stream_head_init`
(core/indexed_thinker_model.py, spec §13.1/13.2) -- avoids training the
~250M-parameter embedding/head from a random init.

Needs a GPU with enough VRAM to load the Teacher (same requirement as
bench_teacher.py/precompute_teacher_targets.py) -- extraction itself
(get_input_embeddings/get_output_embeddings, a projection) is cheap once
loaded, this is not a repeated-inference workload.

Row correspondence requirement (spec §13.1): row i of the Teacher's
embedding is token id i under the TEACHER's OWN tokenizer -- this only
transfers meaningfully to a Thinker whose embedding table uses that exact
same tokenizer/vocabulary (core/model_families.py's tokenizer-matching
constraint, same one KD already depends on). Projecting is about the
FEATURE dimension (hidden_size -> d_model), never about vocab id remapping
across different tokenizers.

Two projection methods (spec §13.1's two options):
  - random: a fixed random projection (orthogonal init) -- fast, no claim
    of optimality.
  - svd: truncated SVD/PCA (torch.pca_lowrank) on the embedding matrix,
    keeps the d_model directions of highest variance in the Teacher's own
    embedding space -- more principled, needs the full (vocab, hidden)
    matrix in memory (~5GB at vocab=248k/hidden=5120/fp32).

Example:
    python learn/distill/extract_teacher_embed_init.py \
      --model_dir /path/to/Qwen3.8-27B-FP8 --d_model 1024 --method svd \
      --out_file data/distill/qwen_big_embed_init_d1024.npz
"""
import argparse

import numpy as np
import torch

from bench_teacher import describe_gpus, load_model_and_tokenizer


def random_projection(hidden_size, d_model, seed, device, dtype):
    g = torch.Generator(device=device).manual_seed(seed)
    P = torch.empty(hidden_size, d_model, device=device, dtype=dtype)
    torch.nn.init.orthogonal_(P, generator=g)
    return P


def project(weight: torch.Tensor, d_model: int, method: str, seed: int) -> torch.Tensor:
    """weight: (vocab, hidden_size) -> (vocab, d_model)."""
    hidden_size = weight.shape[1]
    if method == "random":
        P = random_projection(hidden_size, d_model, seed, weight.device, weight.dtype)
        return weight @ P
    if method == "svd":
        # torch.pca_lowrank centers internally; U*S is the projection onto
        # the top-d_model principal components (highest-variance directions
        # of the Teacher's own embedding space).
        U, S, _ = torch.pca_lowrank(weight.float(), q=d_model)
        return (U * S).to(weight.dtype)
    raise ValueError(f"unknown --method {method!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--method", default="svd", choices=["random", "svd"])
    parser.add_argument("--seed", type=int, default=0, help="only used by --method random")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                         help="see precompute_teacher_targets.py's --dtype docstring -- keep \"auto\" for "
                              "native-FP8 checkpoints, only override with --quantization")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--attn_implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--quantization", default="none", choices=["none", "bnb-4bit", "bnb-8bit"])
    parser.add_argument("--out_file", required=True)
    args = parser.parse_args()

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    print("Detected GPU(s):", flush=True)
    describe_gpus()
    print(f"Loading Teacher {args.model_dir} ...", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, dtype, num_gpus=args.num_gpus, attn_implementation=args.attn_implementation,
        quantization=args.quantization,
    )

    input_embed = model.get_input_embeddings().weight.detach()
    output_embed_module = model.get_output_embeddings()
    tied = output_embed_module is None or output_embed_module.weight is model.get_input_embeddings().weight
    print(f"vocab_size={input_embed.shape[0]} hidden_size={input_embed.shape[1]} tied_word_embeddings={tied}", flush=True)

    print(f"Projecting input embedding ({args.method}) -> d_model={args.d_model} ...", flush=True)
    embed_init = project(input_embed, args.d_model, args.method, args.seed).float().cpu().numpy()

    save_kwargs = {"embed_init": embed_init, "vocab_size": input_embed.shape[0],
                   "hidden_size": input_embed.shape[1], "d_model": args.d_model, "method": args.method}
    if not tied:
        print(f"Projecting output head ({args.method}) -> d_model={args.d_model} ...", flush=True)
        head_weight = output_embed_module.weight.detach()
        save_kwargs["head_init"] = project(head_weight, args.d_model, args.method, args.seed).float().cpu().numpy()

    np.savez_compressed(args.out_file, **save_kwargs)
    print(f"Wrote {args.out_file} ({'embed_init + head_init' if not tied else 'embed_init only (tied Teacher)'})", flush=True)


if __name__ == "__main__":
    import os
    import sys

    main()
    # see scripts/prepare_distill_data.py for why we bypass normal teardown
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
