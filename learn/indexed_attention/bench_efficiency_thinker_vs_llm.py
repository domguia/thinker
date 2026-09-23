"""Inference efficiency comparison: Thinker (retrieval checkpoint) vs a reference
pretrained LLM (default Qwen3.5-0.8B, same family already used for the CE-only
baselines in eval_llm_baseline_retrieval.py), on the SAME prompts/conditions.

Measures, per model: parameter count, greedy-generation wall-clock latency and
tokens/sec (batch_size=1, same N examples, same max_new_tokens, same GPU).

Caveat (report honestly, don't hide): Thinker's generate_thinker() has NO
KV-cache (one full forward() per generated token, re-attending to the whole
context every step -- see generate_qualitative_compare.py's own docstring),
while the HF baseline uses .generate() with its native KV-cache. This is a
real architectural difference worth reporting, not an apples-to-apples
inference-engine comparison -- both numbers are "as actually run", not
"as Thinker could run with a hypothetical cache".
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import RetrievalPromptDataset
from learn.indexed_attention.generate_qualitative_compare import generate_thinker


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def bench_thinker(checkpoint: str, val_data: str, tokenizer, device, n_examples: int,
                   d_model: int, n_head: int, n_step: int, block_size: int, n_docs_max: int,
                   max_answer_len: int, use_ff: bool) -> dict:
    vocab_size = len(tokenizer)
    ds = RetrievalPromptDataset(path=val_data, tokenizer=tokenizer, block_size=block_size,
                                 n_docs_max=n_docs_max, max_answer_len=max_answer_len,
                                 pad_id=tokenizer.pad_token_id)
    stream_dims = {"answer": vocab_size}
    stream_sequence = {"answer": True}
    stream_n_layers = {"answer": 1}
    model = Thinker(
        vocab_size=vocab_size, d_model=d_model, n_register=8,
        block_size=block_size, depth=0, n_slots=1, n_head=n_head,
        disable_kb=False, pool_n_head=1, k_dim=None,
        use_ff=use_ff, ff_hidden_mult=4,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_answer_len,
        stream_n_layers=stream_n_layers,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    n_params = count_params(model)

    indices = list(range(min(n_examples, len(ds))))
    # warmup (first CUDA call pays kernel-compile/allocator cost, exclude from timing)
    generate_thinker(model, ds, indices[:2], device, n_step=n_step, block_size=block_size,
                      max_answer_len=max_answer_len, tokenizer=tokenizer, temperature=0.0, seed=0)
    torch.cuda.synchronize()

    total_tokens, total_s = 0, 0.0
    for i in indices:
        t0 = time.perf_counter()
        answers, _ = generate_thinker(model, ds, [i], device, n_step=n_step, block_size=block_size,
                                       max_answer_len=max_answer_len, tokenizer=tokenizer,
                                       temperature=0.0, seed=0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_tok = len(tokenizer(answers[0])["input_ids"])
        total_tokens += max(n_tok, 1)
        total_s += dt

    return {"n_params": n_params, "n_examples": len(indices), "total_tokens": total_tokens,
            "total_seconds": total_s, "tokens_per_sec": total_tokens / total_s,
            "sec_per_example": total_s / len(indices)}


def bench_llm(model_name: str, val_data: str, tokenizer, device, n_examples: int,
              block_size: int, n_docs_max: int, max_answer_len: int, dtype: str) -> dict:
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(resolve_model_name(model_name), torch_dtype=torch_dtype).to(device)
    model.eval()
    n_params = count_params(model)

    ds = RetrievalPromptDataset(path=val_data, tokenizer=tokenizer, block_size=block_size,
                                 n_docs_max=n_docs_max, max_answer_len=max_answer_len,
                                 pad_id=tokenizer.pad_token_id)
    indices = list(range(min(n_examples, len(ds))))

    def _prompt_ids(i):
        it = ds[i]
        mask = it["kb_leaf_mask"]
        return it["kb_tokens"][mask].unsqueeze(0).to(device)

    # warmup
    with torch.no_grad():
        model.generate(_prompt_ids(indices[0]), max_new_tokens=max_answer_len, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()

    total_tokens, total_s = 0, 0.0
    for i in indices:
        input_ids = _prompt_ids(i)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_answer_len, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_tok = out.shape[1] - input_ids.shape[1]
        total_tokens += max(n_tok, 1)
        total_s += dt

    return {"n_params": n_params, "n_examples": len(indices), "total_tokens": total_tokens,
            "total_seconds": total_s, "tokens_per_sec": total_tokens / total_s,
            "sec_per_example": total_s / len(indices)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinker_checkpoint", required=True)
    ap.add_argument("--llm_model", default="qwen35")
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--n_examples", type=int, default=30)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--use_ff", action="store_true")
    ap.add_argument("--llm_dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"=== Benchmarking Thinker ({args.thinker_checkpoint}) ===", flush=True)
    thinker_stats = bench_thinker(args.thinker_checkpoint, args.val_data, tokenizer, device,
                                   args.n_examples, args.d_model, args.n_head, args.n_step,
                                   args.block_size, args.n_docs_max, args.max_answer_len, args.use_ff)
    print(json.dumps(thinker_stats, indent=2), flush=True)

    print(f"\n=== Benchmarking LLM baseline ({args.llm_model}) ===", flush=True)
    llm_stats = bench_llm(args.llm_model, args.val_data, tokenizer, device, args.n_examples,
                           args.block_size, args.n_docs_max, args.max_answer_len, args.llm_dtype)
    print(json.dumps(llm_stats, indent=2), flush=True)

    ratio_params = llm_stats["n_params"] / thinker_stats["n_params"]
    ratio_speed = thinker_stats["tokens_per_sec"] / llm_stats["tokens_per_sec"]
    print(f"\n=== Summary ===")
    print(f"Params: Thinker={thinker_stats['n_params']/1e6:.1f}M, {args.llm_model}={llm_stats['n_params']/1e6:.1f}M "
          f"({ratio_params:.1f}x fewer params in Thinker)")
    print(f"Speed: Thinker={thinker_stats['tokens_per_sec']:.1f} tok/s, {args.llm_model}={llm_stats['tokens_per_sec']:.1f} tok/s "
          f"(Thinker is {ratio_speed:.2f}x the LLM's tok/s -- NOTE Thinker has no KV-cache, see module docstring)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"thinker": thinker_stats, "llm": llm_stats,
                       "ratio_params_llm_over_thinker": ratio_params,
                       "ratio_speed_thinker_over_llm": ratio_speed}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
