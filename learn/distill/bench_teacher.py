"""Benchmark Teacher inference throughput/latency and extract Top-K logits.

Loads a downloaded Teacher snapshot (see download_teacher.py) as a text-only
causal LM -- image/video inputs are never used, even for a vision-language
checkpoint like Qwen/Qwen3.8-27B-FP8 -- and:
  1. Times generation on a few prompts to get tokens/sec, per the sizing
     discussion in raw/Distill-getting-start.md.
  2. Runs a forward pass on the same prompts and extracts the actual Top-K
     logits (indices + values) per token, to check the storage formula
     (~194 bytes/token at K=32, int32 indices + fp16 values + fp16 renorm
     scalar) from learn/distill/README.md against real measured bytes.

Needs a GPU with enough VRAM for the checkpoint (~31GB+ for Qwen3.8-27B-FP8)
-- run on a GPU reservation, not the CPU node used for downloading.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

SAMPLE_PROMPTS = [
    "What is 12 + 7? Explain your reasoning step by step.",
    "What is the capital of France?",
    "A ship travels 24 km upstream and 28 km downstream. Explain how to "
    "set up the equations to find the speeds involved.",
]


def load_model_and_tokenizer(model_dir, dtype):
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto")
        tokenizer = AutoProcessor.from_pretrained(model_dir).tokenizer
    except (ValueError, OSError):
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def benchmark_generation(model, tokenizer, prompts, max_new_tokens):
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [generation {i}/{len(prompts)}] starting ...", flush=True)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        num_generated = out.shape[1] - inputs["input_ids"].shape[1]
        tokens_per_sec = num_generated / elapsed if elapsed > 0 else float("nan")
        print(
            f"  [generation {i}/{len(prompts)}] prompt_tokens={inputs['input_ids'].shape[1]} "
            f"generated={num_generated} time={elapsed:.2f}s throughput={tokens_per_sec:.1f} tok/s",
            flush=True,
        )
        results.append({
            "prompt_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": num_generated,
            "seconds": elapsed,
            "tokens_per_sec": tokens_per_sec,
        })
    return results


def extract_topk_logits(model, tokenizer, texts, k):
    records = []
    total_bytes = 0
    for i, text in enumerate(texts, 1):
        t0 = time.time()
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]  # (seq_len, vocab)
        torch.topk(logits, k=k, dim=-1)  # exercise the real op the KD pipeline will use
        num_tokens = logits.shape[0]
        bytes_for_seq = num_tokens * (k * (4 + 2) + 2)  # int32 idx + fp16 val + fp16 renorm scalar
        total_bytes += bytes_for_seq
        records.append({"num_tokens": num_tokens, "bytes": bytes_for_seq})
        print(f"  [top-k {i}/{len(texts)}] {num_tokens} tokens in {time.time() - t0:.2f}s", flush=True)
    return records, total_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="local snapshot dir from download_teacher.py, or a Hub repo id")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--out_file", default="teacher_bench_results.json")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading {args.model_dir} in {args.dtype} ...", flush=True)
    print("  (shard-loading progress is printed by transformers itself below)", flush=True)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_dir, dtype)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    print("Benchmarking generation ...", flush=True)
    gen_results = benchmark_generation(model, tokenizer, SAMPLE_PROMPTS, args.max_new_tokens)

    print(f"Extracting Top-{args.top_k} logits on sample prompts ...", flush=True)
    topk_records, total_bytes = extract_topk_logits(model, tokenizer, SAMPLE_PROMPTS, args.top_k)
    total_tokens = sum(r["num_tokens"] for r in topk_records)
    measured_bytes_per_token = total_bytes / total_tokens if total_tokens else float("nan")
    print(
        f"  measured: {measured_bytes_per_token:.1f} bytes/token at K={args.top_k} "
        f"(formula estimate: {args.top_k * 6 + 2} bytes/token)"
    )

    with open(args.out_file, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "dtype": args.dtype,
            "generation_benchmark": gen_results,
            "topk_bytes_per_token_measured": measured_bytes_per_token,
            "topk_k": args.top_k,
        }, f, indent=2)
    print(f"Wrote results to {args.out_file}")


if __name__ == "__main__":
    main()
