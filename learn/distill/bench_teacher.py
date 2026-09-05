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

GPU-adaptive loading (--num_gpus, --attn_implementation), reasoning-effort
control (--reasoning_effort), and on-the-fly quantization (--quantization,
e.g. to load an Unsloth bf16 repo in 4-bit on a smaller GPU) are informed by
learn/distill/qwen3.8-27b-notes.md -- see that file for the reasoning behind
each default.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

SAMPLE_PROMPTS = [
    "What is 12 + 7? Explain your reasoning step by step.",
    "What is the capital of France?",
    "A ship travels 24 km upstream and 28 km downstream. Explain how to "
    "set up the equations to find the speeds involved.",
]

# Hopper (9.x) and Ada (8.9) have native FP8 tensor cores; Ampere (8.0/8.6)
# and older load an FP8 checkpoint fine but dequantize to compute dtype --
# see the "Model identity" section of qwen3.8-27b-notes.md.
_NATIVE_FP8_CAPABILITIES = {(8, 9), (9, 0)}


def build_quantization_config(quantization, dtype):
    """None for the vendor FP8 checkpoint (default path); a bitsandbytes
    config to load a bf16 repo (vendor or Unsloth) in 4-bit/8-bit on a
    smaller GPU instead -- see "Quantization support added to the scripts"
    in qwen3.8-27b-notes.md. Requires `pip install bitsandbytes`.
    """
    if quantization == "none":
        return None
    if quantization == "bnb-4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    if quantization == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unknown quantization: {quantization}")


def describe_gpus():
    if not torch.cuda.is_available():
        print("No CUDA GPU detected -- running on CPU.", flush=True)
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        native_fp8 = (props.major, props.minor) in _NATIVE_FP8_CAPABILITIES or props.major >= 9
        print(
            f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB, "
            f"compute capability {props.major}.{props.minor}, "
            f"native FP8 tensor cores: {'yes' if native_fp8 else 'no (dequantizes to compute dtype)'}",
            flush=True,
        )


def build_max_memory(num_gpus):
    """Restrict device_map="auto" to the first `num_gpus` devices, if given.

    Excluded GPUs get an explicit 0GiB budget -- accelerate otherwise still
    considers every visible device. Leaves room to spill to CPU RAM.
    """
    if num_gpus is None or not torch.cuda.is_available():
        return None
    available = torch.cuda.device_count()
    if num_gpus >= available:
        return None
    max_memory = {}
    for i in range(available):
        if i < num_gpus:
            usable_gb = int(torch.cuda.get_device_properties(i).total_memory / (1024**3) * 0.9)
            max_memory[i] = f"{usable_gb}GiB"
        else:
            max_memory[i] = "0GiB"
    max_memory["cpu"] = "64GiB"
    return max_memory


def load_model_and_tokenizer(model_dir, dtype, num_gpus=None, attn_implementation="auto", quantization="none"):
    """Load the Teacher, adapting to the GPU(s) actually available.

    attn_implementation="auto" tries flash_attention_2 first (needs the
    flash-attn package and Ampere+) and falls back to sdpa if that fails to
    import or load -- pass an explicit value to skip the fallback attempt.

    quantization="none" loads model_dir as-is (the vendor FP8 checkpoint by
    default). "bnb-4bit"/"bnb-8bit" apply bitsandbytes on-the-fly
    quantization -- use with a bf16 repo (vendor or Unsloth), not with the
    already-quantized FP8 checkpoint.
    """
    max_memory = build_max_memory(num_gpus)
    quantization_config = build_quantization_config(quantization, dtype)
    attn_candidates = ["flash_attention_2", "sdpa"] if attn_implementation == "auto" else [attn_implementation]

    model, tokenizer, last_error = None, None, None
    for attn_impl in attn_candidates:
        kwargs = dict(
            torch_dtype=dtype, device_map="auto", max_memory=max_memory, attn_implementation=attn_impl,
        )
        # Only pass quantization_config when we actually want on-the-fly
        # (bnb) quantization. Passing quantization_config=None explicitly
        # was suspected (2026-09-05) to suppress transformers' own
        # auto-detection of a checkpoint's *native* quantization scheme
        # (e.g. the FP8 Teacher's quant_method=fp8/fmt=e4m3) -- omit the key
        # entirely instead so that auto-detection from the checkpoint's own
        # config.json can still kick in.
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        try:
            try:
                model = AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)
                tokenizer = AutoProcessor.from_pretrained(model_dir).tokenizer
            except (ValueError, OSError):
                model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print(f"Loaded with attn_implementation={attn_impl}", flush=True)
            break
        except Exception as e:  # e.g. flash-attn not installed, or unsupported on this GPU
            last_error = e
            print(f"  attn_implementation={attn_impl} failed ({e}); trying next option ...", flush=True)
            model = None
    if model is None:
        raise last_error

    # Guard against a real silent-corruption failure mode found 2026-09-05
    # (see qwen3.8-27b-notes.md's "CRITICAL" section): if the checkpoint's
    # own config declares a quantization scheme but the loaded model wasn't
    # actually wired up with a quantizer, transformers has discarded the
    # dequantization scale tensors (e.g. *.weight_scale_inv) and loaded the
    # raw quantized bytes reinterpreted as the compute dtype -- numerically
    # wrong, not just unoptimized. Fail loudly instead of returning a model
    # that looks fine but produces meaningless outputs.
    declared_quant = getattr(model.config, "quantization_config", None) or getattr(
        getattr(model.config, "text_config", None), "quantization_config", None
    )
    if declared_quant and not getattr(model, "is_quantized", False):
        raise RuntimeError(
            f"{model_dir}'s config declares quantization_config={declared_quant!r} but the "
            "loaded model reports is_quantized=False -- this transformers version is silently "
            "discarding the dequantization scale tensors and loading raw quantized bytes "
            "reinterpreted as the compute dtype (numerically wrong, not just unoptimized). "
            "See qwen3.8-27b-notes.md's 'CRITICAL' section (found 2026-09-05). Use the "
            "checkpoint's bf16 counterpart instead, or fix/upgrade the transformers quantizer "
            "support before trusting outputs from this checkpoint."
        )

    model.eval()
    return model, tokenizer


def apply_chat_template_with_reasoning_effort(tokenizer, messages, reasoning_effort):
    """reasoning_effort: "model_default" (don't override), or one of Qwen3.8's
    four levels ("xhigh", "medium", "low", "none") -- passed as a
    chat-template kwarg, not a plain enable_thinking=True/False boolean (that
    API is for the smaller Qwen3 dense models, not this one -- see
    unsloth.ai/docs/models/qwen3.8, cited in qwen3.8-27b-notes.md). Other
    tokenizers (e.g. the gpt2 local-smoke-test fallback) don't accept this
    kwarg -- fall back silently to the plain call if it's rejected.
    Qwen3.8-27B defaults to "xhigh", which can dominate generation time for
    tasks that don't need deep reasoning.
    """
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if reasoning_effort != "model_default":
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def benchmark_generation(model, tokenizer, prompts, max_new_tokens, reasoning_effort="model_default"):
    # do_sample=False (greedy) is deliberate here for a reproducible,
    # comparable throughput measurement -- it does NOT match Unsloth's
    # recommended sampling params for actual generation with this model
    # (temp=1.0/top_p=0.95/top_k=20 in thinking mode, temp=0.7/top_p=0.80/
    # top_k=20/presence_penalty=1.5 in "none" reasoning_effort mode -- see
    # qwen3.8-27b-notes.md). Apply those separately if this model is ever
    # used to actually generate text rather than just benchmarked.
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  [generation {i}/{len(prompts)}] starting ...", flush=True)
        messages = [{"role": "user", "content": prompt}]
        inputs = apply_chat_template_with_reasoning_effort(tokenizer, messages, reasoning_effort).to(model.device)

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
        if num_generated >= max_new_tokens and reasoning_effort != "none":
            print(
                "    note: generation hit max_new_tokens -- with the model's default "
                "(xhigh) reasoning effort this may have been cut off mid-reasoning; "
                "pass --reasoning_effort none to measure post-thinking throughput instead. "
                "See qwen3.8-27b-notes.md.",
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
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
        help="\"auto\" (default) keeps the checkpoint's own stored dtype -- required for the FP8 checkpoint: "
             "forcing bfloat16 dequantizes the whole model before it touches VRAM (~55.6 GB vs. ~30.9 GB "
             "native FP8), which silently triggers CPU offload and a ~50-100x throughput collapse on GPUs "
             "with less than ~56 GB VRAM (found 2026-09-04 on an A100 40GB; see qwen3.8-27b-notes.md). Only "
             "override this to a concrete dtype when using --quantization (bnb needs a real compute dtype).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="limit to the first N visible GPUs for device_map=\"auto\" sharding "
             "(default: use every visible GPU)",
    )
    parser.add_argument(
        "--attn_implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="\"auto\" tries flash_attention_2 then falls back to sdpa",
    )
    parser.add_argument(
        "--quantization", default="none", choices=["none", "bnb-4bit", "bnb-8bit"],
        help="on-the-fly bitsandbytes quantization (needs pip install bitsandbytes); "
             "use with a bf16 repo (vendor or Unsloth), not with the already-quantized "
             "FP8 checkpoint -- see qwen3.8-27b-notes.md",
    )
    parser.add_argument(
        "--reasoning_effort", default="model_default", choices=["model_default", "xhigh", "medium", "low", "none"],
        help="override Qwen3.8's reasoning_effort chat-template kwarg (not a plain "
             "enable_thinking boolean -- that's for smaller Qwen3 models). The model "
             "defaults to xhigh, which dominates generation time for tasks that don't "
             "need deep reasoning -- see qwen3.8-27b-notes.md",
    )
    parser.add_argument("--out_file", default="teacher_bench_results.json")
    args = parser.parse_args()

    if args.dtype == "auto":
        if args.quantization != "none":
            raise ValueError("--dtype auto is only valid with --quantization none (bnb needs a concrete compute dtype, e.g. --dtype bfloat16)")
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)

    print("Detected GPU(s):", flush=True)
    describe_gpus()

    print(f"Loading {args.model_dir} in {args.dtype} ...", flush=True)
    print("  (shard-loading progress is printed by transformers itself below)", flush=True)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir, dtype, num_gpus=args.num_gpus, attn_implementation=args.attn_implementation,
        quantization=args.quantization,
    )
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Benchmarking generation (reasoning_effort={args.reasoning_effort}) ...", flush=True)
    gen_results = benchmark_generation(
        model, tokenizer, SAMPLE_PROMPTS, args.max_new_tokens, reasoning_effort=args.reasoning_effort
    )

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
            "num_gpus": args.num_gpus,
            "attn_implementation": args.attn_implementation,
            "quantization": args.quantization,
            "reasoning_effort": args.reasoning_effort,
            "generation_benchmark": gen_results,
            "topk_bytes_per_token_measured": measured_bytes_per_token,
            "topk_k": args.top_k,
        }, f, indent=2)
    print(f"Wrote results to {args.out_file}")


if __name__ == "__main__":
    main()
