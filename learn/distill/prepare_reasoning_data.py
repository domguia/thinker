"""Prepare a small SFT sample from OpenR1-Math-220k for distillation onboarding.

Streams N examples (no full dataset/parquet download), keeps the first
math-verified <think> reasoning trace per problem, formats it in ChatML,
tokenizes to filter by length, and writes train/val JSONL splits.

Same code path works for a tiny local smoke-test (e.g. --tokenizer gpt2,
--n_samples 20) and for the real cluster run (--tokenizer Qwen/Qwen3-0.6B,
--n_samples much larger) -- only the CLI flags change.
"""
import argparse
import json
import os
import random
import time

from datasets import load_dataset
from transformers import AutoTokenizer

CHATML_TEMPLATE = "<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n{trace}<|im_end|>"


def build_example(ex, tokenizer, max_length):
    generations = ex.get("generations") or []
    correctness = ex.get("correctness_math_verify") or []

    trace = None
    for gen, ok in zip(generations, correctness):
        if ok:
            trace = gen
            break
    if trace is None and generations:
        trace = generations[0]
    if trace is None:
        return None

    problem = ex["problem"]
    text = CHATML_TEMPLATE.format(problem=problem, trace=trace)
    num_tokens = len(tokenizer(text, truncation=False)["input_ids"])
    if num_tokens > max_length:
        return None

    return {
        "problem": problem,
        "trace": trace,
        "text": text,
        "num_tokens": num_tokens,
        "source": ex.get("source"),
        "uuid": ex.get("uuid"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_samples", type=int, default=100, help="examples to pull via streaming")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out_dir", default="data/distill/openr1_math")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Streaming {args.dataset} split={args.split} (pulling only {args.n_samples} examples) ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    examples = []
    skipped = 0
    seen = 0
    start_time = time.time()
    progress_every = max(1, args.n_samples // 200)  # ~200 progress lines regardless of size
    for ex in ds:
        if len(examples) >= args.n_samples:
            break
        seen += 1
        built = build_example(ex, tokenizer, args.max_length)
        if built is None:
            skipped += 1
        else:
            examples.append(built)
        if seen % progress_every == 0:
            elapsed = time.time() - start_time
            rate = seen / elapsed if elapsed > 0 else 0
            print(
                f"[progress] seen={seen} kept={len(examples)} skipped={skipped} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} ex/s",
                flush=True,
            )

    print(f"Collected {len(examples)} examples ({skipped} skipped: no verified trace or too long).")

    random.seed(args.seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio)) if len(examples) > 1 else 0
    val, train = examples[:n_val], examples[n_val:]

    def dump(name, rows):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {path}")

    dump("train", train)
    dump("val", val)

    token_counts = [e["num_tokens"] for e in examples]
    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        print(f"Token length stats: min={min(token_counts)} max={max(token_counts)} avg={avg:.1f}")


if __name__ == "__main__":
    import sys

    main()
    # datasets' streaming iterator + pyarrow segfault on interpreter teardown
    # (PyGILState_Release race) after all work above is done; skip normal
    # finalization since outputs are already on disk. os._exit() bypasses
    # stdio buffering, so flush explicitly first.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
