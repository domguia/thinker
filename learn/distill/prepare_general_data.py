"""Prepare a small general-language-modeling sample (WikiText + TinyStories)
for distillation onboarding.

Streams N documents per source (no full dataset/parquet download), tokenizes
to filter by length, and writes combined train/val JSONL splits. Same code
path for a tiny local smoke-test (--n_samples 20) and a bigger cluster run
-- only the CLI flags change. Sources are combined because neither alone is
sized like a real pretraining corpus (see raw/Distill-getting-start.md
follow-up on data sizing); fineweb-edu can be added later as a heavier
third source once bandwidth allows.
"""
import argparse
import json
import os
import random
import time

from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = [
    {"name": "wikitext-103", "dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "train"},
    {"name": "tinystories", "dataset": "roneneldan/TinyStories", "config": None, "split": "train"},
]


def build_example(ex, source_name, tokenizer, max_length, min_length):
    text = ex.get("text")
    if not text or not text.strip():
        return None
    num_tokens = len(tokenizer(text, truncation=False)["input_ids"])
    if num_tokens > max_length or num_tokens < min_length:
        return None
    return {"text": text, "num_tokens": num_tokens, "source": source_name}


def collect_from_source(source, tokenizer, n_samples, max_length, min_length):
    print(f"Streaming {source['dataset']}/{source['config']} split={source['split']} (pulling only {n_samples} docs) ...")
    ds = load_dataset(source["dataset"], source["config"], split=source["split"], streaming=True)

    examples, skipped = [], 0
    seen = 0
    start_time = time.time()
    progress_every = max(1, n_samples // 100)
    for ex in ds:
        if len(examples) >= n_samples:
            break
        seen += 1
        built = build_example(ex, source["name"], tokenizer, max_length, min_length)
        if built is None:
            skipped += 1
        else:
            examples.append(built)
        if seen % progress_every == 0:
            elapsed = time.time() - start_time
            rate = seen / elapsed if elapsed > 0 else 0
            print(
                f"  [progress:{source['name']}] seen={seen} kept={len(examples)} "
                f"skipped={skipped} elapsed={elapsed:.1f}s rate={rate:.1f} ex/s",
                flush=True,
            )
    print(f"  -> collected {len(examples)} docs ({skipped} skipped: out of length range).")
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_samples", type=int, default=100, help="documents to pull PER SOURCE via streaming")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--min_length", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out_dir", default="data/distill/general")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    examples = []
    for source in SOURCES:
        examples.extend(collect_from_source(source, tokenizer, args.n_samples, args.max_length, args.min_length))

    print(f"Collected {len(examples)} documents total across {len(SOURCES)} sources.")

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
    # see scripts/prepare_distill_data.py for why we bypass normal teardown
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
