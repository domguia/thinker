"""Prepare a small retrieval/grounded-QA sample (HotpotQA) for distillation onboarding.

Streams N questions (no full dataset/parquet download). Each example bundles
the question with its distractor + supporting context passages (multi-hop,
several documents) so the student must locate the relevant facts among
irrelevant ones -- the text analogue of Thinker's KV retrieval over a KB.
Tokenizes to filter by length, writes train/val JSONL splits. Same code path
for a tiny local smoke-test and the real cluster run.
"""
import argparse
import json
import os
import random
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from core.model_families import resolve_model_name

CHATML_TEMPLATE = "<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"


def format_context(ctx):
    """Returns (flattened_text, docs_list) -- docs_list is the raw
    per-document strings BEFORE joining, added 2026-09-20 for
    data/prompt_response_dataset.py::RetrievalPromptDataset's document-aware
    hierarchical chunking (HotpotQA distractor config bundles ~10 separate
    documents -- treating them as one flat blob loses that structure, see
    that class's docstring). The joined text is kept too, for the existing
    dense-baseline ChatML pipeline (learn/distill/train_sft.py) which reads
    it as one flat string."""
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    docs = [f"[{title}] " + " ".join(sents) for title, sents in zip(titles, sentences)]
    return "\n".join(docs), docs


def build_example(ex, tokenizer, max_length):
    question = ex.get("question")
    answer = ex.get("answer")
    context = ex.get("context")
    if not question or not answer or not context:
        return None

    context_text, context_docs = format_context(context)
    text = CHATML_TEMPLATE.format(context=context_text, question=question, answer=answer)
    num_tokens = len(tokenizer(text, truncation=False)["input_ids"])
    if num_tokens > max_length:
        return None

    # 2026-09-20 (model-design's fine-grained causal control): which of context_docs are
    # gold "supporting" evidence vs distractors -- format_context's docs_list order matches
    # ctx["title"], so this boolean mask lines up 1:1 with context_docs by position.
    supporting_titles = set(ex.get("supporting_facts", {}).get("title", []))
    titles = context.get("title", [])
    is_supporting = [t in supporting_titles for t in titles]

    return {
        "question": question,
        "answer": answer,
        "context": context_text,
        "context_docs": context_docs,
        "is_supporting": is_supporting,
        "text": text,
        "num_tokens": num_tokens,
        "num_hops": len(supporting_titles),
        "id": ex.get("id"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="lfm2", help="HF repo id, or a family alias from core/model_families.py (lfm2/olmo/qwen)")
    parser.add_argument("--n_samples", type=int, default=100, help="questions to pull via streaming")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out_dir", default="data/distill/hotpot_qa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer_name = resolve_model_name(args.tokenizer)
    print(f"Loading tokenizer {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Streaming {args.dataset}/{args.config} split={args.split} (pulling only {args.n_samples} questions) ...")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    examples = []
    skipped = 0
    seen = 0
    start_time = time.time()
    progress_every = max(1, args.n_samples // 200)
    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    with open(raw_path, "w") as raw_f:
        for ex in ds:
            if len(examples) >= args.n_samples:
                break
            seen += 1
            built = build_example(ex, tokenizer, args.max_length)
            # unfiltered original record, written as we go (not held in memory)
            raw_f.write(json.dumps({"kept": built is not None, **ex}, ensure_ascii=False) + "\n")
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

    print(f"Collected {len(examples)} examples ({skipped} skipped: missing fields or too long).")
    print(f"Wrote {seen} unfiltered raw rows to {raw_path}")

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
