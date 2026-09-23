"""
Exact-match / F1 accuracy of Thinker's actual FREE-RUN generated answer against
the ground-truth HotpotQA answer -- supervisor-agent request (2026-09-23), after
the generation-collapse investigation was closed without finding an isolable
cause: does the FINAL answer (often short/factual) stay correct despite the
"safe-bet" noise on generic tokens seen elsewhere in free generation, or does
the collapse also corrupt the answer itself? val_answer (CE) and the
degenerate-output qualitative heuristic (train_prompt_response.py) don't
answer this directly -- CE never checks if the generated text IS the answer,
and the qualitative heuristic only flags low-diversity/high-digit-ratio
strings, not correctness.

Standard SQuAD/HotpotQA normalization (lowercase, strip articles/punctuation,
collapse whitespace) + token-overlap F1, same convention as the original
HotpotQA/SQuAD eval scripts -- EM is exact string match after normalization,
F1 is token-level precision/recall harmonic mean (partial credit for
overlapping-but-not-identical answers, standard for QA with variable phrasing).

Greedy decoding only (temperature=0) via generate_thinker() -- same free-run
generation this whole investigation has been diagnosing, not teacher-forced.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter

import torch
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import RetrievalPromptDataset
from learn.indexed_attention.generate_qualitative_compare import generate_thinker


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_slots", type=int, default=1)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--pool_n_head", type=int, default=1)
    ap.add_argument("--k_dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--use_ff", action="store_true")
    ap.add_argument("--ff_hidden_mult", type=int, default=4)
    ap.add_argument("--answer_n_layers", type=int, default=1)
    ap.add_argument("--disable_kb", action="store_true")
    ap.add_argument("--answer_head_per_position", action="store_true")
    ap.add_argument("--answer_head_lora_rank", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=500, help="0 = full val set")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="2026-09-23, supervisor-agent recovery-path test: 0.0 (default) = greedy, "
                         ">0 = sampled generation via generate_thinker's own _sample_next (moderate "
                         "temperature, e.g. 0.7-0.8, tests whether the argmax 'safe bet' specifically, "
                         "not the whole output distribution, is what's broken).")
    ap.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling cutoff, only used with --temperature > 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    val_ds = RetrievalPromptDataset(args.val_data, tok, block_size=args.block_size, n_docs_max=args.n_docs_max,
                                     max_answer_len=args.max_answer_len, pad_id=tok.pad_token_id)
    print(f"loaded val: {len(val_ds)} examples from {args.val_data}", flush=True)

    n = len(val_ds) if not args.n_samples else min(args.n_samples, len(val_ds))
    indices = list(range(n))

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims={"answer": vocab_size}, stream_sequence={"answer": True}, max_target_len=args.max_answer_len,
        stream_n_layers={"answer": args.answer_n_layers},
        stream_head_per_position=({"answer": True} if args.answer_head_per_position else None),
    ).to(device)
    if args.answer_head_lora_rank > 0:
        from learn.indexed_attention.train_prompt_response import LoRAHead
        model.streams["answer"].head = LoRAHead(model.streams["answer"].head, args.answer_head_lora_rank).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:  # train_sft.py-style wrapped checkpoint, unwrap
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded {args.checkpoint}", flush=True)

    em_sum, f1_sum, count = 0.0, 0.0, 0
    per_example = []
    for start in range(0, n, args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        texts, _ = generate_thinker(model, val_ds, batch_idx, device, args.n_step, args.block_size,
                                     args.max_answer_len, tok, temperature=args.temperature,
                                     top_p=args.top_p, seed=args.seed)
        for i, pred in zip(batch_idx, texts):
            gold = val_ds.examples[i]["answer"]
            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            em_sum += em
            f1_sum += f1
            count += 1
            per_example.append({"index": i, "gold": gold, "pred": pred, "em": em, "f1": round(f1, 4)})
        if (start // args.batch_size) % 5 == 0:
            print(f"[progress] {count}/{n} -- running EM={em_sum/count:.4f} F1={f1_sum/count:.4f}", flush=True)

    result = {"checkpoint": args.checkpoint, "n": count,
              "exact_match_pct": 100 * em_sum / count, "f1_pct": 100 * f1_sum / count,
              "examples": per_example[:30]}
    print(f"\n=== FINAL: checkpoint={args.checkpoint} n={count} EM={result['exact_match_pct']:.2f}% "
          f"F1={result['f1_pct']:.2f}% ===", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
