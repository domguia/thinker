"""Reformat a plain-text prepare_general_data.py JSONL (one {"text": ...} per
line, e.g. wikitext_sample5k/tinystories_sample5k) into ReasoningPromptDataset's
expected schema ("problem"/"thinking"/"answer"/"text"), by splitting each
document into a fixed-length prompt span (first --prompt_tokens tokens) and a
fixed-length answer span (the next --answer_tokens tokens) -- a synthetic
next-span-prediction task, since these datasets have no natural prompt/answer
structure of their own.

Why this exists: train_prompt_response.py (the Thinker/indexed-attention
script) only accepts --dataset_type reasoning/retrieval, both of which require
a real "answer" field that is a VERBATIM substring of "text" (see
data/prompt_response_dataset.py's _locate_token_span/_resolve_span -- KD
target alignment works by re-finding that substring's character offsets and
mapping them to token positions via the tokenizer's offset_mapping, then
reading the matching rows straight out of the already-precomputed top-K .npz).
Decision (supervisor-agent, 2026-09-22): reuse the EXISTING precomputed top-K
targets by keeping "text" identical to the original raw document and choosing
the answer span as an exact token-aligned substring of it, rather than
re-running precompute_teacher_targets.py on a reformatted corpus.

Row order/count is preserved 1:1 (no filtering, no reordering) so doc_id
(= row index in this output file) stays consistent with the topk/ store's
doc_id keys, which were assigned from the ORIGINAL train.jsonl/val.jsonl row
order (see PromptResponseTeacherTargets/TeacherTopKStore docstrings).

"thinking" is left empty ("") for every example -- there is no natural
reasoning trace in this data, and an empty span resolves to an all-masked
thinking stream (zero-length span -> _resolve_span returns tok_end<=tok_start
-> KD/loss stream simply contributes nothing for that example), not an error.

Short documents (fewer than prompt_tokens+answer_tokens tokens total) get a
proportional split (2/3 prompt, 1/3 answer, minimum 1 token each) instead of
being dropped, to preserve the 1:1 row mapping.
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def split_document(tokenizer, text: str, prompt_tokens: int, answer_tokens: int):
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n < 2:
        return None  # nothing usable
    want = prompt_tokens + answer_tokens
    if n >= want:
        p_end_tok, a_end_tok = prompt_tokens, want
    else:
        p_end_tok = max(1, int(n * 2 / 3))
        a_end_tok = min(n, p_end_tok + max(1, n - p_end_tok))
    if p_end_tok >= n:
        return None
    a_end_tok = min(a_end_tok, n)
    if a_end_tok <= p_end_tok:
        return None
    prompt_text = text[offsets[0][0]:offsets[p_end_tok - 1][1]]
    answer_text = text[offsets[p_end_tok][0]:offsets[a_end_tok - 1][1]]
    if not prompt_text.strip() or not answer_text.strip():
        return None
    return prompt_text, answer_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--prompt_tokens", type=int, default=96)
    ap.add_argument("--answer_tokens", type=int, default=48)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_total = 0
    n_written = 0
    n_skipped = 0
    with open(args.in_file) as fin, open(args.out_file, "w") as fout:
        for doc_id, line in enumerate(fin):
            n_total += 1
            row = json.loads(line)
            text = row["text"]
            split = split_document(tok, text, args.prompt_tokens, args.answer_tokens)
            if split is None:
                # write a placeholder so doc_id (= line number) stays aligned with the
                # topk store's doc_id keys -- ReasoningPromptDataset skips rows with no
                # "answer", so this row is simply absent from training, not misaligned.
                fout.write(json.dumps({"problem": text[:1] or " ", "thinking": "", "answer": None, "text": text}) + "\n")
                n_skipped += 1
                continue
            prompt_text, answer_text = split
            fout.write(json.dumps({"problem": prompt_text, "thinking": "", "answer": answer_text, "text": text}) + "\n")
            n_written += 1
    print(f"{args.in_file} -> {args.out_file}: {n_written}/{n_total} usable (doc_id preserved), {n_skipped} skipped (too short)")


if __name__ == "__main__":
    main()
