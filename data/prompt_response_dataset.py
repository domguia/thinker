"""
Prompt/response(+thinking) datasets for Thinker, distinct from
real_text_windows.py's continuous sliding-window scheme: these have an
explicit PROMPT (fed as context/KB, never a prediction target) and one or
two RESPONSE spans, each read out by its own Thinker output stream
(core/indexed_thinker_model.py's multi-stream support, dict target_input).

Confirmed on 2 real OpenR1-Math-220k samples (2026-09-20, before writing
this): `trace` = `<think>\n...reasoning...\n</think>\n\n...clean final
solution...` -- a real, reliable split point, not a guess.

Two dataset classes, matching the two data/prepare_*.py scripts with an
explicit prompt/response structure (prepare_general_data.py's plain text
has neither -- stays on RealTextWindowDataset unchanged):

- ReasoningPromptDataset (learn/distill/prepare_reasoning_data.py output):
  prompt=`problem` (KB), `thinking` stream target = the reasoning trace
  inside <think>...</think>, `answer` stream target = the dataset's own
  canonical `answer` field (short, verified -- NOT a re-parse of the
  generated text after </think>, which is a paraphrase of varying quality).
- RetrievalPromptDataset (learn/distill/prepare_retrieval_data.py output):
  prompt=`context` (source_id=1, KB/long-range) + `question` (source_id=0,
  local) -- the text analogue of Thinker's KV retrieval over a KB, per that
  script's own docstring. `answer` stream target = `answer`. No thinking
  stream (HotpotQA has none).

Fixed-shape leaves (`kb_tokens`/`kb_source_ids`/`kb_leaf_mask`), same
convention as RealTextWindowDataset: prompts are truncated/padded to
`n_ctx` (must equal block_size**depth for whatever HierarchicalMemory
config consumes this). Response spans are truncated/padded to their own
per-stream max length (`max_thinking_len`/`max_answer_len`) -- these are
NOT required to equal n_ctx, unlike the KB leaves.
"""
import json

import torch
from torch.utils.data import Dataset

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _tokenize_padded(tokenizer, text: str, max_len: int, pad_id: int):
    """Returns (ids: LongTensor(max_len), mask: BoolTensor(max_len)) --
    truncated from the END if too long (keep the START, consistent with
    RealTextWindowDataset's left-padding-for-short-context convention:
    padding always goes at the END here since these are direct token
    sequences, not context windows needing right-alignment to "now")."""
    ids = tokenizer(text or "", truncation=True, max_length=max_len)["input_ids"]
    n = len(ids)
    out = torch.full((max_len,), pad_id, dtype=torch.long)
    mask = torch.zeros(max_len, dtype=torch.bool)
    if n > 0:
        out[:n] = torch.tensor(ids, dtype=torch.long)
        mask[:n] = True
    return out, mask


def _teacher_forced_target(ids: torch.Tensor, mask: torch.Tensor, pad_id: int):
    """target_input[t] = token fed as query predicting labels[t]=ids[t] --
    same shift-by-one-with-a-leading-pad convention as
    RealTextWindowDataset._build_window (no "last context token" to prime
    from here, since these spans start fresh, not mid-document)."""
    target_input = torch.cat([torch.tensor([pad_id]), ids[:-1]])
    labels = ids.clone()
    labels[~mask] = -100  # ignored by CE; ALSO must be masked out of KD loss by the caller
    return target_input, labels


class ReasoningPromptDataset(Dataset):
    def __init__(self, path, tokenizer, n_ctx: int, max_thinking_len: int, max_answer_len: int, pad_id: int = None):
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.max_thinking_len = max_thinking_len
        self.max_answer_len = max_answer_len
        self.pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or 0)
        self.examples = []
        n_no_think_tag = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                trace = row.get("trace") or ""
                answer = row.get("answer")
                start = trace.find(THINK_OPEN)
                end = trace.find(THINK_CLOSE)
                if start == -1 or end == -1 or end <= start:
                    n_no_think_tag += 1
                    thinking_text = trace  # fallback: whole trace, no clean split (rare, see prepare_reasoning_data.py)
                else:
                    thinking_text = trace[start + len(THINK_OPEN):end].strip()
                if not answer:
                    continue  # answer stream needs a real target; skip examples missing the canonical answer
                self.examples.append({"problem": row["problem"], "thinking": thinking_text, "answer": str(answer)})
        if n_no_think_tag:
            print(f"WARNING: {n_no_think_tag}/{len(self.examples) + n_no_think_tag} examples in {path} "
                  f"had no <think>/</think> tags -- used the whole trace as 'thinking' with no answer split "
                  f"(check prepare_reasoning_data.py's source data if this is a large fraction).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        kb_tokens, kb_leaf_mask = _tokenize_padded(self.tokenizer, ex["problem"], self.n_ctx, self.pad_id)
        kb_source_ids = torch.zeros(self.n_ctx, dtype=torch.long)  # prompt = local/recency, spec §6.2

        think_ids, think_mask = _tokenize_padded(self.tokenizer, ex["thinking"], self.max_thinking_len, self.pad_id)
        ans_ids, ans_mask = _tokenize_padded(self.tokenizer, ex["answer"], self.max_answer_len, self.pad_id)
        think_input, think_labels = _teacher_forced_target(think_ids, think_mask, self.pad_id)
        ans_input, ans_labels = _teacher_forced_target(ans_ids, ans_mask, self.pad_id)

        return {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "thinking_target_input": think_input, "thinking_labels": think_labels,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
        }


class RetrievalPromptDataset(Dataset):
    """2026-09-20 redesign (user decision: treat HotpotQA separately, not
    flattened like the other two datasets): HotpotQA's distractor config
    bundles ~10 SEPARATE documents (2 supporting + up-to-8 distractors) --
    exactly the "search among candidates" scenario HierarchicalMemory's
    indexing (depth>0) was designed for, unlike a single continuous window.
    The earlier version concatenated everything into one flat blob truncated
    to n_ctx, silently dropping distractors past the token budget and never
    exercising the hierarchy at all (depth=0/1 equivalent in effect).

    Each of up to `n_docs_max` context documents occupies its OWN dedicated
    block of `block_size` tokens (source_id=1, KB) -- pad/truncate per
    document, not globally -- so a short distractor never starves a long
    supporting document's budget and vice versa. Missing documents (fewer
    than n_docs_max in this example) get an entirely-masked block. The
    question occupies one more block (source_id=0, local) as-is. This gives
    `HierarchicalMemory(depth=1, block_size=block_size)` one summary NODE
    per document (spec's actual "index over candidates" mechanism) in
    addition to the raw per-document tokens, both attended in the unified
    softmax -- not just a bigger flat window.

    `data/kb_chain_retrieval.py` is the project's SYNTHETIC analogue of this
    same idea (fixed-size fact blocks); this is its real-text counterpart.

    `n_docs_max=0` (user idea, 2026-09-20): drops every context document,
    keeping only the question -> answer, everything else unchanged.
    Directly tests parametric memorization (can the model's own weights
    learn the answer from training exposure alone, with no document to
    retrieve from at all?) vs retrieval from context -- distinct from
    Thinker's own --disable_kb baseline (which removes the RETRIEVAL
    MECHANISM but keeps documents in the data); this removes the DOCUMENTS
    from the data while keeping the mechanism intact.
    """

    def __init__(self, path, tokenizer, block_size: int, n_docs_max: int, max_answer_len: int, pad_id: int = None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.n_docs_max = n_docs_max
        self.max_answer_len = max_answer_len
        self.pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or 0)
        self.examples = []
        n_truncated_docs = 0
        for line in open(path):
            row = json.loads(line)
            if not row.get("answer"):
                continue
            # Prefer a raw per-document list if present (context_docs, added
            # 2026-09-20 to prepare_retrieval_data.py); fall back to
            # splitting the flattened "context" string on "\n" (format_context's
            # own join separator) for data already generated before that change.
            docs = row.get("context_docs")
            if docs is None:
                docs = row["context"].split("\n")
            if len(docs) > n_docs_max:
                n_truncated_docs += 1
                docs = docs[:n_docs_max]
            self.examples.append({"question": row["question"], "docs": docs, "answer": str(row["answer"])})
        if n_truncated_docs:
            print(f"WARNING: {n_truncated_docs}/{len(self.examples)} examples in {path} had more than "
                  f"n_docs_max={n_docs_max} context documents -- extra documents dropped (increase "
                  f"n_docs_max if this fraction is large).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        n_blocks = self.n_docs_max + 1  # + 1 for the question's own block
        N = n_blocks * self.block_size
        kb_tokens = torch.full((N,), self.pad_id, dtype=torch.long)
        kb_leaf_mask = torch.zeros(N, dtype=torch.bool)
        kb_source_ids = torch.zeros(N, dtype=torch.long)

        for i in range(self.n_docs_max):
            start = i * self.block_size
            if i >= len(ex["docs"]):
                continue  # fewer real documents than n_docs_max -- block stays fully masked
            ids = self.tokenizer(ex["docs"][i], truncation=True, max_length=self.block_size)["input_ids"]
            n = len(ids)
            kb_tokens[start:start + n] = torch.tensor(ids, dtype=torch.long)
            kb_leaf_mask[start:start + n] = True
            kb_source_ids[start:start + self.block_size] = 1  # this document = KB (whole block, incl. its padding)

        q_start = self.n_docs_max * self.block_size
        q_ids = self.tokenizer(ex["question"], truncation=True, max_length=self.block_size)["input_ids"]
        n_q = len(q_ids)
        kb_tokens[q_start:q_start + n_q] = torch.tensor(q_ids, dtype=torch.long)
        kb_leaf_mask[q_start:q_start + n_q] = True
        kb_source_ids[q_start:q_start + self.block_size] = 0  # question block = local

        ans_ids, ans_mask = _tokenize_padded(self.tokenizer, ex["answer"], self.max_answer_len, self.pad_id)
        ans_input, ans_labels = _teacher_forced_target(ans_ids, ans_mask, self.pad_id)

        return {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
        }
