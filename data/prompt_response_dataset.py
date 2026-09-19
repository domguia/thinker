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

Optional KD (2026-09-20, `teacher_targets=...`): both dataset classes can
attach per-span Top-K Teacher targets (a precompute_teacher_targets.py .npz
computed on the SAME jsonl's "text" field) via `PromptResponseTeacherTargets`
-- see that class's docstring for the alignment scheme (response spans are
tokenized standalone, not sliced out of `text`'s own tokenization, so
alignment is verified per example and falls back to CE-only, not assumed).
"""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ASSISTANT_MARKER = "<|im_start|>assistant\n"  # both prepare_reasoning_data.py's and
# prepare_retrieval_data.py's CHATML_TEMPLATE use this marker right before the response --
# used below as a search-start anchor so a KD span lookup can't accidentally match earlier
# text (e.g. a distractor document containing the literal answer string).


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


def _locate_token_span(tokenizer, text: str, span_text: str, search_start: int, max_length: int):
    """Finds `span_text` verbatim in `text` at/after `search_start`, then
    tokenizes the WHOLE `text` (same truncation/max_length as
    precompute_teacher_targets.py, so this re-tokenization lands on the exact
    same rows the Teacher's .npz was built from) to get the token range the
    span occupies in that tokenization -- returns (tok_start, ids_slice) or
    None if span_text isn't found (e.g. a canonical `answer` field that's a
    paraphrase of the generated text, not a verbatim substring, see
    ReasoningPromptDataset's docstring)."""
    char_start = text.find(span_text, search_start)
    if char_start == -1:
        return None
    char_end = char_start + len(span_text)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    tok_start = next((i for i, (s, e) in enumerate(offsets) if e > char_start), None)
    tok_end = next((i for i, (s, e) in enumerate(offsets) if s >= char_end), len(offsets))
    if tok_start is None or tok_end <= tok_start:
        return None
    return tok_start, enc["input_ids"][tok_start:tok_end]


class PromptResponseTeacherTargets:
    """precompute_teacher_targets.py .npz computed on the SAME jsonl's "text"
    field (row order = doc_id, unfiltered -- callers must pass the ORIGINAL
    row index, not a post-filtering position, see ReasoningPromptDataset's
    doc_id bookkeeping).

    Unlike train_real_text.py's TeacherTargets (fixed window position,
    alignment is free), a response span here is tokenized STANDALONE by
    _tokenize_padded, not sliced out of a `text` tokenization -- BPE boundary
    effects (e.g. leading-space merges) mean the standalone tokenization is
    not guaranteed to match the full-text tokenization token-for-token even
    when the underlying string is identical. slice_span verifies this
    explicitly (exact id match, not just length) and falls back to an
    all-False mask (pure CE for that example) rather than risk a silently
    misaligned KD target.
    """

    def __init__(self, npz_path: str):
        npz = np.load(npz_path)
        self.indices = npz["indices"]
        self.values = npz["values"]
        self.residual = npz["residual"]
        self.offsets = npz["offsets"]
        self.k = int(npz["k"])

    def slice_span(self, doc_id: int, tok_start: int, ids_slice, span_ids: torch.Tensor, t_max: int):
        idx = torch.zeros(t_max, self.k, dtype=torch.long)
        val = torch.zeros(t_max, self.k, dtype=torch.float32)
        res = torch.zeros(t_max, dtype=torch.float32)
        mask = torch.zeros(t_max, dtype=torch.bool)
        n = min(len(ids_slice), int(span_ids.shape[0]), t_max)
        if n == 0 or list(ids_slice[:n]) != span_ids[:n].tolist():
            return idx, val, res, mask
        doc_start, doc_end = int(self.offsets[doc_id]), int(self.offsets[doc_id + 1])
        n_doc = doc_end - doc_start
        for t in range(n):
            q = tok_start + t - 1  # Teacher row q predicts token q+1 -- same convention as train_real_text.py's TeacherTargets
            if 0 <= q < n_doc:
                idx[t] = torch.from_numpy(self.indices[doc_start + q].astype(np.int64))
                val[t] = torch.from_numpy(self.values[doc_start + q].astype(np.float32))
                res[t] = float(self.residual[doc_start + q])
                mask[t] = True
        return idx, val, res, mask


def _kd_targets_for_span(teacher, tokenizer, text, span_text, search_start, max_length, doc_id, span_ids, t_max):
    idx = torch.zeros(t_max, teacher.k, dtype=torch.long)
    val = torch.zeros(t_max, teacher.k, dtype=torch.float32)
    res = torch.zeros(t_max, dtype=torch.float32)
    mask = torch.zeros(t_max, dtype=torch.bool)
    if not text or not span_text:
        return idx, val, res, mask
    located = _locate_token_span(tokenizer, text, span_text, search_start, max_length)
    if located is None:
        return idx, val, res, mask
    tok_start, ids_slice = located
    return teacher.slice_span(doc_id, tok_start, ids_slice, span_ids, t_max)


class ReasoningPromptDataset(Dataset):
    def __init__(self, path, tokenizer, n_ctx: int, max_thinking_len: int, max_answer_len: int, pad_id: int = None,
                 teacher_targets: str = None, teacher_max_length: int = 4096):
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.max_thinking_len = max_thinking_len
        self.max_answer_len = max_answer_len
        self.pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or 0)
        self.teacher = PromptResponseTeacherTargets(teacher_targets) if teacher_targets else None
        if self.teacher is not None:
            assert tokenizer.is_fast, "--teacher_targets needs a fast tokenizer (return_offsets_mapping support)"
        self.teacher_max_length = teacher_max_length
        self.examples = []
        n_no_think_tag = 0
        n_total = 0
        with open(path) as f:
            for doc_id, line in enumerate(f):
                n_total += 1
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
                # doc_id = ORIGINAL row index (before this filtering), matching the row order
                # precompute_teacher_targets.py walked over the unfiltered jsonl -- required for
                # --teacher_targets alignment, see PromptResponseTeacherTargets.
                self.examples.append({
                    "problem": row["problem"], "thinking": thinking_text, "answer": str(answer),
                    "doc_id": doc_id, "text": row.get("text"),
                })
        if n_no_think_tag:
            print(f"WARNING: {n_no_think_tag}/{n_total} examples in {path} "
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

        out = {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "thinking_target_input": think_input, "thinking_labels": think_labels,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
        }
        if self.teacher is not None:
            assistant_pos = (ex["text"] or "").find(ASSISTANT_MARKER)
            search_start = assistant_pos + len(ASSISTANT_MARKER) if assistant_pos != -1 else 0
            think_idx, think_val, think_res, think_kmask = _kd_targets_for_span(
                self.teacher, self.tokenizer, ex["text"], ex["thinking"], search_start,
                self.teacher_max_length, ex["doc_id"], think_ids, self.max_thinking_len)
            # answer's search window starts after the thinking span if one was found (plain string
            # search, cheap -- avoids the short canonical answer string spuriously matching inside
            # the (much longer) trace); falls back to the assistant-marker start otherwise.
            think_char_start = (ex["text"] or "").find(ex["thinking"], search_start)
            ans_search_start = think_char_start + len(ex["thinking"]) if think_char_start != -1 else search_start
            ans_idx, ans_val, ans_res, ans_kmask = _kd_targets_for_span(
                self.teacher, self.tokenizer, ex["text"], ex["answer"], ans_search_start,
                self.teacher_max_length, ex["doc_id"], ans_ids, self.max_answer_len)
            out.update({
                "thinking_kd_indices": think_idx, "thinking_kd_values": think_val,
                "thinking_kd_residual": think_res, "thinking_kd_mask": think_kmask,
                "answer_kd_indices": ans_idx, "answer_kd_values": ans_val,
                "answer_kd_residual": ans_res, "answer_kd_mask": ans_kmask,
            })
        return out


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

    def __init__(self, path, tokenizer, block_size: int, n_docs_max: int, max_answer_len: int, pad_id: int = None,
                 teacher_targets: str = None, teacher_max_length: int = 4096):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.n_docs_max = n_docs_max
        self.max_answer_len = max_answer_len
        self.pad_id = pad_id if pad_id is not None else (tokenizer.pad_token_id or 0)
        self.teacher = PromptResponseTeacherTargets(teacher_targets) if teacher_targets else None
        if self.teacher is not None:
            assert tokenizer.is_fast, "--teacher_targets needs a fast tokenizer (return_offsets_mapping support)"
        self.teacher_max_length = teacher_max_length
        self.examples = []
        n_truncated_docs = 0
        for doc_id, line in enumerate(open(path)):
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
            # doc_id = ORIGINAL row index (before this filtering) -- see
            # ReasoningPromptDataset's identical convention for why this matters.
            self.examples.append({
                "question": row["question"], "docs": docs, "answer": str(row["answer"]),
                "doc_id": doc_id, "text": row.get("text"),
            })
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

        out = {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
        }
        if self.teacher is not None:
            # RetrievalPromptDataset's answer, unlike ReasoningPromptDataset's, is substituted
            # verbatim into `text` by CHATML_TEMPLATE (prepare_retrieval_data.py) -- so this span
            # is expected to align cleanly far more often (no paraphrase-vs-canonical mismatch).
            assistant_pos = (ex["text"] or "").find(ASSISTANT_MARKER)
            search_start = assistant_pos + len(ASSISTANT_MARKER) if assistant_pos != -1 else 0
            ans_idx, ans_val, ans_res, ans_kmask = _kd_targets_for_span(
                self.teacher, self.tokenizer, ex["text"], ex["answer"], search_start,
                self.teacher_max_length, ex["doc_id"], ans_ids, self.max_answer_len)
            out.update({
                "answer_kd_indices": ans_idx, "answer_kd_values": ans_val,
                "answer_kd_residual": ans_res, "answer_kd_mask": ans_kmask,
            })
        return out
