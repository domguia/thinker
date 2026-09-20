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
-- see that class's docstring for the alignment scheme.

**Correction 2026-09-20 (real-data bug, found via experiment-manager's
diagnostic on hotpotqa: 0/18000 aligned, deterministically)**: an earlier
version of this alignment tokenized each response span STANDALONE
(`_tokenize_padded` on the extracted substring) and only used the
in-context tokenization (`_locate_token_span`) to verify agreement. That
verification could never pass: (1) standalone tokenization adds a leading
BOS the in-context slice never has, and (2) BPE leading-space merges
differ in/out of context regardless of BOS (e.g. `"São Miguel"` standalone
-> `[1, 560, 2388, 22661]`, in-context (preceded by a space) ->
`[17370, 22661]`) -- the two are simply different token SEQUENCES, not a
comparison bug to patch. Fixed at the root: when a span is located
verbatim in `text` and `--teacher_targets` is set, its ids/mask are now
built DIRECTLY from the in-context slice (`_resolve_span`) instead of a
separate standalone tokenization -- this is also arguably more faithful to
generation (a real continuation right after the prompt, not an
artificially isolated re-tokenization with a spurious leading BOS).
Standalone `_tokenize_padded` remains the fallback exactly as before when
teacher_targets is unset, or when the span isn't found verbatim (e.g.
ReasoningPromptDataset's canonical `answer` vs the generated paraphrase --
a genuinely different failure mode, unaffected by this fix).
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

    slice_span expects (tok_start, n) from _resolve_span -- the span's own
    ids/mask are BY CONSTRUCTION the in-context tokenization's `[tok_start,
    tok_start+n)` slice (see _resolve_span), so no separate identity check
    is needed here (an earlier version re-tokenized the span standalone and
    checked for agreement -- always failed on real data, see this module's
    top docstring)."""

    def __init__(self, npz_path: str):
        npz = np.load(npz_path)
        self.indices = npz["indices"]
        self.values = npz["values"]
        self.residual = npz["residual"]
        self.offsets = npz["offsets"]
        self.k = int(npz["k"])

    def slice_span(self, doc_id: int, tok_start: int, n: int, t_max: int):
        idx = torch.zeros(t_max, self.k, dtype=torch.long)
        val = torch.zeros(t_max, self.k, dtype=torch.float32)
        res = torch.zeros(t_max, dtype=torch.float32)
        mask = torch.zeros(t_max, dtype=torch.bool)
        doc_start, doc_end = int(self.offsets[doc_id]), int(self.offsets[doc_id + 1])
        n_doc = doc_end - doc_start
        for t in range(min(n, t_max)):
            q = tok_start + t - 1  # Teacher row q predicts token q+1 -- same convention as train_real_text.py's TeacherTargets
            if 0 <= q < n_doc:
                idx[t] = torch.from_numpy(self.indices[doc_start + q].astype(np.int64))
                val[t] = torch.from_numpy(self.values[doc_start + q].astype(np.float32))
                res[t] = float(self.residual[doc_start + q])
                mask[t] = True
        return idx, val, res, mask


def _empty_kd(t_max: int, k: int):
    return (torch.zeros(t_max, k, dtype=torch.long), torch.zeros(t_max, k, dtype=torch.float32),
            torch.zeros(t_max, dtype=torch.float32), torch.zeros(t_max, dtype=torch.bool))


def _resolve_span(tokenizer, text, span_text, search_start, max_length, max_len, pad_id, teacher):
    """Returns (ids, mask, kd_info): ids/mask are the span's teacher-forcing
    tokens (padded/truncated to max_len), kd_info is (tok_start, n) if they
    came from `text`'s own in-context tokenization (KD-alignable against
    `teacher`) or None if this fell back to standalone tokenization
    (teacher is None, or span_text isn't found verbatim in text -- e.g.
    ReasoningPromptDataset's paraphrased canonical answer, see this
    module's top docstring)."""
    if teacher is not None and text and span_text:
        located = _locate_token_span(tokenizer, text, span_text, search_start, max_length)
        if located is not None:
            tok_start, ids_slice = located
            n = min(len(ids_slice), max_len)
            ids = torch.full((max_len,), pad_id, dtype=torch.long)
            mask = torch.zeros(max_len, dtype=torch.bool)
            if n > 0:
                ids[:n] = torch.tensor(ids_slice[:n], dtype=torch.long)
                mask[:n] = True
            return ids, mask, (tok_start, n)
    ids, mask = _tokenize_padded(tokenizer, span_text, max_len, pad_id)
    return ids, mask, None


def _kd_targets(teacher, kd_info, doc_id, t_max):
    if teacher is None:
        return None
    if kd_info is None:
        return _empty_kd(t_max, teacher.k)
    tok_start, n = kd_info
    return teacher.slice_span(doc_id, tok_start, n, t_max)


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

        assistant_pos = (ex["text"] or "").find(ASSISTANT_MARKER)
        search_start = assistant_pos + len(ASSISTANT_MARKER) if assistant_pos != -1 else 0
        think_ids, think_mask, think_kd_info = _resolve_span(
            self.tokenizer, ex["text"], ex["thinking"], search_start, self.teacher_max_length,
            self.max_thinking_len, self.pad_id, self.teacher)
        # answer's search window starts after the thinking span if one was found (plain string
        # search, cheap -- avoids the short canonical answer string spuriously matching inside
        # the (much longer) trace); falls back to the assistant-marker start otherwise.
        think_char_start = (ex["text"] or "").find(ex["thinking"], search_start)
        ans_search_start = think_char_start + len(ex["thinking"]) if think_char_start != -1 else search_start
        ans_ids, ans_mask, ans_kd_info = _resolve_span(
            self.tokenizer, ex["text"], ex["answer"], ans_search_start, self.teacher_max_length,
            self.max_answer_len, self.pad_id, self.teacher)
        think_input, think_labels = _teacher_forced_target(think_ids, think_mask, self.pad_id)
        ans_input, ans_labels = _teacher_forced_target(ans_ids, ans_mask, self.pad_id)

        out = {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "thinking_target_input": think_input, "thinking_labels": think_labels,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
        }
        think_kd = _kd_targets(self.teacher, think_kd_info, ex["doc_id"], self.max_thinking_len)
        ans_kd = _kd_targets(self.teacher, ans_kd_info, ex["doc_id"], self.max_answer_len)
        if think_kd is not None:
            think_idx, think_val, think_res, think_kmask = think_kd
            ans_idx, ans_val, ans_res, ans_kmask = ans_kd
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
            is_supporting = row.get("is_supporting")
            if is_supporting is not None:
                is_supporting = is_supporting[:n_docs_max] + [False] * max(0, n_docs_max - len(is_supporting))
            self.examples.append({
                "question": row["question"], "docs": docs, "answer": str(row["answer"]),
                "doc_id": doc_id, "text": row.get("text"),
                # -1 = unknown (data generated before num_hops was added to prepare_retrieval_data.py) --
                # kept distinct from real hop counts (>=0) so downstream stratification (e.g.
                # num_hops>=2 vs <=1, §8ter multi-hop protocol) can exclude unknowns explicitly.
                "num_hops": row.get("num_hops", -1),
                # None = unknown (data generated before is_supporting was added) -- used by
                # eval_causal_control.py's fine-grained corrupt-supporting-only vs
                # corrupt-distractors-only control (model-design, 2026-09-20).
                "is_supporting": is_supporting,
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

        # RetrievalPromptDataset's answer, unlike ReasoningPromptDataset's, is substituted
        # verbatim into `text` by CHATML_TEMPLATE (prepare_retrieval_data.py) -- so _resolve_span's
        # in-context path is expected to succeed far more often here (no paraphrase-vs-canonical
        # mismatch); real-data confirmation and the standalone-tokenization bug this fixed are in
        # this module's top docstring.
        assistant_pos = (ex["text"] or "").find(ASSISTANT_MARKER)
        search_start = assistant_pos + len(ASSISTANT_MARKER) if assistant_pos != -1 else 0
        ans_ids, ans_mask, ans_kd_info = _resolve_span(
            self.tokenizer, ex["text"], ex["answer"], search_start, self.teacher_max_length,
            self.max_answer_len, self.pad_id, self.teacher)
        ans_input, ans_labels = _teacher_forced_target(ans_ids, ans_mask, self.pad_id)

        is_supporting = ex["is_supporting"] if ex["is_supporting"] is not None else [False] * self.n_docs_max
        out = {
            "kb_tokens": kb_tokens, "kb_source_ids": kb_source_ids, "kb_leaf_mask": kb_leaf_mask,
            "answer_target_input": ans_input, "answer_labels": ans_labels,
            "num_hops": torch.tensor(ex["num_hops"], dtype=torch.long),
            "is_supporting": torch.tensor(is_supporting, dtype=torch.bool),
        }
        ans_kd = _kd_targets(self.teacher, ans_kd_info, ex["doc_id"], self.max_answer_len)
        if ans_kd is not None:
            ans_idx, ans_val, ans_res, ans_kmask = ans_kd
            out.update({
                "answer_kd_indices": ans_idx, "answer_kd_values": ans_val,
                "answer_kd_residual": ans_res, "answer_kd_mask": ans_kmask,
            })
        return out
