"""
Sliding-window real-text loader for LM training on `IndexedThinker`
(dev_notes/indexed_attention_spec.md §14, plan Phase 11). Distinct from
data/kb_chain_retrieval.py (fabricated multi-hop facts, single query/answer
per episode): this module slides fixed-shape context/target windows over
real tokenized documents, taken from the "text" field of a JSONL produced by
any of learn/distill/prepare_*_data.py.

Window layout (spec §14.1): for a target span `ids[p : p + t_tgt]`, the
context is the `n_ctx` tokens immediately preceding it, `ids[p - n_ctx : p]`
(left-padded with `pad_id` / masked out via `leaf_mask` if the document
doesn't yet have that many tokens, e.g. near its start). Source id is
assigned by POSITION alone within that fixed-shape context array: the last
`t_local` positions (closest to the target) get source_id=0 ("input",
recency), everything before that gets source_id=1 ("KB", older/long-range) --
no separate long-range fetch is needed, since within one window "older" is
already older relative to "local". This exercises the same unified-KB /
priority-bias mechanism (HierarchicalMemory.build's source_ids) already used
by the synthetic Phase 2 task, on real content.

Because every window has the exact same shape (`n_ctx` context leaves,
`t_tgt` target positions), batches across different windows/documents can be
built with a plain `torch.utils.data.DataLoader`'s default collate -- no
custom padding-by-batch logic needed (unlike train_sft.py's JsonlTextDataset,
whose examples vary in length).

Target teacher-forcing (spec §14.3): `target_input[t]` is the token that
should be embedded and fed as OutputStream's per-position query to predict
`labels[t]` -- the usual next-token shift, continued across the
context/target boundary (`target_input[0]` is the last REAL context token,
or `pad_id` if the window has no real context at all) rather than restarted
at the target.

`n_ctx` must equal `block_size ** depth` for whatever HierarchicalMemory
config consumes this data (a constraint of that module, not enforced here --
this dataset doesn't need to know block_size/depth). Carrying the recurrent
register `R` across a document's windows (spec §14.2) and grouping windows so
same-document windows stay in order within a training loop is the caller's
responsibility -- each window here also carries `doc_id` and
`is_first_window` so a training loop can detect document boundaries (reset R
to `register_init` on `is_first_window`, carry it forward otherwise) even if
the DataLoader shuffles across documents.
"""
import json

import torch
from torch.utils.data import Dataset


class RealTextWindowDataset(Dataset):
    def __init__(self, path, tokenizer, n_ctx: int, t_local: int, t_tgt: int,
                 stride: int = None, min_real_context: int = 1, pad_id: int = None):
        assert 1 <= t_local <= n_ctx, "t_local must be in [1, n_ctx]"
        assert t_tgt >= 1
        assert 1 <= min_real_context <= n_ctx
        self.n_ctx = n_ctx
        self.t_local = t_local
        self.t_tgt = t_tgt
        self.stride = stride if stride is not None else t_tgt
        assert self.stride >= 1
        self.min_real_context = min_real_context

        if pad_id is not None:
            self.pad_id = pad_id
        else:
            tok_pad = getattr(tokenizer, "pad_token_id", None)
            self.pad_id = tok_pad if tok_pad is not None else 0

        self.docs = []          # list[list[int]], one token-id list per document
        self.windows = []        # list[(doc_id, p, is_first_window)]
        with open(path) as f:
            for doc_id, line in enumerate(f):
                row = json.loads(line)
                ids = tokenizer(row["text"], truncation=False)["input_ids"]
                self.docs.append(ids)
                self._add_windows(doc_id, ids)

    def _add_windows(self, doc_id, ids):
        L = len(ids)
        p = 0
        is_first = True
        while p + self.t_tgt <= L:
            real_context = min(p, self.n_ctx)
            if real_context >= self.min_real_context:
                self.windows.append((doc_id, p, is_first))
                is_first = False
            p += self.stride

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        doc_id, p, is_first_window = self.windows[idx]
        ids = self.docs[doc_id]
        return self._build_window(ids, p, doc_id, is_first_window)

    def _build_window(self, ids, p, doc_id, is_first_window):
        n_ctx, t_local, t_tgt = self.n_ctx, self.t_local, self.t_tgt

        real_ctx_start = max(p - n_ctx, 0)
        real_ctx = ids[real_ctx_start:p]  # length <= n_ctx, oldest-to-newest
        n_pad = n_ctx - len(real_ctx)

        kb_tokens = torch.full((n_ctx,), self.pad_id, dtype=torch.long)
        if real_ctx:
            kb_tokens[n_pad:] = torch.tensor(real_ctx, dtype=torch.long)
        kb_leaf_mask = torch.zeros(n_ctx, dtype=torch.bool)
        kb_leaf_mask[n_pad:] = True

        n_kb_region = n_ctx - t_local  # first n_kb_region positions = "older" (source_id=1)
        kb_source_ids = torch.zeros(n_ctx, dtype=torch.long)
        kb_source_ids[:n_kb_region] = 1

        target = ids[p:p + t_tgt]
        labels = torch.tensor(target, dtype=torch.long)
        last_ctx_token = real_ctx[-1] if real_ctx else self.pad_id
        target_input = torch.tensor([last_ctx_token] + target[:-1], dtype=torch.long)

        return {
            "kb_tokens": kb_tokens,
            "kb_source_ids": kb_source_ids,
            "kb_leaf_mask": kb_leaf_mask,
            "target_input": target_input,
            "labels": labels,
            "doc_id": doc_id,
            "is_first_window": is_first_window,
            "window_pos": p,  # absolute position in ids[] where the target span starts --
                               # needed to align this window against a precomputed Teacher's
                               # per-document, per-position KD targets (see train_real_text.py's
                               # --teacher_targets: teacher row q=window_pos+t-1 predicts labels[t]).
        }
