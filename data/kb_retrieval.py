"""
Synthetic "needle in an indexed haystack" task for testing HierarchicalMemory
(core/indexed_memory.py) and IndexedThinker (core/indexed_thinker_model.py).

Not a benchmark — a minimal correctness/sanity task exercising exactly what the
existing toy tasks (data/numbers.py: copy, addition, factorization) never do:
retrieving a value from a large-ish structured external memory rather than
reasoning purely over the given input. See dev_notes/indexed_attention_spec.md
§9 for how this relates to the "pure retrieval" evaluation split.

Layout: `n_facts` facts of exactly 4 leaf tokens each:
    [KEY_MARK, key_id, VAL_MARK, value_id]
`n_facts` must equal (block_size ** depth) / 4 for the natural alignment where
each block_size=4 leaf block is exactly one fact (so a depth-1 parent node is a
per-fact summary, and so on up the tree) — this alignment is a convenience for
interpretability/testing, not a hard requirement of HierarchicalMemory itself.
"""

import torch
from torch.utils.data import IterableDataset

KEY_MARK = 0
VAL_MARK = 1
NUM_OFFSET = 2


class KBRetrievalDataset(IterableDataset):
    def __init__(self, n_facts: int = 4, vocab_size: int = 32, seed: int = None):
        self.n_facts = n_facts
        self.vocab_size = vocab_size
        self.n_leaves = n_facts * 4
        self.total_vocab_size = vocab_size + NUM_OFFSET
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def _sample_one(self):
        keys = torch.randperm(self.vocab_size, generator=self._rng)[: self.n_facts]
        values = torch.randint(0, self.vocab_size, (self.n_facts,), generator=self._rng)

        leaves = torch.empty(self.n_leaves, dtype=torch.long)
        for i in range(self.n_facts):
            leaves[i * 4 + 0] = KEY_MARK
            leaves[i * 4 + 1] = keys[i] + NUM_OFFSET
            leaves[i * 4 + 2] = VAL_MARK
            leaves[i * 4 + 3] = values[i] + NUM_OFFSET

        target_idx = torch.randint(0, self.n_facts, (1,), generator=self._rng).item()
        query = (keys[target_idx] + NUM_OFFSET).view(1)
        label = values[target_idx] + NUM_OFFSET
        source_ids = torch.ones(self.n_leaves, dtype=torch.long)  # all leaves are "KB"

        return leaves, source_ids, query, label

    def __iter__(self):
        while True:
            yield self._sample_one()

    def sample_batch(self, batch_size: int):
        leaves, source_ids, query, label = zip(*[self._sample_one() for _ in range(batch_size)])
        return (
            torch.stack(leaves),
            torch.stack(source_ids),
            torch.stack(query),
            torch.stack(label),
        )
