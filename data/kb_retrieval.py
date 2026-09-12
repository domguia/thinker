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

Curriculum support (`max_facts`): dev_notes/indexed_attention_experiment_plan.md
Phase -1 found that training collapses when scaling n_facts up directly (16
works, 64 doesn't, independent of dimensions/LR), matching a pattern already
solved in this project's own history (dev_notes/experiment.log.md, Dec 2023,
ToyThinker copy task) via curriculum learning. `max_facts` fixes the leaf
sequence length (and therefore `block_size`/`depth`) to the *target* curriculum
scale, padding unused fact slots — the same fixed-oversized-slot pattern the
original ToyThinker used (`max_input_len = seq_len * 4`). `n_facts` can then be
increased across a curriculum without changing the model's hierarchy shape.
"""

import torch
from torch.utils.data import IterableDataset

KEY_MARK = 0
VAL_MARK = 1
NUM_OFFSET = 2


class KBRetrievalDataset(IterableDataset):
    def __init__(self, n_facts: int, vocab_size: int, max_facts: int = None, seed: int = None):
        self.n_facts = n_facts
        self.max_facts = max_facts if max_facts is not None else n_facts
        assert self.max_facts >= n_facts, "max_facts must be >= n_facts"
        self.vocab_size = vocab_size
        self.n_leaves = self.max_facts * 4  # fixed shape across curriculum stages
        self.total_vocab_size = vocab_size + NUM_OFFSET
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def _sample_one(self):
        keys = torch.randperm(self.vocab_size, generator=self._rng)[: self.n_facts]
        values = torch.randint(0, self.vocab_size, (self.n_facts,), generator=self._rng)

        leaves = torch.zeros(self.n_leaves, dtype=torch.long)  # padding = token id 0 (unused, masked out)
        mask = torch.zeros(self.n_leaves, dtype=torch.bool)
        for i in range(self.n_facts):
            leaves[i * 4 + 0] = KEY_MARK
            leaves[i * 4 + 1] = keys[i] + NUM_OFFSET
            leaves[i * 4 + 2] = VAL_MARK
            leaves[i * 4 + 3] = values[i] + NUM_OFFSET
            mask[i * 4: i * 4 + 4] = True

        target_idx = torch.randint(0, self.n_facts, (1,), generator=self._rng).item()
        query = (keys[target_idx] + NUM_OFFSET).view(1)
        label = values[target_idx] + NUM_OFFSET
        source_ids = torch.ones(self.n_leaves, dtype=torch.long)  # all leaves are "KB"

        return leaves, source_ids, mask, query, label

    def __iter__(self):
        while True:
            yield self._sample_one()

    def sample_batch(self, batch_size: int):
        leaves, source_ids, mask, query, label = zip(*[self._sample_one() for _ in range(batch_size)])
        return (
            torch.stack(leaves),
            torch.stack(source_ids),
            torch.stack(mask),
            torch.stack(query),
            torch.stack(label),
        )
