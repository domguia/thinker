"""
Multi-hop "chain of facts" retrieval task — dev_notes/indexed_attention_experiment_plan.md
Phase 2 / Phase 1quater. Tests whether iterative reasoning (N_step > 1) provides
real value, unlike data/kb_retrieval.py's single-lookup task where N_step beyond
1-2 has nothing to add.

Layout: same 4-token fact format as kb_retrieval.py, [KEY_MARK, key_id, VAL_MARK,
value_id]. `n_hops` facts form a chain — the value of fact i is the key of fact
i+1 — plus `n_distractors` unrelated facts padding out the KB. The query is the
chain's starting key; the label is the value at the end of the chain. Answering
correctly requires following the chain hop by hop, not a single lookup.

Fact blocks (not individual tokens) are shuffled so the chain can't be found by
a positional shortcut ("the next fact in memory order continues the chain") —
the model must match on content (key == previous value) at every hop.

`max_facts` fixes the leaf count for a fixed-shape curriculum across n_hops
values, same convention as KBRetrievalDataset.
"""

import torch
from torch.utils.data import IterableDataset

KEY_MARK = 0
VAL_MARK = 1
NUM_OFFSET = 2


class KBChainDataset(IterableDataset):
    def __init__(self, n_hops: int, n_distractors: int, vocab_size: int,
                 max_facts: int = None, seed: int = None):
        assert n_hops >= 1
        self.n_hops = n_hops
        self.n_distractors = n_distractors
        self.n_facts = n_hops + n_distractors
        self.max_facts = max_facts if max_facts is not None else self.n_facts
        assert self.max_facts >= self.n_facts, "max_facts must be >= n_hops + n_distractors"
        self.vocab_size = vocab_size
        self.n_leaves = self.max_facts * 4
        self.total_vocab_size = vocab_size + NUM_OFFSET
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        n_ids_needed = (n_hops + 1) + 2 * n_distractors
        assert n_ids_needed <= vocab_size, (
            f"vocab_size={vocab_size} too small for n_hops={n_hops}/n_distractors={n_distractors} "
            f"(need {n_ids_needed} distinct ids)"
        )

    def _sample_one(self):
        n_ids_needed = (self.n_hops + 1) + 2 * self.n_distractors
        ids = torch.randperm(self.vocab_size, generator=self._rng)[:n_ids_needed]
        chain_ids = ids[: self.n_hops + 1]         # key_0, value_0=key_1, ..., value_{n_hops-1}
        distractor_ids = ids[self.n_hops + 1:]      # 2 * n_distractors unrelated ids

        n_real_facts = self.n_facts
        leaves = torch.zeros(self.n_leaves, dtype=torch.long)
        mask = torch.zeros(self.n_leaves, dtype=torch.bool)
        mask[: n_real_facts * 4] = True

        facts_k = torch.empty(n_real_facts, dtype=torch.long)
        facts_v = torch.empty(n_real_facts, dtype=torch.long)
        for i in range(self.n_hops):
            facts_k[i] = chain_ids[i] + NUM_OFFSET
            facts_v[i] = chain_ids[i + 1] + NUM_OFFSET
        for i in range(self.n_distractors):
            facts_k[self.n_hops + i] = distractor_ids[2 * i] + NUM_OFFSET
            facts_v[self.n_hops + i] = distractor_ids[2 * i + 1] + NUM_OFFSET

        perm = torch.randperm(n_real_facts, generator=self._rng)
        facts_k, facts_v = facts_k[perm], facts_v[perm]

        for i in range(n_real_facts):
            leaves[i * 4 + 0] = KEY_MARK
            leaves[i * 4 + 1] = facts_k[i]
            leaves[i * 4 + 2] = VAL_MARK
            leaves[i * 4 + 3] = facts_v[i]

        query = (chain_ids[0] + NUM_OFFSET).view(1)
        label = chain_ids[-1] + NUM_OFFSET
        source_ids = torch.ones(self.n_leaves, dtype=torch.long)

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
