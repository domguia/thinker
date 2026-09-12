import unittest

import torch

from data.kb_chain_retrieval import KBChainDataset, KEY_MARK, VAL_MARK


class TestKBChainDataset(unittest.TestCase):
    def _decode_facts(self, leaves, n_real_facts):
        facts = {}
        for i in range(n_real_facts):
            self.assertEqual(leaves[i * 4 + 0].item(), KEY_MARK)
            self.assertEqual(leaves[i * 4 + 2].item(), VAL_MARK)
            facts[leaves[i * 4 + 1].item()] = leaves[i * 4 + 3].item()
        return facts

    def test_chain_is_actually_followable_to_the_label(self):
        torch.manual_seed(0)
        n_hops, n_distractors, vocab_size = 3, 2, 32
        ds = KBChainDataset(n_hops=n_hops, n_distractors=n_distractors, vocab_size=vocab_size, seed=0)
        for _ in range(20):
            leaves, source_ids, mask, query, label = ds._sample_one()
            n_real_facts = n_hops + n_distractors
            facts = self._decode_facts(leaves, n_real_facts)

            cur = query.item()
            for _ in range(n_hops):
                self.assertIn(cur, facts, "chain broken: key not found among the facts")
                cur = facts[cur]
            self.assertEqual(cur, label.item(), "following the chain n_hops times must reach the label")

    def test_padding_mask_matches_real_fact_count(self):
        torch.manual_seed(0)
        ds = KBChainDataset(n_hops=2, n_distractors=1, vocab_size=32, max_facts=8, seed=1)
        self.assertEqual(ds.n_leaves, 8 * 4)
        leaves, source_ids, mask, query, label = ds._sample_one()
        self.assertEqual(mask.sum().item(), (2 + 1) * 4)
        self.assertTrue(torch.all(mask[: (2 + 1) * 4]))
        self.assertFalse(torch.any(mask[(2 + 1) * 4:]))

    def test_sample_batch_shapes(self):
        ds = KBChainDataset(n_hops=2, n_distractors=2, vocab_size=32, seed=0)
        leaves, source_ids, mask, query, label = ds.sample_batch(batch_size=5)
        self.assertEqual(leaves.shape, (5, ds.n_leaves))
        self.assertEqual(source_ids.shape, (5, ds.n_leaves))
        self.assertEqual(mask.shape, (5, ds.n_leaves))
        self.assertEqual(query.shape, (5, 1))
        self.assertEqual(label.shape, (5,))

    def test_vocab_too_small_raises(self):
        with self.assertRaises(AssertionError):
            KBChainDataset(n_hops=10, n_distractors=10, vocab_size=5)


if __name__ == '__main__':
    unittest.main()
