"""
Correctness tests for HierarchicalMemory / IndexedThinker, all CPU-only.
See dev_notes/indexed_attention_spec.md for the math these cross-check against.
"""

import math
import unittest

import torch
import torch.nn.functional as F

from core.indexed_memory import HierarchicalMemory, LevelCompressor
from core.indexed_thinker_model import IndexedThinker, OutputStream
from data.kb_retrieval import KBRetrievalDataset


class TestHierarchicalMemoryShapes(unittest.TestCase):
    def test_level_shapes(self):
        d, block_size, depth, n_slots, B = 16, 4, 2, 1, 3
        mem = HierarchicalMemory(d, block_size, depth, n_slots=n_slots)
        N = block_size ** depth
        leaves = torch.randn(B, N, d)
        source_ids = torch.randint(0, 2, (B, N))
        mem.build(leaves, source_ids)

        self.assertEqual(len(mem._levels_k), depth + 1)
        expected_sizes = [N, N // block_size, N // block_size ** 2]
        for level, expected in zip(mem._levels_k, expected_sizes):
            self.assertEqual(level.shape, (B, expected, d))

    def test_depth_zero_is_flat_attention(self):
        d, B, N = 8, 2, 5
        mem = HierarchicalMemory(d, block_size=4, depth=0)
        leaves = torch.randn(B, N, d)
        source_ids = torch.zeros(B, N, dtype=torch.long)
        mem.build(leaves, source_ids)
        self.assertEqual(len(mem._levels_k), 1)
        self.assertEqual(mem._levels_k[0].shape, (B, N, d))

        out = mem.attend(torch.randn(B, 3, d))
        self.assertEqual(out.shape, (B, 3, d))

    def test_wrong_leaf_count_raises(self):
        mem = HierarchicalMemory(d_model=8, block_size=4, depth=2)
        leaves = torch.randn(1, 10, 8)  # should be 16
        source_ids = torch.zeros(1, 10, dtype=torch.long)
        with self.assertRaises(AssertionError):
            mem.build(leaves, source_ids)


class TestGradientFlow(unittest.TestCase):
    def test_gradient_reaches_every_level_and_both_source_biases(self):
        d, block_size, depth, B = 8, 2, 3, 2
        mem = HierarchicalMemory(d, block_size, depth, n_slots=1)
        N = block_size ** depth
        leaves = torch.randn(B, N, d, requires_grad=True)
        # ensure both source ids (0 and 1) are present so both bias rows get grad
        source_ids = torch.zeros(B, N, dtype=torch.long)
        source_ids[:, N // 2:] = 1

        mem.build(leaves, source_ids)
        out = mem.attend(torch.randn(B, 1, d))
        out.sum().backward()

        self.assertIsNotNone(leaves.grad)
        self.assertTrue(torch.any(leaves.grad != 0))

        self.assertIsNotNone(mem.compressor.query.grad)
        self.assertTrue(torch.any(mem.compressor.query.grad != 0))

        for name in ['ff_k_in', 'ff_k_out', 'ff_v_in', 'ff_v_out']:
            w = getattr(mem.compressor, name).weight
            self.assertIsNotNone(w.grad, f"compressor.{name} got no gradient")
            self.assertTrue(torch.any(w.grad != 0), f"compressor.{name} gradient is all-zero")

        self.assertIsNotNone(mem.source_bias.weight.grad)
        self.assertTrue(torch.all(mem.source_bias.weight.grad.abs().sum(dim=-1) > 0),
                         "both source_bias rows (input=0, kb=1) must receive gradient")

        for i, norm in enumerate(mem.level_norms):
            self.assertIsNotNone(norm.weight.grad, f"level_norms[{i}] got no gradient")
            self.assertTrue(torch.any(norm.weight.grad != 0), f"level_norms[{i}] gradient is all-zero")

        for name, p in [('k_proj', mem.k_proj.weight), ('v_proj', mem.v_proj.weight), ('q_proj', mem.q_proj.weight)]:
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertTrue(torch.any(p.grad != 0), f"{name} gradient is all-zero")

    def test_no_stop_gradient_leaves_are_fully_differentiable(self):
        # Explicitly documents the spec §4.1 "reading B" choice: no stop-gradient
        # anywhere in the leaf -> level -> attend path.
        d, block_size, depth = 4, 2, 2
        mem = HierarchicalMemory(d, block_size, depth)
        N = block_size ** depth
        leaves = torch.randn(1, N, d, requires_grad=True)
        source_ids = torch.zeros(1, N, dtype=torch.long)
        mem.build(leaves, source_ids)
        # every level must trace back to `leaves` in the autograd graph
        for level_k in mem._levels_k:
            self.assertTrue(level_k.requires_grad)


class TestNumericalReference(unittest.TestCase):
    def test_compressor_pooling_matches_manual_reference(self):
        # Tests the attention-weighted pooling in isolation (LevelCompressor.pool),
        # independently of the residual FF added on top of it (see the dedicated
        # FF test below) — keeps this check focused on the highest-risk part
        # (indexing/transpose bugs in the einsum-based pooling).
        torch.manual_seed(0)
        d, C, P, B = 4, 3, 2, 1
        compressor = LevelCompressor(d, n_slots=1)
        children_k = torch.randn(B, P, C, d)
        children_v = torch.randn(B, P, C, d)

        pooled_k, pooled_v = compressor.pool(children_k, children_v)

        # manual reference: explicit python loops, no einsum/view, using the
        # exact same learned query parameter.
        q = compressor.query[0]  # (d,)
        ref_k = torch.zeros(B, P, 1, d)
        ref_v = torch.zeros(B, P, 1, d)
        for b in range(B):
            for p in range(P):
                scores = torch.stack([
                    (q * children_k[b, p, c]).sum() / math.sqrt(d) for c in range(C)
                ])
                w = torch.softmax(scores, dim=0)
                acc_k = torch.zeros(d)
                acc_v = torch.zeros(d)
                for c in range(C):
                    acc_k += w[c] * children_k[b, p, c]
                    acc_v += w[c] * children_v[b, p, c]
                ref_k[b, p, 0] = acc_k
                ref_v[b, p, 0] = acc_v

        torch.testing.assert_close(pooled_k, ref_k, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pooled_v, ref_v, atol=1e-5, rtol=1e-5)

    def test_compressor_applies_residual_ff_on_top_of_pooling(self):
        torch.manual_seed(0)
        d, C, P, B = 4, 3, 2, 1
        compressor = LevelCompressor(d, n_slots=1)
        children_k = torch.randn(B, P, C, d)
        children_v = torch.randn(B, P, C, d)

        pooled_k, pooled_v = compressor.pool(children_k, children_v)
        parent_k, parent_v = compressor(children_k, children_v)

        expected_k = pooled_k + compressor.ff_k_out(F.gelu(compressor.ff_k_in(compressor.norm_k(pooled_k))))
        expected_v = pooled_v + compressor.ff_v_out(F.gelu(compressor.ff_v_in(compressor.norm_v(pooled_v))))
        torch.testing.assert_close(parent_k, expected_k, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(parent_v, expected_v, atol=1e-6, rtol=1e-6)

        # sanity: the FF must actually change the output (not a no-op), and the
        # residual connection means output stays close-ish to the pooled input.
        self.assertFalse(torch.allclose(parent_k, pooled_k))
        self.assertFalse(torch.allclose(parent_v, pooled_v))

    def test_unified_attention_matches_manual_reference(self):
        torch.manual_seed(0)
        d, block_size, depth = 4, 2, 1
        mem = HierarchicalMemory(d, block_size, depth, n_head=1)
        N = block_size ** depth
        leaves = torch.randn(1, N, d)
        source_ids = torch.zeros(1, N, dtype=torch.long)
        mem.build(leaves, source_ids)

        query = torch.randn(1, 1, d)
        out = mem.attend(query)

        # manual reference over the concatenated (normalized) levels
        k_all = torch.cat([mem.level_norms[i](k) for i, k in enumerate(mem._levels_k)], dim=1)[0]
        v_all = torch.cat(mem._levels_v, dim=1)[0]
        q = mem.q_proj(query)[0, 0]

        S = k_all.shape[0]
        scores = torch.stack([(q * k_all[s]).sum() / math.sqrt(d) for s in range(S)])
        w = torch.softmax(scores, dim=0)
        ref_out = torch.zeros(d)
        for s in range(S):
            ref_out += w[s] * v_all[s]

        torch.testing.assert_close(out[0, 0], ref_out, atol=1e-4, rtol=1e-4)


class TestRMSNormScaleBias(unittest.TestCase):
    def test_per_level_rmsnorm_equalizes_scale(self):
        d, block_size, depth = 8, 4, 2
        mem = HierarchicalMemory(d, block_size, depth)

        # simulate the realistic scale mismatch: raw leaves large norm,
        # compressed levels smaller norm (as an average of several vectors would be).
        leaf_k = torch.randn(1, block_size ** depth, d) * 10.0
        level1_k = torch.randn(1, block_size ** (depth - 1), d) * 1.0
        level2_k = torch.randn(1, 1, d) * 0.1
        mem._levels_k = [leaf_k, level1_k, level2_k]
        mem._levels_v = [leaf_k, level1_k, level2_k]  # values unused here

        raw_norms = [k.norm(dim=-1).mean().item() for k in mem._levels_k]
        normed_norms = [mem.level_norms[i](k).norm(dim=-1).mean().item() for i, k in enumerate(mem._levels_k)]

        raw_ratio = max(raw_norms) / min(raw_norms)
        normed_ratio = max(normed_norms) / min(normed_norms)

        self.assertGreater(raw_ratio, 5.0, "test setup should have a large raw scale mismatch")
        self.assertLess(normed_ratio, 1.5, "RMSNorm per level should equalize key scale across levels")


class TestOverfitSanityCheck(unittest.TestCase):
    def test_indexed_thinker_overfits_small_kb_retrieval_batch(self):
        torch.manual_seed(0)
        n_facts, vocab_size = 4, 16
        ds = KBRetrievalDataset(n_facts=n_facts, vocab_size=vocab_size, seed=0)
        kb_tokens, kb_source_ids, query_tokens, labels = ds.sample_batch(batch_size=8)

        model = IndexedThinker(
            vocab_size=ds.total_vocab_size, d_model=32, n_register=1,
            block_size=4, depth=2, n_slots=1, n_head=1, sm_cap=8,
        )
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

        losses = []
        for step in range(300):
            opt.zero_grad()
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=2)
            logits = streams['answer']
            loss = F.cross_entropy(logits[:, 0, :], labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=2)
            preds = streams['answer'][:, 0, :].argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        self.assertLess(losses[-1], losses[0] * 0.1, "loss should drop sharply on this fixed tiny batch")
        self.assertGreaterEqual(acc, 0.9, f"expected near-perfect overfit accuracy, got {acc}")

    def test_flat_baseline_depth_zero_also_overfits(self):
        # Baseline C (spec §9): depth=0 degenerates HierarchicalMemory to plain
        # dense attention over all leaves, no hierarchy. Sanity check that the
        # same IndexedThinker loop isn't broken/regressed in this degenerate case.
        torch.manual_seed(0)
        n_facts, vocab_size = 4, 16
        ds = KBRetrievalDataset(n_facts=n_facts, vocab_size=vocab_size, seed=1)
        kb_tokens, kb_source_ids, query_tokens, labels = ds.sample_batch(batch_size=8)

        model = IndexedThinker(
            vocab_size=ds.total_vocab_size, d_model=32, n_register=1,
            block_size=16, depth=0, n_slots=1, n_head=1, sm_cap=8,
        )
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

        losses = []
        for step in range(300):
            opt.zero_grad()
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=2)
            logits = streams['answer']
            loss = F.cross_entropy(logits[:, 0, :], labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=2)
            preds = streams['answer'][:, 0, :].argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        self.assertGreaterEqual(acc, 0.9, f"flat (depth=0) baseline should also overfit this tiny batch, got {acc}")


class TestOutputStreamsIndependence(unittest.TestCase):
    def test_streams_have_disjoint_parameters(self):
        model = IndexedThinker(
            vocab_size=20, d_model=16, n_register=1, block_size=4, depth=1,
            stream_dims={'answer': 20, 'thinking': 8},
        )
        answer_params = set(id(p) for p in model.streams['answer'].parameters())
        thinking_params = set(id(p) for p in model.streams['thinking'].parameters())
        self.assertTrue(answer_params.isdisjoint(thinking_params),
                         "output streams must not share any parameter tensor")

    def test_backward_on_one_stream_does_not_touch_the_other(self):
        torch.manual_seed(0)
        ds = KBRetrievalDataset(n_facts=1, vocab_size=8, seed=2)  # n_leaves=4, block_size=4, depth=1
        kb_tokens, kb_source_ids, query_tokens, labels = ds.sample_batch(batch_size=4)

        model = IndexedThinker(
            vocab_size=ds.total_vocab_size, d_model=16, n_register=1,
            block_size=4, depth=1, stream_dims={'answer': ds.total_vocab_size, 'thinking': 8},
        )
        _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=2)
        loss = streams['answer'].sum()
        loss.backward()

        for p in model.streams['answer'].parameters():
            self.assertIsNotNone(p.grad)
        for p in model.streams['thinking'].parameters():
            self.assertIsNone(p.grad, "a loss on 'answer' must not populate gradients on 'thinking' stream weights")

    def test_stream_supports_1_to_3_layers(self):
        d, B, S = 8, 2, 5
        sm_k = torch.randn(B, S, d)
        sm_v = torch.randn(B, S, d)
        for n_layers in (1, 2, 3):
            stream = OutputStream(d, out_dim=6, n_layers=n_layers)
            self.assertEqual(len(stream.layers), n_layers)
            out = stream(sm_k, sm_v)
            self.assertEqual(out.shape, (B, 1, 6))

        with self.assertRaises(AssertionError):
            OutputStream(d, out_dim=6, n_layers=4)

    def test_gradient_reaches_every_layer_of_a_multilayer_stream(self):
        d, B, S = 8, 2, 5
        sm_k = torch.randn(B, S, d, requires_grad=True)
        sm_v = torch.randn(B, S, d, requires_grad=True)
        stream = OutputStream(d, out_dim=6, n_layers=3)
        out = stream(sm_k, sm_v)
        out.sum().backward()

        for i, layer in enumerate(stream.layers):
            self.assertIsNotNone(layer.q_proj.weight.grad, f"layer {i} q_proj got no gradient")
            self.assertTrue(torch.any(layer.q_proj.weight.grad != 0), f"layer {i} q_proj gradient is all-zero")
            self.assertIsNotNone(layer.ff_in.weight.grad, f"layer {i} ff_in got no gradient")
            self.assertTrue(torch.any(layer.ff_in.weight.grad != 0), f"layer {i} ff_in gradient is all-zero")


if __name__ == '__main__':
    unittest.main()
