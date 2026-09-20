"""
Correctness tests for learn/distill/chunked_loss.py (spec §13.3): the chunked
CE+KD loss must match the unchunked F.cross_entropy/topk_kd_loss call exactly
(loss value and gradient), for any chunk_size and with/without checkpoint --
see that module's docstring for why row-chunking needs no cross-chunk
accumulator (per-token normalization).
"""
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from learn.distill.chunked_loss import chunked_ce_kd_loss
from learn.distill.train_sft import topk_kd_loss
from core.indexed_thinker_model import OutputStream, Thinker


class TestChunkedCEKDLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.N, self.d, self.V, self.K = 37, 8, 50, 5
        self.head = nn.Linear(self.d, self.V)
        self.hidden = torch.randn(self.N, self.d)
        self.labels = torch.randint(0, self.V, (self.N,))
        self.labels[3] = -100
        self.ti = torch.randint(0, self.V, (self.N, self.K))
        self.tv = torch.randn(self.N, self.K)
        self.tr = torch.randn(self.N)
        self.tm = torch.ones(self.N, dtype=torch.bool)
        self.tm[5] = False

    def _reference(self):
        h = self.hidden.clone().requires_grad_(True)
        logits = self.head(h)
        ce = F.cross_entropy(logits, self.labels, ignore_index=-100)
        kd = topk_kd_loss(logits.unsqueeze(0), self.ti.unsqueeze(0), self.tv.unsqueeze(0),
                           self.tr.unsqueeze(0), self.tm.unsqueeze(0))
        (ce + kd).backward()
        return ce.detach(), kd.detach(), h.grad.clone()

    def test_matches_unchunked_for_various_chunk_sizes_and_checkpoint(self):
        ce_ref, kd_ref, grad_ref = self._reference()
        for chunk_size in (1, 7, 100):
            for use_checkpoint in (True, False):
                for p in self.head.parameters():
                    p.grad = None
                h = self.hidden.clone().requires_grad_(True)
                ce, kd = chunked_ce_kd_loss(h, self.head, self.labels, chunk_size=chunk_size,
                                             teacher_indices=self.ti, teacher_values=self.tv,
                                             teacher_residual=self.tr, teacher_mask=self.tm,
                                             use_checkpoint=use_checkpoint)
                (ce + kd).backward()
                with self.subTest(chunk_size=chunk_size, use_checkpoint=use_checkpoint):
                    self.assertTrue(torch.allclose(ce, ce_ref, atol=1e-5))
                    self.assertTrue(torch.allclose(kd, kd_ref, atol=1e-5))
                    self.assertTrue(torch.allclose(h.grad, grad_ref, atol=1e-4))

    def test_ce_only_path_when_teacher_targets_omitted(self):
        h = self.hidden.clone().requires_grad_(True)
        ce, kd = chunked_ce_kd_loss(h, self.head, self.labels, chunk_size=7)
        logits = self.head(self.hidden)
        ce_ref = F.cross_entropy(logits, self.labels, ignore_index=-100)
        self.assertTrue(torch.allclose(ce, ce_ref, atol=1e-5))
        self.assertEqual(kd.item(), 0.0)

    def test_chunk_size_zero_or_negative_falls_back_to_single_chunk(self):
        ce_ref, kd_ref, _ = self._reference()
        h = self.hidden.clone().requires_grad_(True)
        ce, kd = chunked_ce_kd_loss(h, self.head, self.labels, chunk_size=0,
                                     teacher_indices=self.ti, teacher_values=self.tv,
                                     teacher_residual=self.tr, teacher_mask=self.tm)
        self.assertTrue(torch.allclose(ce, ce_ref, atol=1e-5))
        self.assertTrue(torch.allclose(kd, kd_ref, atol=1e-5))


class TestReturnHiddenMatchesLogits(unittest.TestCase):
    """return_hidden=True (spec §13.3) must give exactly `head(hidden) ==
    forward()`'s own logits -- it's a pure refactor of where the head is
    applied, not a behavior change."""

    def test_output_stream_return_hidden(self):
        torch.manual_seed(0)
        d_model, out_dim = 16, 12
        stream = OutputStream(d_model, out_dim, n_layers=1)
        sm_k = torch.randn(2, 5, d_model)
        sm_v = torch.randn(2, 5, d_model)
        logits = stream(sm_k, sm_v)
        hidden = stream(sm_k, sm_v, return_hidden=True)
        self.assertTrue(torch.allclose(stream.head(hidden), logits, atol=1e-6))

    def test_thinker_return_hidden(self):
        torch.manual_seed(0)
        d_model, vocab = 16, 20
        model = Thinker(vocab_size=vocab, d_model=d_model, block_size=4, depth=0,
                         n_register=1, stream_dims={"answer": vocab}, stream_n_layers={"answer": 1},
                         stream_sequence={"answer": True}, max_target_len=6)
        B, N, T = 2, 4, 3
        kb_tokens = torch.randint(0, vocab, (B, N))
        kb_source_ids = torch.zeros(B, N, dtype=torch.long)
        query_tokens = torch.randint(0, vocab, (B, N))
        target_input = torch.randint(0, vocab, (B, T))
        _, logits_out = model(kb_tokens, kb_source_ids, query_tokens, n_step=2, target_input=target_input)
        _, hidden_out = model(kb_tokens, kb_source_ids, query_tokens, n_step=2, target_input=target_input,
                               return_hidden=True)
        head = model.streams["answer"].head
        self.assertTrue(torch.allclose(head(hidden_out["answer"]), logits_out["answer"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
