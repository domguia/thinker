"""
Correctness tests for representation-distillation (spec Q2,
dev_notes/indexed_attention_experiment_plan.md): PromptResponseReprTargets /
_repr_targets (data/prompt_response_dataset.py) and their wiring into
RetrievalPromptDataset/ReasoningPromptDataset. GPT2-based, offline, no
network/GPU needed.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from data.prompt_response_dataset import (
    PromptResponseReprTargets, ReasoningPromptDataset, RetrievalPromptDataset, _repr_targets,
)


class TestPromptResponseReprTargets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        hidden_size, self.n_tokens = 8, 20
        self.raw = np.random.RandomState(0).randn(self.n_tokens, hidden_size).astype(np.float16)
        self.npz_path = os.path.join(self.tmpdir, "repr.npz")
        np.savez_compressed(self.npz_path, hidden_5=self.raw, offsets=np.array([0, self.n_tokens], dtype=np.int64))

    def test_projection_shape_and_determinism(self):
        r1 = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=0)
        r2 = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=0)
        self.assertEqual(r1.projected.shape, (self.n_tokens, 4))
        # Same seed -> byte-identical projection (fixed, never fit/trained).
        self.assertTrue(np.allclose(r1.projected.astype(np.float32), r2.projected.astype(np.float32)))

    def test_different_seed_gives_different_projection(self):
        r1 = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=0)
        r2 = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=1)
        self.assertFalse(np.allclose(r1.projected.astype(np.float32), r2.projected.astype(np.float32)))

    def test_missing_layer_key_raises(self):
        with self.assertRaises(AssertionError):
            PromptResponseReprTargets(self.npz_path, layer=99, proj_dim=4)

    def test_slice_span_masks_out_of_range_positions(self):
        r = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=0)
        # tok_start=0, n=5 -- q = tok_start+t-1 is -1 at t=0 (out of range, masked),
        # then 0..3 for t=1..4 (in range).
        out, mask = r.slice_span(doc_id=0, tok_start=0, n=5, t_max=5)
        self.assertEqual(list(mask.tolist()), [False, True, True, True, True])
        for t in range(1, 5):
            q = t - 1
            self.assertTrue(torch.allclose(out[t], torch.from_numpy(r.projected[q].astype(np.float32))))

    def test_repr_targets_helper_none_teacher(self):
        self.assertIsNone(_repr_targets(None, (0, 5), 0, 5))

    def test_repr_targets_helper_none_kd_info_returns_empty(self):
        r = PromptResponseReprTargets(self.npz_path, layer=5, proj_dim=4, seed=0)
        out, mask = _repr_targets(r, None, 0, 5)
        self.assertEqual(out.shape, (5, 4))
        self.assertFalse(mask.any())


class TestDatasetWiring(unittest.TestCase):
    def setUp(self):
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.tmpdir = tempfile.mkdtemp()

    def _write_jsonl(self, rows, name):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    def _make_teacher_npz(self, texts, hidden_size=8, k=4):
        all_ids = [self.tok(t)["input_ids"] for t in texts]
        offsets = [0]
        for ids in all_ids:
            offsets.append(offsets[-1] + len(ids))
        n_total = offsets[-1]
        indices = np.random.RandomState(0).randint(0, 100, size=(n_total, k)).astype(np.int32)
        values = np.random.RandomState(0).randn(n_total, k).astype(np.float16)
        residual = np.random.RandomState(0).randn(n_total).astype(np.float16)
        path = os.path.join(self.tmpdir, "topk.npz")
        np.savez_compressed(path, indices=indices, values=values, residual=residual,
                             offsets=np.array(offsets, dtype=np.int64), k=k)
        return path, n_total, hidden_size

    def _make_repr_npz(self, n_total, hidden_size, layer=5):
        path = os.path.join(self.tmpdir, "repr.npz")
        raw = np.random.RandomState(1).randn(n_total, hidden_size).astype(np.float16)
        np.savez_compressed(path, **{f"hidden_{layer}": raw}, offsets=np.array([0, n_total], dtype=np.int64))
        return path

    def test_retrieval_dataset_produces_repr_target(self):
        text = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis is nice<|im_end|>"
        row = {"question": "What is the capital of France?", "context_docs": ["France is in Europe."],
               "answer": "Paris is nice", "text": text}
        path = self._write_jsonl([row], "retrieval.jsonl")
        topk_path, n_total, hidden_size = self._make_teacher_npz([text])
        repr_path = self._make_repr_npz(n_total, hidden_size)

        ds = RetrievalPromptDataset(
            path, self.tok, block_size=16, n_docs_max=2, max_answer_len=8,
            teacher_targets=topk_path, repr_teacher_hidden=repr_path, repr_teacher_layer=5, repr_proj_dim=4,
        )
        ex = ds[0]
        self.assertIn("answer_repr_target", ex)
        self.assertIn("answer_repr_mask", ex)
        self.assertEqual(ex["answer_repr_target"].shape, (8, 4))
        # Answer is verbatim in text -> in-context alignment should succeed -> some real signal.
        self.assertTrue(ex["answer_repr_mask"].any())

    def test_repr_requires_teacher_targets(self):
        with self.assertRaises(AssertionError):
            RetrievalPromptDataset(
                self._write_jsonl([{"question": "q", "context_docs": ["d"], "answer": "a", "text": "t"}], "r2.jsonl"),
                self.tok, block_size=8, n_docs_max=1, max_answer_len=4,
                repr_teacher_hidden="/nonexistent.npz", repr_teacher_layer=5,
            )

    def test_reasoning_dataset_produces_repr_targets_for_both_streams(self):
        text = ("<|im_start|>user\nSolve 2+2<|im_end|>\n<|im_start|>assistant\n"
                "<think>\nadd them\n</think>\nThe answer is 4<|im_end|>")
        row = {"problem": "Solve 2+2", "trace": "<think>\nadd them\n</think>\nThe answer is 4",
               "answer": "The answer is 4", "text": text}
        path = self._write_jsonl([row], "reasoning.jsonl")
        topk_path, n_total, hidden_size = self._make_teacher_npz([text])
        repr_path = self._make_repr_npz(n_total, hidden_size)

        ds = ReasoningPromptDataset(
            path, self.tok, n_ctx=16, max_thinking_len=8, max_answer_len=8,
            teacher_targets=topk_path, repr_teacher_hidden=repr_path, repr_teacher_layer=5, repr_proj_dim=4,
        )
        ex = ds[0]
        self.assertIn("thinking_repr_target", ex)
        self.assertIn("answer_repr_target", ex)
        self.assertEqual(ex["thinking_repr_target"].shape, (8, 4))
        self.assertEqual(ex["answer_repr_target"].shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
