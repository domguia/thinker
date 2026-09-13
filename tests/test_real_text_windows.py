"""
CPU-only correctness tests for data/real_text_windows.py (spec §14, plan
Phase 11). Uses a trivial whitespace/int tokenizer instead of a real HF
tokenizer -- this module only needs a `tokenizer(text, truncation=False) ->
{"input_ids": [...]}`-shaped callable, no HF/network dependency to test it.
"""
import json
import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from data.real_text_windows import RealTextWindowDataset

PAD_ID = -1


class ListTokenizer:
    """Splits on whitespace, tokens are plain ints -- makes expected window
    content trivially predictable/hand-checkable in the tests below."""

    pad_token_id = PAD_ID

    def __call__(self, text, truncation=False):
        return {"input_ids": [int(tok) for tok in text.split()]}


def write_jsonl(path, docs):
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}) + "\n")


def doc_of(n, start=0):
    """A document of n tokens, values start..start+n-1 (so content at any
    position is directly checkable without re-deriving the tokenizer)."""
    return " ".join(str(start + i) for i in range(n))


class RealTextWindowsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "train.jsonl")
        self.tokenizer = ListTokenizer()

    def _make(self, docs, **kwargs):
        write_jsonl(self.path, docs)
        return RealTextWindowDataset(self.path, self.tokenizer, **kwargs)


def manual_window_count(L, t_tgt, stride, n_ctx, min_real_context):
    """Reference implementation of RealTextWindowDataset._add_windows's loop,
    independent of the module under test, including the min_real_context
    filter (a window needs >= min_real_context real tokens before it)."""
    count = 0
    p = 0
    while p + t_tgt <= L:
        if min(p, n_ctx) >= min_real_context:
            count += 1
        p += stride
    return count


class TestWindowCountAndStride(RealTextWindowsTestCase):
    def test_window_count_matches_manual_stride_loop(self):
        L, n_ctx, t_local, t_tgt = 37, 8, 3, 4
        ds = self._make([doc_of(L)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        expected = manual_window_count(L, t_tgt, t_tgt, n_ctx, min_real_context=1)
        self.assertEqual(len(ds), expected)

    def test_custom_stride_changes_window_count(self):
        L, n_ctx, t_local, t_tgt, stride = 40, 8, 3, 4, 2
        ds = self._make([doc_of(L)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt, stride=stride)
        expected = manual_window_count(L, t_tgt, stride, n_ctx, min_real_context=1)
        self.assertEqual(len(ds), expected)

        ds_default_stride = self._make([doc_of(L)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        self.assertGreater(len(ds), len(ds_default_stride))  # smaller stride -> strictly more windows


class TestShapes(RealTextWindowsTestCase):
    def test_shapes_and_dtypes(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds = self._make([doc_of(30)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        w = ds[0]
        self.assertEqual(w["kb_tokens"].shape, (n_ctx,))
        self.assertEqual(w["kb_source_ids"].shape, (n_ctx,))
        self.assertEqual(w["kb_leaf_mask"].shape, (n_ctx,))
        self.assertEqual(w["target_input"].shape, (t_tgt,))
        self.assertEqual(w["labels"].shape, (t_tgt,))
        self.assertEqual(w["kb_tokens"].dtype, torch.long)
        self.assertEqual(w["kb_leaf_mask"].dtype, torch.bool)
        self.assertEqual(w["kb_source_ids"].dtype, torch.long)


class TestPaddingAtDocumentStart(RealTextWindowsTestCase):
    def test_zero_real_context_window_is_excluded_by_default(self):
        # p=0 would have zero real context tokens -- default min_real_context=1
        # excludes it, so the first yielded window is p=stride (=t_tgt here), not p=0.
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds = self._make([doc_of(20)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        self.assertEqual(ds.windows[0][1], t_tgt, "first window should start at p=stride, p=0 excluded")

    def test_first_window_is_partially_padded(self):
        # first yielded window: p=4 (stride=t_tgt=4), only 4 real context tokens
        # available (0..3), n_ctx=8 -> 4 padded + 4 real.
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds = self._make([doc_of(20)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        w = ds[0]
        mask = w["kb_leaf_mask"]
        self.assertEqual(mask.tolist(), [False, False, False, False, True, True, True, True])
        self.assertEqual(w["kb_tokens"][4:].tolist(), [0, 1, 2, 3])
        self.assertTrue(torch.all(w["kb_tokens"][:4] == PAD_ID))
        self.assertEqual(w["labels"].tolist(), [4, 5, 6, 7])
        self.assertEqual(w["target_input"].tolist(), [3, 4, 5, 6])  # last real ctx token + target[:-1]


class TestFullyRealMiddleWindow(RealTextWindowsTestCase):
    def test_no_padding_once_document_has_enough_history(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        L = 40
        ds = self._make([doc_of(L)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        # second yielded window: p = 8, exactly n_ctx real tokens available before it -> no padding.
        w = ds[1]
        p = 8
        self.assertEqual(ds.windows[1][1], p)
        self.assertTrue(w["kb_leaf_mask"].all())
        self.assertEqual(w["kb_tokens"].tolist(), list(range(p - n_ctx, p)))
        self.assertEqual(w["labels"].tolist(), list(range(p, p + t_tgt)))
        self.assertEqual(w["target_input"][0].item(), p - 1)


class TestSourceIdSplitByPosition(RealTextWindowsTestCase):
    def test_source_ids_partition_kb_vs_local_regardless_of_padding(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds = self._make([doc_of(40)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        for w in (ds[0], ds[1], ds[2]):  # padded, partially padded, fully real
            source_ids = w["kb_source_ids"]
            self.assertEqual(source_ids[: n_ctx - t_local].tolist(), [1] * (n_ctx - t_local))
            self.assertEqual(source_ids[n_ctx - t_local:].tolist(), [0] * t_local)

    def test_t_local_equal_n_ctx_has_no_kb_region(self):
        n_ctx, t_tgt = 6, 3
        ds = self._make([doc_of(30)], n_ctx=n_ctx, t_local=n_ctx, t_tgt=t_tgt)
        w = ds[2]
        self.assertTrue(torch.all(w["kb_source_ids"] == 0))


class TestMultipleDocuments(RealTextWindowsTestCase):
    def test_windows_do_not_cross_document_boundary_and_doc_ids_correct(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        docs = [doc_of(20, start=0), doc_of(24, start=1000)]
        ds = self._make(docs, n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)

        doc_ids = [ds.windows[i][0] for i in range(len(ds))]
        self.assertIn(0, doc_ids)
        self.assertIn(1, doc_ids)

        for i in range(len(ds)):
            w = ds[i]
            real = w["kb_tokens"][w["kb_leaf_mask"]]
            if w["doc_id"] == 0:
                self.assertTrue(torch.all(real < 1000))
            else:
                self.assertTrue(torch.all(real >= 1000))

    def test_is_first_window_flag_resets_per_document(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        docs = [doc_of(20, start=0), doc_of(20, start=1000)]
        ds = self._make(docs, n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)

        seen_doc = set()
        for i in range(len(ds)):
            w = ds[i]
            if w["doc_id"] not in seen_doc:
                self.assertTrue(w["is_first_window"], f"first window of doc {w['doc_id']} must be flagged")
                seen_doc.add(w["doc_id"])
            else:
                self.assertFalse(w["is_first_window"])
        self.assertEqual(seen_doc, {0, 1})


class TestMinRealContext(RealTextWindowsTestCase):
    def test_windows_below_min_real_context_are_excluded(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds_default = self._make([doc_of(20)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt, min_real_context=1)
        write_jsonl(self.path, [doc_of(20)])
        ds_strict = RealTextWindowDataset(
            self.path, self.tokenizer, n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt, min_real_context=5,
        )
        self.assertLess(len(ds_strict), len(ds_default))
        for i in range(len(ds_strict)):
            w = ds_strict[i]
            self.assertGreaterEqual(w["kb_leaf_mask"].sum().item(), 5)


class TestArgValidation(RealTextWindowsTestCase):
    def test_t_local_larger_than_n_ctx_raises(self):
        write_jsonl(self.path, [doc_of(20)])
        with self.assertRaises(AssertionError):
            RealTextWindowDataset(self.path, self.tokenizer, n_ctx=4, t_local=5, t_tgt=2)

    def test_t_local_zero_raises(self):
        write_jsonl(self.path, [doc_of(20)])
        with self.assertRaises(AssertionError):
            RealTextWindowDataset(self.path, self.tokenizer, n_ctx=4, t_local=0, t_tgt=2)

    def test_min_real_context_out_of_range_raises(self):
        write_jsonl(self.path, [doc_of(20)])
        with self.assertRaises(AssertionError):
            RealTextWindowDataset(self.path, self.tokenizer, n_ctx=4, t_local=2, t_tgt=2, min_real_context=0)
        with self.assertRaises(AssertionError):
            RealTextWindowDataset(self.path, self.tokenizer, n_ctx=4, t_local=2, t_tgt=2, min_real_context=5)


class TestPadIdSource(RealTextWindowsTestCase):
    def test_pad_id_from_tokenizer_attribute_by_default(self):
        ds = self._make([doc_of(20)], n_ctx=8, t_local=3, t_tgt=4)
        self.assertEqual(ds.pad_id, PAD_ID)
        w = ds[0]
        self.assertTrue(torch.all(w["kb_tokens"][~w["kb_leaf_mask"]] == PAD_ID))

    def test_explicit_pad_id_overrides_tokenizer(self):
        ds = self._make([doc_of(20)], n_ctx=8, t_local=3, t_tgt=4, pad_id=-999)
        self.assertEqual(ds.pad_id, -999)
        w = ds[0]
        self.assertTrue(torch.all(w["kb_tokens"][~w["kb_leaf_mask"]] == -999))


class TestDataLoaderIntegration(RealTextWindowsTestCase):
    def test_default_collate_batches_fixed_shape_windows(self):
        n_ctx, t_local, t_tgt = 8, 3, 4
        ds = self._make([doc_of(60), doc_of(60, start=1000)], n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        self.assertEqual(batch["kb_tokens"].shape, (4, n_ctx))
        self.assertEqual(batch["kb_source_ids"].shape, (4, n_ctx))
        self.assertEqual(batch["kb_leaf_mask"].shape, (4, n_ctx))
        self.assertEqual(batch["target_input"].shape, (4, t_tgt))
        self.assertEqual(batch["labels"].shape, (4, t_tgt))
        self.assertEqual(batch["doc_id"].shape, (4,))
        self.assertEqual(batch["is_first_window"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
