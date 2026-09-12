"""
Hierarchical indexed KV memory for the "Indexed Attention" Thinker variant.

Implements dev_notes/indexed_attention_spec.md §5-§6: a shared-weight compressor
builds a fixed-depth tree over a unified input∪KB leaf sequence, and a single
softmax attends over the concatenation of raw leaves and all compressed levels
(per-level RMSNorm applied to keys beforehand to correct the scale bias between
raw and compressed vectors).

This module is intentionally independent from core/toy_model.py / core/layers.py's
CustomFlexDecoder machinery — per the spec, the existing implementation is not
treated as ground truth here.

Not implemented (out of MVP scope, see spec §7): No-Op / adaptive width. `build()`'s
`leaf_mask` allows padding a variable number of real leaves up to a fixed
`block_size ** depth` (needed for a curriculum that keeps the same hierarchy shape
across stages); Q_KB/Q_SM are already decoupled (HierarchicalMemory vs IndexedThinker's
SM query use separate projections); `level_dropout_p` implements stochastic level
dropping (train-time only, see HierarchicalMemory.attend). Stop-gradient on SM keys
lives in IndexedThinker (core/indexed_thinker_model.py), not here.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from core.layers import RMSNorm


class LevelCompressor(nn.Module):
    """
    Pools blocks of `block_size` children (K, V) pairs into `n_slots` parent
    (K, V) pairs via learned queries (Perceiver-style cross-attention pooling,
    M=1 by default — spec §5.1). Pure attention-weighted pooling, no
    feed-forward: per spec §-1, the compressor's job is to summarize what is
    already stored in the KB's leaves, not to hold learned factual content of
    its own — a FF here would let the compressor's weights start encoding
    facts about specific compressed patterns rather than just recombining
    whatever content came from the KB itself (an earlier version added a
    residual FF; removed after review, see spec §11bis). The *same* instance
    is reused at every level of the hierarchy, i.e. weights are shared across
    levels ("share weight of the compressor no matter the stage" — spec §1/§5.1).

    An intra-block position embedding is added to children K/V *before*
    pooling (NSA-style — cross-checked against lucidrains/native-sparse-attention-pytorch,
    see spec §11bis): pure content-weighted pooling is a set function, blind to
    the order of children within a block, so two blocks with the same content
    in a different order would compress to the exact same parent. This is a
    genuine correctness gap (not a "no FF" violation — an additive embedding
    holds no associative-memory capacity), caught by comparing against a
    reference implementation rather than found by any test written from the
    spec alone.

    A single attention score (query vs. position-biased children keys) is
    used to pool both K and V: this is what makes the parent key a genuine
    summary of "what this block is about" while the parent value stays
    consistent with the same weighting.
    """

    def __init__(self, d_model: int, block_size: int, n_slots: int = 1):
        super().__init__()
        self.n_slots = n_slots
        self.query = nn.Parameter(torch.randn(n_slots, d_model) * d_model ** -0.5)
        self.intrablock_pos = nn.Embedding(block_size, d_model)

    def forward(self, children_k: torch.Tensor, children_v: torch.Tensor,
                children_mask: torch.Tensor = None):
        # children_k, children_v: (B, P, C, d) -> parent_k, parent_v: (B, P, M, d)
        # children_mask (optional): (B, P, C) bool, True = real leaf, False = padding.
        B, P, C, d = children_k.shape
        pos = self.intrablock_pos.weight[:C].view(1, 1, C, d)
        biased_k = children_k + pos
        biased_v = children_v + pos

        q = self.query.view(1, 1, self.n_slots, d).expand(B, P, self.n_slots, d)
        scores = torch.einsum('bpmd,bpcd->bpmc', q, biased_k) / math.sqrt(d)
        if children_mask is not None:
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~children_mask.unsqueeze(2), neg_inf)
        weights = F.softmax(scores, dim=-1)
        if children_mask is not None:
            # a block that is entirely padding has all -inf scores -> softmax gives NaN;
            # such a parent is itself masked out one level up, so its content only needs
            # to be non-NaN (not meaningful) to avoid poisoning gradients through the mask.
            weights = torch.nan_to_num(weights, nan=0.0)
        parent_k = torch.einsum('bpmc,bpcd->bpmd', weights, biased_k)
        parent_v = torch.einsum('bpmc,bpcd->bpmd', weights, biased_v)
        return parent_k, parent_v


class HierarchicalMemory(nn.Module):
    """
    Unified indexed KV memory over a leaf sequence combining input and KB
    tokens in the same space, distinguished by an additive priority-bias
    embedding (spec §6.2, strategy 1 of §8, chosen for the MVP).

    Usage:
        mem = HierarchicalMemory(d_model=64, block_size=4, depth=2)
        mem.build(leaf_embeddings, source_ids)   # once per example/batch
        out = mem.attend(query_hidden_state)      # once per reasoning step

    When depth == 0, no compression happens and this degenerates to plain
    dense attention over the raw leaves (useful as a flat, no-hierarchy
    baseline — spec §9, Baseline C).
    """

    def __init__(self, d_model: int, block_size: int, depth: int, n_slots: int = 1, n_head: int = 1,
                 level_dropout_p: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.block_size = block_size
        self.depth = depth
        self.n_slots = n_slots
        self.n_head = n_head
        # spec §11bis / plan Phase 1bis: stochastic level dropping (train-time only,
        # never applied to level 0 leaves so there is always something real to attend
        # to). Probability increases with level, matching the idea's own description
        # ("plus un niveau est haut, plus il est masqué aléatoirement souvent").
        # Reuses the leaf-padding mask machinery below rather than a separate mechanism.
        self.level_dropout_p = level_dropout_p

        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # priority-bias: 0 = input, 1 = KB (spec §6.2)
        self.source_bias = nn.Embedding(2, d_model)

        self.compressor = LevelCompressor(d_model, block_size, n_slots=n_slots)
        self.level_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(depth + 1)])

        self._levels_k = None
        self._levels_v = None
        self._levels_mask = None

    def build(self, leaf_embeddings: torch.Tensor, source_ids: torch.Tensor,
              leaf_mask: torch.Tensor = None) -> None:
        """
        leaf_embeddings: (B, N, d) raw token embeddings (pre K/V-projection).
        source_ids: (B, N) long tensor in {0, 1} (0 = input, 1 = KB).
        leaf_mask: optional (B, N) bool tensor, True = real leaf, False = padding.
            Enables a fixed-shape curriculum (same block_size/depth across
            curriculum stages, fewer real facts padded up to N — the same
            pattern already used by the original ToyThinker's fixed
            max_input_len slots, see dev_notes/indexed_attention_experiment_plan.md
            Phase -1). Padding never contributes to any pooled/attended output
            (masked before every softmax, at every level) and is never required
            when leaf_mask is None (all leaves treated as real, prior behavior
            unchanged).
        N must equal block_size ** depth when depth > 0.
        """
        B, N, d = leaf_embeddings.shape
        if self.depth > 0:
            expected_n = self.block_size ** self.depth
            assert N == expected_n, (
                f"HierarchicalMemory: expected {expected_n} leaves "
                f"(block_size={self.block_size} ** depth={self.depth}), got {N}"
            )

        biased = leaf_embeddings + self.source_bias(source_ids)
        cur_k = self.k_proj(biased)
        cur_v = self.v_proj(biased)
        cur_mask = leaf_mask if leaf_mask is not None else torch.ones(B, N, dtype=torch.bool, device=leaf_embeddings.device)

        levels_k = [cur_k]
        levels_v = [cur_v]
        levels_mask = [cur_mask]
        for _ in range(self.depth):
            Bc, Nc, dc = cur_k.shape
            P = Nc // self.block_size
            children_k = cur_k.view(Bc, P, self.block_size, dc)
            children_v = cur_v.view(Bc, P, self.block_size, dc)
            children_mask = cur_mask.view(Bc, P, self.block_size)
            parent_k, parent_v = self.compressor(children_k, children_v, children_mask=children_mask)
            parent_k = parent_k.reshape(Bc, P * self.n_slots, dc)
            parent_v = parent_v.reshape(Bc, P * self.n_slots, dc)
            parent_mask = children_mask.any(dim=-1)  # (B, P): real if >=1 real child
            parent_mask = parent_mask.unsqueeze(-1).expand(-1, -1, self.n_slots).reshape(Bc, P * self.n_slots)
            levels_k.append(parent_k)
            levels_v.append(parent_v)
            levels_mask.append(parent_mask)
            cur_k, cur_v, cur_mask = parent_k, parent_v, parent_mask

        self._levels_k = levels_k
        self._levels_v = levels_v
        self._levels_mask = levels_mask

    def attend(self, query_input: torch.Tensor) -> torch.Tensor:
        """
        query_input: (B, T, d) raw register state (pre Q-projection).
        Returns (B, T, d): unified softmax attention output over all leaves
        and all compressed levels (spec §5.2). Padded leaves/nodes (see
        `build`'s `leaf_mask`) never receive attention weight.
        """
        assert self._levels_k is not None, "call build() before attend()"

        normed_k = [self.level_norms[i](k) for i, k in enumerate(self._levels_k)]
        k_all = torch.cat(normed_k, dim=1)
        v_all = torch.cat(self._levels_v, dim=1)

        levels_mask = self._levels_mask
        if self.training and self.level_dropout_p > 0 and self.depth > 0:
            levels_mask = list(levels_mask)
            for i in range(1, len(levels_mask)):  # never drop level 0 (leaves)
                p_i = self.level_dropout_p * (i / self.depth)
                if torch.rand(()) < p_i:
                    levels_mask[i] = torch.zeros_like(levels_mask[i])
        mask_all = torch.cat(levels_mask, dim=1)  # (B, S) bool, True = attend

        B, T, d = query_input.shape
        q = self.q_proj(query_input)
        S = k_all.shape[1]

        if self.n_head > 1:
            hd = d // self.n_head
            q_h = q.view(B, T, self.n_head, hd).transpose(1, 2)
            k_h = k_all.view(B, S, self.n_head, hd).transpose(1, 2)
            v_h = v_all.view(B, S, self.n_head, hd).transpose(1, 2)
            attn_mask = mask_all.view(B, 1, 1, S)
            out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask)
            out = out.transpose(1, 2).contiguous().view(B, T, d)
        else:
            attn_mask = mask_all.view(B, 1, S)
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        return out
