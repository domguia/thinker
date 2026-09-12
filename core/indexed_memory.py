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

Not implemented (out of MVP scope, see spec §7): No-Op / adaptive width, stop-gradient,
decoupled Q_KB/Q_SM, stochastic level dropping, padding/masking of leaves (caller must
supply exactly block_size ** depth leaves when depth > 0).
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
    M=1 by default — spec §5.1), followed by a residual feed-forward on each
    of the pooled K and V (a real Perceiver block is attention *and* FF — the
    first version of this module only had the attention half, which limited a
    parent node to a convex combination of its children with no nonlinear
    transform capacity; caught in review, see spec §11bis). The *same*
    instance is reused at every level of the hierarchy, i.e. all weights
    (pooling query and both FFs) are shared across levels ("share weight of
    the compressor no matter the stage" — spec §1/§5.1).

    A single attention score (query vs. children keys) is used to pool both K
    and V: this is what makes the parent key a genuine summary of "what this
    block is about" while the parent value stays consistent with the same
    weighting.
    """

    def __init__(self, d_model: int, n_slots: int = 1, d_hid: int = None):
        super().__init__()
        d_hid = d_hid or 4 * d_model
        self.n_slots = n_slots
        self.query = nn.Parameter(torch.randn(n_slots, d_model) * d_model ** -0.5)

        self.norm_k = RMSNorm(d_model)
        self.ff_k_in = nn.Linear(d_model, d_hid)
        self.ff_k_out = nn.Linear(d_hid, d_model)

        self.norm_v = RMSNorm(d_model)
        self.ff_v_in = nn.Linear(d_model, d_hid)
        self.ff_v_out = nn.Linear(d_hid, d_model)

    def pool(self, children_k: torch.Tensor, children_v: torch.Tensor):
        """Attention-weighted pooling only (no FF) — exposed separately for testing."""
        # children_k, children_v: (B, P, C, d) -> pooled_k, pooled_v: (B, P, M, d)
        B, P, C, d = children_k.shape
        q = self.query.view(1, 1, self.n_slots, d).expand(B, P, self.n_slots, d)
        scores = torch.einsum('bpmd,bpcd->bpmc', q, children_k) / math.sqrt(d)
        weights = F.softmax(scores, dim=-1)
        pooled_k = torch.einsum('bpmc,bpcd->bpmd', weights, children_k)
        pooled_v = torch.einsum('bpmc,bpcd->bpmd', weights, children_v)
        return pooled_k, pooled_v

    def forward(self, children_k: torch.Tensor, children_v: torch.Tensor):
        pooled_k, pooled_v = self.pool(children_k, children_v)
        parent_k = pooled_k + self.ff_k_out(F.gelu(self.ff_k_in(self.norm_k(pooled_k))))
        parent_v = pooled_v + self.ff_v_out(F.gelu(self.ff_v_in(self.norm_v(pooled_v))))
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

    def __init__(self, d_model: int, block_size: int, depth: int, n_slots: int = 1, n_head: int = 1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.block_size = block_size
        self.depth = depth
        self.n_slots = n_slots
        self.n_head = n_head

        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # priority-bias: 0 = input, 1 = KB (spec §6.2)
        self.source_bias = nn.Embedding(2, d_model)

        self.compressor = LevelCompressor(d_model, n_slots=n_slots)
        self.level_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(depth + 1)])

        self._levels_k = None
        self._levels_v = None

    def build(self, leaf_embeddings: torch.Tensor, source_ids: torch.Tensor) -> None:
        """
        leaf_embeddings: (B, N, d) raw token embeddings (pre K/V-projection).
        source_ids: (B, N) long tensor in {0, 1} (0 = input, 1 = KB).
        N must equal block_size ** depth when depth > 0 (no padding/masking
        implemented — see module docstring).
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

        levels_k = [cur_k]
        levels_v = [cur_v]
        for _ in range(self.depth):
            Bc, Nc, dc = cur_k.shape
            P = Nc // self.block_size
            children_k = cur_k.view(Bc, P, self.block_size, dc)
            children_v = cur_v.view(Bc, P, self.block_size, dc)
            parent_k, parent_v = self.compressor(children_k, children_v)
            parent_k = parent_k.reshape(Bc, P * self.n_slots, dc)
            parent_v = parent_v.reshape(Bc, P * self.n_slots, dc)
            levels_k.append(parent_k)
            levels_v.append(parent_v)
            cur_k, cur_v = parent_k, parent_v

        self._levels_k = levels_k
        self._levels_v = levels_v

    def attend(self, query_input: torch.Tensor) -> torch.Tensor:
        """
        query_input: (B, T, d) raw register state (pre Q-projection).
        Returns (B, T, d): unified softmax attention output over all leaves
        and all compressed levels (spec §5.2).
        """
        assert self._levels_k is not None, "call build() before attend()"

        normed_k = [self.level_norms[i](k) for i, k in enumerate(self._levels_k)]
        k_all = torch.cat(normed_k, dim=1)
        v_all = torch.cat(self._levels_v, dim=1)

        B, T, d = query_input.shape
        q = self.q_proj(query_input)

        if self.n_head > 1:
            hd = d // self.n_head
            S = k_all.shape[1]
            q_h = q.view(B, T, self.n_head, hd).transpose(1, 2)
            k_h = k_all.view(B, S, self.n_head, hd).transpose(1, 2)
            v_h = v_all.view(B, S, self.n_head, hd).transpose(1, 2)
            out = F.scaled_dot_product_attention(q_h, k_h, v_h)
            out = out.transpose(1, 2).contiguous().view(B, T, d)
        else:
            out = F.scaled_dot_product_attention(q, k_all, v_all)
        return out
