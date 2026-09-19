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
across stages); Q_KB/Q_SM are already decoupled (HierarchicalMemory vs Thinker's
SM query use separate projections); `level_dropout_p` implements stochastic level
dropping (train-time only, see HierarchicalMemory.attend). Stop-gradient on SM keys
lives in Thinker (core/indexed_thinker_model.py), not here.
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

    `decouple_kv` (default True since 2026-09-13) controls whether the parent
    key and the parent value are pooled with the SAME attention weights or
    with two independently learned queries.

    Sharing one weight vector (the original behavior, `decouple_kv=False`,
    kept only as an ablation) makes an associative memory structurally
    impossible, and was the root cause of the n_hops>=2 plateau documented in
    dev_notes/experiment.log.md. A fact block is laid out as
    [KEY_MARK, key_id, VAL_MARK, val_id] (data/kb_chain_retrieval.py): to be
    usable as a memory entry, the parent must be FINDABLE by its key
    (parent_k ~ f(key_id)) and must RETURN its value (parent_v ~ g(val_id)) --
    two opposite weightings over the same children. With a single softmax the
    compressor can only pick one, or settle on a blurred compromise that is
    brute-forceable at one hop and unchainable beyond. Decoupling costs
    `n_slots * d_model` extra parameters (one more learned query) and lifts
    the constraint; both scores are still computed against the children's
    KEYS (biased_k), only the pooling target differs.

    Verified on CPU (d_model=32, n_hops=2, 3000 steps, 2 seeds): shared
    pooling plateaus at 26-35% (= 1/n_facts, i.e. "copy some KB value at
    random"), decoupled reaches 100% with loss 0.000 on both seeds.

    Two further variants, both opt-in and both independent of `decouple_kv`
    (2026-09-14, dev_notes/indexed_attention_spec.md §5.1bis "piste
    alternative" and §5.4):

    - `pool_n_head` (default 1, the user's own question): splits the
      pooling query/keys into independent heads, each with its own
      attention distribution over the block's children, instead of one
      score per child computed on the full width. Documented reasoning
      (spec §5.1bis) for why this is a *partial*, not equivalent,
      alternative to `decouple_kv`: within a single head the SAME weights
      still drive both the K-slice and V-slice average, so the structural
      key/value conflict persists locally even if different heads
      specialize toward "key position" vs "value position" -- no guarantee
      the way a dedicated second query provides. Zero new parameters
      (reshapes the existing query, doesn't add one).
    - `k_dim` (default None = d_model, i.e. symmetric, unchanged behavior):
      lets K be narrower than V, per the literature precedent already
      researched for this project (Product-Key Memory / Memory Layers at
      Scale, He et al. 2024 -- dim(V) carries the information that mixes
      into the residual stream and stays at d_model; dim(K) is chosen
      smaller purely for addressing tractability, not because the key
      "matters less"). When `k_dim != d_model`, `intrablock_pos_v` becomes
      a second, separate position-bias embedding for the V branch (the
      original single `intrablock_pos` can no longer be added to both K
      and V children once they have different widths) -- the symmetric
      default path is untouched (still one shared `intrablock_pos`, exact
      prior behavior, no new parameters), so this is purely additive.
    """

    def __init__(self, d_model: int, block_size: int, n_slots: int = 1,
                 decouple_kv: bool = True, pool_n_head: int = 1, k_dim: int = None):
        super().__init__()
        self.n_slots = n_slots
        self.decouple_kv = decouple_kv
        self.pool_n_head = pool_n_head
        self.d_model = d_model
        self.k_dim = k_dim if k_dim is not None else d_model
        assert self.k_dim % pool_n_head == 0, "k_dim must be divisible by pool_n_head"
        assert d_model % pool_n_head == 0, "d_model must be divisible by pool_n_head"

        self.query = nn.Parameter(torch.randn(n_slots, self.k_dim) * self.k_dim ** -0.5)
        if decouple_kv:
            self.query_v = nn.Parameter(torch.randn(n_slots, self.k_dim) * self.k_dim ** -0.5)
        # K-branch position bias, always present, sized k_dim (== d_model in the
        # default symmetric case -- same tensor doubles as the V-branch bias
        # too, see forward(), so the default path allocates nothing extra).
        self.intrablock_pos = nn.Embedding(block_size, self.k_dim)
        if self.k_dim != d_model:
            self.intrablock_pos_v = nn.Embedding(block_size, d_model)

    def _pool(self, query, biased_k, target, children_mask):
        B, P, C, dk = biased_k.shape
        H = self.pool_n_head
        q = query.view(1, 1, self.n_slots, dk).expand(B, P, self.n_slots, dk)
        if H > 1:
            hdk = dk // H
            q_h = q.view(B, P, self.n_slots, H, hdk)
            k_h = biased_k.view(B, P, C, H, hdk)
            scores = torch.einsum('bpmhd,bpchd->bpmhc', q_h, k_h) / math.sqrt(hdk)  # (B,P,M,H,C)
            if children_mask is not None:
                neg_inf = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~children_mask.unsqueeze(2).unsqueeze(3), neg_inf)
            weights = F.softmax(scores, dim=-1)
            if children_mask is not None:
                weights = torch.nan_to_num(weights, nan=0.0)
            dt = target.shape[-1]
            assert dt % H == 0, "target's last dim must be divisible by pool_n_head"
            target_h = target.view(B, P, C, H, dt // H)
            out_h = torch.einsum('bpmhc,bpchd->bpmhd', weights, target_h)  # (B,P,M,H,dt/H)
            return out_h.reshape(B, P, self.n_slots, dt)

        scores = torch.einsum('bpmd,bpcd->bpmc', q, biased_k) / math.sqrt(dk)
        if children_mask is not None:
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~children_mask.unsqueeze(2), neg_inf)
        weights = F.softmax(scores, dim=-1)
        if children_mask is not None:
            # a block that is entirely padding has all -inf scores -> softmax gives NaN;
            # such a parent is itself masked out one level up, so its content only needs
            # to be non-NaN (not meaningful) to avoid poisoning gradients through the mask.
            weights = torch.nan_to_num(weights, nan=0.0)
        return torch.einsum('bpmc,bpcd->bpmd', weights, target)

    def forward(self, children_k: torch.Tensor, children_v: torch.Tensor,
                children_mask: torch.Tensor = None):
        # children_k: (B, P, C, k_dim), children_v: (B, P, C, d_model) ->
        # parent_k: (B, P, M, k_dim), parent_v: (B, P, M, d_model).
        # children_mask (optional): (B, P, C) bool, True = real leaf, False = padding.
        C = children_k.shape[2]
        pos_k = self.intrablock_pos.weight[:C].view(1, 1, C, -1)
        biased_k = children_k + pos_k
        if self.k_dim == self.d_model:
            biased_v = children_v + pos_k  # symmetric default: exact prior behavior, one shared bias
        else:
            pos_v = self.intrablock_pos_v.weight[:C].view(1, 1, C, -1)
            biased_v = children_v + pos_v

        parent_k = self._pool(self.query, biased_k, biased_k, children_mask)
        value_query = self.query_v if self.decouple_kv else self.query
        parent_v = self._pool(value_query, biased_k, biased_v, children_mask)
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
                 level_dropout_p: float = 0.0, decouple_kv: bool = True,
                 pool_n_head: int = 1, k_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.k_dim = k_dim if k_dim is not None else d_model
        assert self.k_dim % n_head == 0, "k_dim must be divisible by n_head"
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
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

        # k_dim (spec §5.4, 2026-09-14): K narrower than V is a deliberate,
        # literature-backed asymmetry (Product-Key Memory / Memory Layers at
        # Scale), not implemented until now. Defaults to d_model (symmetric,
        # exact prior behavior/shapes) -- k_proj/q_proj only shrink when
        # k_dim is explicitly set smaller.
        self.k_proj = nn.Linear(d_model, self.k_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, self.k_dim, bias=False)

        # priority-bias: 0 = input, 1 = KB (spec §6.2)
        self.source_bias = nn.Embedding(2, d_model)

        # decouple_kv: see LevelCompressor's docstring -- default True since
        # 2026-09-13 (the shared-pooling variant cannot represent a key->value
        # association at all); decouple_kv=False is kept only as an ablation.
        # pool_n_head/k_dim: see LevelCompressor's docstring, both 2026-09-14,
        # both default to prior (symmetric, single-head) behavior when unset.
        self.compressor = LevelCompressor(d_model, block_size, n_slots=n_slots,
                                          decouple_kv=decouple_kv, pool_n_head=pool_n_head,
                                          k_dim=k_dim)
        self.level_norms = nn.ModuleList([RMSNorm(self.k_dim) for _ in range(depth + 1)])

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
        N must be a multiple of block_size ** depth when depth > 0 (each of
            the `depth` compression steps reshapes its current node count by
            block_size, see the loop below) -- NOT N == block_size ** depth.
            The stricter equality (fixed until 2026-09-13) forced the
            hierarchy to always reduce all the way down to a single top
            node, which conflates "how many compression steps" with "what
            block_size**depth happens to equal N" -- e.g. it made
            "one node per fact, no further hierarchy" (depth=1, block_size=4,
            N=4*n_facts) inexpressible for n_facts>1, since block_size would
            have to absorb every fact into one block to satisfy equality
            (dev_notes/indexed_attention_experiment_plan.md, Phase 0bis).
            Nothing downstream (attend()'s unified softmax, §5.2) assumes a
            single root -- concatenating a "forest" of N/block_size**depth
            top-level nodes works exactly the same as concatenating one.
        """
        B, N, d = leaf_embeddings.shape
        if self.depth > 0:
            divisor = self.block_size ** self.depth
            assert N % divisor == 0, (
                f"HierarchicalMemory: N={N} leaves must be a multiple of "
                f"block_size={self.block_size} ** depth={self.depth} = {divisor}"
            )

        biased = leaf_embeddings + self.source_bias(source_ids)
        cur_k = self.k_proj(biased)
        cur_v = self.v_proj(biased)
        cur_mask = leaf_mask if leaf_mask is not None else torch.ones(B, N, dtype=torch.bool, device=leaf_embeddings.device)

        levels_k = [cur_k]
        levels_v = [cur_v]
        levels_mask = [cur_mask]
        for _ in range(self.depth):
            Bc, Nc, dk = cur_k.shape
            dv = cur_v.shape[-1]  # k_dim (§5.4) may differ from d_model -- track separately
            P = Nc // self.block_size
            children_k = cur_k.view(Bc, P, self.block_size, dk)
            children_v = cur_v.view(Bc, P, self.block_size, dv)
            children_mask = cur_mask.view(Bc, P, self.block_size)
            parent_k, parent_v = self.compressor(children_k, children_v, children_mask=children_mask)
            parent_k = parent_k.reshape(Bc, P * self.n_slots, dk)
            parent_v = parent_v.reshape(Bc, P * self.n_slots, dv)
            parent_mask = children_mask.any(dim=-1)  # (B, P): real if >=1 real child
            parent_mask = parent_mask.unsqueeze(-1).expand(-1, -1, self.n_slots).reshape(Bc, P * self.n_slots)
            levels_k.append(parent_k)
            levels_v.append(parent_v)
            levels_mask.append(parent_mask)
            cur_k, cur_v, cur_mask = parent_k, parent_v, parent_mask

        self._levels_k = levels_k
        self._levels_v = levels_v
        self._levels_mask = levels_mask

    def build_static(self, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> None:
        """
        Phase 12 (dev_notes/indexed_attention_experiment_plan.md, S0/curriculum
        ffn2attn) escape hatch: injects a single, pre-computed level of (K, V)
        pairs directly, bypassing `k_proj`/`v_proj`/`compressor` entirely.

        `build()` always derives K and V from the SAME `leaf_embeddings` tensor
        via two learned projections -- correct when K/V come from a shared
        embedding space, but S0's exact FFN->attention conversion needs K and V
        to be two INDEPENDENT fixed matrices sharing only a neuron index
        (`K = W_in^T`/`W_gate^T`, `V = W_out`/`W_down`, spec §"Conversion
        analytique sans Softmax") -- not expressible as two projections of one
        input. `attend()` only ever reads `self._levels_k/_v/_mask`, so this is
        purely additive: no change to any existing `build()` caller or shape.

        K: (B, S, k_dim), V: (B, S, d_model) -- already in this memory's K/V
            spaces (no k_proj/v_proj/source_bias applied). For S0's frozen,
            per-layer FFN-as-KB use case, B is typically 1 (broadcast by the
            caller) and S = d_ff (one static (k_i, v_i) pair per FFN neuron).
        mask: optional (B, S) bool, True = real entry -- same convention as
            `build`'s `leaf_mask`.
        depth must be 0 for this path (a single level, no compressor tree) --
        asserted below since compressing static, weight-derived K/V through
        the leaf-embedding compressor would silently reintroduce a learned
        transform on top of what is meant to stay an exact/frozen conversion.
        """
        assert self.depth == 0, (
            "build_static is for a single static K/V level (S0's exact FFN "
            "conversion); depth>0 would run these through LevelCompressor, "
            "which expects leaf embeddings, not frozen weight-derived K/V"
        )
        B, S, _ = K.shape
        self._levels_k = [K]
        self._levels_v = [V]
        self._levels_mask = [mask if mask is not None else torch.ones(B, S, dtype=torch.bool, device=K.device)]

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

        B, T, _ = query_input.shape
        q = self.q_proj(query_input)  # (B, T, k_dim) -- may differ from d_model, see k_dim (spec §5.4)
        S = k_all.shape[1]

        if self.n_head > 1:
            # k_dim (Q/K) and d_model (V) can differ (§5.4) -- SDPA supports
            # Ev != Eqk natively, but the per-head split size must be computed
            # from EACH tensor's own last dim, not a single shared `hd`.
            hd_qk = self.k_dim // self.n_head
            hd_v = self.d_model // self.n_head
            q_h = q.view(B, T, self.n_head, hd_qk).transpose(1, 2)
            k_h = k_all.view(B, S, self.n_head, hd_qk).transpose(1, 2)
            v_h = v_all.view(B, S, self.n_head, hd_v).transpose(1, 2)
            attn_mask = mask_all.view(B, 1, 1, S)
            out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask)
            out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        else:
            attn_mask = mask_all.view(B, 1, S)
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        return out
