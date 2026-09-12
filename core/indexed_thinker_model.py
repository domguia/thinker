"""
Minimal end-to-end model wiring HierarchicalMemory (core/indexed_memory.py) into
the "Indexed Attention" main loop described in dev_notes/indexed_attention_spec.md §2:

    R_t = R_{t-1} + Linear([O_kb_t ; O_sm_t ; R_{t-1}])   (no FF by default — spec §-1;
                                                            `use_ff=True` reintroduces one
                                                            here only, plan Phase 1bis)
    O_kb_t = HierarchicalMemory.attend(R_{t-1})          (input ∪ KB, hierarchical)
    O_sm_t = Attn(Q_sm(R_{t-1}), K^s, V^s)                (short-term memory, flat)
    new_K, new_V = split(W_sm(R_t))  -> appended to SM     (no stop-gradient by default,
                                                            spec §4.1 reading B;
                                                            `detach_sm_keys=True` tests
                                                            reading A)

Output is produced by one or more independent "Output Streams" (spec §11bis,
source: raw/Distill-reasonning-stream.md) reading the *full* accumulated SM
trajectory by their own dedicated cross-attention — each stream owns its own
query + head weights, disjoint from every other stream and from the core loop
above. Only the SM tensor itself (an activation, not a weight) is shared.

Deliberately a *new*, separate model from core/toy_model.py::ToyThinker rather
than a modification of it: ToyThinker's forward has many special-cased flags
(autoregressive output, probes, curriculum perturbation) unrelated to this
architecture, and the spec explicitly says not to treat the existing
implementation as ground truth. This keeps the new mechanism testable in
isolation.

Not implemented (out of MVP scope, see spec §7, §11bis): No-Op / adaptive
width, SM capacity eviction beyond an optional hard cap. Q_KB vs Q_SM are
already decoupled (`HierarchicalMemory.q_proj` vs `sm_q_proj` below are
separate weights). `level_dropout_p` (stochastic level dropping) lives in
HierarchicalMemory, threaded through here; `use_ff` and `detach_sm_keys` are
the two Phase 1bis variant flags for this module specifically.
"""

import torch
from torch import nn
import torch.nn.functional as F

from core.indexed_memory import HierarchicalMemory
from core.layers import RMSNorm


class OutputStreamLayer(nn.Module):
    """
    One cross-attention block of an OutputStream (pre-norm residual). No FF:
    per spec §-1, a stream's job is to read out what the core already
    extracted into SM, not to hold its own learned factual content — an FF
    here would give the stream a place to memorize associations independently
    of what is actually present in SM (an earlier version had one; removed
    after review, see spec §11bis).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, sm_k: torch.Tensor, sm_v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.norm1(x))
        x = x + F.scaled_dot_product_attention(q, sm_k, sm_v)
        return x


class OutputStream(nn.Module):
    """
    One independent, lightweight output stream (spec §11bis): 1-3 stacked
    cross-attention layers (per raw/Distill-reasonning-stream.md's "streams
    légers (1-2 couches)", extended to allow 3) plus a head, all disjoint from
    any other stream's weights and from the core recurrent loop. Reads the
    *entire* accumulated SM trajectory via cross-attention, not just the final
    register state — the stream's own attention learns which recurrent steps
    matter for its task, rather than assuming the core's iteration count lines
    up with anything.
    """

    def __init__(self, d_model: int, out_dim: int, n_layers: int = 1):
        super().__init__()
        assert 1 <= n_layers <= 3, "output streams are meant to stay lightweight (1-3 layers)"
        self.query_seed = nn.Parameter(torch.randn(1, d_model) * d_model ** -0.5)
        self.layers = nn.ModuleList([OutputStreamLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, sm_k: torch.Tensor, sm_v: torch.Tensor) -> torch.Tensor:
        B = sm_k.shape[0]
        x = self.query_seed.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, sm_k, sm_v)
        return self.head(x)


class IndexedThinker(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_register: int,
                 block_size: int, depth: int, n_slots: int = 1, n_head: int = 1,
                 sm_cap: int = None, stream_dims: dict = None,
                 stream_n_layers: dict = None, level_dropout_p: float = 0.0,
                 detach_sm_keys: bool = False, use_ff: bool = False, ff_hidden_mult: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_register = n_register
        self.sm_cap = sm_cap
        self.detach_sm_keys = detach_sm_keys
        self.use_ff = use_ff

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_init = nn.Parameter(torch.randn(n_register, d_model) * d_model ** -0.5)

        self.memory = HierarchicalMemory(d_model, block_size, depth, n_slots=n_slots, n_head=n_head,
                                          level_dropout_p=level_dropout_p)

        self.sm_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.sm_write_proj = nn.Linear(d_model, 2 * d_model, bias=False)  # -> new K, new V

        # Register update. Default (spec §-1): a single linear recombination of
        # [O_kb; O_sm; R], no hidden-expansion FF — the core recombines what it
        # retrieved, it does not hold its own learned factual associations.
        # `use_ff=True` (plan Phase 1bis, priority #1 variant) reintroduces a
        # 2-layer GELU MLP here specifically, to test whether the "no FF
        # anywhere" premise costs composition/computation capacity — not
        # re-added to the compressor or the output streams, see spec §11bis.
        self.fuse_norm = RMSNorm(3 * d_model)
        if use_ff:
            ff_hidden = d_model * ff_hidden_mult
            self.fuse_in = nn.Linear(3 * d_model, ff_hidden)
            self.fuse_out = nn.Linear(ff_hidden, d_model)
        else:
            self.fuse_proj = nn.Linear(3 * d_model, d_model, bias=False)

        stream_dims = stream_dims if stream_dims is not None else {'answer': vocab_size}
        stream_n_layers = stream_n_layers or {}
        self.streams = nn.ModuleDict({
            name: OutputStream(d_model, dim, n_layers=stream_n_layers.get(name, 1))
            for name, dim in stream_dims.items()
        })

    def forward(self, kb_tokens: torch.Tensor, kb_source_ids: torch.Tensor,
                query_tokens: torch.Tensor, n_step: int, kb_leaf_mask: torch.Tensor = None):
        """
        kb_tokens: (B, N) leaf token ids for the unified input∪KB sequence
            (N must equal block_size ** depth when depth > 0).
        kb_source_ids: (B, N) in {0, 1} (0 = input, 1 = KB).
        kb_leaf_mask: optional (B, N) bool, True = real leaf, False = padding —
            enables a curriculum over the number of real facts while keeping
            `block_size`/`depth` fixed (see HierarchicalMemory.build).
        query_tokens: (B, Tq) token ids used to seed the register (mean-pooled
            embedding added to the learned initial register — a default choice,
            not specified by the spec).
        n_step: number of reasoning iterations (shared weights across steps).

        Returns: (R, stream_outputs) with R: (B, n_register, d_model) the final
                 core register state, and stream_outputs a dict {name: (B, 1, out_dim)}
                 — one entry per registered output stream (spec §11bis).
        """
        B = kb_tokens.shape[0]
        device = kb_tokens.device

        leaf_emb = self.embed(kb_tokens)
        self.memory.build(leaf_emb, kb_source_ids, leaf_mask=kb_leaf_mask)

        q_emb = self.embed(query_tokens).mean(dim=1, keepdim=True)  # (B, 1, d)
        R = self.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb

        sm_k = torch.zeros(B, 0, self.d_model, device=device, dtype=R.dtype)
        sm_v = torch.zeros(B, 0, self.d_model, device=device, dtype=R.dtype)

        for _ in range(n_step):
            o_kb = self.memory.attend(R)

            if sm_k.shape[1] > 0:
                q_sm = self.sm_q_proj(R)
                o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
            else:
                o_sm = torch.zeros_like(R)

            fused = torch.cat([o_kb, o_sm, R], dim=-1)
            if self.use_ff:
                delta = self.fuse_out(F.gelu(self.fuse_in(self.fuse_norm(fused))))
            else:
                delta = self.fuse_proj(self.fuse_norm(fused))
            R = R + delta

            new_k, new_v = self.sm_write_proj(R).chunk(2, dim=-1)
            if self.detach_sm_keys:
                # spec §4.1 "reading A": stop-gradient on keys entering SM only
                # (plan Phase 1bis variant) — values stay fully differentiable.
                new_k = new_k.detach()
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)
            if self.sm_cap is not None and sm_k.shape[1] > self.sm_cap:
                sm_k = sm_k[:, -self.sm_cap:]
                sm_v = sm_v[:, -self.sm_cap:]

        stream_outputs = {name: stream(sm_k, sm_v) for name, stream in self.streams.items()}
        return R, stream_outputs
