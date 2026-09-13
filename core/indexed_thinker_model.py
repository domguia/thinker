"""
`Thinker` -- the project's actual model (renamed from `IndexedThinker`
2026-09-13: this IS the Thinker described by the project's thesis, not a
variant of it). `core/toy_model.py::ToyThinker` is a deliberately simplified
placeholder used for earlier toy-task debugging (flat concatenated memory,
basic transformer layer, no indexing) -- not an earlier version of this class
and not renamed. `core/thinker_model.py::Th1nker` is a separate, older,
unused/inactive file -- do not confuse the two.

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

Deliberately a *separate* class from core/toy_model.py::ToyThinker rather
than a modification of it: ToyThinker's forward has many special-cased flags
(autoregressive output, probes, curriculum perturbation) unrelated to this
architecture, and the spec explicitly says not to treat the existing
implementation as ground truth. This kept the new mechanism testable in
isolation while under active development; `Thinker` here is the real target
architecture going forward, not a parallel experiment to eventually merge back.

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

    Default mode: a single learned query (`query_seed`) -> one output vector
    per forward call, as used by the synthetic single-query/single-answer
    tasks (data/kb_retrieval.py, data/kb_chain_retrieval.py).

    `sequence_mode=True` (spec §14.3, real-text integration, plan Phase 11):
    generalizes this to `T` independent per-position queries instead of one,
    for a per-token LM objective. Each query is built by the caller (Thinker,
    from teacher-forced target-token embeddings) and passed in as
    `query_input`; this class only adds a learned per-position embedding
    (`pos_embed`) on top, since a raw token embedding alone carries no
    position information. Every position still attends to `sm_k`/`sm_v`
    *independently* -- no self-attention is added between positions, so this
    stays exactly the same lightweight cross-attention mechanism as the
    default mode, merely batched over a query dimension of size `T` instead
    of 1 (`F.scaled_dot_product_attention` already treats queries at
    different positions independently when there's no causal/self mask).
    Teacher forcing means training is fully parallel across positions;
    generation is necessarily autoregressive (spec §14.3), same as any
    standard LM decoder.
    """

    def __init__(self, d_model: int, out_dim: int, n_layers: int = 1,
                 sequence_mode: bool = False, max_seq_len: int = None):
        super().__init__()
        assert 1 <= n_layers <= 3, "output streams are meant to stay lightweight (1-3 layers)"
        self.sequence_mode = sequence_mode
        if sequence_mode:
            assert max_seq_len is not None and max_seq_len > 0, (
                "sequence_mode requires max_seq_len (spec §14.3, an upper bound on T_tgt "
                "for the learned per-position embedding table)"
            )
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
        else:
            self.query_seed = nn.Parameter(torch.randn(1, d_model) * d_model ** -0.5)
        self.layers = nn.ModuleList([OutputStreamLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, sm_k: torch.Tensor, sm_v: torch.Tensor,
                query_input: torch.Tensor = None) -> torch.Tensor:
        """
        query_input (sequence_mode only): (B, T, d_model) teacher-forced
        target-token embeddings (spec §14.3's q_t = embed(target_token_{t-1}),
        computed by the caller since only Thinker owns `self.embed`) -- this
        method adds the learned position embedding on top.
        """
        B = sm_k.shape[0]
        if self.sequence_mode:
            assert query_input is not None, (
                "sequence_mode stream requires query_input (spec §14.3 teacher forcing)"
            )
            T = query_input.shape[1]
            pos = self.pos_embed(torch.arange(T, device=query_input.device)).unsqueeze(0)
            x = query_input + pos
        else:
            x = self.query_seed.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, sm_k, sm_v)
        return self.head(x)


class Thinker(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_register: int,
                 block_size: int, depth: int, n_slots: int = 1, n_head: int = 1,
                 sm_cap: int = None, stream_dims: dict = None,
                 stream_n_layers: dict = None, level_dropout_p: float = 0.0,
                 detach_sm_keys: bool = False, use_ff: bool = False, ff_hidden_mult: int = 4,
                 decouple_kv: bool = True,
                 stream_sequence: dict = None, max_target_len: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_register = n_register
        self.sm_cap = sm_cap
        self.detach_sm_keys = detach_sm_keys
        self.use_ff = use_ff

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_init = nn.Parameter(torch.randn(n_register, d_model) * d_model ** -0.5)

        self.memory = HierarchicalMemory(d_model, block_size, depth, n_slots=n_slots, n_head=n_head,
                                          level_dropout_p=level_dropout_p, decouple_kv=decouple_kv)

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
        # spec §14.3 (plan Phase 11): per-stream opt-in to the multi-position
        # generalization of OutputStream, for a per-token LM objective on real
        # text -- e.g. stream_sequence={'answer': True} while a 'thinking'
        # stream (if any) stays in the default single-query mode. Unset by
        # default so every existing single-query/single-answer usage
        # (data/kb_retrieval.py, data/kb_chain_retrieval.py) is unaffected.
        stream_sequence = stream_sequence or {}
        self.streams = nn.ModuleDict({
            name: OutputStream(
                d_model, dim, n_layers=stream_n_layers.get(name, 1),
                sequence_mode=stream_sequence.get(name, False),
                max_seq_len=max_target_len if stream_sequence.get(name, False) else None,
            )
            for name, dim in stream_dims.items()
        })

    def forward(self, kb_tokens: torch.Tensor, kb_source_ids: torch.Tensor,
                query_tokens: torch.Tensor, n_step: int, kb_leaf_mask: torch.Tensor = None,
                register_init_override: torch.Tensor = None, target_input: torch.Tensor = None):
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
        register_init_override: optional (B, n_register, d_model) — spec §14.2
            (plan Phase 11, real-text sliding windows): replaces the learned
            `self.register_init` as the base the current window's mean query
            embedding is added to. Lets a training loop carry the previous
            window's final `R` (typically stop-gradient'd by the caller, per
            §14.2's default) forward as this window's starting point, instead
            of always restarting from the learned initial register — pass it
            only for non-first windows of a document (the first window of
            each document still uses the learned `self.register_init`, see
            `RealTextWindowDataset`'s `is_first_window` flag).
        target_input: optional (B, T_tgt) — spec §14.3 (plan Phase 11):
            teacher-forced target-token ids (`target_input[t]` is the token
            that should be embedded and fed as the query predicting
            `labels[t]`, see data/real_text_windows.py). Required exactly
            when at least one registered stream has `sequence_mode=True`;
            embedded here (via `self.embed`, the same table as the KB/input
            leaves) and passed to those streams as their per-position query
            input, since only `Thinker` owns `self.embed` (OutputStream
            itself only knows how to add position information on top, see
            OutputStream's docstring).

        Returns: (R, stream_outputs) with R: (B, n_register, d_model) the final
                 core register state, and stream_outputs a dict {name: (B, 1, out_dim)}
                 (or (B, T_tgt, out_dim) for a sequence_mode stream) — one
                 entry per registered output stream (spec §11bis).
        """
        B = kb_tokens.shape[0]
        device = kb_tokens.device

        leaf_emb = self.embed(kb_tokens)
        self.memory.build(leaf_emb, kb_source_ids, leaf_mask=kb_leaf_mask)

        q_emb = self.embed(query_tokens).mean(dim=1, keepdim=True)  # (B, 1, d)
        register_base = (
            register_init_override if register_init_override is not None
            else self.register_init.unsqueeze(0).expand(B, -1, -1)
        )
        R = register_base + q_emb

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

        # spec §14.3: sequence_mode streams need teacher-forced target-token
        # embeddings as their per-position query input; embedded once here
        # (shared self.embed) and reused by every such stream, rather than
        # each stream re-embedding target_input independently.
        needs_target_embed = any(stream.sequence_mode for stream in self.streams.values())
        assert not needs_target_embed or target_input is not None, (
            "target_input is required when at least one stream has sequence_mode=True (spec §14.3)"
        )
        target_embed = self.embed(target_input) if needs_target_embed else None

        stream_outputs = {
            name: (stream(sm_k, sm_v, query_input=target_embed) if stream.sequence_mode else stream(sm_k, sm_v))
            for name, stream in self.streams.items()
        }
        return R, stream_outputs
