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

`stream_vocab_sizes` (spec §11ter, 2026-09-20): an output stream can target a
DIFFERENT tokenizer than the core's own `vocab_size` -- its teacher-forced
target tokens are embedded via a dedicated `self.stream_embed[name]` table
instead of the shared `self.embed`. Lets several streams each decode into
(and receive KD from) a different Teacher family's native vocabulary at
once, without the KD tokenizer-matching constraint this project has had to
work around so far (core/model_families.py). Streams absent from this dict
are unaffected.

`tie_stream_embed`/`embed_init`/`stream_embed_init`/`stream_head_init`
(spec §13.1/13.2, 2026-09-20): a large vocab's embedding+head are the
biggest parameter block by far (§13). `tie_stream_embed` shares a stream's
head weight with its own input embedding table (self.embed or
stream_embed[name]) instead of a second independent matrix -- halves that
block at zero behavior cost beyond the tied gradient. The `*_init` args let
a caller (offline, from an actual Teacher checkpoint -- not this module's
job to load one) seed these tables from a projected Teacher matrix instead
of random init; `freeze_embed`/`freeze_stream_embed` then stop the
optimizer from tracking state for them (does not reduce forward/backward
FLOPs through them, only optimizer memory/update cost).
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
                 sequence_mode: bool = False, max_seq_len: int = None,
                 per_position_head: bool = False):
        """
        per_position_head (2026-09-23, supervisor-agent request, generation-collapse
        investigation): the single shared `self.head` nn.Linear is applied identically
        at every output position -- a literal per-position UNSHARED head (a separate
        (d_model, vocab) matrix per position) would multiply the head's ~63.7M params
        (at this project's 248k vocab) by max_seq_len (e.g. 64x -> ~4B), infeasible for
        a diagnostic run. Cheaper proxy tested instead: a per-position (d_model,
        d_model) linear transform applied to the hidden state right before the
        (still-shared) vocab head -- adds position-specific capacity without touching
        the vocab-sized parameter block. Tests the same qualitative hypothesis
        ("does forcing every position through one identical transform limit
        calibration") at a tractable parameter cost (max_seq_len * d_model^2, a few
        million params, not billions).
        """
        super().__init__()
        assert 1 <= n_layers <= 3, "output streams are meant to stay lightweight (1-3 layers)"
        self.sequence_mode = sequence_mode
        self.per_position_head = per_position_head
        if sequence_mode:
            assert max_seq_len is not None and max_seq_len > 0, (
                "sequence_mode requires max_seq_len (spec §14.3, an upper bound on T_tgt "
                "for the learned per-position embedding table)"
            )
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
        else:
            self.query_seed = nn.Parameter(torch.randn(1, d_model) * d_model ** -0.5)
        if per_position_head:
            assert sequence_mode, "per_position_head only makes sense in sequence_mode"
            self.pos_head_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(max_seq_len)])
        self.layers = nn.ModuleList([OutputStreamLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, sm_k: torch.Tensor, sm_v: torch.Tensor,
                query_input: torch.Tensor = None, return_hidden: bool = False) -> torch.Tensor:
        """
        query_input (sequence_mode only): (B, T, d_model) teacher-forced
        target-token embeddings (spec §14.3's q_t = embed(target_token_{t-1}),
        computed by the caller since only Thinker owns `self.embed`) -- this
        method adds the learned position embedding on top.

        return_hidden=True returns the pre-head hidden state (B, T, d_model)
        instead of `self.head(x)` -- lets a caller run the head itself
        through a chunked loss (spec §13.3, learn/distill/chunked_loss.py)
        instead of materializing the full (B, T, vocab) logits tensor here.
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
        if self.per_position_head:
            T = x.shape[1]
            x = torch.stack([self.pos_head_proj[t](x[:, t, :]) for t in range(T)], dim=1)
        return x if return_hidden else self.head(x)


class Thinker(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_register: int,
                 block_size: int, depth: int, n_slots: int = 1, n_head: int = 1,
                 sm_cap: int = None, stream_dims: dict = None,
                 stream_n_layers: dict = None, level_dropout_p: float = 0.0,
                 detach_sm_keys: bool = False, use_ff: bool = False, ff_hidden_mult: int = 4,
                 decouple_kv: bool = True, pool_n_head: int = 1, k_dim: int = None,
                 disable_kb: bool = False, disable_sm: bool = False, outer_norm: bool = False,
                 outer_norm_type: str = "rmsnorm",
                 stream_sequence: dict = None, max_target_len: int = None,
                 stream_vocab_sizes: dict = None, use_ingest_token: bool = False,
                 stream_head_per_position: dict = None,
                 tie_stream_embed: set = None, embed_init: torch.Tensor = None,
                 freeze_embed: bool = False, stream_embed_init: dict = None,
                 freeze_stream_embed: set = None, stream_head_init: dict = None):
        super().__init__()
        # spec §8ter: a dedicated ingestion-marker embedding, id reserved just
        # past the tokenizer's own vocab_size (same pattern as the synthetic
        # tasks' KEY_MARK/VAL_MARK, data/kb_retrieval.py) rather than a
        # tokenizer.add_special_tokens() call -- no precedent for the latter
        # in this repo, and it would require resizing/retraining the
        # tokenizer's own embedding. Purely additive: vocab_size grows by 1
        # only when use_ingest_token=True, every existing caller unaffected.
        self.use_ingest_token = use_ingest_token
        if use_ingest_token:
            self.ingest_token_id = vocab_size
            vocab_size = vocab_size + 1
        self.d_model = d_model
        self.n_register = n_register
        self.sm_cap = sm_cap
        self.detach_sm_keys = detach_sm_keys
        self.use_ff = use_ff
        # spec §9 Baseline B ("boucle pure, sans mémoire"): the recurrent
        # register still loops n_step times, but external-memory (KB) access
        # is disabled -- isolates whether any gain comes from the loop itself
        # or from the memory. [DEFAUT] scope choice, 2026-09-14: disables
        # only the KB term (o_kb), not the SM (o_sm) -- the SM is populated
        # from the loop's own trajectory, not an external source, so it reads
        # as part of "the loop mechanism" rather than "the memory" per the
        # spec's own framing of this baseline's question.
        self.disable_kb = disable_kb
        # 2026-09-14: sm_cap=1 vs sm_cap=None came back indistinguishable on
        # n_hops=2 (mirrors the toy-memory n_memory=1 vs n_memory=10000
        # finding) -- multi-slot SM accumulation isn't adding anything
        # detectable here. This flag tests the more radical version of the
        # same question: does the SM mechanism (write+read at all) add
        # anything beyond the recurrent register R itself, which already
        # persists/accumulates via R = R + delta every step? Skips both the
        # write (sm_write_proj) and the read (o_sm) entirely when True --
        # a stronger test than sm_cap=1, which still writes+reads one slot.
        self.disable_sm = disable_sm

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_init = nn.Parameter(torch.randn(n_register, d_model) * d_model ** -0.5)

        # pool_n_head/k_dim: see HierarchicalMemory/LevelCompressor docstrings
        # (spec §5.1bis, §5.4, 2026-09-14) -- both default to prior behavior.
        self.memory = HierarchicalMemory(d_model, block_size, depth, n_slots=n_slots, n_head=n_head,
                                          pool_n_head=pool_n_head, k_dim=k_dim,
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
        # 2026-09-23, supervisor-agent request (thesis/paper/WRITING_PLAN.md §9 E13) --
        # see _step's docstring for the full rationale. None (default) = old behavior,
        # unchanged for every existing checkpoint.
        # outer_norm_type (2026-09-23, supervisor-agent, 2nd variant in parallel with
        # the RMSNorm default): "layernorm" tests whether mean-centering (not just
        # rescaling) matters for stabilizing R across iterations, vs RMSNorm which
        # only rescales.
        if not outer_norm:
            self.outer_norm = None
        elif outer_norm_type == "layernorm":
            self.outer_norm = nn.LayerNorm(d_model)
        else:
            assert outer_norm_type == "rmsnorm", f"unknown outer_norm_type {outer_norm_type!r}"
            self.outer_norm = RMSNorm(d_model)
        if use_ff:
            ff_hidden = d_model * ff_hidden_mult
            self.fuse_in = nn.Linear(3 * d_model, ff_hidden)
            self.fuse_out = nn.Linear(ff_hidden, d_model)
        else:
            self.fuse_proj = nn.Linear(3 * d_model, d_model, bias=False)

        stream_dims = stream_dims if stream_dims is not None else {'answer': vocab_size}
        stream_n_layers = stream_n_layers or {}
        # spec §11ter: a stream targeting a DIFFERENT tokenizer (cross-model
        # KD without a shared-tokenizer constraint) needs its teacher-forced
        # target tokens embedded in that tokenizer's own id space, not
        # self.embed (sized to this Thinker's own vocab_size, §14.3's core
        # KB/input embedding table). Streams absent from this dict keep using
        # self.embed exactly as before -- purely additive, no behavior change
        # when unset.
        stream_vocab_sizes = stream_vocab_sizes or {}
        self.stream_embed = nn.ModuleDict({
            name: nn.Embedding(size, d_model) for name, size in stream_vocab_sizes.items()
        })
        # spec §14.3 (plan Phase 11): per-stream opt-in to the multi-position
        # generalization of OutputStream, for a per-token LM objective on real
        # text -- e.g. stream_sequence={'answer': True} while a 'thinking'
        # stream (if any) stays in the default single-query mode. Unset by
        # default so every existing single-query/single-answer usage
        # (data/kb_retrieval.py, data/kb_chain_retrieval.py) is unaffected.
        stream_sequence = stream_sequence or {}
        stream_head_per_position = stream_head_per_position or {}
        self.streams = nn.ModuleDict({
            name: OutputStream(
                d_model, dim, n_layers=stream_n_layers.get(name, 1),
                sequence_mode=stream_sequence.get(name, False),
                max_seq_len=max_target_len if stream_sequence.get(name, False) else None,
                per_position_head=stream_head_per_position.get(name, False),
            )
            for name, dim in stream_dims.items()
        })

        # spec §13.2: a stream's head and its input embedding table are the
        # same shape ((vocab, d_model)) whenever the stream's out_dim equals
        # that table's vocab size -- tying them (sharing the same Parameter,
        # not just copying values) roughly halves the single biggest
        # parameter block at a large vocab size, standard LM practice. Named
        # per-stream (not a single tie_embed_head bool) because §11ter's
        # per-stream vocabularies mean there can be several independent
        # (embed, head) pairs to tie, not just one.
        tie_stream_embed = tie_stream_embed or set()
        for name in tie_stream_embed:
            assert name in self.streams, f"tie_stream_embed: unknown stream {name!r}"
            table = self.stream_embed[name] if name in self.stream_embed else self.embed
            head = self.streams[name].head
            assert head.weight.shape == table.weight.shape, (
                f"tie_stream_embed[{name!r}]: stream_dims[{name!r}]={head.weight.shape[0]} must equal "
                f"the embedding table's vocab size ({table.weight.shape[0]}) to tie weights"
            )
            head.weight = table.weight

        # spec §13.1/13.2: initialize self.embed/stream_embed/a stream's head
        # from an already-projected Teacher matrix instead of random init --
        # the projection itself (Teacher hidden_size -> d_model, random or
        # SVD, spec §13.1) is the CALLER's job (needs the actual Teacher
        # checkpoint, offline, not something Thinker itself should load).
        # freeze_embed/freeze_stream_embed matter for optimizer state size,
        # not FLOPs (the forward/backward matmul through a frozen table still
        # happens) -- per §13.1, freezing the head is deliberately not
        # offered: it must keep adapting to this model's own internal state,
        # unlike the input embedding's largely architecture-independent
        # token-to-representation role.
        if embed_init is not None:
            assert embed_init.shape == self.embed.weight.shape, (
                f"embed_init shape {tuple(embed_init.shape)} != self.embed.weight shape "
                f"{tuple(self.embed.weight.shape)}"
            )
            with torch.no_grad():
                self.embed.weight.copy_(embed_init)
            self.embed.weight.requires_grad = not freeze_embed
        if freeze_embed:
            tied_to_embed = [name for name in tie_stream_embed if name not in self.stream_embed]
            assert not tied_to_embed, (
                f"freeze_embed=True would also freeze {tied_to_embed}'s head (tie_stream_embed shares "
                f"the same Parameter) -- spec §13.1 deliberately recommends against freezing a stream's "
                f"head, only its input embedding; untie {tied_to_embed} from self.embed if this is intended"
            )

        # (tying + stream_embed_init on the same name is fine and expected --
        # it initializes the shared table both the embedding and, since
        # tie_stream_embed makes them the same Parameter, the head read from
        # for free. Only tying + stream_head_init together is a real
        # conflict, guarded below: that would ambiguously specify two
        # different init tensors for what is now a single shared weight.)
        stream_embed_init = stream_embed_init or {}
        freeze_stream_embed = freeze_stream_embed or set()
        for name, init in stream_embed_init.items():
            table = self.stream_embed[name]
            assert init.shape == table.weight.shape, (
                f"stream_embed_init[{name!r}] shape {tuple(init.shape)} != {tuple(table.weight.shape)}"
            )
            with torch.no_grad():
                table.weight.copy_(init)
            table.weight.requires_grad = name not in freeze_stream_embed
        tied_stream_embed_frozen = [name for name in tie_stream_embed
                                     if name in self.stream_embed and name in freeze_stream_embed]
        assert not tied_stream_embed_frozen, (
            f"freeze_stream_embed would also freeze {tied_stream_embed_frozen}'s head (tied via "
            f"tie_stream_embed) -- spec §13.1 recommends against freezing a stream's head"
        )

        stream_head_init = stream_head_init or {}
        for name, init in stream_head_init.items():
            assert name not in tie_stream_embed, (
                f"stream_head_init[{name!r}]: this stream's head is tied to its embedding, "
                f"initializing it here would silently also overwrite the embedding weight"
            )
            head = self.streams[name].head
            assert init.shape == head.weight.shape, (
                f"stream_head_init[{name!r}] shape {tuple(init.shape)} != {tuple(head.weight.shape)}"
            )
            with torch.no_grad():
                head.weight.copy_(init)

    def _step(self, R: torch.Tensor, sm_k: torch.Tensor, sm_v: torch.Tensor, external_kb: tuple = None):
        """
        One iteration of the core recurrent loop (spec §2), factored out of
        forward() so ingest() (spec §8ter) can reuse the exact same weights
        (memory.attend/attend_static, sm_q_proj, fuse_*, sm_write_proj)
        instead of duplicating the loop body.

        external_kb: None (default, forward()'s normal reasoning path) reads
            self.memory.attend(R) as before. A (K, V, mask) tuple (ingest()'s
            path) instead calls self.memory.attend_static(R, K, V, mask) --
            a STATELESS read of exactly those tensors, not of self.memory's
            current global state. This distinction is what makes ingest()
            safe to wrap in torch.utils.checkpoint.checkpoint() (see
            HierarchicalMemory.attend_static's docstring for the bug this
            avoids: self.memory can be mutated between checkpoint's forward
            and its backward-time recompute, but explicit tensor arguments
            can't).
        """
        if self.disable_kb:
            o_kb = torch.zeros_like(R)
        elif external_kb is not None:
            o_kb = self.memory.attend_static(R, *external_kb)
        else:
            o_kb = self.memory.attend(R)

        if not self.disable_sm and sm_k.shape[1] > 0:
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
        if self.outer_norm is not None:
            # 2026-09-23, supervisor-agent request (thesis/paper/WRITING_PLAN.md §9 E13,
            # citing Labovich "Stability and Generalization in Looped Transformers"):
            # `fuse_norm` above only normalizes the INPUT to the delta computation
            # (pre-norm style) -- R itself is a raw, unnormalized residual accumulation
            # across all n_step iterations of a WEIGHT-SHARED loop, unlike a normal deep
            # transformer where each layer has distinct weights and pre-norm alone is
            # enough. Nothing bounds ||R|| as n_step grows. This "outer" norm (applied to
            # R itself, at the end of each iteration, not just to a sub-block's input) is
            # the candidate stabilizer the paper argues is necessary for looped/recurrent
            # transformers specifically. Opt-in (existing checkpoints have no such
            # weights) -- see --outer_norm in train_prompt_response.py.
            R = self.outer_norm(R)

        if not self.disable_sm:
            new_k, new_v = self.sm_write_proj(R).chunk(2, dim=-1)
            if self.detach_sm_keys:
                new_k = new_k.detach()
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)
            if self.sm_cap is not None and sm_k.shape[1] > self.sm_cap:
                sm_k = sm_k[:, -self.sm_cap:]
                sm_v = sm_v[:, -self.sm_cap:]

        return R, sm_k, sm_v

    def _ingest_impl(self, doc_tokens: torch.Tensor, n_step: int,
                      prior_k: torch.Tensor, prior_v: torch.Tensor, prior_mask: torch.Tensor) -> tuple:
        """
        Pure function of its explicit arguments only (no read of
        self.memory) -- see ingest()'s docstring for why this matters. Split
        out from ingest() only so torch.utils.checkpoint.checkpoint() has a
        plain function to wrap (checkpoint needs positional tensor args, not
        a method carrying extra non-tensor bookkeeping).
        """
        B, L = doc_tokens.shape
        device = doc_tokens.device
        marker = torch.full((B, 1), self.ingest_token_id, dtype=doc_tokens.dtype, device=device)
        tokens_with_marker = torch.cat([marker, doc_tokens], dim=1)
        q_emb = self.embed(tokens_with_marker).mean(dim=1, keepdim=True)  # (B, 1, d)
        R = self.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb

        sm_k = torch.zeros(B, 0, self.d_model, device=device, dtype=R.dtype)
        sm_v = torch.zeros(B, 0, self.d_model, device=device, dtype=R.dtype)
        external_kb = (prior_k, prior_v, prior_mask)
        for _ in range(n_step):
            R, sm_k, sm_v = self._step(R, sm_k, sm_v, external_kb=external_kb)
        return sm_k, sm_v

    def ingest(self, doc_tokens: torch.Tensor, n_step: int, prior_k: torch.Tensor = None,
               prior_v: torch.Tensor = None, prior_mask: torch.Tensor = None,
               use_checkpoint: bool = False) -> tuple:
        """
        Spec §8ter: runs a document through the SAME recurrent loop used for
        reasoning (self._step, shared weights, in particular sm_write_proj)
        instead of the direct k_proj/v_proj projection of §8 -- the
        (new_k, new_v) written to SM at each step become, instead, the
        document's own entry in the KB (injected by the caller via
        `self.memory.add_static_level(sm_k, sm_v, mask)`, not left in the
        ephemeral per-episode SM).

        Requires use_ingest_token=True (a dedicated marker embedding prepended
        to doc_tokens, spec §8ter point 1) -- assert below since ingest()
        without it would silently reuse a real vocabulary token's embedding
        as the ingestion marker, indistinguishable from actual document
        content to the model.

        Per the user's 2026-09-20 decision, KB access stays ON during
        ingestion -- but (2026-09-20, checkpoint-safety fix) this method is
        now a PURE function of its explicit tensor arguments, not of
        self.memory's current mutable state: `prior_k`/`prior_v`/`prior_mask`
        are the CALLER-supplied concatenation of whatever was ingested
        earlier (e.g. accumulated by learn/indexed_attention/train_prompt_
        response.py's ingest_documents() as it loops over a batch's
        documents in order), read via HierarchicalMemory.attend_static (a
        stateless sibling of attend()) instead of self.memory.attend(R).
        Omit all three (or pass None) for the first document -- equivalent
        to empty (B, 0, d) tensors, attend_static returns zeros. The caller
        is still responsible for eventually calling
        `self.memory.add_static_level(sm_k, sm_v, mask)` once ingestion of
        all documents is done, so the QA forward pass can read them via the
        normal (stateful) attend() path.

        use_checkpoint (2026-09-20, spec §8ter "grand volume" discussion,
        option A): wraps the actual computation in
        torch.utils.checkpoint.checkpoint(), trading recompute-on-backward
        for not keeping this document's ingestion activations resident for
        the rest of the batch's forward pass. Safe ONLY because this method
        no longer reads self.memory (see above) -- checkpoint's recompute
        during backward() would otherwise silently read whatever self.memory
        looks like AT THAT LATER TIME (more documents added, or cleared for
        the next batch), not what it looked like during the original
        forward, corrupting the gradient without any error or NaN to signal
        it. use_checkpoint=False (default) is the plain/no-recompute MVP
        path -- identical numerically, just keeps all activations.

        doc_tokens: (B, L) token ids of ONE document (no ingestion marker --
            added here).
        n_step: number of ingestion iterations (may differ from the QA
            n_step; the spec notes this could later control how much KB
            capacity a document ends up occupying, §8ter [OUVERT]).

        Returns (sm_k, sm_v): (B, n_step, k_dim)/(B, n_step, d_model) -- the
            full SM trajectory produced while ingesting this one document,
            meant to be added as one KB level via
            `self.memory.add_static_level(sm_k, sm_v)`.
        """
        assert self.use_ingest_token, (
            "Thinker.ingest requires use_ingest_token=True (spec §8ter point 1: "
            "a dedicated marker embedding, not a reused vocabulary token)"
        )
        B, device, dtype = doc_tokens.shape[0], doc_tokens.device, self.register_init.dtype
        if prior_k is None:
            # sm_write_proj (2 * d_model -> chunked) always produces d_model-wide
            # K and V, matching add_static_level's existing assumption that
            # ingested K/V share d_model regardless of HierarchicalMemory's own
            # k_dim (only meaningful for k_proj/v_proj-derived levels, §5.4).
            prior_k = torch.zeros(B, 0, self.d_model, device=device, dtype=dtype)
            prior_v = torch.zeros(B, 0, self.d_model, device=device, dtype=dtype)
            prior_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)

        if use_checkpoint:
            import torch.utils.checkpoint as ckpt
            return ckpt.checkpoint(self._ingest_impl, doc_tokens, n_step, prior_k, prior_v, prior_mask,
                                    use_reentrant=False)
        return self._ingest_impl(doc_tokens, n_step, prior_k, prior_v, prior_mask)

    def forward(self, kb_tokens: torch.Tensor, kb_source_ids: torch.Tensor,
                query_tokens: torch.Tensor, n_step: int, kb_leaf_mask: torch.Tensor = None,
                register_init_override: torch.Tensor = None, target_input: torch.Tensor = None,
                kb_prebuilt: bool = False, return_hidden: bool = False,
                residual_tokens: torch.Tensor = None):
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
        target_input: optional (B, T_tgt) tensor, OR a dict {stream_name: (B, T_name)}
            — spec §14.3 (plan Phase 11): teacher-forced target-token ids
            (`target_input[t]` is the token that should be embedded and fed as
            the query predicting `labels[t]`, see data/real_text_windows.py).
            Required exactly when at least one registered stream has
            `sequence_mode=True`; embedded here (via `self.embed`, the same
            table as the KB/input leaves) and passed to those streams as
            their per-position query input, since only `Thinker` owns
            `self.embed` (OutputStream itself only knows how to add position
            information on top, see OutputStream's docstring).

            A single tensor is embedded ONCE and shared by every sequence_mode
            stream -- the original real-text use case (one continuous target
            span, spec §14.1-14.3), unchanged. A dict lets DIFFERENT streams
            read DIFFERENT target sequences of DIFFERENT lengths (2026-09-20,
            prompt/response/thinking data: the `thinking` stream's target is a
            reasoning trace, the `answer` stream's target is the final answer
            -- unrelated text, can't share one embedding). Every sequence_mode
            stream must have a matching key in the dict.
        kb_prebuilt: spec §8ter -- when True, skips the internal
            `self.memory.build(...)` call and uses whatever levels the caller
            already populated via `self.memory.add_static_level(...)` (one
            call per ingested document, using `self.ingest(...)`'s output),
            instead of deriving K/V from `kb_tokens` via k_proj/v_proj.
        return_hidden: spec §13.3 -- when True, every stream in the returned
            dict yields its pre-head hidden state (B, T, d_model) instead of
            logits (B, T, out_dim), letting the caller run the head itself
            through a chunked loss (learn/distill/chunked_loss.py) instead of
            materializing the full (B, T, vocab) logits tensor here.
        residual_tokens: optional (B, Tres) -- X1_DISPATCH.md X2(c) "lecture
            depuis un résiduel non récurrent": raw input token ids, embedded
            here and appended to sm_k/sm_v as extra keys/values AFTER the
            n_step loop, so every output stream's cross-attention has direct
            access to the untouched input embedding regardless of what the
            recurrent core computed -- a hard non-recurrent bypass, distinct
            from sm_k/sm_v's existing per-step trajectory (which is already a
            "read from all latents" of the recurrent state, not a residual
            around it) and from the KB mechanism (X2(b), disable_kb). Default
            None preserves prior behavior exactly (no other caller passes
            this yet).

        Returns: (R, stream_outputs) with R: (B, n_register, d_model) the final
                 core register state, and stream_outputs a dict {name: (B, 1, out_dim)}
                 (or (B, T_tgt, out_dim) for a sequence_mode stream) — one
                 entry per registered output stream (spec §11bis).
        """
        B = kb_tokens.shape[0]
        device = kb_tokens.device

        if not self.disable_kb and not kb_prebuilt:
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
            R, sm_k, sm_v = self._step(R, sm_k, sm_v)

        if residual_tokens is not None:
            res_emb = self.embed(residual_tokens)  # (B, Tres, d) -- untouched by the loop above
            sm_k = torch.cat([sm_k, res_emb], dim=1)
            sm_v = torch.cat([sm_v, res_emb], dim=1)

        # spec §14.3: sequence_mode streams need teacher-forced target-token
        # embeddings as their per-position query input; embedded here (shared
        # self.embed). A plain tensor is embedded once and shared by every
        # sequence_mode stream (original behavior); a dict lets each stream
        # embed its OWN target sequence (2026-09-20, prompt/thinking/answer
        # data -- see this method's docstring).
        needs_target_embed = any(stream.sequence_mode for stream in self.streams.values())
        assert not needs_target_embed or target_input is not None, (
            "target_input is required when at least one stream has sequence_mode=True (spec §14.3)"
        )
        if isinstance(target_input, dict):
            target_embed_by_stream = {
                name: (self.stream_embed[name](t) if name in self.stream_embed else self.embed(t))
                for name, t in target_input.items()
            }
            shared_target_embed = None
        else:
            assert not self.stream_embed or not any(
                self.streams[name].sequence_mode for name in self.stream_embed
            ), (
                "a stream with its own stream_vocab_sizes entry needs its own target_input "
                "tensor (dict target_input) -- a single shared tensor can't be valid ids in "
                "two different tokenizers' id spaces at once"
            )
            shared_target_embed = self.embed(target_input) if needs_target_embed else None
            target_embed_by_stream = None

        def stream_query_input(name):
            if target_embed_by_stream is not None:
                assert name in target_embed_by_stream, (
                    f"target_input dict is missing an entry for sequence_mode stream '{name}'"
                )
                return target_embed_by_stream[name]
            return shared_target_embed

        stream_outputs = {
            name: (
                stream(sm_k, sm_v, query_input=stream_query_input(name), return_hidden=return_hidden)
                if stream.sequence_mode else stream(sm_k, sm_v, return_hidden=return_hidden)
            )
            for name, stream in self.streams.items()
        }
        return R, stream_outputs
