"""
Minimal end-to-end model wiring HierarchicalMemory (core/indexed_memory.py) into
the "Indexed Attention" main loop described in dev_notes/indexed_attention_spec.md §2:

    R_t = R_{t-1} + MLP([O_kb_t ; O_sm_t ; R_{t-1}])
    O_kb_t = HierarchicalMemory.attend(R_{t-1})          (input ∪ KB, hierarchical)
    O_sm_t = Attn(Q_sm(R_{t-1}), K^s, V^s)                (short-term memory, flat)
    new_K, new_V = split(W_sm(R_t))  -> appended to SM     (no stop-gradient, spec §4.1 reading B)

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
width, decoupled Q_KB vs Q_SM projections, stochastic level dropping, SM
capacity eviction beyond an optional hard cap, embedding-valued "thinking"
streams (need a real Teacher, not available for the synthetic kb_retrieval task).
"""

import torch
from torch import nn
import torch.nn.functional as F

from core.indexed_memory import HierarchicalMemory
from core.layers import RMSNorm


class OutputStream(nn.Module):
    """
    One independent, lightweight output stream (spec §11bis): its own query
    projection and its own head, both disjoint from any other stream's weights
    and from the core recurrent loop. Reads the *entire* accumulated SM
    trajectory via cross-attention, not just the final register state — the
    stream's own attention learns which recurrent steps matter for its task,
    rather than assuming the core's iteration count lines up with anything.
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.query_seed = nn.Parameter(torch.randn(1, d_model) * d_model ** -0.5)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, sm_k: torch.Tensor, sm_v: torch.Tensor) -> torch.Tensor:
        B = sm_k.shape[0]
        q = self.q_proj(self.query_seed.unsqueeze(0).expand(B, -1, -1))
        out = F.scaled_dot_product_attention(q, sm_k, sm_v)
        return self.head(out)


class IndexedThinker(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_register: int,
                 block_size: int, depth: int, n_slots: int = 1, n_head: int = 1,
                 d_hid: int = None, sm_cap: int = None, stream_dims: dict = None):
        super().__init__()
        d_hid = d_hid or 4 * d_model
        self.d_model = d_model
        self.n_register = n_register
        self.sm_cap = sm_cap

        self.embed = nn.Embedding(vocab_size, d_model)
        self.register_init = nn.Parameter(torch.randn(n_register, d_model) * d_model ** -0.5)

        self.memory = HierarchicalMemory(d_model, block_size, depth, n_slots=n_slots, n_head=n_head)

        self.sm_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.sm_write_proj = nn.Linear(d_model, 2 * d_model, bias=False)  # -> new K, new V

        self.fuse_norm = RMSNorm(3 * d_model)
        self.fuse_in = nn.Linear(3 * d_model, d_hid)
        self.fuse_out = nn.Linear(d_hid, d_model)

        stream_dims = stream_dims if stream_dims is not None else {'answer': vocab_size}
        self.streams = nn.ModuleDict({
            name: OutputStream(d_model, dim) for name, dim in stream_dims.items()
        })

    def forward(self, kb_tokens: torch.Tensor, kb_source_ids: torch.Tensor,
                query_tokens: torch.Tensor, n_step: int):
        """
        kb_tokens: (B, N) leaf token ids for the unified input∪KB sequence
            (N must equal block_size ** depth when depth > 0).
        kb_source_ids: (B, N) in {0, 1} (0 = input, 1 = KB).
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
        self.memory.build(leaf_emb, kb_source_ids)

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
            delta = self.fuse_out(F.gelu(self.fuse_in(self.fuse_norm(fused))))
            R = R + delta

            new_k, new_v = self.sm_write_proj(R).chunk(2, dim=-1)
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)
            if self.sm_cap is not None and sm_k.shape[1] > self.sm_cap:
                sm_k = sm_k[:, -self.sm_cap:]
                sm_v = sm_v[:, -self.sm_cap:]

        stream_outputs = {name: stream(sm_k, sm_v) for name, stream in self.streams.items()}
        return R, stream_outputs
