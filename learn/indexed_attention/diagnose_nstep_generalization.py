"""
I7 (relayed 2026-09-19/20 night): does Thinker's loop generalize in N_step
without retraining, or has it learned a step-count-indexed stopping rule
(a "counter" strategy that would be functionally equivalent to a step
embedding, just derived from SM size instead of a parameter -- spec Sec.7.2bis's
distinction between "t knowable via state" (fine) and "t parameterized"
(the thing that must not happen))?

Two measurements, same eval pass, no training:
  (1) N_step generalization: evaluate an N_step=16-trained checkpoint at
      N_step_test in {2,4,6,8,12,16,20,24} -- zero retraining.
  (2) Read-step curve: a single long unrolled run, reading out task accuracy
      at EVERY intermediate step t=1..N (not just the final one) on the same
      eval batch -- does accuracy plateau (implicit content-based stop) or
      drift/degrade past the expected hop count (never learned to stop,
      just externally read at the right step by the harness)?

**Scope caveat, important**: Etape 4's actual hardened-generator checkpoints
(n_hops in {2,3,4}, d_model=256, GPU) were never saved to disk -- confirmed
2026-09-19, train_kb_chain.py never called torch.save until tonight's fix
(--save_checkpoint_path). This script instead reuses the CPU n_hops=2,
d_model=32 checkpoint already regenerated for I4/I6
(/tmp/diagnose_n_hops2_no_ff_checkpoint.pt) -- NOT the hardened, larger-scale
config the night's brief actually asked about. A proper re-run at Etape 4's
real scale is queued once GPU frees up; this is a partial, small-scale
stand-in to get a first signal without blocking on GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import learn.indexed_attention.diagnose_no_ff_composition as base

CHECKPOINT = "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt"
TRAINED_N_STEP = base.N_STEP  # 16


@torch.no_grad()
def eval_at_n_step(model, ds, n_step: int, n_eval: int) -> tuple[float, float]:
    """Full unroll for n_step iterations, task accuracy at the end + chance level."""
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    B = leaves.shape[0]
    leaf_emb = model.embed(leaves)
    model.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model.embed(query).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model.d_model)
    sm_v = torch.zeros(B, 0, model.d_model)

    for t in range(n_step):
        o_kb = model.memory.attend(R)
        o_sm = (F.scaled_dot_product_attention(model.sm_q_proj(R), sm_k, sm_v)
                if sm_k.shape[1] > 0 else torch.zeros_like(R))
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        R = R + model.fuse_proj(model.fuse_norm(fused))
        new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)

    stream_out = model.streams["answer"](sm_k, sm_v)[:, 0]
    acc = (stream_out.argmax(-1) == label).float().mean().item()
    return acc, label


@torch.no_grad()
def readstep_curve(model, ds, n_step_max: int, n_eval: int) -> list[float]:
    """One long unrolled run, accuracy readout AT EVERY step t=1..n_step_max
    (reads the SM-so-far as if training had stopped there)."""
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    B = leaves.shape[0]
    leaf_emb = model.embed(leaves)
    model.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model.embed(query).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model.d_model)
    sm_v = torch.zeros(B, 0, model.d_model)

    accs = []
    for t in range(n_step_max):
        o_kb = model.memory.attend(R)
        o_sm = (F.scaled_dot_product_attention(model.sm_q_proj(R), sm_k, sm_v)
                if sm_k.shape[1] > 0 else torch.zeros_like(R))
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        R = R + model.fuse_proj(model.fuse_norm(fused))
        new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)
        stream_out = model.streams["answer"](sm_k, sm_v)[:, 0]
        accs.append((stream_out.argmax(-1) == label).float().mean().item())
    return accs


def main() -> None:
    torch.manual_seed(1)
    ds, model = base.make_model(n_hops=2)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    n_eval = 512
    chance = 1.0 / (2 + base.N_DISTRACTORS)  # conditional chance, n_facts candidates

    print(f"=== I7a: N_step generalization (trained at N_step={TRAINED_N_STEP}) ===")
    print(f"chance_level={chance:.4f}")
    for n_step_test in [2, 4, 6, 8, 12, 16, 20, 24]:
        acc, _ = eval_at_n_step(model, ds, n_step_test, n_eval)
        flag = " <-- trained N_step" if n_step_test == TRAINED_N_STEP else ""
        print(f"N_step_test={n_step_test:3d}  acc={acc:.4f}{flag}")

    print(f"\n=== I7b: read-step accuracy curve (single unroll to N_step={TRAINED_N_STEP + 8}) ===")
    accs = readstep_curve(model, ds, TRAINED_N_STEP + 8, n_eval)
    for t, a in enumerate(accs, start=1):
        flag = " <-- trained N_step" if t == TRAINED_N_STEP else ""
        print(f"t={t:3d}  acc={a:.4f}{flag}")


if __name__ == "__main__":
    main()
