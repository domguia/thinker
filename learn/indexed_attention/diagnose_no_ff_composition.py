"""
Mechanistic diagnostic for the n_hops=2 composition plateau (~25-34%, see
dev_notes/indexed_attention_experiment_plan.md Phase 2), on the no-FF
Thinker specifically (user's explicit priority: study the no-FF failure
mode directly before further architectural variants).

CPU-only, small scale, no Grid'5000/experiment-manager involvement needed --
same "test locally first" philosophy already used for
tests/test_indexed_memory.py's overfit sanity checks.

Question: does the register R ever come to represent the chain's
INTERMEDIATE value (the answer to hop 1 / key of hop 2) at some point during
the n_step loop, for a 2-hop example? Three parts:

1. Train a small no-FF Thinker on KBChainDataset(n_hops=2) to its plateau.
2. Probe: at every iteration t, nearest-neighbor-decode R_t against the
   embedding table and check whether it matches the ground-truth
   intermediate value -- gives a probe-accuracy-per-step curve. Cross-checked
   against a positive control (n_hops=1, known to converge to ~98%, spec/plan
   "Overfit sanity check"/Phase 2) to validate the probe methodology itself
   before trusting a negative result on n_hops=2.
3. Intervention: at the step with peak probe accuracy, forcibly overwrite R
   with the correct intermediate value's embedding and let the loop
   continue -- if final accuracy recovers to ~100%, the bottleneck is
   specifically DERIVING the intermediate value (the linear fuse step can't
   extract-and-hold a retrieved value as a reusable key), not USING it once
   available (retrieval/readout of the second hop is fine).
"""
import time

import torch
import torch.nn.functional as F

from core.indexed_thinker_model import Thinker
from data.kb_chain_retrieval import KBChainDataset, KEY_MARK, VAL_MARK

torch.manual_seed(0)

VOCAB_SIZE = 32
N_DISTRACTORS = 2
D_MODEL = 32
N_REGISTER = 4
BLOCK_SIZE = 4
N_STEP = 16
BATCH_SIZE = 64
TRAIN_STEPS = 4000
LOG_EVERY = 500


def decode_mid_val(leaves, source_ids, query, n_facts):
    """Follows the chain ONE hop from `query`, per the KEY_MARK/VAL_MARK
    layout (data/kb_chain_retrieval.py), to recover the ground-truth
    intermediate value -- same decoding logic as
    tests/test_kb_chain_retrieval.py::_decode_facts, not stored by the
    dataset itself."""
    facts = {}
    for i in range(n_facts):
        assert leaves[i * 4 + 0].item() == KEY_MARK
        assert leaves[i * 4 + 2].item() == VAL_MARK
        facts[leaves[i * 4 + 1].item()] = leaves[i * 4 + 3].item()
    return torch.tensor([facts[q.item()] for q in query])


MAX_FACTS = N_DISTRACTORS + 2  # fixed leaf-shape across n_hops in {1, 2}: max_facts*4 == block_size**depth


def make_model(n_hops, n_distractors=N_DISTRACTORS, use_ff=False):
    ds = KBChainDataset(n_hops=n_hops, n_distractors=n_distractors, vocab_size=VOCAB_SIZE,
                         max_facts=MAX_FACTS, seed=0)
    n_facts = n_hops + n_distractors  # real facts (<= MAX_FACTS, the rest is padding)
    depth = 2
    assert ds.n_leaves == BLOCK_SIZE ** depth, f"n_leaves={ds.n_leaves} != block_size**depth, adjust MAX_FACTS"
    model = Thinker(
        vocab_size=ds.total_vocab_size, d_model=D_MODEL, n_register=N_REGISTER,
        block_size=BLOCK_SIZE, depth=depth, use_ff=use_ff,
        stream_dims={'answer': ds.total_vocab_size},
    )
    return ds, model


def train(ds, model, steps=TRAIN_STEPS, lr=3e-4, log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(1, steps + 1):
        leaves, source_ids, mask, query, label = ds.sample_batch(BATCH_SIZE)
        _, streams = model(leaves, source_ids, query, n_step=N_STEP, kb_leaf_mask=mask)
        logits = streams['answer'][:, 0]
        loss = F.cross_entropy(logits, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == 1:
            acc = (logits.argmax(-1) == label).float().mean().item()
            print(f"{log_prefix}step {step:5d}  loss {loss.item():.4f}  acc {acc:.3f}  ({time.time()-t0:.1f}s)")
    return model


@torch.no_grad()
def eval_accuracy(ds, model, n_eval=512):
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    _, streams = model(leaves, source_ids, query, n_step=N_STEP, kb_leaf_mask=mask)
    logits = streams['answer'][:, 0]
    return (logits.argmax(-1) == label).float().mean().item()


@torch.no_grad()
def probe_curve(ds, model, n_hops, n_eval=256):
    """Manually unrolls Thinker.forward's loop (same equations as
    core/indexed_thinker_model.py) to capture R at every iteration, then
    nearest-neighbor-decodes it against self.embed.weight. Returns
    (probe_acc_per_step: list[float], final_task_acc: float)."""
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    n_facts = n_hops + N_DISTRACTORS
    # decode_mid_val must run per-example (leaves/source_ids differ per row)
    mid_vals = torch.stack([
        decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(n_eval)
    ]).squeeze(-1)

    B = leaves.shape[0]
    device = leaves.device
    leaf_emb = model.embed(leaves)
    model.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model.embed(query).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model.d_model, device=device, dtype=R.dtype)
    sm_v = torch.zeros(B, 0, model.d_model, device=device, dtype=R.dtype)

    embed_table = model.embed.weight  # (vocab, d)
    probe_acc = []
    for t in range(N_STEP):
        # probe BEFORE this iteration's update: does current R already encode mid_val?
        r_mean = R.mean(dim=1)  # (B, d)
        sims = F.normalize(r_mean, dim=-1) @ F.normalize(embed_table, dim=-1).T  # (B, vocab)
        pred = sims.argmax(-1)
        probe_acc.append((pred == mid_vals).float().mean().item())

        o_kb = model.memory.attend(R)
        if sm_k.shape[1] > 0:
            q_sm = model.sm_q_proj(R)
            o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
        else:
            o_sm = torch.zeros_like(R)
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        if model.use_ff:
            delta = model.fuse_out(F.gelu(model.fuse_in(model.fuse_norm(fused))))
        else:
            delta = model.fuse_proj(model.fuse_norm(fused))
        R = R + delta
        new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)

    stream_out = model.streams['answer'](sm_k, sm_v)[:, 0]
    final_acc = (stream_out.argmax(-1) == label).float().mean().item()
    return probe_acc, final_acc


@torch.no_grad()
def intervention_at_step(ds, model, n_hops, force_step, n_eval=256, add_register_init=False):
    """Same unroll as probe_curve, but at `force_step`, R is overwritten with
    the GROUND-TRUTH intermediate value's embedding (broadcast to all
    n_register slots) before continuing -- a causal test: if final accuracy
    recovers, the bottleneck is deriving mid_val, not using it.

    add_register_init=True (correction after the first run's near-null
    result): the naive R := embed(mid_val) may be out-of-distribution for
    q_proj/retrieval, since the register is NEVER naturally seeded that way
    -- at t=0 it's always register_init + mean(query embedding), so a
    "resolved hop" plausibly lives in that same register_init-shifted
    subspace, not in raw embedding space. This variant forces
    R := register_init + embed(mid_val) instead, matching that natural
    distribution more closely."""
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    n_facts = n_hops + N_DISTRACTORS
    mid_vals = torch.stack([
        decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(n_eval)
    ]).squeeze(-1)

    B = leaves.shape[0]
    device = leaves.device
    leaf_emb = model.embed(leaves)
    model.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model.embed(query).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model.d_model, device=device, dtype=R.dtype)
    sm_v = torch.zeros(B, 0, model.d_model, device=device, dtype=R.dtype)

    mid_emb = model.embed(mid_vals).unsqueeze(1).expand(-1, model.n_register, -1)  # (B, n_register, d)
    if add_register_init:
        mid_emb = mid_emb + model.register_init.unsqueeze(0).expand(B, -1, -1)

    for t in range(N_STEP):
        if t == force_step:
            R = mid_emb.clone()
        o_kb = model.memory.attend(R)
        if sm_k.shape[1] > 0:
            q_sm = model.sm_q_proj(R)
            o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
        else:
            o_sm = torch.zeros_like(R)
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        if model.use_ff:
            delta = model.fuse_out(F.gelu(model.fuse_in(model.fuse_norm(fused))))
        else:
            delta = model.fuse_proj(model.fuse_norm(fused))
        R = R + delta
        new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)

    stream_out = model.streams['answer'](sm_k, sm_v)[:, 0]
    return (stream_out.argmax(-1) == label).float().mean().item()


def main():
    print("=== Positive control: n_hops=1 (known to converge ~98%) ===")
    ds1, model1 = make_model(n_hops=1)
    train(ds1, model1, log_prefix="[n_hops=1] ")
    acc1 = eval_accuracy(ds1, model1)
    print(f"[n_hops=1] final eval acc: {acc1:.3f}")
    probe1, final1 = probe_curve(ds1, model1, n_hops=1)
    print(f"[n_hops=1] probe acc per step: {[round(a, 2) for a in probe1]}")
    print(f"[n_hops=1] probe-consistent final acc: {final1:.3f}")

    print("\n=== Target: n_hops=2 (the plateau under investigation) ===")
    ds2, model2 = make_model(n_hops=2)
    train(ds2, model2, log_prefix="[n_hops=2] ")
    acc2 = eval_accuracy(ds2, model2)
    print(f"[n_hops=2] final eval acc: {acc2:.3f}")
    probe2, final2 = probe_curve(ds2, model2, n_hops=2)
    print(f"[n_hops=2] probe acc per step: {[round(a, 2) for a in probe2]}")
    print(f"[n_hops=2] probe-consistent final acc: {final2:.3f}")

    torch.save(model2.state_dict(), "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt")
    print("\n[n_hops=2] checkpoint saved to /tmp/diagnose_n_hops2_no_ff_checkpoint.pt")

    best_step = max(range(N_STEP), key=lambda t: probe2[t])
    print(f"[n_hops=2] peak probe accuracy at step {best_step}: {probe2[best_step]:.3f}")

    print("=== Intervention v1 (naive): force R := embed(mid_val) ===")
    for step in (0, best_step, N_STEP // 2, N_STEP - 1):
        acc = intervention_at_step(ds2, model2, n_hops=2, force_step=step, add_register_init=False)
        print(f"[n_hops=2] v1 force_step={step:2d}  post-intervention final acc: {acc:.3f}")

    print("=== Intervention v2 (corrected): force R := register_init + embed(mid_val) ===")
    for step in (0, best_step, N_STEP // 2, N_STEP - 1):
        acc = intervention_at_step(ds2, model2, n_hops=2, force_step=step, add_register_init=True)
        print(f"[n_hops=2] v2 force_step={step:2d}  post-intervention final acc: {acc:.3f}")


if __name__ == "__main__":
    main()
