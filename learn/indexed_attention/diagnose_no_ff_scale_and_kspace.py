"""
Second follow-up to diagnose_no_ff_composition.py's causal intervention
result (see dev_notes/indexed_attention_experiment_plan.md Phase 2).

Two things investigated here, both using the checkpoint saved by
diagnose_no_ff_composition.py's rerun
(/tmp/diagnose_n_hops2_no_ff_checkpoint.pt -- run that script first if
missing):

1. R's norm across a natural (unforced) unroll -- discovered to grow
   unboundedly (~4-5x from t=0 to t=15, no normalization is ever applied to
   R itself, only to the concatenation fed into fuse_proj). This raised a
   confound: the original intervention (replace R with register_init +
   embed(mid_val), a small/t=0-scale vector) might have failed simply
   because it injected a value at the WRONG SCALE for a mid-trajectory step,
   not because retrieval is actually broken. Tested here via two corrected
   ADDITIVE interventions (add on top of the natural trajectory instead of
   replacing it -- raw, and rescaled to match R's current norm) -- result:
   still no recovery, ruling out scale mismatch as the explanation.

2. A K-space (not V-space) soft-rank test: cosine similarity of
   `q_proj(R)` against the actual (level-normed) K vectors
   HierarchicalMemory stored for the episode's own candidate facts, mirroring
   diagnose_no_ff_soft_retrieval.py's V-space test but on the query/key
   matching side. Result: BELOW-chance similarity for the correct key at
   both hops (not just noise/at-chance) -- suggestive of the q_proj/k_proj
   matching having collapsed to some generic pattern rather than simply
   lacking capacity, a hypothesis not yet investigated further.
"""
import torch
import torch.nn.functional as F

import learn.indexed_attention.diagnose_no_ff_composition as base

CHECKPOINT = "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt"


def kspace_rank_test(mem, R, k_leaf_normed, leaves, n_facts, n_eval, target_key_fn, label_text):
    q = mem.q_proj(R).mean(dim=1)  # (B, d); register rows are identical when R is uniformly broadcast
    correct_sim, ranks = [], []
    for b in range(n_eval):
        key_positions = [i * 4 + 1 for i in range(n_facts)]
        cand_k = k_leaf_normed[b, key_positions]
        sims = F.cosine_similarity(q[b].unsqueeze(0), cand_k, dim=-1)
        target_key = target_key_fn(b)
        correct_idx = [i for i, pos in enumerate(key_positions) if leaves[b, pos].item() == target_key][0]
        correct_sim.append(sims[correct_idx].item())
        order = sims.argsort(descending=True).tolist()
        ranks.append(order.index(correct_idx))
    correct_sim, ranks = torch.tensor(correct_sim), torch.tensor(ranks)
    print(f"=== {label_text} ===")
    print(f"mean cos_sim(q_proj(R), correct key's true K vector): {correct_sim.mean().item():.4f}")
    print(f"top-1 rate among {n_facts}: {(ranks == 0).float().mean().item():.3f}  (chance={1 / n_facts:.3f})")
    print(f"mean rank: {ranks.float().mean().item():.3f}  (chance={(n_facts - 1) / 2:.2f})")


@torch.no_grad()
def norm_matched_intervention(ds2, model2, n_facts, force_step, mode, n_eval=256):
    leaves, source_ids, mask, query, label = ds2.sample_batch(n_eval)
    mid_vals = torch.stack([
        base.decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(n_eval)
    ]).squeeze(-1)
    B = leaves.shape[0]
    leaf_emb = model2.embed(leaves)
    model2.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model2.embed(query).mean(dim=1, keepdim=True)
    R = model2.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model2.d_model)
    sm_v = torch.zeros(B, 0, model2.d_model)
    mid_emb = model2.embed(mid_vals).unsqueeze(1).expand(-1, model2.n_register, -1)

    for t in range(base.N_STEP):
        if t == force_step:
            if mode == "replace":
                R = mid_emb + model2.register_init.unsqueeze(0).expand(B, -1, -1)
            elif mode == "additive_raw":
                R = R + mid_emb
            elif mode == "additive_norm_matched":
                r_norm = R.norm(dim=-1, keepdim=True)
                e_norm = mid_emb.norm(dim=-1, keepdim=True)
                R = R + mid_emb * (r_norm / e_norm)
        o_kb = model2.memory.attend(R)
        if sm_k.shape[1] > 0:
            q_sm = model2.sm_q_proj(R)
            o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
        else:
            o_sm = torch.zeros_like(R)
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        delta = model2.fuse_proj(model2.fuse_norm(fused))
        R = R + delta
        new_k, new_v = model2.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)

    stream_out = model2.streams['answer'](sm_k, sm_v)[:, 0]
    return (stream_out.argmax(-1) == label).float().mean().item()


def main():
    torch.manual_seed(1)
    ds2, model2 = base.make_model(n_hops=2)
    model2.load_state_dict(torch.load(CHECKPOINT))
    model2.eval()
    n_facts = 2 + base.N_DISTRACTORS
    n_eval = 256

    print("=== R norm growth across a natural (unforced) unroll ===")
    leaves, source_ids, mask, query, label = ds2.sample_batch(n_eval)
    with torch.no_grad():
        B = leaves.shape[0]
        leaf_emb = model2.embed(leaves)
        model2.memory.build(leaf_emb, source_ids, leaf_mask=mask)
        mem = model2.memory
        k_leaf_normed = mem.level_norms[0](mem._levels_k[0])

        q_emb = model2.embed(query).mean(dim=1, keepdim=True)
        R0 = model2.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
        kspace_rank_test(mem, R0, k_leaf_normed, leaves, n_facts, n_eval,
                          lambda b: query[b, 0].item(), "hop1 K-space (natural R0)")

        mid_vals = torch.stack([
            base.decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(n_eval)
        ]).squeeze(-1)
        mid_emb_bcast = model2.embed(mid_vals).unsqueeze(1).expand(-1, model2.n_register, -1)
        R_forced = mid_emb_bcast + model2.register_init.unsqueeze(0).expand(B, -1, -1)
        kspace_rank_test(mem, R_forced, k_leaf_normed, leaves, n_facts, n_eval,
                          lambda b: mid_vals[b].item(), "hop2 K-space (forced R)")

        R = model2.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
        sm_k = torch.zeros(B, 0, model2.d_model)
        sm_v = torch.zeros(B, 0, model2.d_model)
        for t in range(base.N_STEP):
            o_kb = mem.attend(R)
            if sm_k.shape[1] > 0:
                q_sm = model2.sm_q_proj(R)
                o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
            else:
                o_sm = torch.zeros_like(R)
            fused = torch.cat([o_kb, o_sm, R], dim=-1)
            delta = model2.fuse_proj(model2.fuse_norm(fused))
            R = R + delta
            new_k, new_v = model2.sm_write_proj(R).chunk(2, dim=-1)
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)
            if t % 4 == 0 or t == base.N_STEP - 1:
                print(f"t={t:2d}  ||R_t|| mean={R.norm(dim=-1).mean().item():.3f}")

    print("\n=== Norm-matched intervention (rules out scale mismatch as the explanation) ===")
    for mode in ("replace", "additive_raw", "additive_norm_matched"):
        print(f"--- mode={mode} ---")
        for step in (0, 4, 8, 11, 15):
            acc = norm_matched_intervention(ds2, model2, n_facts, step, mode)
            print(f"  force_step={step:2d}  final acc: {acc:.3f}")


if __name__ == "__main__":
    main()
