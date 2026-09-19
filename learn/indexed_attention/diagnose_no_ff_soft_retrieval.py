"""
Follow-up to diagnose_no_ff_composition.py's causal intervention result (see
dev_notes/indexed_attention_experiment_plan.md Phase 2): a "soft" retrieval
quality measure, since the raw top-1 attention-argmax audit tried first was
inconclusive (retrieval here is soft/implicit by design, spec §5.3 -- a hard
argmax metric likely under-measures it).

Methodological correction made while building this (worth keeping visible):
a first attempt compared `o_kb` (HierarchicalMemory.attend's output) against
RAW `self.embed(value)` vectors -- invalid, because `o_kb` is a weighted
average of `v_proj`-transformed (+ source-bias-shifted) leaf values, not raw
embeddings, exactly the same pitfall that invalidated the nearest-neighbor
probe on R in diagnose_no_ff_composition.py. Fixed here by comparing `o_kb`
against the ACTUAL v_proj-space vectors HierarchicalMemory.build() stored
(`memory._levels_v[0]`), and by ranking only among the episode's own
candidate fact values (chance = 1/n_facts) rather than the full vocabulary.

Requires the checkpoint saved by diagnose_no_ff_composition.py's rerun
(/tmp/diagnose_n_hops2_no_ff_checkpoint.pt) -- run that script first if it's
missing.
"""
import numpy as np
import torch
import torch.nn.functional as F

import learn.indexed_attention.diagnose_no_ff_composition as base

CHECKPOINT = "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt"


def fact_positions(leaves, n_facts, b):
    """(key_token, val_token, val_leaf_idx) for each real fact in example b."""
    return [(leaves[b, i * 4 + 1].item(), leaves[b, i * 4 + 3].item(), i * 4 + 3) for i in range(n_facts)]


def _dist_str(t: torch.Tensor) -> str:
    q = torch.quantile(t.float(), torch.tensor([0.1, 0.5, 0.9]))
    return (f"mean={t.mean().item():+.4f} std={t.std().item():.4f} "
            f"p10={q[0].item():+.4f} p50={q[1].item():+.4f} p90={q[2].item():+.4f}")


def soft_rank_test(o_kb_batch, v_leaf, leaves, n_facts, n_eval, target_idx_fn, label_text,
                    model, non_key_source_ids, vocab_size, rng):
    """target_idx_fn(b, facts) -> index into `facts` of the correct candidate.

    Two trivial controls (both in v_proj space, same pitfall as the module
    docstring notes -- comparing against raw embeddings would be invalid):

    - control_random_kb: cos_sim against a WRONG candidate value's true
      v_proj vector, drawn at random from this same episode's own KB (i.e.
      "a value drawn at random from the KB" -- in-context negative).
    - control_non_key: cos_sim against a value that was never inserted into
      this episode's memory at all -- a uniformly random vocab token pushed
      through the same v_proj(embed(x) + source_bias(KB)) transform as a
      real KB value leaf would be, but absent from `memory.build()`'s actual
      leaves (i.e. a "non-key" value: not keyed to anything here).
    """
    correct_sim, random_kb_sim, non_key_sim, rank_of_correct = [], [], [], []
    for b in range(n_eval):
        facts = fact_positions(leaves, n_facts, b)
        correct_idx = target_idx_fn(b, facts)
        cand_v = torch.stack([v_leaf[b, pos] for (_, _, pos) in facts])  # (n_facts, d), true v_proj space
        sims = F.cosine_similarity(o_kb_batch[b].unsqueeze(0), cand_v, dim=-1)
        correct_sim.append(sims[correct_idx].item())
        order = sims.argsort(descending=True).tolist()
        rank_of_correct.append(order.index(correct_idx))

        wrong_idxs = [i for i in range(n_facts) if i != correct_idx]
        random_kb_sim.append(sims[wrong_idxs[rng.integers(len(wrong_idxs))]].item())

    with torch.no_grad():
        rand_tok = torch.from_numpy(rng.integers(0, vocab_size, size=(n_eval,))).long()
        non_key_v = model.memory.v_proj(model.embed(rand_tok) + model.memory.source_bias(non_key_source_ids))
        non_key_sim = F.cosine_similarity(o_kb_batch, non_key_v, dim=-1)

    correct_sim = torch.tensor(correct_sim)
    random_kb_sim = torch.tensor(random_kb_sim)
    rank_of_correct = torch.tensor(rank_of_correct)
    print(f"=== {label_text} ===")
    print(f"correct value      : {_dist_str(correct_sim)}")
    print(f"control random_kb  : {_dist_str(random_kb_sim)}  (wrong candidate, same episode's own KB)")
    print(f"control non_key    : {_dist_str(non_key_sim)}  (random token, never inserted this episode)")
    print(f"gap correct-random_kb: {(correct_sim - random_kb_sim).mean().item():+.4f}   "
          f"gap correct-non_key: {(correct_sim - non_key_sim.float()).mean().item():+.4f}")
    print(f"top-1 rate among {n_facts} candidates: {(rank_of_correct == 0).float().mean().item():.3f}  (chance={1 / n_facts:.3f})")
    print(f"mean rank (0=best, {n_facts - 1}=worst): {rank_of_correct.float().mean().item():.3f}  (chance={(n_facts - 1) / 2:.2f})")


def main():
    torch.manual_seed(1)
    ds2, model2 = base.make_model(n_hops=2)
    model2.load_state_dict(torch.load(CHECKPOINT))
    model2.eval()

    n_eval = 256
    n_facts = 2 + base.N_DISTRACTORS
    leaves, source_ids, mask, query, label = ds2.sample_batch(n_eval)
    rng = np.random.default_rng(0)
    non_key_source_ids = torch.ones(n_eval, dtype=torch.long)  # KB source id (data/kb_chain_retrieval.py: all-ones)

    with torch.no_grad():
        B = leaves.shape[0]
        leaf_emb = model2.embed(leaves)
        model2.memory.build(leaf_emb, source_ids, leaf_mask=mask)
        v_leaf = model2.memory._levels_v[0]  # (B, N, d) -- same space o_kb is built from

        # Hop 1: natural R0 (the query the model actually produces), target = fact whose KEY == query
        q_emb = model2.embed(query).mean(dim=1, keepdim=True)
        R0 = model2.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
        o_kb0 = model2.memory.attend(R0).mean(dim=1)
        soft_rank_test(
            o_kb0, v_leaf, leaves, n_facts, n_eval,
            target_idx_fn=lambda b, facts: [i for i, (k, v, _) in enumerate(facts) if k == query[b, 0].item()][0],
            label_text="hop1: natural R0, target=fact whose KEY==query",
            model=model2, non_key_source_ids=non_key_source_ids, vocab_size=base.VOCAB_SIZE, rng=rng,
        )

        # Hop 2: forced R (register_init + embed(mid_val), spec-consistent seeding), target = fact whose VALUE == label
        mid_vals = torch.stack([
            base.decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(n_eval)
        ]).squeeze(-1)
        mid_emb_bcast = model2.embed(mid_vals).unsqueeze(1).expand(-1, model2.n_register, -1)
        R_forced = mid_emb_bcast + model2.register_init.unsqueeze(0).expand(B, -1, -1)
        o_kb_forced = model2.memory.attend(R_forced).mean(dim=1)
        soft_rank_test(
            o_kb_forced, v_leaf, leaves, n_facts, n_eval,
            target_idx_fn=lambda b, facts: [i for i, (k, v, _) in enumerate(facts) if v == label[b].item()][0],
            label_text="hop2: forced R (clean mid_val query), target=fact whose VALUE==label",
            model=model2, non_key_source_ids=non_key_source_ids, vocab_size=base.VOCAB_SIZE, rng=rng,
        )


if __name__ == "__main__":
    main()
