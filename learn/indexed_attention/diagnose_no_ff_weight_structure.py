"""
Third follow-up to diagnose_no_ff_composition.py's causal intervention
result (see dev_notes/indexed_attention_experiment_plan.md Phase 2):
inspects the learned q_proj/k_proj weight structure directly, to
characterize the "mode collapse" hypothesis raised by the K-space test's
below-chance similarity (diagnose_no_ff_scale_and_kspace.py).

Requires the checkpoint saved by diagnose_no_ff_composition.py's rerun
(/tmp/diagnose_n_hops2_no_ff_checkpoint.pt).

Four checks, in increasing specificity:
1. Effective rank (participation ratio of singular values) of q_proj/k_proj/
   v_proj/sm_q_proj/fuse_proj -- rules out simple low-rank collapse.
2. Whether the shared source_bias(KB=1) additive term dominates k_proj's
   output relative to content-driven variation across distinct tokens --
   rules out bias-swamps-content.
3. Whether register_init dominates q_proj's input relative to an injected
   token embedding (norm comparison + per-token direction cosine) -- rules
   out register_init washing out content identity.
4. A context-free "self-match" test: for many random distinct tokens, is
   q_proj(embed(x)) closer to k_proj(embed(x)+source_bias) (the correct
   self-match) than to k_proj(embed(y)+source_bias) for other y?

BUG FOUND AND CORRECTED (2026-09-13): this script originally requested
`torch.randperm(vocab_size)[:300]` where `vocab_size` was
`ds.total_vocab_size` (34 for this toy setup: VOCAB_SIZE=32 + 2 marker
tokens) -- `randperm(34)` can only ever return 34 elements, and Python
slicing past a tensor's length silently returns what exists rather than
raising, so this was actually testing among 34 tokens (the model's ENTIRE
vocabulary -- there is no larger pool to sample from), not 300. The
originally reported "mean rank ~21/300, chance=150 -- above chance" was
wrong on both the N and the chance level. Corrected below (arange(vocab_size)
instead of a randperm slice, explicit real N): mean rank is ~21.4 out of the
ACTUAL 34, chance=16.5 -- i.e. slightly WORSE than chance, consistent with
(not in tension with) the in-episode K-space test in
diagnose_no_ff_scale_and_kspace.py. There is no unresolved gap between
"isolated" and "in-episode" matching after this fix -- both are at-or-below
chance; the earlier "tension" was an artifact of this bug, not a real
phenomenon.
"""
import torch
import torch.nn.functional as F

import learn.indexed_attention.diagnose_no_ff_composition as base

CHECKPOINT = "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt"


def participation_ratio(W):
    s = torch.linalg.svdvals(W)
    return (s.sum() ** 2 / s.pow(2).sum()).item(), s


def main():
    torch.manual_seed(1)
    ds2, model2 = base.make_model(n_hops=2)
    model2.load_state_dict(torch.load(CHECKPOINT))
    model2.eval()
    mem = model2.memory
    d_model = model2.d_model

    print("=== 1. Effective rank (participation ratio, max = d_model) ===")
    for name, W in [("q_proj (memory)", mem.q_proj.weight), ("k_proj (memory)", mem.k_proj.weight),
                     ("v_proj (memory)", mem.v_proj.weight), ("sm_q_proj", model2.sm_q_proj.weight),
                     ("fuse_proj", model2.fuse_proj.weight)]:
        pr, s = participation_ratio(W)
        print(f"{name:20s}  shape={tuple(W.shape)}  participation_ratio={pr:.2f} / d_model={d_model}  "
              f"top sv={s[0].item():.3f}  smallest sv={s[-1].item():.4f}")

    with torch.no_grad():
        print("\n=== 2. Shared source_bias term vs. content variation ===")
        kb_bias_k = mem.k_proj(mem.source_bias.weight[1])
        vocab_size = ds2.total_vocab_size  # 34 for this toy setup -- the model's ENTIRE vocabulary, no larger pool exists
        sample_ids = torch.arange(vocab_size)  # all of them; requesting more via randperm()[:N>vocab_size] silently truncates, see module docstring
        content_k = mem.k_proj(model2.embed(sample_ids))
        content_k_centered = content_k - content_k.mean(dim=0, keepdim=True)
        print(f"||k_proj(source_bias[KB=1])|| = {kb_bias_k.norm().item():.4f}  (shared/constant across all KB keys)")
        print(f"content-only variation norm (post-centering) = {content_k_centered.norm(dim=-1).mean().item():.4f}")
        print(f"ratio shared/content: {kb_bias_k.norm().item() / content_k_centered.norm(dim=-1).mean().item():.3f}  (>>1 would mean bias swamps content)")

        print("\n=== 3. Does register_init wash out injected token identity? ===")
        ri = model2.register_init
        print(f"||register_init|| per slot: {[round(x, 3) for x in ri.norm(dim=-1).tolist()]}")
        tok_emb = model2.embed(sample_ids)
        print(f"||embed(token)|| mean: {tok_emb.norm(dim=-1).mean().item():.4f}")
        q_with_ri = mem.q_proj(ri[0].unsqueeze(0) + tok_emb)
        q_without_ri = mem.q_proj(tok_emb)
        per_token_cos = F.cosine_similarity(q_with_ri, q_without_ri, dim=-1)
        print(f"per-token cos_sim(q_proj(register_init+embed(x)), q_proj(embed(x))): {per_token_cos.mean().item():.4f}  (1.0 = content fully preserved)")

        print("\n=== 4. Context-free self-match test ===")
        N = vocab_size  # the real ceiling -- see module docstring for the bug this corrects
        ids = torch.arange(N)
        tok_emb = model2.embed(ids)
        q = mem.q_proj(tok_emb)
        k = mem.level_norms[0](mem.k_proj(tok_emb + mem.source_bias.weight[1]))
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = qn @ kn.T
        self_sim = sim.diag()
        rank_of_self = (sim > self_sim.unsqueeze(1)).sum(dim=1).float()
        print(f"mean self-match sim q_proj(x).k_proj(x+bias): {self_sim.mean().item():.4f}")
        print(f"mean rank of self-match among {N} candidates (0=best, chance={(N - 1) / 2:.1f}): {rank_of_self.mean().item():.2f}")
        print(f"top-1 rate (chance={1 / N:.4f}): {(rank_of_self == 0).float().mean().item():.3f}")

        print("\n=== 5. Same self-match test, candidates = the 4 tokens actually co-sampled per real episode ===")
        print("(the natural follow-up to 4: does grouping matter, independent of pool size? see module docstring)")
        n_facts = 2 + base.N_DISTRACTORS
        n_eval = 256
        leaves, source_ids, mask, query, label = ds2.sample_batch(n_eval)
        self_sim_ep, rank_ep = [], []
        for b in range(n_eval):
            key_tokens = torch.tensor([leaves[b, i * 4 + 1].item() for i in range(n_facts)])
            tok_emb_ep = model2.embed(key_tokens)
            q_ep = mem.q_proj(tok_emb_ep)
            k_ep = mem.level_norms[0](mem.k_proj(tok_emb_ep + mem.source_bias.weight[1]))
            qn_ep, kn_ep = F.normalize(q_ep, dim=-1), F.normalize(k_ep, dim=-1)
            sim_ep = qn_ep @ kn_ep.T
            s_diag = sim_ep.diag()
            self_sim_ep.append(s_diag)
            rank_ep.append((sim_ep > s_diag.unsqueeze(1)).sum(dim=1).float())
        self_sim_ep = torch.cat(self_sim_ep)
        rank_ep = torch.cat(rank_ep)
        print(f"n = {len(self_sim_ep)} ({n_eval} episodes x {n_facts} facts)")
        print(f"mean self-match similarity: {self_sim_ep.mean().item():.4f}")
        print(f"mean rank among {n_facts} (0=best, chance={(n_facts - 1) / 2:.2f}): {rank_ep.mean().item():.3f}")
        print(f"top-1 rate (chance={1 / n_facts:.3f}): {(rank_ep == 0).float().mean().item():.3f}")
        print("-> consistent with part 4 (both at-or-below chance): no real gap between 'isolated' and "
              "'in-episode' matching once part 4's bug is fixed -- both show broken matching.")


if __name__ == "__main__":
    main()
