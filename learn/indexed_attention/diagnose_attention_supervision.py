"""
Follow-up to diagnose_no_ff_scale_and_kspace.py / train_kb_chain_sharpened.py:
tests recommendation 3-4 from the literature research report (see
dev_notes/indexed_attention_experiment_plan.md Phase 2) after the
temperature/query-norm "sharpening" fix failed -- an auxiliary loss that
DIRECTLY supervises the retrieval matching (contrastive, InfoNCE-style),
rather than only hoping it emerges from the downstream task loss.

Motivation (from the sharpening result): a sharpening/temperature parameter
can only AMPLIFY an existing preference signal, it can't create one from
nothing. Since self-match rank was already at-or-below chance even
unsharpened, the hypothesis tested here is that there's no preference signal
to amplify in the first place -- q_proj/k_proj need a training signal that
pushes them toward correct matching directly, not just via the diluted
gradient that reaches them through fuse_proj -> SM -> OutputStream.

Two supervision points, both well-defined on data/kb_chain_retrieval.py's
known chain structure:
- hop 1 at t=0: R0 = register_init + mean(embed(query)) is exactly the
  natural "query" for the first hop -- ground truth is the fact whose KEY
  equals the query token. Supervised with a single CE/InfoNCE term.
- hop 2 at every t=1..n_step-1: unlike hop 1, there's no fixed step where the
  model "should" resolve the second hop -- averaging the loss across all
  later steps supervises whichever step the model actually uses for it,
  without us having to guess/force which one.

Both losses are restricted to the episode's own n_facts candidate keys (not
the full leaf+level attention span), matching the self-match diagnostic used
throughout this investigation, and use the SAME q_proj/k_proj/level_norms[0]
the model already has -- no new parameters, this only changes what the loss
function supervises.
"""
import time

import torch
import torch.nn.functional as F

import learn.indexed_attention.diagnose_no_ff_composition as base
from data.kb_chain_retrieval import KEY_MARK, VAL_MARK

AUX_WEIGHT = 1.0


def key_positions(leaves, n_facts):
    """(B, n_facts) leaf index of each fact's KEY token, batched."""
    return torch.tensor([i * 4 + 1 for i in range(n_facts)])


def candidate_match_loss(mem, embed_fn, query_vec, leaves, key_pos, target_token):
    """CE loss pushing q_proj(query_vec) to prefer the candidate whose KEY
    equals target_token, among this episode's n_facts candidate keys.
    query_vec: (B, d). leaves: (B, N). key_pos: (n_facts,). target_token: (B,).
    """
    B = leaves.shape[0]
    cand_keys = leaves[:, key_pos]  # (B, n_facts)
    cand_emb = embed_fn(cand_keys + 0)  # (B, n_facts, d) -- KB convention: source_bias added below
    cand_k = mem.level_norms[0](mem.k_proj(cand_emb + mem.source_bias.weight[1]))  # (B, n_facts, d)
    q = mem.q_proj(query_vec).unsqueeze(1)  # (B, 1, d)
    scores = torch.einsum('bod,bnd->bn', q, cand_k) / (mem.d_model ** 0.5)  # (B, n_facts)
    target_idx = (cand_keys == target_token.unsqueeze(1)).float().argmax(dim=1)  # (B,)
    return F.cross_entropy(scores, target_idx)


def train_with_supervision(ds, model, n_facts, steps, lr=3e-4, aux_weight=AUX_WEIGHT, log_prefix=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    kp = key_positions(None, n_facts)
    for step in range(1, steps + 1):
        leaves, source_ids, mask, query, label = ds.sample_batch(base.BATCH_SIZE)
        mid_vals = torch.stack([
            base.decode_mid_val(leaves[b], source_ids[b], query[b], n_facts) for b in range(leaves.shape[0])
        ]).squeeze(-1)

        B = leaves.shape[0]
        device = leaves.device
        mem = model.memory
        leaf_emb = model.embed(leaves)
        mem.build(leaf_emb, source_ids, leaf_mask=mask)
        q_emb = model.embed(query).mean(dim=1, keepdim=True)
        R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
        sm_k = torch.zeros(B, 0, model.d_model, device=device)
        sm_v = torch.zeros(B, 0, model.d_model, device=device)

        aux_loss = candidate_match_loss(mem, model.embed, R.mean(dim=1), leaves, kp, query[:, 0])  # hop 1 at t=0
        n_later_steps = 0
        for t in range(base.N_STEP):
            o_kb = mem.attend(R)
            if t >= 1:
                aux_loss = aux_loss + candidate_match_loss(mem, model.embed, R.mean(dim=1), leaves, kp, mid_vals)
                n_later_steps += 1
            if sm_k.shape[1] > 0:
                q_sm = model.sm_q_proj(R)
                o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
            else:
                o_sm = torch.zeros_like(R)
            fused = torch.cat([o_kb, o_sm, R], dim=-1)
            delta = model.fuse_proj(model.fuse_norm(fused))
            R = R + delta
            new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)
        aux_loss = aux_loss / (1 + n_later_steps)

        stream_out = model.streams['answer'](sm_k, sm_v)[:, 0]
        task_loss = F.cross_entropy(stream_out, label)
        loss = task_loss + aux_weight * aux_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            acc = (stream_out.argmax(-1) == label).float().mean().item()
            print(f"{log_prefix}step {step:5d}  task_loss {task_loss.item():.4f}  aux_loss {aux_loss.item():.4f}  "
                  f"acc {acc:.3f}  ({time.time() - t0:.1f}s)")
    return model


@torch.no_grad()
def self_match_diagnostic(model, ds, n_facts, n_eval=512):
    mem = model.memory
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    self_sim, ranks = [], []
    for b in range(n_eval):
        key_tokens = torch.tensor([leaves[b, i * 4 + 1].item() for i in range(n_facts)])
        tok_emb = model.embed(key_tokens)
        q = mem.q_proj(tok_emb)
        k = mem.level_norms[0](mem.k_proj(tok_emb + mem.source_bias.weight[1]))
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = qn @ kn.T
        s_diag = sim.diag()
        self_sim.append(s_diag)
        ranks.append((sim > s_diag.unsqueeze(1)).sum(dim=1).float())
    self_sim, ranks = torch.cat(self_sim), torch.cat(ranks)
    print(f"[self-match] n={len(self_sim)} mean_sim={self_sim.mean().item():.4f} "
          f"mean_rank={ranks.mean().item():.3f} (chance={(n_facts - 1) / 2:.2f}) "
          f"top1={(ranks == 0).float().mean().item():.3f} (chance={1 / n_facts:.3f})")


def main():
    torch.manual_seed(0)
    ds2, model2 = base.make_model(n_hops=2)
    n_facts = 2 + base.N_DISTRACTORS
    print("=== baseline (no supervision) self-match, untrained-for-reference ===")
    self_match_diagnostic(model2, ds2, n_facts)

    print("\n=== training WITH attention-matching supervision (aux_weight=1.0) ===")
    train_with_supervision(ds2, model2, n_facts, steps=base.TRAIN_STEPS, log_prefix="[supervised] ")
    acc = base.eval_accuracy(ds2, model2)
    print(f"\n[supervised] final eval acc: {acc:.3f}")
    self_match_diagnostic(model2, ds2, n_facts)


if __name__ == "__main__":
    main()
