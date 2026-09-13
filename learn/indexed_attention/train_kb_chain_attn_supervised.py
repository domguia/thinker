"""
GPU-scale version of learn/indexed_attention/diagnose_attention_supervision.py
-- the fix that broke the n_hops=2 composition plateau at CPU scale (see
dev_notes/indexed_attention_experiment_plan.md Phase 2): a contrastive-style
CE auxiliary loss directly supervising q_proj(R).k_proj(candidate_key)
toward the correct fact, at two points -- hop 1 at t=0 (R0 is exactly the
natural hop-1 query) and hop 2 averaged across every later step
t=1..n_step-1 (no fixed step "should" resolve hop 2, so this supervises
whichever step the model actually uses for it). No new parameters -- only
the loss function changes, unlike the (failed) sharpening/temperature fix.

CPU result (4000 steps, d_model=32, batch_size=64): self-match top-1 88.7%
(chance 25%), task accuracy 72.9% (vs ~25-32% baseline), still rising when
the budget ran out. This script re-tests the same idea at a bigger scale and
longer budget -- same CLI/task/eval conventions as train_kb_chain.py and
train_kb_chain_sharpened.py. `--attn_supervised` toggles the auxiliary loss;
omitting it reproduces the exact unsupervised baseline (plain cross-entropy
on the final answer only), so this file covers both arms of the comparison.

Requires access to KBChainDataset's raw fact layout (KEY_MARK/VAL_MARK) to
build the per-episode candidate set and decode the ground-truth intermediate
value -- see decode_mid_val below, same logic as
tests/test_kb_chain_retrieval.py / diagnose_no_ff_composition.py.
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data.kb_chain_retrieval import KBChainDataset, KEY_MARK, VAL_MARK
from core.indexed_thinker_model import Thinker


def build_model(args, total_vocab_size, device):
    return Thinker(
        vocab_size=total_vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        sm_cap=args.sm_cap, use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        detach_sm_keys=args.detach_sm_keys, level_dropout_p=args.level_dropout_p,
        decouple_kv=not args.shared_kv_pooling,
    ).to(device)


def decode_mid_val(leaves, source_ids, query, n_facts):
    facts = {}
    for i in range(n_facts):
        facts[leaves[i * 4 + 1].item()] = leaves[i * 4 + 3].item()
    return facts[query.item()]


def key_positions(n_facts):
    return torch.tensor([i * 4 + 1 for i in range(n_facts)])


def target_fact_index(leaves, key_pos, target_token):
    """(B,) index of the fact whose KEY token equals `target_token`."""
    cand_keys = leaves[:, key_pos]
    return (cand_keys == target_token.unsqueeze(1)).float().argmax(dim=1)


def fact_node_match_loss(mem, query_vec, leaves, key_pos, target_token, n_facts):
    """CORRECT SUPERVISION TARGET (2026-09-13): pushes q_proj(query_vec) to
    select the target fact's level-1 NODE -- the only object in the memory
    that carries a key->value association, and what
    `HierarchicalMemory.attend()` must actually pick. Requires
    `decouple_kv=True` to be useful (with shared pooling a node's value is
    tied to the same weighting as its key, so selecting it correctly still
    cannot return the fact's value).

    Replaces `leaf_key_match_loss` below as the default; see that function's
    docstring for why the original target was self-defeating."""
    assert mem.depth >= 1, "node-level supervision needs a hierarchy (depth >= 1)"
    nodes_k = mem.level_norms[1](mem._levels_k[1])[:, : n_facts * mem.n_slots]
    q = mem.q_proj(query_vec).unsqueeze(1)
    scores = torch.einsum('bod,bnd->bn', q, nodes_k) / (mem.d_model ** 0.5)
    if mem.n_slots > 1:
        scores = scores.view(scores.shape[0], n_facts, mem.n_slots).max(dim=-1).values
    return F.cross_entropy(scores, target_fact_index(leaves, key_pos, target_token))


def leaf_key_match_loss(mem, embed_fn, query_vec, leaves, key_pos, target_token):
    """DEPRECATED TARGET (`--supervise leaf`), kept only to reproduce the
    pre-2026-09-13 GPU grid. Pushes the query toward the fact's KEY LEAF --
    but attending to a key leaf returns `v_proj` of that same key token, i.e.
    exactly what the model already had, never the fact's value. This is why
    that grid drove the self-match diagnostic to a perfect 0.000 mean_rank /
    100% top-1 across all 3 seeds while task accuracy stayed flat at the
    plateau: the auxiliary objective was fully satisfiable AND orthogonal to
    the task. See dev_notes/experiment.log.md ("contre-expertise")."""
    cand_keys = leaves[:, key_pos]  # (B, n_facts)
    cand_emb = embed_fn(cand_keys)
    cand_k = mem.level_norms[0](mem.k_proj(cand_emb + mem.source_bias.weight[1]))  # (B, n_facts, d)
    q = mem.q_proj(query_vec).unsqueeze(1)  # (B, 1, d)
    scores = torch.einsum('bod,bnd->bn', q, cand_k) / (mem.d_model ** 0.5)
    target_idx = (cand_keys == target_token.unsqueeze(1)).float().argmax(dim=1)
    return F.cross_entropy(scores, target_idx)


def aux_match_loss(supervise, mem, embed_fn, query_vec, leaves, key_pos, target_token, n_facts):
    if supervise == "node":
        return fact_node_match_loss(mem, query_vec, leaves, key_pos, target_token, n_facts)
    return leaf_key_match_loss(mem, embed_fn, query_vec, leaves, key_pos, target_token)


def forward_with_optional_supervision(model, kb_tokens, kb_source_ids, query_tokens, kb_mask,
                                       n_step, mid_vals, key_pos, attn_supervised, aux_weight, device,
                                       supervise="node", n_facts=None):
    """Manual unroll of Thinker.forward's loop (same equations as
    core/indexed_thinker_model.py), needed to insert the auxiliary loss at
    each step -- the public forward() doesn't expose intermediate R."""
    B = kb_tokens.shape[0]
    d_model = model.d_model
    leaf_emb = model.embed(kb_tokens)
    model.memory.build(leaf_emb, kb_source_ids, leaf_mask=kb_mask)
    q_emb = model.embed(query_tokens).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, d_model, device=device, dtype=R.dtype)
    sm_v = torch.zeros(B, 0, d_model, device=device, dtype=R.dtype)

    aux_loss = None
    if attn_supervised:
        aux_loss = aux_match_loss(supervise, model.memory, model.embed, R.mean(dim=1),
                                  kb_tokens, key_pos, query_tokens[:, 0], n_facts)

    n_later = 0
    for t in range(n_step):
        o_kb = model.memory.attend(R)
        if attn_supervised and t >= 1:
            aux_loss = aux_loss + aux_match_loss(supervise, model.memory, model.embed, R.mean(dim=1),
                                                 kb_tokens, key_pos, mid_vals, n_facts)
            n_later += 1
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
    if attn_supervised:
        aux_loss = aux_loss / (1 + n_later)
    return stream_out, aux_loss


def evaluate(model, ds, args, device, n_batches=10):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
            kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
            kb_mask, query_tokens, labels = kb_mask.to(device), query_tokens.to(device), labels.to(device)
            logits, _ = forward_with_optional_supervision(
                model, kb_tokens, kb_source_ids, query_tokens, kb_mask, args.n_step,
                None, None, False, 0.0, device,
            )
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    model.train()
    return correct / total


@torch.no_grad()
def self_match_diagnostic(model, ds, n_facts, device, n_eval=512):
    model.eval()
    mem = model.memory
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    leaves = leaves.to(device)
    self_sim, ranks = [], []
    for b in range(n_eval):
        key_tokens = leaves[b, [i * 4 + 1 for i in range(n_facts)]]
        tok_emb = model.embed(key_tokens)
        q = mem.q_proj(tok_emb)
        k = mem.level_norms[0](mem.k_proj(tok_emb + mem.source_bias.weight[1]))
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = qn @ kn.T
        s_diag = sim.diag()
        self_sim.append(s_diag)
        ranks.append((sim > s_diag.unsqueeze(1)).sum(dim=1).float())
    model.train()
    self_sim, ranks = torch.cat(self_sim), torch.cat(ranks)
    print(f"[self-match diag] n={len(self_sim)} ({n_eval} episodes x {n_facts} facts) "
          f"mean_sim={self_sim.mean().item():.4f} mean_rank={ranks.mean().item():.3f} "
          f"(chance={(n_facts - 1) / 2:.2f}) top1_rate={(ranks == 0).float().mean().item():.3f} "
          f"(chance={1 / n_facts:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--n_distractors", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_register", type=int, default=1)
    parser.add_argument("--n_slots", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--n_step", type=int, default=16)
    parser.add_argument("--use_ff", action="store_true")
    parser.add_argument("--ff_hidden_mult", type=int, default=4)
    parser.add_argument("--detach_sm_keys", action="store_true")
    parser.add_argument("--level_dropout_p", type=float, default=0.0)
    parser.add_argument("--sm_cap", type=int, default=None)
    parser.add_argument("--attn_supervised", action="store_true", help="add the contrastive attention-matching auxiliary loss")
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--supervise", choices=["node", "leaf"], default="node",
                         help="what the auxiliary loss targets. 'node' (default, correct): the "
                              "target fact's level-1 node, what attend() must select. 'leaf' "
                              "(deprecated): the fact's KEY leaf -- reproduces the pre-2026-09-13 "
                              "grid whose auxiliary loss was perfectly satisfiable AND orthogonal "
                              "to the task. See this file's loss docstrings.")
    parser.add_argument("--shared_kv_pooling", action="store_true",
                         help="ABLATION: pre-2026-09-13 LevelCompressor (one softmax pools parent "
                              "K and parent V). Node supervision cannot help under this variant.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--max_time_minutes", type=float, default=30.0)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    max_facts = args.n_hops + args.n_distractors
    expected_leaves = args.block_size ** args.depth
    assert expected_leaves == max_facts * 4, (
        f"max_facts={max_facts} (-> {max_facts * 4} leaves) doesn't match "
        f"block_size={args.block_size} ** depth={args.depth} = {expected_leaves}"
    )
    key_pos = key_positions(max_facts).to(device)

    ds = KBChainDataset(n_hops=args.n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                         max_facts=max_facts, seed=args.seed)
    eval_ds = KBChainDataset(n_hops=args.n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                              max_facts=max_facts, seed=args.seed + 10_000)

    model = build_model(args, ds.total_vocab_size, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"attn_supervised={args.attn_supervised} d_model={args.d_model} n_hops={args.n_hops} n_step={args.n_step} "
          f"batch_size={args.batch_size} lr={args.lr} seed={args.seed} num_params={num_params / 1e6:.3f}M device={device}")

    best_loss_t = None
    start = time.time()
    step = 0
    while True:
        kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
        kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
        kb_mask, query_tokens, labels = kb_mask.to(device), query_tokens.to(device), labels.to(device)

        mid_vals = None
        if args.attn_supervised:
            leaves_cpu, source_ids_cpu, query_cpu = kb_tokens.cpu(), kb_source_ids.cpu(), query_tokens.cpu()
            mid_vals = torch.tensor([
                decode_mid_val(leaves_cpu[b], source_ids_cpu[b], query_cpu[b], max_facts)
                for b in range(kb_tokens.shape[0])
            ], device=device)

        opt.zero_grad()
        logits, aux_loss = forward_with_optional_supervision(
            model, kb_tokens, kb_source_ids, query_tokens, kb_mask, args.n_step,
            mid_vals, key_pos, args.attn_supervised, args.aux_weight, device,
            supervise=args.supervise, n_facts=max_facts,
        )
        task_loss = F.cross_entropy(logits, labels)
        loss = task_loss + args.aux_weight * aux_loss if args.attn_supervised else task_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            best_loss_t = task_loss.detach() if best_loss_t is None else torch.minimum(best_loss_t, task_loss.detach())
        step += 1

        if step % args.log_every == 0:
            aux_str = f" aux_loss {aux_loss.item():.4f}" if args.attn_supervised else ""
            print(f"step {step:5d} task_loss {task_loss.item():.4f}{aux_str} elapsed {time.time() - start:.1f}s")
        if step % args.eval_every == 0:
            acc = evaluate(model, eval_ds, args, device)
            print(f"step {step:5d} eval_acc {acc:.4f}")

        elapsed_min = (time.time() - start) / 60
        if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
            break

    final_acc = evaluate(model, eval_ds, args, device, n_batches=20)
    elapsed = time.time() - start
    print("---")
    print(f"best_task_loss:   {best_loss_t.item():.6f}")
    print(f"final_acc:        {final_acc:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"attn_supervised:  {args.attn_supervised}")
    print(f"seed:             {args.seed}")
    self_match_diagnostic(model, eval_ds, n_facts=max_facts, device=device)


if __name__ == "__main__":
    main()
