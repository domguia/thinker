"""
Phase 0 sanity-check / curriculum training loop
(dev_notes/indexed_attention_experiment_plan.md Phase 0 / Phase -1): compares
`Thinker` with a real hierarchy (depth > 0) against the flat Baseline C
(depth = 0, spec §9) on `data/kb_retrieval.py`'s synthetic retrieval task, at a
larger scale than the CPU overfit test in tests/test_indexed_memory.py (more
distractor facts, deeper hierarchy, wider model). Since KBRetrievalDataset is
infinite/IID (a fresh random KB is drawn every sample_batch() call), training
accuracy measured on freshly sampled batches doubles as a generalization
estimate -- there is no fixed train set to overfit here, unlike the CPU
sanity test.

Curriculum (--curriculum "16,32,64"): dev_notes/indexed_attention_experiment_plan.md
Phase -1 found training collapses when n_facts jumps straight to 64 (works at
16, doesn't at 64, independent of d_model/lr/n_step tried). Matches this
project's own Dec-2023 ToyThinker curriculum precedent. Uses
KBRetrievalDataset's `max_facts` (fixed leaf-sequence shape == the LAST
curriculum stage, so the model's hierarchy shape never changes) with
`n_facts` (real facts present, rest padded+masked) increasing per stage once
held-out accuracy crosses --curriculum_promote_acc.

Output format matches program.md / learn/distill/train_sft.py's convention
(best_loss / training_seconds / num_steps / num_params_M) for consistency
with the rest of the repo's experiment scripts.
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data.kb_retrieval import KBRetrievalDataset
from core.indexed_thinker_model import Thinker
from learn.indexed_attention.eval_metrics import (
    prediction_stats, trivial_baselines, format_report,
)


def build_model(args, total_vocab_size, device):
    model = Thinker(
        vocab_size=total_vocab_size,
        d_model=args.d_model,
        n_register=args.n_register,
        block_size=args.block_size,
        depth=args.depth,
        n_slots=args.n_slots,
        n_head=args.n_head,
        sm_cap=args.sm_cap,
        use_ff=args.use_ff,
        ff_hidden_mult=args.ff_hidden_mult,
        detach_sm_keys=args.detach_sm_keys,
        level_dropout_p=args.level_dropout_p,
        decouple_kv=not args.shared_kv_pooling,
    ).to(device)
    return model


def evaluate(model, ds, args, device, n_batches=10, with_stats=False):
    """See train_kb_chain.py::evaluate -- `with_stats` additionally returns
    `pred_in_kb_rate`, without which an accuracy number cannot be compared
    against the right chance level (learn/indexed_attention/eval_metrics.py)."""
    model.eval()
    correct, in_kb, total = 0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
            kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
            kb_mask = kb_mask.to(device)
            query_tokens, labels = query_tokens.to(device), labels.to(device)
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=args.n_step, kb_leaf_mask=kb_mask)
            preds = streams["answer"][:, 0, :].argmax(dim=-1)
            st = prediction_stats(preds.cpu(), labels.cpu(), kb_tokens.cpu(), ds.n_facts)
            correct += st["correct"]
            in_kb += st["in_kb"]
            total += st["total"]
    model.train()
    acc = correct / total
    return (acc, in_kb / total) if with_stats else acc


def attention_attribution(model, ds, args, device, n_examples=4):
    """
    Diagnostic requested by dev_notes/indexed_attention_experiment_plan.md's
    revised methodology: without this, high accuracy could come from a
    shortcut (the register/SM memorizing the answer without ever routing
    through the KB) rather than the hierarchy actually being used. Recomputes
    HierarchicalMemory.attend()'s softmax manually (same k_proj/q_proj/
    level_norms already on the model -- no core/ changes needed) to report,
    for the LAST core iteration, how much attention mass lands on the leaf
    block containing the target fact's KEY_MARK/key token vs. everywhere
    else, per hierarchy level. Best-effort/approximate (recomputes only the
    final register state's query, not a full step-by-step trace); meant as a
    sanity signal, not a rigorous causal proof.
    """
    model.eval()
    with torch.no_grad():
        kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(n_examples)
        kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
        kb_mask = kb_mask.to(device)
        query_tokens, labels = query_tokens.to(device), labels.to(device)

        leaf_emb = model.embed(kb_tokens)
        model.memory.build(leaf_emb, kb_source_ids, leaf_mask=kb_mask)

        q_emb = model.embed(query_tokens).mean(dim=1, keepdim=True)
        R = model.register_init.unsqueeze(0).expand(kb_tokens.shape[0], -1, -1) + q_emb
        sm_k = torch.zeros(kb_tokens.shape[0], 0, model.d_model, device=device, dtype=R.dtype)
        sm_v = torch.zeros(kb_tokens.shape[0], 0, model.d_model, device=device, dtype=R.dtype)
        for _ in range(args.n_step):
            o_kb = model.memory.attend(R)
            if sm_k.shape[1] > 0:
                q_sm = model.sm_q_proj(R)
                o_sm = torch.nn.functional.scaled_dot_product_attention(q_sm, sm_k, sm_v)
            else:
                o_sm = torch.zeros_like(R)
            fused = torch.cat([o_kb, o_sm, R], dim=-1)
            normed_fused = model.fuse_norm(fused)
            if model.use_ff:
                delta = model.fuse_out(torch.nn.functional.gelu(model.fuse_in(normed_fused)))
            else:
                delta = model.fuse_proj(normed_fused)
            R = R + delta
            new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
            if model.detach_sm_keys:
                new_k = new_k.detach()
            sm_k = torch.cat([sm_k, new_k], dim=1)
            sm_v = torch.cat([sm_v, new_v], dim=1)

        mem = model.memory
        normed_k = [mem.level_norms[i](k) for i, k in enumerate(mem._levels_k)]
        k_all = torch.cat(normed_k, dim=1)
        q = mem.q_proj(R)
        import math
        scores = torch.einsum("btd,bsd->bts", q, k_all) / math.sqrt(mem.d_model)
        weights = torch.softmax(scores, dim=-1)[:, 0, :]  # (B, S) at n_register=1's first slot

        level_sizes = [k.shape[1] for k in mem._levels_k]
        # locate, per example, the leaf index of the target fact's key token
        # (KEY_MARK, key, VAL_MARK, value quadruples -- find the leaf whose
        # (key) token equals the query token, i.e. the fact being retrieved).
        results = []
        for b in range(kb_tokens.shape[0]):
            leaf_ids = kb_tokens[b]
            query_tok = query_tokens[b, 0]
            match = (leaf_ids == query_tok).nonzero(as_tuple=True)[0]
            leaf_idx = match[0].item() if len(match) > 0 else None
            offset = 0
            per_level_mass = []
            for lvl, size in enumerate(level_sizes):
                lvl_weights = weights[b, offset: offset + size]
                per_level_mass.append(lvl_weights.sum().item())
                offset += size
            target_leaf_weight = weights[b, leaf_idx].item() if leaf_idx is not None else None
            results.append({"per_level_mass": per_level_mass, "target_leaf_weight": target_leaf_weight})
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_facts", type=int, default=64, help="distractor facts in the KB (ignored if --curriculum is set)")
    parser.add_argument("--curriculum", default=None,
                         help="comma-separated increasing n_facts stages, e.g. '16,32,64'; "
                              "model/hierarchy shape is fixed to the LAST stage (max_facts)")
    parser.add_argument("--curriculum_promote_acc", type=float, default=0.95,
                         help="held-out accuracy needed to advance to the next curriculum stage")
    parser.add_argument("--curriculum_min_steps", type=int, default=200,
                         help="minimum steps spent in a stage before an accuracy check can promote it")
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3, help="hierarchy depth; 0 = flat Baseline C")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_register", type=int, default=1)
    parser.add_argument("--n_slots", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_step", type=int, default=2, help="core loop iterations")
    parser.add_argument("--sm_cap", type=int, default=None)
    parser.add_argument("--use_ff", action="store_true", help="Phase 1bis: 2-layer GELU MLP in the main loop's fuse step instead of linear")
    parser.add_argument("--ff_hidden_mult", type=int, default=4)
    parser.add_argument("--detach_sm_keys", action="store_true", help="Phase 1bis: stop-gradient on SM keys before append")
    parser.add_argument("--level_dropout_p", type=float, default=0.0, help="Phase 1bis: stochastic dropping of high hierarchy levels (train-time only)")
    parser.add_argument("--shared_kv_pooling", action="store_true",
                         help="ABLATION: pre-2026-09-13 LevelCompressor (one softmax pools both "
                              "parent K and parent V). Cannot represent a key->value association; "
                              "see learn/indexed_attention/eval_metrics.py and the spec §5.1.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--max_time_minutes", type=float, default=20.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.curriculum:
        stages = [int(x) for x in args.curriculum.split(",")]
        assert stages == sorted(stages), "--curriculum stages must be increasing"
    else:
        stages = [args.n_facts]
    max_facts = stages[-1]

    if args.depth == 0:
        # Baseline C: no compression, so block_size must equal n_leaves
        # exactly (HierarchicalMemory.build() only checks N==block_size**depth
        # when depth>0, so block_size is otherwise unused at depth=0 -- keep
        # it consistent anyway for clarity).
        args.block_size = max_facts * 4
    else:
        expected_leaves = args.block_size ** args.depth
        assert expected_leaves == max_facts * 4, (
            f"max_facts={max_facts} (-> {max_facts * 4} leaves) doesn't match "
            f"block_size={args.block_size} ** depth={args.depth} = {expected_leaves}; "
            f"adjust --curriculum/--n_facts/--block_size/--depth so block_size**depth == max_facts*4"
        )

    def make_datasets(n_facts):
        # Held-out generalization check (plan's revised methodology): a
        # SEPARATE dataset instance seeded disjointly from training, so eval
        # batches are never drawn from the training RNG stream.
        ds = KBRetrievalDataset(n_facts=n_facts, vocab_size=args.vocab_size, max_facts=max_facts, seed=args.seed)
        eval_ds = KBRetrievalDataset(n_facts=n_facts, vocab_size=args.vocab_size, max_facts=max_facts, seed=args.seed + 10_000)
        return ds, eval_ds

    stage_idx = 0
    ds, eval_ds = make_datasets(stages[stage_idx])
    model = build_model(args, ds.total_vocab_size, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"depth={args.depth} block_size={args.block_size} max_facts={max_facts} "
          f"curriculum_stages={stages} d_model={args.d_model} num_params={num_params / 1e6:.3f}M device={device}")

    best_loss_t = None  # kept on-device; avoids a CPU<->GPU sync every step
    start = time.time()
    step = 0
    stage_start_step = 0
    while True:
        kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
        kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
        kb_mask = kb_mask.to(device)
        query_tokens, labels = query_tokens.to(device), labels.to(device)

        opt.zero_grad()
        _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=args.n_step, kb_leaf_mask=kb_mask)
        logits = streams["answer"][:, 0, :]
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        with torch.no_grad():
            best_loss_t = loss.detach() if best_loss_t is None else torch.minimum(best_loss_t, loss.detach())
        step += 1

        if step % args.log_every == 0:
            elapsed = time.time() - start
            print(f"step {step:5d} stage_n_facts={stages[stage_idx]} loss {loss.item():.4f} elapsed {elapsed:.1f}s")

        if step % args.eval_every == 0:
            acc = evaluate(model, eval_ds, args, device)
            print(f"step {step:5d} stage_n_facts={stages[stage_idx]} eval_acc(held_out) {acc:.4f}")
            if args.depth > 0:
                attrib = attention_attribution(model, eval_ds, args, device)
                for i, r in enumerate(attrib):
                    print(f"  attrib ex{i}: per_level_mass={['%.3f' % m for m in r['per_level_mass']]} "
                          f"target_leaf_weight={r['target_leaf_weight']}")

            if (stage_idx < len(stages) - 1
                    and step - stage_start_step >= args.curriculum_min_steps
                    and acc >= args.curriculum_promote_acc):
                stage_idx += 1
                stage_start_step = step
                ds, eval_ds = make_datasets(stages[stage_idx])
                print(f"step {step:5d} CURRICULUM PROMOTE -> n_facts={stages[stage_idx]}")

        elapsed_min = (time.time() - start) / 60
        if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
            break

    final_acc, pred_in_kb_rate = evaluate(model, eval_ds, args, device, n_batches=20, with_stats=True)
    baselines = trivial_baselines(eval_ds, batch_size=args.batch_size)
    elapsed = time.time() - start

    print("---")
    print(f"best_loss:        {best_loss_t.item():.6f}")
    print(f"final_acc:        {final_acc:.6f}")
    print(f"final_stage_n_facts: {stages[stage_idx]}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"seed:             {args.seed}")
    print(f"depth:            {args.depth}")
    print(f"decouple_kv:      {not args.shared_kv_pooling}")
    print(f"pred_in_kb_rate:  {pred_in_kb_rate:.6f}")
    print(f"conditional_chance: {baselines['conditional_chance']:.6f}")
    print(format_report(final_acc, pred_in_kb_rate, baselines, n_hops=1))


if __name__ == "__main__":
    main()
