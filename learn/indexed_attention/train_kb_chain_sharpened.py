"""
GPU-scale test of the two cheapest literature-suggested fixes for the
n_hops=2 composition plateau's root cause (see
dev_notes/indexed_attention_experiment_plan.md Phase 2's "Inspection des
poids" diagnostic, 2026-09-13): `q_proj`/`k_proj` content-based matching was
found to be at-or-below chance even in the simplest isolated case (a plain
self-match test among an episode's own candidate facts).

A literature review (NTM/DNC's content-addressing, Product-Key Memory's
"catastrophic drift" finding, DPR/contrastive retrieval training) converged
on two cheap, directly-applicable fixes, tested here:

1. A learned "sharpening"/temperature scalar (NTM's key-strength beta,
   Graves et al. 2014) multiplying the query before the dot product,
   initialized > 1 so scores start already sharpened rather than flat.
2. RMSNorm on `q_proj`'s OUTPUT (Product-Key Memory's fix, Lample et al.
   2019, for "catastrophic drift" -- there only a minority of memory slots
   got used without normalizing the query NETWORK's output; their fix was
   BatchNorm on that output, RMSNorm matches this project's existing
   normalization convention elsewhere). Placement matters: normalizing
   q_proj's INPUT instead (tried first, at CPU scale) showed no improvement
   -- see dev_notes/indexed_attention_experiment_plan.md Phase 2.

Otherwise identical to train_kb_chain.py (same CLI, same KBChainDataset task)
-- kept as a SEPARATE file rather than editing train_kb_chain.py in place, to
avoid clobbering any in-flight run of that script. `--sharpened` toggles the
fix; omitting it reproduces the exact baseline behavior (plain
HierarchicalMemory), so this file can be used for both arms of a comparison.

At the end of training, in addition to the usual eval_acc, runs the same
episode-structured self-match diagnostic used in
learn/indexed_attention/diagnose_no_ff_weight_structure.py (part 5): checks
whether q_proj(R)/k_proj now actually discriminate the correct key among an
episode's own candidate facts, above chance -- the direct mechanistic
question, not just downstream accuracy (which could improve for unrelated
reasons, e.g. the model finding a shortcut).
"""
import argparse
import time

import torch
from torch import nn
import torch.nn.functional as F

from data.kb_chain_retrieval import KBChainDataset
from core.indexed_thinker_model import Thinker
from core.indexed_memory import HierarchicalMemory
from core.layers import RMSNorm
from core.run_logging import add_run_args, logger_from_args


class SharpenedHierarchicalMemory(HierarchicalMemory):
    """RMSNorm is applied to q_proj's OUTPUT (not its input) -- this matches
    Product-Key Memory's actual fix (BatchNorm on the query NETWORK's output,
    Lample et al. 2019), a meaningfully different placement from normalizing
    the input. An earlier CPU-scale attempt with norm-before-projection
    showed no improvement (dev_notes/indexed_attention_experiment_plan.md
    Phase 2); norm-after-projection is the literature-faithful version,
    tested here for the first time at this (bigger) scale."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_out_norm = RMSNorm(self.d_model)
        self.temperature = nn.Parameter(torch.tensor(4.0))

    def attend(self, query_input):
        assert self._levels_k is not None, "call build() before attend()"
        normed_k = [self.level_norms[i](k) for i, k in enumerate(self._levels_k)]
        k_all = torch.cat(normed_k, dim=1)
        v_all = torch.cat(self._levels_v, dim=1)

        levels_mask = self._levels_mask
        if self.training and self.level_dropout_p > 0 and self.depth > 0:
            levels_mask = list(levels_mask)
            for i in range(1, len(levels_mask)):
                p_i = self.level_dropout_p * (i / self.depth)
                if torch.rand(()) < p_i:
                    levels_mask[i] = torch.zeros_like(levels_mask[i])
        mask_all = torch.cat(levels_mask, dim=1)

        B, T, d = query_input.shape
        q = self.query_out_norm(self.q_proj(query_input)) * self.temperature
        S = k_all.shape[1]
        if self.n_head > 1:
            hd = d // self.n_head
            q_h = q.view(B, T, self.n_head, hd).transpose(1, 2)
            k_h = k_all.view(B, S, self.n_head, hd).transpose(1, 2)
            v_h = v_all.view(B, S, self.n_head, hd).transpose(1, 2)
            attn_mask = mask_all.view(B, 1, 1, S)
            out = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=attn_mask)
            out = out.transpose(1, 2).contiguous().view(B, T, d)
        else:
            attn_mask = mask_all.view(B, 1, S)
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        return out


def build_model(args, total_vocab_size, device):
    model = Thinker(
        vocab_size=total_vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        sm_cap=args.sm_cap, use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        detach_sm_keys=args.detach_sm_keys, level_dropout_p=args.level_dropout_p,
    )
    if args.sharpened:
        model.memory = SharpenedHierarchicalMemory(
            args.d_model, args.block_size, args.depth, n_slots=args.n_slots, n_head=args.n_head,
            level_dropout_p=args.level_dropout_p,
        )
    return model.to(device)


def evaluate(model, ds, args, device, n_step, n_batches=10):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            kb_tokens, kb_source_ids, kb_mask, query_tokens, labels = ds.sample_batch(args.batch_size)
            kb_tokens, kb_source_ids = kb_tokens.to(device), kb_source_ids.to(device)
            kb_mask = kb_mask.to(device)
            query_tokens, labels = query_tokens.to(device), labels.to(device)
            _, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step=n_step, kb_leaf_mask=kb_mask)
            preds = streams["answer"][:, 0, :].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    model.train()
    return correct / total


@torch.no_grad()
def self_match_diagnostic(model, ds, n_facts, device, n_eval=512):
    """Episode-structured self-match test (mirrors
    diagnose_no_ff_weight_structure.py part 5): does q_proj(embed(key)) rank
    its own correct key above the episode's other candidate keys?"""
    model.eval()
    mem = model.memory
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    leaves, source_ids = leaves.to(device), source_ids.to(device)
    self_sim, ranks = [], []
    for b in range(n_eval):
        key_tokens = leaves[b, [i * 4 + 1 for i in range(n_facts)]]
        tok_emb = model.embed(key_tokens)
        if hasattr(mem, "query_out_norm"):
            q = mem.query_out_norm(mem.q_proj(tok_emb)) * mem.temperature
        else:
            q = mem.q_proj(tok_emb)
        k = mem.level_norms[0](mem.k_proj(tok_emb + mem.source_bias.weight[1]))
        qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = qn @ kn.T
        s_diag = sim.diag()
        self_sim.append(s_diag)
        ranks.append((sim > s_diag.unsqueeze(1)).sum(dim=1).float())
    model.train()
    self_sim, ranks = torch.cat(self_sim), torch.cat(ranks)
    chance_rank = (n_facts - 1) / 2
    print(f"[self-match diag] n={len(self_sim)} ({n_eval} episodes x {n_facts} facts) "
          f"mean_sim={self_sim.mean().item():.4f} mean_rank={ranks.mean().item():.3f} "
          f"(chance={chance_rank:.2f}) top1_rate={(ranks == 0).float().mean().item():.3f} "
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
    parser.add_argument("--sharpened", action="store_true", help="use SharpenedHierarchicalMemory (temperature + query RMSNorm)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--max_time_minutes", type=float, default=20.0)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_args(parser)
    args = parser.parse_args()
    logger = logger_from_args(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    max_facts = args.n_hops + args.n_distractors
    # N must be a MULTIPLE of block_size**depth, not equal to it -- see
    # core/indexed_memory.py::HierarchicalMemory.build()'s divisibility fix.
    divisor = args.block_size ** args.depth
    assert (max_facts * 4) % divisor == 0, (
        f"max_facts={max_facts} (-> {max_facts * 4} leaves) is not a multiple of "
        f"block_size={args.block_size} ** depth={args.depth} = {divisor}"
    )

    ds = KBChainDataset(n_hops=args.n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                         max_facts=max_facts, seed=args.seed)
    eval_ds = KBChainDataset(n_hops=args.n_hops, n_distractors=args.n_distractors, vocab_size=args.vocab_size,
                              max_facts=max_facts, seed=args.seed + 10_000)

    model = build_model(args, ds.total_vocab_size, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"sharpened={args.sharpened} d_model={args.d_model} n_hops={args.n_hops} n_step={args.n_step} "
          f"batch_size={args.batch_size} lr={args.lr} seed={args.seed} num_params={num_params / 1e6:.3f}M device={device}")

    best_loss_t = None
    start = time.time()
    step = 0
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
            print(f"step {step:5d} loss {loss.item():.4f} elapsed {time.time() - start:.1f}s")
            logger.progress(step, loss=loss.item())
        if step % args.eval_every == 0:
            acc = evaluate(model, eval_ds, args, device, n_step=args.n_step)
            temp_str = f" temperature={model.memory.temperature.item():.3f}" if args.sharpened else ""
            print(f"step {step:5d} eval_acc {acc:.4f}{temp_str}")

        elapsed_min = (time.time() - start) / 60
        if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
            break

    final_acc = evaluate(model, eval_ds, args, device, n_step=args.n_step, n_batches=20)
    elapsed = time.time() - start
    print("---")
    print(f"best_loss:        {best_loss_t.item():.6f}")
    print(f"final_acc:        {final_acc:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"sharpened:        {args.sharpened}")
    if args.sharpened:
        print(f"final_temperature: {model.memory.temperature.item():.4f}")
    print(f"seed:             {args.seed}")
    self_match_diagnostic(model, eval_ds, n_facts=max_facts, device=device)

    logger.finish(
        summary={"final_acc": final_acc, "best_loss": best_loss_t.item(),
                 "num_steps": step, "num_params_M": num_params / 1e6,
                 "sharpened": args.sharpened, "training_seconds": elapsed},
        # Ce script n'instrumente pas de taux de fuite : son contrôle est le
        # self_match_diagnostic imprimé juste au-dessus, qui n'est pas réduit
        # à un scalaire. Déclaré None explicitement plutôt qu'omis, pour que
        # collect.py signale la lacune au lieu de la masquer.
        controls={"chance": 1.0 / args.vocab_size,
                  "margin": final_acc - 1.0 / args.vocab_size,
                  "leak_check": None},
    )


if __name__ == "__main__":
    main()
