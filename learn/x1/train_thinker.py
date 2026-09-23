"""X1 M1: Thinker ("Baseline B", disable_kb=True -- spec §9: pure recurrent
loop, no external memory) on the T1/T3 synthetic tasks -- gate G3
(X1_DISPATCH.md §3): compared against M3 (gate G2, already >=95% EM
in-distribution on T3). Per the dispatch decision table, a low G3 EM while
M3 passes is itself a valid, expected result (same defect as hypothesis H8)
-- not a blocker, continue the grid regardless.

Architecture (agreed with experiment-agent, 2026-09-23): no KB to load for
these tasks (the whole "document" IS the query), so kb_tokens = the full
prompt token ids and kb_source_ids = all zeros (kb_tokens/kb_source_ids are
unused anyway when disable_kb=True -- core/indexed_thinker_model.py's
forward() only reads kb_tokens.shape[0] for B/device in that path -- kept
only to satisfy the API). query_tokens = the full prompt too, mean-pooled
into the register's initial state (core/indexed_thinker_model.py Thinker.
forward, `R = register_base + q_emb`) -- this mean-pooling is the actual
bottleneck this baseline is testing: unlike M3/M4's per-position sequence
processing, Thinker/Baseline-B sees the whole problem only as one pooled
vector plus n_step iterations of a shared-weight loop, no self-attention
over the input tokens at all.

A single output stream "answer" (sequence_mode=True, spec §14.3) is
teacher-forced over the answer digits + EOS during training, using the same
pad-filled-placeholder-as-first-query convention as
learn/indexed_attention/generate_qualitative_compare_reasoning.py's
`generate_stream` (target_input[0] = PAD_ID, target_input[t+1] = the
token generated/observed at position t) -- reused verbatim here for greedy
eval/generation, since it already implements exactly this per-position
teacher-forcing convention against a fixed sm_k/sm_v (computed once per
n_step loop, cross-attended by every generation step -- no self-attention
between output positions, so recomputing the full forward per step, while
wasteful, is correct and cheap enough at this model's tiny scale)."""
from __future__ import annotations

import argparse
import random
import time

import torch

from core.indexed_thinker_model import Thinker
from learn.x1.tasks import TASKS, PAD_ID, VOCAB_SIZE, TOK2ID, EOS


def collate(examples, device):
    max_prompt = max(e.prompt_len for e in examples)
    max_ans = max(len(e.target_ids) - e.prompt_len for e in examples)
    B = len(examples)
    prompt_ids = torch.full((B, max_prompt), PAD_ID, dtype=torch.long)
    answer_ids = torch.full((B, max_ans), PAD_ID, dtype=torch.long)
    labels = torch.full((B, max_ans), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        prompt_ids[i, :e.prompt_len] = torch.tensor(e.input_ids[:e.prompt_len])
        ans = e.target_ids[e.prompt_len:]
        n = len(ans)
        answer_ids[i, :n] = torch.tensor(ans)
        labels[i, :n] = torch.tensor(ans)
    target_input = torch.full((B, max_ans), PAD_ID, dtype=torch.long)
    target_input[:, 1:] = answer_ids[:, :-1]
    return {k: v.to(device) for k, v in
            {"prompt_ids": prompt_ids, "answer_ids": answer_ids, "labels": labels,
             "target_input": target_input}.items()}


@torch.no_grad()
def greedy_generate_batch(model, prompt_ids, n_step: int, max_len: int, eos_id: int, device):
    """Same convention as generate_qualitative_compare_reasoning.py's
    generate_stream: target_input[0] = PAD_ID, recompute the full forward
    (including the n_step core loop) at every position -- sm_k/sm_v only
    depend on prompt_ids/n_step, not on target_input, so this is wasteful
    but correct."""
    B = prompt_ids.shape[0]
    kb_source_ids = torch.zeros_like(prompt_ids)
    target_input = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    generated = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for t in range(max_len):
        _, stream_outputs = model(kb_tokens=prompt_ids, kb_source_ids=kb_source_ids,
                                   query_tokens=prompt_ids, n_step=n_step,
                                   target_input={"answer": target_input})
        next_token = stream_outputs["answer"][:, t, :].argmax(dim=-1)
        next_token = torch.where(done, torch.full_like(next_token, PAD_ID), next_token)
        generated[:, t] = next_token
        done = done | (next_token == eos_id)
        if t + 1 < max_len:
            target_input[:, t + 1] = next_token
        if bool(done.all()):
            break
    return generated


def exact_match_eval(model, examples, device, eos_id, n_step: int, max_new_tokens=40):
    batch = collate(examples, device)
    generated = greedy_generate_batch(model, batch["prompt_ids"], n_step, max_new_tokens, eos_id, device)
    correct = 0
    for i, e in enumerate(examples):
        target = e.target_ids[e.prompt_len:]
        target_no_eos = [t for t in target if t != eos_id]
        pred = generated[i].tolist()
        if eos_id in pred:
            pred = pred[:pred.index(eos_id)]
        pred = [p for p in pred if p != PAD_ID]
        if pred == target_no_eos:
            correct += 1
    return correct / len(examples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--train_size_range", required=True, help="e.g. 32,32")
    ap.add_argument("--test_size_range", required=True, help="e.g. 64,512")
    ap.add_argument("--position_offset_max", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_register", type=int, default=8)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--pool_n_head", type=int, default=4)
    ap.add_argument("--max_answer_len", type=int, default=48)
    ap.add_argument("--n_step_train_max", type=int, default=8, help="upper bound of the per-step "
                     "random.randint(1, n) curriculum")
    ap.add_argument("--n_step_test", type=int, default=None, help="default: n_step_train_max")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--max_time_minutes", type=float, default=60.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    n_step_test = args.n_step_test or args.n_step_train_max

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    gen_fn = TASKS[args.task]
    train_lo, train_hi = map(int, args.train_size_range.split(","))
    test_lo, test_hi = map(int, args.test_size_range.split(","))
    eos_id = TOK2ID[EOS]

    model = Thinker(
        vocab_size=VOCAB_SIZE, d_model=args.d_model, n_register=args.n_register,
        block_size=2, depth=0, n_head=args.n_head, pool_n_head=args.pool_n_head,
        disable_kb=True, stream_dims={"answer": VOCAB_SIZE},
        stream_n_layers={"answer": 1}, stream_sequence={"answer": True},
        max_target_len=args.max_answer_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model (M1 Thinker, disable_kb=True/Baseline B): {n_params / 1e6:.2f}M params "
          f"d_model={args.d_model} n_register={args.n_register} n_step_train_max={args.n_step_train_max}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    step = 0
    best_id_em = 0.0
    while step < args.max_steps and (time.time() - start) / 60 < args.max_time_minutes:
        batch_examples = gen_fn(args.batch_size, (train_lo, train_hi), seed=args.seed * 1_000_003 + step,
                                 position_offset_max=args.position_offset_max)
        batch = collate(batch_examples, device)
        n_step = random.randint(1, args.n_step_train_max)
        kb_source_ids = torch.zeros_like(batch["prompt_ids"])
        _, stream_outputs = model(kb_tokens=batch["prompt_ids"], kb_source_ids=kb_source_ids,
                                   query_tokens=batch["prompt_ids"], n_step=n_step,
                                   target_input={"answer": batch["target_input"]})
        logits = stream_outputs["answer"]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch["labels"].reshape(-1),
                                                   ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.4f} n_step={n_step} elapsed={(time.time()-start)/60:.2f}m",
                  flush=True)
        if step % args.eval_every == 0:
            model.eval()
            id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_000 + step, position_offset_max=0)
            id_em = exact_match_eval(model, id_examples, device, eos_id, n_step_test, args.max_answer_len)
            print(f"step={step} IN-DIST EM={id_em:.4f} (n_step_test={n_step_test})", flush=True)
            model.train()
            if id_em > best_id_em:
                best_id_em = id_em
                if args.save_dir:
                    import os
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pt")

    model.eval()
    id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_001, position_offset_max=0)
    id_em = exact_match_eval(model, id_examples, device, eos_id, n_step_test, args.max_answer_len)
    ood_examples = gen_fn(args.n_eval, (test_lo, test_hi), seed=999_002, position_offset_max=0)
    ood_em = exact_match_eval(model, ood_examples, device, eos_id, n_step_test, args.max_answer_len)
    print(f"FINAL in-distribution EM={id_em:.4f} ({args.train_size_range}) n_step_test={n_step_test}")
    print(f"FINAL OOD EM={ood_em:.4f} ({args.test_size_range}) n_step_test={n_step_test}")
    print(f"best_in_dist_em_during_training={best_id_em:.4f}")


if __name__ == "__main__":
    main()
