"""X1 M4: dense (non-recurrent) transformer baseline on the T1/T3 synthetic
tasks -- gate G1 (X1_DISPATCH.md §3): must reach >=95% EM in-distribution
before the rest of the grid is trusted (rules out a data/eval bug before
blaming any model architecture).

Tiny custom vocab (17 tokens, see tasks.py) means the embedding/head no
longer dominates params (unlike every qwen35-vocab experiment this session)
-- model size is controlled directly via n_layer/n_embd for the ~5-20M target
(X1_DISPATCH.md §2). Explicit place-value `position_ids` (not the default
sequential ones) are passed through GPT2Model at every step, both during
training (teacher-forced) and generation (manual loop, continuing the
place-value sequence -- HF's own .generate() auto-increments position_ids
sequentially, incompatible with this scheme).
"""
from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from learn.x1.tasks import TASKS, PAD_ID, VOCAB_SIZE, ID2TOK


def collate(examples, device):
    max_len = max(len(e.input_ids) for e in examples)
    B = len(examples)
    input_ids = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    position_ids = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e.input_ids)
        input_ids[i, :n] = torch.tensor(e.input_ids)
        position_ids[i, :n] = torch.tensor(e.position_ids)
        labels[i, :n] = torch.tensor(e.target_ids)
        attention_mask[i, :n] = 1
    return {k: v.to(device) for k, v in
            {"input_ids": input_ids, "position_ids": position_ids, "labels": labels,
             "attention_mask": attention_mask}.items()}


@torch.no_grad()
def greedy_generate(model, prompt_ids: list, prompt_pos: list, max_new_tokens: int, eos_id: int, device):
    """Manual greedy loop with explicit place-value position ids (continuing
    the same 0,1,2,... sequence the prompt's answer positions would have used
    -- see tasks.py's target_pos construction, always starts back at 0)."""
    ids = torch.tensor([prompt_ids], device=device)
    generated = []
    next_pos = 0
    for _ in range(max_new_tokens):
        pos = torch.tensor([prompt_pos + [p for p in range(next_pos)]], device=device) if generated else \
              torch.tensor([prompt_pos], device=device)
        out = model(input_ids=ids, position_ids=pos, use_cache=False)
        next_id = out.logits[0, -1].argmax().item()
        if next_id == eos_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        next_pos += 1
    return generated


def exact_match_eval(model, examples, device, eos_id, max_new_tokens=40):
    correct = 0
    for e in examples:
        target = e.target_ids[e.prompt_len:]
        target_no_eos = [t for t in target if t != eos_id]
        prompt_ids = e.input_ids[:e.prompt_len]
        prompt_pos = e.position_ids[:e.prompt_len]
        pred = greedy_generate(model, prompt_ids, prompt_pos, max_new_tokens, eos_id, device)
        if pred == target_no_eos:
            correct += 1
    return correct / len(examples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--train_size_range", required=True, help="e.g. 1,20")
    ap.add_argument("--test_size_range", required=True, help="e.g. 21,100")
    ap.add_argument("--position_offset_max", type=int, default=0)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=192)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_positions", type=int, default=600)
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    gen_fn = TASKS[args.task]
    train_lo, train_hi = map(int, args.train_size_range.split(","))
    test_lo, test_hi = map(int, args.test_size_range.split(","))
    eos_id = None
    from learn.x1.tasks import TOK2ID, EOS
    eos_id = TOK2ID[EOS]

    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = args.n_layer
    config.n_embd = args.n_embd
    config.n_head = args.n_head
    config.vocab_size = VOCAB_SIZE
    config.n_positions = args.n_positions
    config.n_ctx = args.n_positions
    model = AutoModelForCausalLM.from_config(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model (M4 dense): {n_params / 1e6:.2f}M params, n_layer={args.n_layer} n_embd={args.n_embd}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    step = 0
    best_id_em = 0.0
    while step < args.max_steps and (time.time() - start) / 60 < args.max_time_minutes:
        batch_examples = gen_fn(args.batch_size, (train_lo, train_hi), seed=args.seed * 1_000_003 + step,
                                 position_offset_max=args.position_offset_max)
        batch = collate(batch_examples, device)
        out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                     attention_mask=batch["attention_mask"], labels=batch["labels"])
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        step += 1
        if step % 20 == 0:
            print(f"step={step} loss={out.loss.item():.4f} elapsed={(time.time()-start)/60:.2f}m", flush=True)
        if step % args.eval_every == 0:
            model.eval()
            id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_000 + step, position_offset_max=0)
            id_em = exact_match_eval(model, id_examples, device, eos_id)
            print(f"step={step} IN-DIST EM={id_em:.4f}", flush=True)
            model.train()
            if id_em > best_id_em:
                best_id_em = id_em
                if args.save_dir:
                    import os
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pt")

    model.eval()
    id_examples = gen_fn(args.n_eval, (train_lo, train_hi), seed=999_001, position_offset_max=0)
    id_em = exact_match_eval(model, id_examples, device, eos_id)
    ood_examples = gen_fn(args.n_eval, (test_lo, test_hi), seed=999_002, position_offset_max=0)
    ood_em = exact_match_eval(model, ood_examples, device, eos_id)
    print(f"FINAL in-distribution EM={id_em:.4f} ({args.train_size_range})")
    print(f"FINAL OOD EM={ood_em:.4f} ({args.test_size_range})")
    print(f"best_in_dist_em_during_training={best_id_em:.4f}")


if __name__ == "__main__":
    main()
