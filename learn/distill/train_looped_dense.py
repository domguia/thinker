"""E3 (paper adversarial-review priority, 2026-09-23): a dense transformer
baseline with a SINGLE shared block reused across `n_step` recurrent
iterations -- like Thinker's core (one set of weights, applied n_step times),
but with a plain GPT2-style self-attention block instead of Thinker's
register/KB/output-stream machinery. Tests whether the calibration-collapse
(agent2's teacher-forced argmax diagnostic, ~0.3-0.5% on every Thinker
variant, 21.2% on the non-recurrent Baseline C) is a general property of
weight-shared recurrence, or specific to Thinker's architecture.

Design (approved by supervisor-agent before implementation):
- Same base config as Baseline C (`train_sft.py`): gpt2, n_embd=256, n_head=4,
  vocab from --tokenizer.
- Build with n_layer=--n_step_train_max GPT2Blocks, then ALIAS blocks 1..N-1
  to literally be the SAME nn.Module object as block 0 (weight tying, not a
  copy) -- so there is only ONE distinct block's worth of trainable
  parameters, reused every iteration. `model.parameters()` already
  deduplicates by object identity (same mechanism as HF's tied
  embed/lm_head), so the reported param count is correct without extra
  bookkeeping.
- Per training step, `n_step = random.randint(1, args.n_step_train_max)`
  (exact copy of train_prompt_response.py's own curriculum line) -- the
  transformer's ModuleList is temporarily SLICED to the first n_step entries
  (all aliases of the same block) before the forward call, then restored to
  the full list before checkpointing, so the saved checkpoint always has a
  complete, loadable n_layer=--n_step_train_max structure (needed by
  diagnose_generation_divergence_flat.py's own model construction, which
  expects a full config-shaped checkpoint).
- CE-only (same deviation as Baseline C: fast diagnostic control, not a
  production run -- flagged per CLAUDE.md's KD-by-default policy).
- Param count is NOT exactly matched to Baseline C (68.5M: 5.0M core +
  63.5M head) -- with only 1 distinct block instead of 6, core drops to
  ~0.83M, total ~64.3M (93.9% of Baseline C). The vocab head dominates both,
  so the difference is small and reported transparently rather than chasing
  an exact match via width-rescaling (judged too risky/slow for the deadline,
  approved as an explicit tradeoff).
"""
import argparse
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core.model_families import resolve_model_name
from learn.distill.train_sft import JsonlTextDataset, collate, save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", default=None)
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--base_config", default="gpt2")
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_step_train_max", type=int, default=8, help="ModuleList aliased to this many "
                     "entries; also the upper bound of the per-step random.randint(1, n) curriculum")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--max_time_minutes", type=float, default=150.0)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_config)
    config.n_layer = args.n_step_train_max
    config.n_embd = args.n_embd
    config.n_head = args.n_head
    config.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(config)

    # Weight-tie every block to block 0 -- same underlying nn.Module/Parameters,
    # not copies. model.parameters() dedupes by object identity, so the
    # trainable param count reflects ONE shared block, matching Thinker's
    # "one core reused across n_step iterations" design.
    base_block = model.transformer.h[0]
    for i in range(1, args.n_step_train_max):
        model.transformer.h[i] = base_block
    full_h = model.transformer.h
    model.to(device)

    core_params = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "lm_head" not in n)
    head_params = sum(p.numel() for n, p in model.named_parameters() if "wte" in n or "lm_head" in n)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params total = {core_params / 1e6:.1f}M core (1 shared block) + "
          f"{head_params / 1e6:.1f}M head (random init, arch={args.base_config}, looped up to "
          f"n_step_train_max={args.n_step_train_max}), device={device}")

    train_ds = JsonlTextDataset(args.train_file, tokenizer, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    print(f"Train examples: {len(train_ds)}")

    val_loader = None
    if args.val_file and args.val_every:
        val_ds = JsonlTextDataset(args.val_file, tokenizer, args.block_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
        print(f"Val examples: {len(val_ds)} (evaluated every {args.val_every} steps)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    @torch.no_grad()
    def evaluate_at_n_step(n_step: int) -> float:
        model.eval()
        model.transformer.h = full_h[:n_step]
        total_ce, total_n = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                out = model(**batch, use_cache=False)
            n_valid = (batch["labels"] != -100).sum().item()
            total_ce += out.loss.item() * n_valid
            total_n += n_valid
        model.transformer.h = full_h
        model.train()
        return total_ce / total_n if total_n else float("nan")

    step, losses, best_loss = 0, [], float("inf")
    start = time.time()
    done = False
    while not done:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            n_step = random.randint(1, args.n_step_train_max)
            model.transformer.h = full_h[:n_step]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                out = model(**batch, use_cache=False)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            model.transformer.h = full_h
            step += 1
            loss_val = out.loss.item()
            losses.append(loss_val)
            if step % args.log_every == 0:
                elapsed = time.time() - start
                print(f"step {step} loss {loss_val:.4f} n_step={n_step} elapsed={elapsed / 60:.2f}m")
                print(f"progress: {{\"step\": {step}, \"loss\": {loss_val}, \"n_step\": {n_step}, "
                      f"\"elapsed_s\": {elapsed:.1f}, \"max_steps\": {args.max_steps}}}", flush=True)
            if val_loader is not None and step % args.val_every == 0:
                val_ce = evaluate_at_n_step(args.n_step_train_max)
                print(f"step {step} VAL n_step={args.n_step_train_max} loss {val_ce:.4f}", flush=True)
                if args.save_dir and val_ce < best_loss:
                    best_loss = val_ce
                    save_checkpoint(f"{args.save_dir}/checkpoint.pt", model, optimizer, step, losses,
                                     args, width_mult=1.0, depth_mult=1.0)
                    print(f"  new best val_loss={val_ce:.4f} -> checkpoint saved to {args.save_dir}/checkpoint.pt")
            if step >= args.max_steps or (time.time() - start) / 60 >= args.max_time_minutes:
                done = True
                print(f"Budget reached at step {step}. Stopping.")
                break

    final_loss = losses[-1] if losses else float("nan")
    training_seconds = time.time() - start
    print(f"final_loss: {final_loss:.4f}")
    print(f"num_steps: {step}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"num_params_M: {num_params / 1e6:.2f}")
    if args.save_dir:
        model.transformer.h = full_h
        save_checkpoint(f"{args.save_dir}/checkpoint.pt", model, optimizer, step, losses, args,
                         width_mult=1.0, depth_mult=1.0)
        print(f"final checkpoint saved to {args.save_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
