"""
Trains Thinker (core/indexed_thinker_model.py) on prompt/thinking/answer
data (data/prompt_response_dataset.py) -- distinct from train_real_text.py's
continuous sliding-window scheme, per user request (2026-09-20): our model
does not ingest data the way a plain decoder-only LLM does (every position
learning on its whole context); it needs an explicit prompt (fed as KB),
and a response -- optionally split into a `thinking` stream (reasoning
trace) and an `answer` stream (final answer), each Thinker's own output
stream with its own teacher-forced target (core/indexed_thinker_model.py's
per-stream target_input, added for exactly this).

Two of the project's three data.distill/prepare_*_data.py sources have
this structure and are covered here (`--dataset_type`):
  - reasoning (learn/distill/prepare_reasoning_data.py, OpenR1-Math-220k):
    ReasoningPromptDataset, TWO streams (thinking + answer).
  - retrieval (learn/distill/prepare_retrieval_data.py, HotpotQA):
    RetrievalPromptDataset, ONE stream (answer only, no reasoning trace
    available in this dataset).
The third source (general, WikiText/TinyStories) has no prompt/response
structure -- stays on train_real_text.py's sliding-window pipeline,
unchanged, not duplicated here.

Each example is an independent episode (no cross-example document
continuity, unlike real-text's LockstepLaneBatcher/register carry-over) --
a plain shuffled DataLoader, register always starts fresh from
register_init. This is deliberately simpler than train_real_text.py.

KD (logit-level, on both streams) is NOT wired here yet -- landing the
CE-only pipeline first per the user's own request to test each dataset
individually before layering KD and joint training on top. See
dev_notes/experiments/real_text_baselines.md, 2026-09-20 entries, for the
staged plan and why KD alignment is more involved for this data shape
(char-to-token offset mapping against precompute_teacher_targets.py's
per-position arrays) than train_real_text.py's fixed-window-position case.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args
from data.prompt_response_dataset import ReasoningPromptDataset, RetrievalPromptDataset


def build_dataset(dataset_type: str, path: str, tokenizer, args):
    if dataset_type == "reasoning":
        return ReasoningPromptDataset(path, tokenizer, n_ctx=args.n_ctx,
                                       max_thinking_len=args.max_thinking_len,
                                       max_answer_len=args.max_answer_len, pad_id=tokenizer.pad_token_id)
    if dataset_type == "retrieval":
        # block_size/n_docs_max, NOT n_ctx/t_local (2026-09-20 redesign, user
        # decision: treat HotpotQA's multiple documents as distinct indexable
        # blocks, depth=1, rather than one flattened truncated window --
        # see RetrievalPromptDataset's docstring). --block_size here is the
        # SAME value as Thinker's own --block_size constructor arg (must
        # match for HierarchicalMemory's depth=1 to compress one node per
        # document correctly).
        return RetrievalPromptDataset(path, tokenizer, block_size=args.block_size, n_docs_max=args.n_docs_max,
                                       max_answer_len=args.max_answer_len, pad_id=tokenizer.pad_token_id)
    raise ValueError(f"unknown --dataset_type {dataset_type!r} (expected 'reasoning' or 'retrieval' -- "
                      f"'general' stays on train_real_text.py's sliding-window pipeline, not this script)")


def query_tokens_for(dataset_type: str, batch, block_size: int):
    """What seeds the register (spec: mean-pooled embedding added to
    register_init) -- real-text uses the last t_local context tokens
    (recency); here, the natural analogue per dataset:
      - retrieval: the question's own block (the LAST block_size positions
        of kb_tokens, see RetrievalPromptDataset -- always the question,
        regardless of n_docs_max).
      - reasoning: the WHOLE problem (no local/long-range split there --
        source_id is uniformly 0, see ReasoningPromptDataset)."""
    if dataset_type == "retrieval":
        return batch["kb_tokens"][:, -block_size:]
    return batch["kb_tokens"]


@torch.no_grad()
def evaluate(model, loader, device, dataset_type: str, n_step: int, block_size: int, n_batches: int = 20):
    model.eval()
    losses = {"answer": [], "thinking": []} if dataset_type == "reasoning" else {"answer": []}
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        query_tokens = query_tokens_for(dataset_type, batch, block_size)
        target_input = {"answer": batch["answer_target_input"]}
        if dataset_type == "reasoning":
            target_input["thinking"] = batch["thinking_target_input"]
        _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, n_step,
                            kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)
        ce_answer = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
        losses["answer"].append(ce_answer.item())
        if dataset_type == "reasoning":
            ce_thinking = F.cross_entropy(streams["thinking"].transpose(1, 2), batch["thinking_labels"], ignore_index=-100)
            losses["thinking"].append(ce_thinking.item())
    model.train()
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in losses.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_type", required=True, choices=["reasoning", "retrieval"])
    p.add_argument("--data", required=True)
    p.add_argument("--val_data", default=None)
    p.add_argument("--tokenizer", default="lfm2")
    p.add_argument("--n_ctx", type=int, default=256, help="reasoning only: prompt length (flat, depth=0)")
    p.add_argument("--n_docs_max", type=int, default=10,
                   help="retrieval only: max context documents per example, each its own block_size-token "
                        "block -- HotpotQA distractor config has ~10 (2 supporting + up to 8 distractors)")
    p.add_argument("--max_thinking_len", type=int, default=1024, help="reasoning only")
    p.add_argument("--max_answer_len", type=int, default=64)
    p.add_argument("--depth", type=int, default=0,
                   help="reasoning: 0 (flat, single prompt block, no multi-document structure to index). "
                        "retrieval: MUST be 1 -- see RetrievalPromptDataset, one summary node per document.")
    p.add_argument("--block_size", type=int, default=16,
                   help="reasoning: HierarchicalMemory's block_size (depth=0 -> unused). "
                        "retrieval: tokens per document block -- MUST match what RetrievalPromptDataset "
                        "used to build kb_tokens (same --block_size value drives both).")
    p.add_argument("--n_register", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=1)
    p.add_argument("--n_step", type=int, default=6)
    p.add_argument("--pool_n_head", type=int, default=1)
    p.add_argument("--k_dim", type=int, default=None)
    p.add_argument("--disable_kb", action="store_true")
    p.add_argument("--use_ff", action="store_true")
    p.add_argument("--ff_hidden_mult", type=int, default=4)
    p.add_argument("--thinking_weight", type=float, default=1.0,
                   help="reasoning only: loss = ce_answer + thinking_weight * ce_thinking")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_warmup_init", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--extrapolate_n_steps", default=None,
                   help="comma-separated n_step_test values, probed on held-out --val_data after training "
                        "-- does 'thinking longer' at inference help solve reasoning/retrieval examples "
                        "it was never trained with that many steps on?")
    p.add_argument("--save_checkpoint_path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_args(p)
    args = p.parse_args()
    logger = logger_from_args(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = build_dataset(args.dataset_type, args.data, tok, args)
    print(f"loaded {len(train_ds)} {args.dataset_type} examples from {args.data}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data:
        val_ds = build_dataset(args.dataset_type, args.val_data, tok, args)
        print(f"loaded held-out val: {len(val_ds)} examples from {args.val_data}", flush=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.dataset_type == "reasoning":
        stream_dims = {"thinking": vocab_size, "answer": vocab_size}
        stream_sequence = {"thinking": True, "answer": True}
        max_target_len = max(args.max_thinking_len, args.max_answer_len)
    else:
        stream_dims = {"answer": vocab_size}
        stream_sequence = {"answer": True}
        max_target_len = args.max_answer_len

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"dataset_type={args.dataset_type} d_model={args.d_model} n_step={args.n_step} "
          f"params={n_params/1e6:.2f}M device={device}", flush=True)

    lr_warmup_init = args.lr_warmup_init if args.lr_warmup_init is not None else args.lr / 10

    def lr_at(step: int) -> float:
        if args.lr_warmup_steps <= 0 or step >= args.lr_warmup_steps:
            return args.lr
        return lr_warmup_init + (args.lr - lr_warmup_init) * (step / args.lr_warmup_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=1e-2)
    model.train()

    step, loss_hist = 0, []
    start_time = time.time()
    done = False
    while not done:
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed > args.max_time_minutes * 60 or step >= args.max_steps:
                print(f"Budget reached at step {step}. Stopping.", flush=True)
                done = True
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            query_tokens = query_tokens_for(args.dataset_type, batch, args.block_size)
            target_input = {"answer": batch["answer_target_input"]}
            if args.dataset_type == "reasoning":
                target_input["thinking"] = batch["thinking_target_input"]

            _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, args.n_step,
                               kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)

            ce_answer = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
            if args.dataset_type == "reasoning":
                ce_thinking = F.cross_entropy(streams["thinking"].transpose(1, 2), batch["thinking_labels"], ignore_index=-100)
                loss = ce_answer + args.thinking_weight * ce_thinking
            else:
                ce_thinking = None
                loss = ce_answer

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_hist.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
                mean_loss = sum(loss_hist[-args.log_every:]) / len(loss_hist[-args.log_every:])
                extra = f" ce_thinking={ce_thinking.item():.4f}" if ce_thinking is not None else ""
                print(f"step={step:6d} elapsed={elapsed/60:.2f}m loss={mean_loss:.4f} "
                      f"ce_answer={ce_answer.item():.4f}{extra} lr={lr_at(step):.2e}", flush=True)
                logger.progress(step, loss=mean_loss, ce_answer=ce_answer.item(), lr=lr_at(step))
            if val_loader is not None and (step % args.val_every == 0 or step == 1):
                val_losses = evaluate(model, val_loader, device, args.dataset_type, args.n_step,
                                       args.block_size, n_batches=args.val_batches)
                print(f"step={step:6d} VAL {val_losses}", flush=True)
                logger.progress(step, **{f"val_{k}": v for k, v in val_losses.items()})

    elapsed = time.time() - start_time
    print("---", flush=True)
    print(f"final_loss: {sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1):.4f}", flush=True)
    print(f"num_steps: {step}", flush=True)
    print(f"training_seconds: {elapsed:.1f}", flush=True)

    if args.save_checkpoint_path:
        import os
        ckpt_path = args.save_checkpoint_path
        if ckpt_path.endswith("/") or os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_path = os.path.join(ckpt_path, f"{args.run_id or 'adhoc'}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint saved to {ckpt_path}", flush=True)

    extrapolation_results = {}
    if args.extrapolate_n_steps and val_loader is not None:
        print("--- extrapolation probe (n_step_test vs training n_step), held-out ---", flush=True)
        for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
            r = evaluate(model, val_loader, device, args.dataset_type, n_step_test, args.block_size, n_batches=args.val_batches)
            extrapolation_results[n_step_test] = r
            marker = " <- training n_step" if n_step_test == args.n_step else ""
            print(f"  n_step_test={n_step_test:3d} {r}{marker}", flush=True)

    logger.finish(summary={
        "final_loss": sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1),
        "num_steps": step, "training_seconds": elapsed,
        "extrapolation": extrapolation_results or None,
    })


if __name__ == "__main__":
    main()
