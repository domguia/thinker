"""
Causal control for the noctx/retrieval budget-escalation inversion
(2026-09-20, requested by model-design): does a trained retrieval checkpoint
actually USE the document content, or does the gain over noctx come from
processing documents slowing down memorization (a regularization/compute-
load effect) rather than real retrieval?

Protocol: load a checkpoint trained with --dataset_type retrieval, run the
same val_answer evaluation twice --
  (a) real documents (baseline, matches the training-time val curve), and
  (b) documents SHUFFLED ACROSS EXAMPLES within each batch (same real
      HotpotQA text, same length/structure, but no longer relevant to that
      example's question) -- a cheap, faithful "irrelevant documents" swap
      since it reuses genuine documents rather than synthesizing fake ones.
A clear degradation under (b) means the model is reading document content;
little change means the gain is not from real retrieval.

Usage:
    python -m learn.indexed_attention.eval_causal_control \
      --checkpoint checkpoints/noctx_budget_ext24k_seed0_retrieval.pt \
      --val_data data/distill/hotpotqa/val.jsonl \
      --d_model 256 --n_head 4 --n_step 4 --block_size 64 --n_docs_max 10 --max_answer_len 32
(all model-shape args must match the checkpoint's training config exactly.)
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import RetrievalPromptDataset


def shuffle_documents(batch, block_size: int, n_docs_max: int, target: str = "all"):
    """Replace document blocks with another example's (same batch, random
    permutation with no fixed points when batch_size > 1) -- keeps the
    question/answer/labels untouched, only the targeted KB content changes.

    target: "all" (every document block, the coarse control) / "supporting"
    (only gold supporting-fact blocks, per-example `is_supporting` mask) /
    "distractor" (only non-supporting blocks) -- model-design's fine-grained
    control (2026-09-20): if corrupting "supporting" hurts much more than
    corrupting "distractor", that's evidence of TARGETED retrieval rather
    than generic sensitivity to any coherent text.
    """
    b = batch["kb_tokens"].shape[0]
    if b < 2:
        return batch  # nothing to shuffle against
    perm = torch.randperm(b)
    while (perm == torch.arange(b)).any():  # avoid any example mapping to itself
        perm = torch.randperm(b)
    out = {k: v.clone() for k, v in batch.items()}
    for i in range(n_docs_max):
        start, end = i * block_size, (i + 1) * block_size
        if target == "supporting":
            sel = batch["is_supporting"][:, i]
        elif target == "distractor":
            sel = ~batch["is_supporting"][:, i]
        else:
            sel = torch.ones(b, dtype=torch.bool)
        if not sel.any():
            continue
        out["kb_tokens"][sel, start:end] = batch["kb_tokens"][perm][sel, start:end]
        out["kb_leaf_mask"][sel, start:end] = batch["kb_leaf_mask"][perm][sel, start:end]
    # kb_source_ids unchanged (still marks doc-block positions as KB=1, question stays local=0)
    return out


@torch.no_grad()
def run_eval(model, loader, device, block_size, n_docs_max, n_step, shuffle: str = None, n_batches=None):
    """shuffle: None (real documents) / "all" / "supporting" / "distractor"."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if n_batches and i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if shuffle:
            batch = shuffle_documents(batch, block_size, n_docs_max, target=shuffle)
        query_tokens = batch["kb_tokens"][:, -block_size:]
        target_input = {"answer": batch["answer_target_input"]}
        _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, n_step,
                            kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)
        ce = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
        losses.append(ce.item())
    return sum(losses) / len(losses) if losses else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--tokenizer", default="lfm2")
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--n_head", type=int, required=True)
    p.add_argument("--n_step", type=int, required=True)
    p.add_argument("--block_size", type=int, required=True)
    p.add_argument("--n_docs_max", type=int, required=True)
    p.add_argument("--max_answer_len", type=int, required=True)
    p.add_argument("--n_register", type=int, default=1)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--n_slots", type=int, default=1)
    p.add_argument("--pool_n_head", type=int, default=1)
    p.add_argument("--k_dim", type=int, default=None)
    p.add_argument("--answer_n_layers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_batches", type=int, default=None, help="cap eval batches; default = whole val set")
    p.add_argument("--fine_grained", action="store_true",
                   help="also run supporting-only and distractor-only corruption (needs is_supporting in data)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    vocab_size = len(tok)

    val_ds = RetrievalPromptDataset(args.val_data, tok, block_size=args.block_size,
                                     n_docs_max=args.n_docs_max, max_answer_len=args.max_answer_len,
                                     pad_id=tok.pad_token_id)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"loaded {len(val_ds)} val examples from {args.val_data}", flush=True)

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        stream_dims={"answer": vocab_size}, stream_sequence={"answer": True},
        max_target_len=args.max_answer_len, stream_n_layers={"answer": args.answer_n_layers},
    ).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"loaded checkpoint {args.checkpoint} ({n_params/1e6:.2f}M params)", flush=True)

    real = run_eval(model, val_loader, args.device, args.block_size, args.n_docs_max, args.n_step,
                     shuffle=None, n_batches=args.n_batches)
    shuffled_all = run_eval(model, val_loader, args.device, args.block_size, args.n_docs_max, args.n_step,
                             shuffle="all", n_batches=args.n_batches)
    print(f"val_answer (real documents)         : {real:.4f}", flush=True)
    print(f"val_answer (all docs shuffled)      : {shuffled_all:.4f}", flush=True)
    print(f"degradation (all - real)            : {shuffled_all - real:.4f}", flush=True)

    if args.fine_grained:
        shuffled_supporting = run_eval(model, val_loader, args.device, args.block_size, args.n_docs_max,
                                        args.n_step, shuffle="supporting", n_batches=args.n_batches)
        shuffled_distractor = run_eval(model, val_loader, args.device, args.block_size, args.n_docs_max,
                                        args.n_step, shuffle="distractor", n_batches=args.n_batches)
        print(f"val_answer (supporting shuffled only) : {shuffled_supporting:.4f}", flush=True)
        print(f"val_answer (distractor shuffled only) : {shuffled_distractor:.4f}", flush=True)
        print(f"degradation (supporting - real)       : {shuffled_supporting - real:.4f}", flush=True)
        print(f"degradation (distractor - real)       : {shuffled_distractor - real:.4f}", flush=True)


if __name__ == "__main__":
    main()
