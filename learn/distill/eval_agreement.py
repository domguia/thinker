"""Evaluate a distilled student against the Teacher on held-out data, using
distillation-specific metrics instead of a downstream benchmark.

Why not just compare to a benchmark (MMLU, GSM8K, ...) or to the Teacher's
own benchmark scores directly: these student sizes (tens to hundreds of M
core params) trained on a tiny validation slice are nowhere near benchmark-
capable yet, and comparing a small student's absolute task score to a 27B
Teacher's isn't apples-to-apples anyway. What IS answerable now, cheaply, is
"how well did the distillation itself transfer" -- does the student's output
distribution track the Teacher's on data it wasn't trained on:

- Top-1 agreement rate: fraction of held-out token positions where the
  student's argmax prediction matches the Teacher's top (stored Top-K index
  0, since torch.topk returns indices sorted by value descending).
- Mean KL(Teacher || Student): the same (K+1)-way KL used as the KD training
  loss (see train_sft.py's topk_kd_loss), but computed on held-out data with
  no gradient step -- distinguishes "generalized the Teacher's distribution"
  from "memorized the training KD targets".

Needs a --checkpoint from train_sft.py's --save_dir, and --teacher_targets
precomputed (via precompute_teacher_targets.py) for the SAME --eval_file --
these must be a held-out split the checkpoint was never trained on.

Example:
    python learn/distill/eval_agreement.py \
      --checkpoint runs/500Mcore/checkpoint.pt \
      --eval_file data/distill_cluster_run/reasoning/val_sample.jsonl \
      --teacher_targets data/distill_cluster_run/reasoning/val_sample_topk32.npz
"""
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train_sft import JsonlTextDataset, apply_depth_mup_scaling, collate, topk_kd_loss


def rebuild_model(saved_args, tokenizer, state_dict):
    """Reconstructs the exact architecture train_sft.py's main() built at
    training time, from the args it saved in the checkpoint -- so eval uses
    the identical config (n_layer/n_embd/n_head, vocab_size, tied/untied
    head) the weights were actually trained under.
    """
    config = AutoConfig.from_pretrained(saved_args["base_config"])
    for attr in ("n_layer", "n_embd", "n_head"):
        val = saved_args.get(attr)
        if val is not None:
            setattr(config, attr, val)
    config.vocab_size = len(tokenizer)
    if saved_args.get("mup") and saved_args.get("mup_untie_head"):
        config.tie_word_embeddings = False
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="a train_sft.py --save_dir checkpoint.pt")
    parser.add_argument("--eval_file", required=True, help="held-out JSONL, NOT used for training the checkpoint")
    parser.add_argument("--teacher_targets", required=True, help="precompute_teacher_targets.py .npz output for --eval_file")
    parser.add_argument("--tokenizer", default=None, help="override the tokenizer path stored in the checkpoint's args")
    parser.add_argument("--max_examples", type=int, default=None, help="limit how many held-out examples to evaluate (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt["args"]
    width_mult = ckpt["width_mult"]
    depth_mult = ckpt["depth_mult"]

    tokenizer_path = args.tokenizer or saved_args["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = rebuild_model(saved_args, tokenizer, ckpt["state_dict"])
    if saved_args.get("depth_mup"):
        apply_depth_mup_scaling(model, depth_mult)  # hooks aren't part of state_dict, must reapply
    model.to(device)
    model.eval()
    print(
        f"Loaded checkpoint: {saved_args['n_layer']}L/{saved_args['n_embd']}D "
        f"(mup={saved_args.get('mup')}, width_mult={width_mult:.2f}, depth_mup={saved_args.get('depth_mup')}, depth_mult={depth_mult:.2f})"
    )

    eval_ds = JsonlTextDataset(args.eval_file, tokenizer, saved_args["block_size"], teacher_targets=args.teacher_targets)
    if args.max_examples is not None:
        eval_ds.examples = eval_ds.examples[: args.max_examples]
    print(f"Eval examples: {len(eval_ds)} (K={eval_ds.k})")

    total_tokens, total_correct, total_kl = 0, 0, 0.0
    with torch.no_grad():
        for example in eval_ds.examples:
            batch = collate([example], tokenizer.pad_token_id, k=eval_ds.k)
            batch = {k: v.to(device) for k, v in batch.items()}
            teacher_indices = batch.pop("teacher_indices")
            teacher_values = batch.pop("teacher_values")
            teacher_residual = batch.pop("teacher_residual")
            teacher_mask = batch.pop("teacher_mask")
            batch.pop("labels")

            logits = model(**batch).logits
            if saved_args.get("mup"):
                logits = logits / width_mult

            kl = topk_kd_loss(logits, teacher_indices, teacher_values, teacher_residual, teacher_mask)
            n_tokens = teacher_mask.sum().item()
            total_kl += kl.item() * n_tokens
            total_tokens += n_tokens

            student_top1 = logits.argmax(dim=-1)  # (1, T)
            teacher_top1 = teacher_indices[..., 0]  # torch.topk sorts descending -> index 0 is the Teacher's own argmax
            correct = ((student_top1 == teacher_top1) & teacher_mask).sum().item()
            total_correct += correct

    agreement_rate = total_correct / total_tokens if total_tokens else float("nan")
    mean_kl = total_kl / total_tokens if total_tokens else float("nan")
    print("---")
    print(f"eval_tokens:        {total_tokens}")
    print(f"top1_agreement:     {agreement_rate:.4f}")
    print(f"mean_kl_vs_teacher: {mean_kl:.4f}")


if __name__ == "__main__":
    main()
