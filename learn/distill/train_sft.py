"""Minimal from-scratch SFT/KD loop for distillation onboarding.

Trains a RANDOMLY INITIALIZED model (architecture only sourced from
--base_config, weights are never loaded) on the "text" field of a JSONL file
produced by any of the scripts/prepare_*_data.py scripts (reasoning /
general / retrieval all share that field, so this loop is dataset-agnostic).

Two modes:
- Plain Sequence-level SFT (default, no --teacher_targets): next-token
  cross-entropy only. See raw/Distill-getting-start.md for why we start here
  before introducing logit-level KD.
- Logit-level KD (--teacher_targets pointing to a precompute_teacher_targets.py
  .npz output): adds a KL term against the Teacher's Top-K logits, restricted
  to a (K+1)-way categorical over the Teacher's Top-K token indices plus one
  merged "everything else" bucket (reconstructed from the stored residual
  log-sum-exp) -- the only distribution shape recoverable from Top-K-only
  storage. Requires --tokenizer to be the SAME tokenizer used to produce
  --teacher_targets (so token positions and vocab indices line up), and
  --block_size to match (or be <=) the --max_length used for that precompute
  run, since alignment is purely positional (no re-tokenization/alignment
  step is done here).

Output format at the end matches program.md's convention (best_loss /
training_seconds / num_steps / num_params_M) for consistency with the rest
of the repo's experiment scripts.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class JsonlTextDataset(Dataset):
    """Tokenizes "text" and, if teacher_targets is given, attaches that
    example's precomputed Top-K indices/values/residual (sliced from the
    flat (total_tokens, ...) npz arrays via `offsets`). Truncates to the
    shorter of block_size and the precomputed token count so student/teacher
    sequences always line up position-for-position.
    """

    def __init__(self, path, tokenizer, block_size, teacher_targets=None):
        self.examples = []
        teacher = None
        if teacher_targets is not None:
            npz = np.load(teacher_targets)
            teacher = {
                "indices": npz["indices"], "values": npz["values"],
                "residual": npz["residual"], "offsets": npz["offsets"],
            }
            self.k = int(npz["k"])

        with open(path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                ids = tokenizer(row["text"], truncation=True, max_length=block_size)["input_ids"]
                target = None
                if teacher is not None:
                    start, end = teacher["offsets"][i], teacher["offsets"][i + 1]
                    n = min(len(ids), end - start)
                    if n < 2:
                        continue
                    ids = ids[:n]
                    target = {
                        "indices": teacher["indices"][start:start + n],
                        "values": teacher["values"][start:start + n],
                        "residual": teacher["residual"][start:start + n],
                    }
                if len(ids) >= 2:
                    self.examples.append((ids, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id, k=None):
    ids_batch = [ids for ids, _ in batch]
    max_len = max(len(x) for x in ids_batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        labels[i, :n] = torch.tensor(ids)
        attention_mask[i, :n] = 1
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if k is not None:
        teacher_indices = torch.zeros((len(batch), max_len, k), dtype=torch.long)
        teacher_values = torch.zeros((len(batch), max_len, k), dtype=torch.float32)
        teacher_residual = torch.zeros((len(batch), max_len), dtype=torch.float32)
        teacher_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, (_, target) in enumerate(batch):
            n = len(target["values"])
            teacher_indices[i, :n] = torch.from_numpy(target["indices"].astype(np.int64))
            teacher_values[i, :n] = torch.from_numpy(target["values"].astype(np.float32))
            teacher_residual[i, :n] = torch.from_numpy(target["residual"].astype(np.float32))
            teacher_mask[i, :n] = True
        out.update(
            teacher_indices=teacher_indices, teacher_values=teacher_values,
            teacher_residual=teacher_residual, teacher_mask=teacher_mask,
        )
    return out


def topk_kd_loss(student_logits, teacher_indices, teacher_values, teacher_residual, teacher_mask):
    """KL(teacher || student) over the (K+1)-way categorical formed by the
    Teacher's Top-K token indices plus one merged "everything else" bucket.

    This is the only distribution exactly recoverable from Top-K-only
    storage (see precompute_teacher_targets.py's topk_with_residual): the
    residual is the log-sum-exp of every non-Top-K teacher logit, so
    logsumexp(cat([values, residual])) reconstructs the true full-vocab
    normalizer. The student's matching "everything else" mass is computed
    the same way (mask the student logits at the teacher's chosen indices,
    logsumexp what's left), so the comparison is apples-to-apples even
    though the two models don't share the same Top-K token set.

    student_logits: (B, T, V). teacher_*: (B, T, K) / (B, T) as built by
    collate(). teacher_mask: (B, T) bool, True where a target exists (i.e.
    not padding, and within the precomputed sequence length).
    """
    student_topk = torch.gather(student_logits, dim=-1, index=teacher_indices)  # (B,T,K)
    scatter_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    scatter_mask.scatter_(-1, teacher_indices, True)
    student_masked = student_logits.masked_fill(scatter_mask, float("-inf"))
    student_residual = torch.logsumexp(student_masked, dim=-1)  # (B,T)

    student_logZ = torch.logsumexp(student_logits, dim=-1)  # (B,T)
    teacher_logZ = torch.logsumexp(
        torch.cat([teacher_values, teacher_residual.unsqueeze(-1)], dim=-1), dim=-1
    )  # (B,T)

    teacher_logp_topk = teacher_values - teacher_logZ.unsqueeze(-1)  # (B,T,K)
    teacher_logp_bucket = teacher_residual - teacher_logZ  # (B,T)
    student_logp_topk = student_topk - student_logZ.unsqueeze(-1)  # (B,T,K)
    student_logp_bucket = student_residual - student_logZ  # (B,T)

    teacher_p_topk = teacher_logp_topk.exp()
    teacher_p_bucket = teacher_logp_bucket.exp()
    per_token_kl = (
        (teacher_p_topk * (teacher_logp_topk - student_logp_topk)).sum(dim=-1)
        + teacher_p_bucket * (teacher_logp_bucket - student_logp_bucket)
    )  # (B,T)

    per_token_kl = per_token_kl.masked_fill(~teacher_mask, 0.0)
    denom = teacher_mask.sum().clamp(min=1)
    return per_token_kl.sum() / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--base_config", default="gpt2", help="HF model id to source the architecture config from")
    parser.add_argument("--n_layer", type=int, default=None, help="override config n_layer (smaller/faster smoke-test model)")
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_time_minutes", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher_targets", default=None,
        help="precompute_teacher_targets.py .npz output -- enables KD (KL on Top-K logits) "
             "on top of the plain CE loss. --tokenizer must match the one used to produce it.",
    )
    parser.add_argument("--kd_alpha", type=float, default=0.5, help="weight on the KD/KL term; (1-alpha) on CE")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_config)
    for attr, val in [("n_layer", args.n_layer), ("n_embd", args.n_embd), ("n_head", args.n_head)]:
        if val is not None:
            setattr(config, attr, val)
    config.vocab_size = len(tokenizer)  # base_config's own vocab size is wrong whenever --tokenizer differs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_config(config)  # random init: architecture only, no pretrained weights
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params (random init, arch={args.base_config}), device={device}")

    train_ds = JsonlTextDataset(args.train_file, tokenizer, args.block_size, teacher_targets=args.teacher_targets)
    print(f"Train examples: {len(train_ds)}" + (f" (KD against {args.teacher_targets}, K={train_ds.k})" if args.teacher_targets else ""))
    k = train_ds.k if args.teacher_targets else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id, k=k),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    step, losses, done = 0, [], False
    while not done:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.teacher_targets:
                labels = batch.pop("labels")
                teacher_indices = batch.pop("teacher_indices")
                teacher_values = batch.pop("teacher_values")
                teacher_residual = batch.pop("teacher_residual")
                teacher_mask = batch.pop("teacher_mask")
                logits = model(**batch).logits
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                kd = topk_kd_loss(logits, teacher_indices, teacher_values, teacher_residual, teacher_mask)
                loss = (1 - args.kd_alpha) * ce + args.kd_alpha * kd
            else:
                loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
                if args.teacher_targets:
                    print(f"step {step} loss {loss.item():.4f} (ce {ce.item():.4f} kd {kd.item():.4f})")
                else:
                    print(f"step {step} loss {loss.item():.4f}")
            elapsed_min = (time.time() - start) / 60
            if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
                done = True
                break

    elapsed = time.time() - start
    print("---")
    print(f"best_loss:        {min(losses):.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")


if __name__ == "__main__":
    main()
