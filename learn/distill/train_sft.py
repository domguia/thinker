"""Minimal from-scratch SFT loop for distillation onboarding.

Trains a RANDOMLY INITIALIZED model (architecture only sourced from
--base_config, weights are never loaded) with plain next-token cross-entropy
on the "text" field of a JSONL file produced by any of the
scripts/prepare_*_data.py scripts (reasoning / general / retrieval all share
that field, so this loop is dataset-agnostic).

This is Sequence-level SFT (no Teacher, no KL on logits) -- see
raw/Distill-getting-start.md for why we start here before introducing
logit-level KD. Output format at the end matches program.md's convention
(best_loss / training_seconds / num_steps / num_params_M) for consistency
with the rest of the repo's experiment scripts.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class JsonlTextDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.examples = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                ids = tokenizer(row["text"], truncation=True, max_length=block_size)["input_ids"]
                if len(ids) >= 2:
                    self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id):
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        labels[i, :n] = torch.tensor(ids)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_config)
    for attr, val in [("n_layer", args.n_layer), ("n_embd", args.n_embd), ("n_head", args.n_head)]:
        if val is not None:
            setattr(config, attr, val)
    model = AutoModelForCausalLM.from_config(config)  # random init: architecture only, no pretrained weights
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params (random init, arch={args.base_config})")

    train_ds = JsonlTextDataset(args.train_file, tokenizer, args.block_size)
    print(f"Train examples: {len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    step, losses, done = 0, [], False
    while not done:
        for batch in train_loader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
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
