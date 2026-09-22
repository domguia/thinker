"""
Reference-LLM baseline on the retrieval1/HotpotQA prompt-response dataset
(RetrievalPromptDataset, same --data/--val_data as train_prompt_response.py
--dataset_type retrieval) -- inference-only (no training), "how well does an
off-the-shelf small pretrained LLM answer given the SAME truncated context
blocks/question the Thinker student sees" reference point.

Method: mirrors train_prompt_response.py's evaluate() "answer" metric --
feed [context blocks ; question ; answer] to the LLM with labels=-100 on
everything but the answer span (HF's built-in shift-and-cross-entropy with
ignore_index), context padding masked out via attention_mask (kb_leaf_mask).
Directly comparable to Thinker's own val_answer number on the same file.

--block_size/--n_docs_max/--max_answer_len default to train_prompt_response.py's
own defaults (16/10/64) so the context truncation matches what the student
sees -- change only if the comparison run used non-default values.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model_families import resolve_model_name
from data.prompt_response_dataset import RetrievalPromptDataset


@torch.no_grad()
def evaluate_llm(model, val_ds, device, batch_size: int = 8, n_eval_batches: int = None) -> float:
    losses, weights = [], []
    n = len(val_ds)
    n_batches = (n + batch_size - 1) // batch_size
    if n_eval_batches is not None:
        n_batches = min(n_batches, n_eval_batches)
    for b in range(n_batches):
        items = [val_ds[i] for i in range(b * batch_size, min((b + 1) * batch_size, n))]
        input_ids, attn_mask, label_ids = [], [], []
        for it in items:
            seq = torch.cat([it["kb_tokens"], it["answer_target_input"]])
            mask = torch.cat([it["kb_leaf_mask"], torch.ones_like(it["answer_target_input"], dtype=torch.bool)])
            lab = torch.cat([torch.full_like(it["kb_tokens"], -100), it["answer_labels"]])
            input_ids.append(seq)
            attn_mask.append(mask)
            label_ids.append(lab)
        input_ids = torch.stack(input_ids).to(device)
        attn_mask = torch.stack(attn_mask).to(device)
        label_ids = torch.stack(label_ids).to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask.long(), labels=label_ids)
        n_valid = (label_ids != -100).sum().item()
        if n_valid == 0:
            continue
        losses.append(out.loss.item() * n_valid)  # HF's loss is already mean over valid tokens -- undo to weight batches by valid-token count
        weights.append(n_valid)
        if (b + 1) % 20 == 0:
            print(f"[progress] {b + 1}/{n_batches} batches, running loss={sum(losses) / sum(weights):.4f}", flush=True)
    return sum(losses) / sum(weights) if weights else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_data", required=True, help="held-out JSONL, same file used for Thinker's own eval")
    ap.add_argument("--model", default="lfm2", help="HF repo id or a family alias from core/model_families.py")
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_eval_batches", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                     help="load dtype -- use bfloat16/float16 for models too large to fit in float32 "
                          "on the target GPU (inference-only, precision loss is not a concern here)")
    ap.add_argument("--num_threads", type=int, default=None,
                     help="torch.set_num_threads() cap for CPU inference -- keep low on a shared/contended "
                          "node (other besteffort jobs, or this project's own concurrent GPU run whose "
                          "DataLoader workers also need CPU) rather than grabbing every core by default.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    device = torch.device(args.device)
    if device.type == "cpu":
        # causal_conv1d's CUDA extension is picked regardless of tensor device (transformers'
        # use_kernel_func_from_hub_with_fallback only checks "is the package importable", not the
        # device) -- LFM2 crashes on CPU otherwise (RuntimeError: Expected x.is_cuda() to be true).
        # Force the ImportError path so it falls back to the reference PyTorch implementation.
        sys.modules["causal_conv1d"] = None
    model_id = resolve_model_name(args.model)
    print(f"loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype).to(device)
    model.eval()

    val_ds = RetrievalPromptDataset(args.val_data, tok, block_size=args.block_size, n_docs_max=args.n_docs_max,
                                     max_answer_len=args.max_answer_len, pad_id=tok.pad_token_id)
    print(f"loaded held-out val: {len(val_ds)} examples", flush=True)

    loss = evaluate_llm(model, val_ds, device, batch_size=args.batch_size, n_eval_batches=args.n_eval_batches)
    print(f"\n[LLM baseline: {model_id}] answer_ce={loss:.4f} "
          f"(directly comparable to Thinker's val_answer on the same file/block_size/n_docs_max)", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": model_id, "answer_ce": loss}, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
