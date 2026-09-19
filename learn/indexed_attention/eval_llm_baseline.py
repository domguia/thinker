"""
Reference-LLM baseline on the SAME held-out windows Thinker is evaluated on
(train_real_text.py's --val_data / eval_checkpoint.py) -- the direct answer
to "does Thinker do the tasks a real LLM does, and how far behind/ahead is
it" (user's framing), not just an internal loss number.

Default reference: LFM2-350M -- already this project's own default
tokenizer alias (core/model_families.py's "lfm2"), so no tokenizer
mismatch/retokenization needed for a fair comparison: same vocabulary,
same window boundaries (RealTextWindowDataset), same held-out data,
directly comparable `loss`/`ppl` numbers against Thinker's own
final_val_loss (train_real_text.py) or eval_checkpoint.py's output.

Method: for each window, feed [context ; target] to the LLM with
labels=[-100]*len(context) + target (HF's built-in shift-and-cross-entropy
with ignore_index handles the "predict target given context" loss
correctly) -- context padding (RealTextWindowDataset left-pads short
documents) is masked out via attention_mask, consistent with how Thinker
itself treats padding (kb_leaf_mask).

CPU-feasible at this model size (350M, small batches) but GPU recommended
for a full pass -- same cost order as an OLMo-2-1B diagnostic pass.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model_families import resolve_model_name
from data.real_text_windows import RealTextWindowDataset


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
            seq = torch.cat([it["kb_tokens"], it["labels"]])
            mask = torch.cat([it["kb_leaf_mask"], torch.ones_like(it["labels"], dtype=torch.bool)])
            lab = torch.cat([torch.full_like(it["kb_tokens"], -100), it["labels"]])
            input_ids.append(seq)
            attn_mask.append(mask)
            label_ids.append(lab)
        input_ids = torch.stack(input_ids).to(device)
        attn_mask = torch.stack(attn_mask).to(device)
        label_ids = torch.stack(label_ids).to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask.long(), labels=label_ids)
        n_valid = (label_ids != -100).sum().item()
        losses.append(out.loss.item() * n_valid)  # HF's loss is already mean over valid tokens -- undo to weight batches by valid-token count
        weights.append(n_valid)
    return sum(losses) / sum(weights) if weights else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val_data", required=True, help="held-out JSONL, same file used for Thinker's own eval")
    ap.add_argument("--model", default="lfm2", help="HF repo id or a family alias from core/model_families.py")
    ap.add_argument("--n_ctx", type=int, default=256)
    ap.add_argument("--t_tgt", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_eval_batches", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model_id = resolve_model_name(args.model)
    print(f"loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    model.eval()

    val_ds = RealTextWindowDataset(args.val_data, tok, n_ctx=args.n_ctx, t_local=args.n_ctx,
                                    t_tgt=args.t_tgt, pad_id=tok.pad_token_id)
    print(f"loaded held-out val: {len(val_ds.docs)} docs, {len(val_ds.windows)} windows", flush=True)

    loss = evaluate_llm(model, val_ds, device, batch_size=args.batch_size, n_eval_batches=args.n_eval_batches)
    ppl = torch.exp(torch.tensor(loss)).item()
    print(f"\n[LLM baseline: {model_id}] loss={loss:.4f} ppl={ppl:.2f} "
          f"(directly comparable to Thinker's final_val_loss on the same tokenizer/windows)", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": model_id, "loss": loss, "ppl": ppl}, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
