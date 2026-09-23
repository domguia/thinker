"""Qualitative degeneracy check for Baseline C (dense transformer control,
same heuristics as qualitative_eval_wikitext_manual.py) -- greedy generation,
first 30 val examples, using HF's native .generate() (no custom decode loop
needed here, unlike Thinker).
"""
import argparse
import json
import sys
sys.path.insert(0, ".")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core.model_families import resolve_model_name

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--config_dir", required=True)
ap.add_argument("--val_data", required=True)
ap.add_argument("--tokenizer", default="qwen35")
ap.add_argument("--n_samples", type=int, default=30)
ap.add_argument("--prompt_tokens", type=int, default=48)
ap.add_argument("--max_new_tokens", type=int, default=48)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

config = AutoConfig.from_pretrained(args.config_dir)
model = AutoModelForCausalLM.from_config(config)
ckpt = torch.load(args.checkpoint, map_location=device)
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model = model.to(device).eval()

examples = []
with open(args.val_data) as f:
    for line in f:
        if len(examples) >= args.n_samples:
            break
        row = json.loads(line)
        ids = tok(row["text"], truncation=True, max_length=args.prompt_tokens)["input_ids"]
        if len(ids) >= 8:
            examples.append(ids)

n_degenerate = 0
print(f"=== Baseline C qualitative eval, {len(examples)} examples, greedy ===\n")
for i, ids in enumerate(examples):
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id)
    gen_ids = out[0, len(ids):].tolist()
    a = tok.decode(gen_ids, skip_special_tokens=True)
    toks = a.split()
    word_degenerate = len(toks) >= 4 and len(set(toks)) / len(toks) < 0.5
    digit_ratio = sum(c.isdigit() for c in a) / max(len(a), 1)
    digit_degenerate = len(a) >= 15 and digit_ratio > 0.5
    flag = " [DEGENERATE]" if (word_degenerate or digit_degenerate) else ""
    if flag:
        n_degenerate += 1
    prompt_text = tok.decode(ids, skip_special_tokens=True)[:60]
    print(f"--- example {i} (prompt: {prompt_text!r}...) ---")
    print(f"  {flag or '[ok]'}: {a!r}\n")

print("=== Summary ===")
print(f"Baseline C: {n_degenerate}/{len(examples)} degenerate")
