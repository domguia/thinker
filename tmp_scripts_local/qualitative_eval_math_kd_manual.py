"""Manual qualitative check for math phase17b/c KD-arm checkpoint --
train_prompt_response.py's own --qualitative_eval_at_end was skipped because
the Rennes code was stale (missing the reasoning dispatch added this session,
commit 6e3b866). Same heuristics as qualitative_eval_wikitext_manual.py.
"""
import argparse
import sys
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import ReasoningPromptDataset
from learn.indexed_attention.generate_qualitative_compare import generate_thinker_reasoning

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--val_data", required=True)
ap.add_argument("--tokenizer", default="qwen35")
ap.add_argument("--n_samples", type=int, default=30)
ap.add_argument("--max_thinking_len", type=int, default=1024)
ap.add_argument("--max_answer_len", type=int, default=64)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
ds = ReasoningPromptDataset(
    path=args.val_data, tokenizer=tok, n_ctx=1024, max_thinking_len=args.max_thinking_len,
    max_answer_len=args.max_answer_len, pad_id=tok.pad_token_id,
)
indices = list(range(min(args.n_samples, len(ds))))

vocab_size = len(tok)
stream_dims = {"thinking": vocab_size, "answer": vocab_size}
stream_sequence = {"thinking": True, "answer": True}
stream_n_layers = {"thinking": 1, "answer": 1}
max_target_len = max(args.max_thinking_len, args.max_answer_len)

model = Thinker(
    vocab_size=vocab_size, d_model=256, n_register=8,
    block_size=16, depth=0, n_slots=1, n_head=4,
    disable_kb=False, pool_n_head=1, k_dim=None,
    use_ff=True, ff_hidden_mult=4,
    stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
    stream_n_layers=stream_n_layers,
).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

answers = {}
for i in indices:
    with torch.no_grad():
        answers.update(zip([i], generate_thinker_reasoning(model, ds, [i], device, n_step=4,
                                                            max_answer_len=args.max_answer_len, tokenizer=tok,
                                                            temperature=0.0, top_p=1.0, seed=0)))
    torch.cuda.empty_cache()

n_degenerate = 0
print(f"=== Math KD-arm qualitative eval, {len(indices)} examples, greedy ===\n")
for i in indices:
    prompt_text = ds.examples[i]["problem"][:80]
    a = answers[i]
    toks = a.split()
    word_degenerate = len(toks) >= 4 and len(set(toks)) / len(toks) < 0.5
    digit_ratio = sum(c.isdigit() for c in a) / max(len(a), 1)
    digit_degenerate = len(a) >= 15 and digit_ratio > 0.5
    flag = " [DEGENERATE]" if (word_degenerate or digit_degenerate) else ""
    if flag:
        n_degenerate += 1
    print(f"--- example {i} (prompt: {prompt_text!r}...) ---")
    print(f"  {flag or '[ok]'}: {a!r}\n")

print("=== Summary ===")
print(f"Math KD-arm: {n_degenerate}/{len(indices)} degenerate")
