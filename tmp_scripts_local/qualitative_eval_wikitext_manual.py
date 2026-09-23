"""Manual qualitative check (supervisor-agent, 2026-09-22): verify the wikitext
CE-vs-KD result (KD=6.6285 < CE-only=6.8462) isn't hiding a degenerate-generation
collapse the CE number alone wouldn't show (cf. the retrieval <think>-collapse
precedent, where good CE masked a collapsed generation). Greedy decoding, first
30 val examples (fixed, deterministic), both checkpoints, side by side.
"""
import sys
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import ReasoningPromptDataset
from learn.indexed_attention.generate_qualitative_compare import generate_thinker_reasoning

WIKI_ROOT = "/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/wikitext"
VAL_PR = f"{WIKI_ROOT}/val_pr.jsonl"
N_SAMPLES = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(resolve_model_name("qwen35"))
ds = ReasoningPromptDataset(
    path=VAL_PR, tokenizer=tok, n_ctx=256, max_thinking_len=8, max_answer_len=48,
    pad_id=tok.pad_token_id,
)
indices = list(range(min(N_SAMPLES, len(ds))))

vocab_size = len(tok)
# Same defaults as train_prompt_response.py's argparse (none overridden by phase18's launch
# command: n_register=8, block_size=16, depth=0, n_slots=1, pool_n_head=1, k_dim=None,
# ff_hidden_mult=4, thinking_n_layers=1, answer_n_layers=1) -- copied from
# eval_thinker_full_val.py's own model-construction block, not guessed.
stream_dims = {"thinking": vocab_size, "answer": vocab_size}
stream_sequence = {"thinking": True, "answer": True}
stream_n_layers = {"thinking": 1, "answer": 1}
max_target_len = max(8, 48)  # max_thinking_len, max_answer_len used at launch

def load_model(ckpt_path):
    model = Thinker(
        vocab_size=vocab_size, d_model=256, n_register=8,
        block_size=16, depth=0, n_slots=1, n_head=4,
        disable_kb=False, pool_n_head=1, k_dim=None,
        use_ff=True, ff_hidden_mult=4,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
        stream_n_layers=stream_n_layers,
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

results = {}
for label, ckpt in [("CE-only", "checkpoints/wikitext_ceonly_best.pt"),
                     ("KD", "checkpoints/wikitext_kd_best.pt")]:
    model = load_model(ckpt)
    answers = generate_thinker_reasoning(model, ds, indices, device, n_step=4,
                                          max_answer_len=48, tokenizer=tok,
                                          temperature=0.0, top_p=1.0, seed=0)
    results[label] = answers
    del model
    torch.cuda.empty_cache()

n_degenerate = {"CE-only": 0, "KD": 0}
print(f"=== Qualitative eval, {len(indices)} examples, greedy ===\n")
for i in indices:
    prompt_text = ds.examples[i]["problem"][:80]
    print(f"--- example {i} (prompt: {prompt_text!r}...) ---")
    for label in ("CE-only", "KD"):
        a = results[label][i]
        toks = a.split()
        word_degenerate = len(toks) >= 4 and len(set(toks)) / len(toks) < 0.5
        digit_ratio = sum(c.isdigit() for c in a) / max(len(a), 1)
        digit_degenerate = len(a) >= 15 and digit_ratio > 0.5
        flag = " [DEGENERATE]" if (word_degenerate or digit_degenerate) else ""
        if flag:
            n_degenerate[label] += 1
        print(f"  {label}{flag}: {a!r}")
    print()

print("=== Summary ===")
for label in ("CE-only", "KD"):
    print(f"{label}: {n_degenerate[label]}/{len(indices)} degenerate")
