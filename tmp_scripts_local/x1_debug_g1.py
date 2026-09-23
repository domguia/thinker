"""Debug: train M4 addition briefly, then print raw predicted vs target
token sequences for a few EASY (short) in-distribution examples, to find
the EM=0.0000 bug (X1 gate G1)."""
import sys
sys.path.insert(0, ".")
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from learn.x1.tasks import gen_addition, TOK2ID, EOS, ID2TOK, VOCAB_SIZE
from learn.x1.train_dense import collate, greedy_generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eos_id = TOK2ID[EOS]

config = AutoConfig.from_pretrained("gpt2")
config.n_layer, config.n_embd, config.n_head = 6, 320, 5
config.vocab_size = VOCAB_SIZE
config.n_positions = config.n_ctx = 600
model = AutoModelForCausalLM.from_config(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()

import time
start = time.time()
for step in range(1, 1501):
    batch_examples = gen_addition(64, (1, 20), seed=step, position_offset_max=80)
    batch = collate(batch_examples, device)
    out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                attention_mask=batch["attention_mask"], labels=batch["labels"])
    opt.zero_grad(); out.loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"step={step} loss={out.loss.item():.4f} elapsed={(time.time()-start)/60:.2f}m", flush=True)

model.eval()

# Teacher-forced check: does the model correctly predict the first target
# digit when given the FULL (prompt+target) sequence, exactly as trained?
# If this also fails, the bug is in data/labels, not in greedy_generate.
tf_examples = gen_addition(6, (1, 2), seed=42, position_offset_max=0)
tf_batch = collate(tf_examples, device)
with torch.no_grad():
    tf_out = model(input_ids=tf_batch["input_ids"], position_ids=tf_batch["position_ids"],
                    attention_mask=tf_batch["attention_mask"])
for i, e in enumerate(tf_examples):
    pl = e.prompt_len
    pred_first = tf_out.logits[i, pl - 1].argmax().item()
    true_first = e.target_ids[pl]
    print(f"TEACHER-FORCED ex{i}: pred_first_tok={ID2TOK[pred_first]!r} true_first_tok={ID2TOK[true_first]!r} match={pred_first==true_first}", flush=True)


print("--- isolating divergence: single-example forward (no attention_mask) ---", flush=True)
for i, e in enumerate(tf_examples):
    pl = e.prompt_len
    prompt_ids = e.input_ids[:pl]
    prompt_pos = e.position_ids[:pl]
    ids1 = torch.tensor([prompt_ids], device=device)
    pos1 = torch.tensor([prompt_pos], device=device)
    with torch.no_grad():
        out1 = model(input_ids=ids1, position_ids=pos1, use_cache=False)
    pred1 = out1.logits[0, -1].argmax().item()
    mask1 = torch.ones_like(ids1)
    with torch.no_grad():
        out2 = model(input_ids=ids1, position_ids=pos1, attention_mask=mask1, use_cache=False)
    pred2 = out2.logits[0, -1].argmax().item()
    with torch.no_grad():
        out3 = model(input_ids=ids1, position_ids=pos1)  # default use_cache
    pred3 = out3.logits[0, -1].argmax().item()
    true_first = e.target_ids[pl]
    print(f"ex{i}: no_mask_no_cacheflag={ID2TOK[pred1]!r} with_mask={ID2TOK[pred2]!r} default_cache={ID2TOK[pred3]!r} true={ID2TOK[true_first]!r}", flush=True)

for offset_test in (0, 40):
    print(f"--- offset={offset_test} ---", flush=True)
    easy = gen_addition(6, (1, 2), seed=42, position_offset_max=0)
    easy = [type(e)(e.input_ids, [p + offset_test for p in e.position_ids], e.target_ids, e.prompt_len)
            for e in easy]
    for e in easy:
        prompt_ids = e.input_ids[:e.prompt_len]
        prompt_pos = e.position_ids[:e.prompt_len]
        target = [t for t in e.target_ids[e.prompt_len:] if t != eos_id]
        pred = greedy_generate(model, prompt_ids, prompt_pos, 40, eos_id, device)
        q = "".join(ID2TOK[t] for t in prompt_ids)
        tgt = "".join(ID2TOK[t] for t in target)
        prd = "".join(ID2TOK[t] for t in pred)
        print(f"Q={q!r} TARGET={tgt!r} PRED={prd!r} MATCH={pred==target}", flush=True)
