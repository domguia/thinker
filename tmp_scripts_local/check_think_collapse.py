import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from data.prompt_response_dataset import RetrievalPromptDataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
think_id = tok.convert_tokens_to_ids("<think>")
print(f"<think> token id: {think_id}")

ROOT = "/srv/storage/killerdroid@storage3.rennes.grid5000.fr/thinker-distill/data/distill/hotpotqa"
ds = RetrievalPromptDataset(
    path=f"{ROOT}/val.jsonl",
    tokenizer=tok, block_size=16, n_docs_max=10, max_answer_len=32, pad_id=tok.pad_token_id,
    teacher_targets=f"{ROOT}/topk/val/thinkfix_n2000.qwen_big.npz",
    teacher_max_length=4096, teacher_name="qwen_big",
)

N = 200
n_think_present = 0
think_mass = []
for i in range(N):
    ex = ds[i]
    idx0 = ex["answer_kd_indices"][0]
    val0 = ex["answer_kd_values"][0]
    idx0_list = idx0.tolist()
    if think_id in idx0_list:
        n_think_present += 1
        pos = idx0_list.index(think_id)
        think_mass.append(float(val0[pos]))

print(f"<think> present in top-K at answer position 0: {n_think_present}/{N}")
if think_mass:
    print(f"mass when present: min={min(think_mass):.4f} max={max(think_mass):.4f} mean={sum(think_mass)/len(think_mass):.4f}")
else:
    print("<think> NEVER present in top-K at answer position 0 -- collapse mechanism confirmed gone.")
