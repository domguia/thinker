"""
Item [4] (overnight queue): extract OLMo-2-1B's per-layer residual-stream
activations x_l (input to each of the 16 layers) over real corpus text,
to NFS -- derisks Phase 12 S1 regardless of which branch (i)/(ii) or which
result A1/A2 give, since S1 needs these activations no matter what.

Stores ONLY x_l (fp16), never FFN_l(x_l) -- the latter recomputes cheaply
from frozen weights at S1 time and would cost ~as much disk as x_l itself
for no benefit. Budget: ~64KB/token * 16 layers... actually 64KB/token
already quotes the FULL per-token cost across all layers (dev_notes'
"4 * d * n_layers bytes/token, fp16" -- d=2048, n_layers=16 -> 64KB/token
total, i.e. this already includes every layer). Target ~1M tokens -> ~64GB.

Smoke-tested (10 tokens, forward pass) before committing to the full run --
required by this project's launch discipline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill/general_realtext/train.jsonl")
    ap.add_argument("--out_dir", default="/home/jdomguia/thinker/activation_cache/olmo2_1b")
    ap.add_argument("--target_tokens", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--smoke_test", action="store_true")
    args = ap.parse_args()

    print(f"loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()
    n_layers = len(model.model.layers)

    if args.smoke_test:
        ids = tok("Hello, this is a smoke test.", return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        assert len(out.hidden_states) == n_layers + 1
        print(f"smoke test OK: {len(out.hidden_states)} hidden_states, "
              f"shape {out.hidden_states[0].shape}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"

    total_tokens = 0
    shard_idx = 0
    t0 = time.time()
    buffer_texts = []

    def flush_shard(texts):
        nonlocal total_tokens, shard_idx
        if not texts:
            return
        ids = tok("\n\n".join(texts), return_tensors="pt", truncation=True,
                   max_length=args.seq_len)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        # hidden_states[l] for l in 0..n_layers-1 is the INPUT to layer l
        # (hidden_states[n_layers] is the final norm's input, not needed here).
        x = torch.stack(out.hidden_states[:n_layers], dim=0)  # (n_layers, 1, T, d)
        x = x[:, 0].to(torch.float16)  # (n_layers, T, d)
        n_tok = x.shape[1]
        torch.save(x, out_dir / f"shard_{shard_idx:05d}.pt")
        total_tokens += n_tok
        shard_idx += 1
        elapsed = time.time() - t0
        print(f"progress: {{\"shard\": {shard_idx}, \"total_tokens\": {total_tokens}, "
              f"\"elapsed_s\": {elapsed:.1f}, \"target_tokens\": {args.target_tokens}}}",
              flush=True)

    # Wraps around the source file if it's exhausted before target_tokens --
    # the 2700-document corpus this project uses elsewhere hit exactly this
    # single-pass-exhaustion trap before (train_real_text.py's
    # LockstepLaneBatcher, 2026-09-14 entry in experiment.log.md); a second
    # pass over the same docs still gives fresh, real activations (the model
    # is frozen, not overfitting to anything), just not bit-identical to a
    # once-through corpus this size -- acceptable for a "get enough
    # activations to work with" cache, not a training run.
    epoch = 0
    while total_tokens < args.target_tokens:
        epoch += 1
        any_line = False
        for line in open(args.data):
            any_line = True
            if not line.strip():
                continue
            buffer_texts.append(json.loads(line)["text"])
            if len(buffer_texts) >= 4:
                flush_shard(buffer_texts)
                buffer_texts = []
            if total_tokens >= args.target_tokens:
                break
        if not any_line:
            print(f"WARNING: {args.data} is empty, stopping.")
            break
        print(f"progress: {{\"epoch\": {epoch}, \"total_tokens\": {total_tokens}}}", flush=True)
    if buffer_texts and total_tokens < args.target_tokens:
        flush_shard(buffer_texts)

    meta = {"model": MODEL_ID, "n_layers": n_layers, "d_model": model.config.hidden_size,
            "total_tokens": total_tokens, "n_shards": shard_idx, "dtype": "float16",
            "note": "shard_XXXXX.pt is (n_layers, T_shard, d_model) fp16, x_l only (input to layer l)"}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nsummary: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
