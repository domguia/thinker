"""E5 (thesis/paper/WRITING_PLAN.md, P1): mechanistic diagnostic of the
calibration-collapse -- tests the "convergence to a fixed point/attractor"
hypothesis directly, per-iteration of the recurrent core loop (inference-only,
instrumentation of an already-trained checkpoint, no new training).

Manually replicates Thinker.forward()'s loop (register init -> n_step calls to
model._step()) from OUTSIDE the model, capturing R and (sm_k, sm_v) after
EVERY iteration (forward() itself only returns the FINAL R/streams). For each
iteration i we compute:

  - ||R_i||: mean L2 norm of the register state (flattened over n_register x
    d_model, averaged over the batch) -- does it blow up / saturate?
  - effective rank of R_i (participation ratio of the SVD singular values of
    the (n_register, d_model) register matrix, averaged over the batch) --
    does the register collapse onto fewer effective directions over
    iterations (a literal "rank collapse", consistent with convergence to a
    low-dimensional attractor)?
  - cosine similarity between R_i and R_{i-1} (flattened, per-example) --
    -> 1.0 means the register has stopped changing (a fixed point reached).
  - "logit lens": the output stream normally cross-attends over the FULL
    accumulated (sm_k, sm_v) trajectory after all n_step iterations. Here we
    call it early, after only the first i entries of sm_k/sm_v (exactly the
    trajectory the model would have if the loop stopped after i steps) --
    this uses the stream's REAL read mechanism (cross-attention over sm_k/sm_v,
    not R directly), so it's a faithful "what would the model predict if it
    stopped here" probe, not a hypothetical projection. Reports predicted
    entropy (nats) of the first answer-position's distribution.

n_step is swept well past any checkpoint's training n_step (up to 32, same
range as E1's extrapolation sweep) since the attractor hypothesis is most
visible in that extrapolation regime.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import RetrievalPromptDataset


def effective_rank(mat: torch.Tensor) -> float:
    """Participation ratio of the singular value spectrum: (sum s)^2 / sum(s^2).
    Ranges from 1 (rank-1, all mass on one direction) to min(shape) (flat spectrum).
    """
    s = torch.linalg.svdvals(mat.float())
    return ((s.sum() ** 2) / (s.pow(2).sum() + 1e-12)).item()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_data", required=True)
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_docs_max", type=int, default=10)
    ap.add_argument("--max_answer_len", type=int, default=64)
    ap.add_argument("--use_ff", action="store_true")
    ap.add_argument("--n_step_max", type=int, default=32, help="sweep register dynamics up to this many iterations")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    ds = RetrievalPromptDataset(path=args.val_data, tokenizer=tokenizer, block_size=args.block_size,
                                 n_docs_max=args.n_docs_max, max_answer_len=args.max_answer_len,
                                 pad_id=tokenizer.pad_token_id)

    stream_dims = {"answer": vocab_size}
    stream_sequence = {"answer": True}
    stream_n_layers = {"answer": 1}
    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=8,
        block_size=args.block_size, depth=0, n_slots=1, n_head=args.n_head,
        disable_kb=False, pool_n_head=1, k_dim=None,
        use_ff=args.use_ff, ff_hidden_mult=4,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=args.max_answer_len,
        stream_n_layers=stream_n_layers,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    indices = list(range(min(args.n_samples, len(ds))))
    items = [ds[i] for i in indices]
    kb_tokens = torch.stack([it["kb_tokens"] for it in items]).to(device)
    kb_source_ids = torch.stack([it["kb_source_ids"] for it in items]).to(device)
    kb_leaf_mask = torch.stack([it["kb_leaf_mask"] for it in items]).to(device)
    query_tokens = kb_tokens[:, -args.block_size:]  # retrieval convention: last block = the question
    target_input = torch.stack([it["answer_target_input"] for it in items]).to(device)

    if not model.disable_kb:
        leaf_emb = model.embed(kb_tokens)
        model.memory.build(leaf_emb, kb_source_ids, leaf_mask=kb_leaf_mask)

    q_emb = model.embed(query_tokens).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(kb_tokens.shape[0], -1, -1) + q_emb
    sm_k = torch.zeros(kb_tokens.shape[0], 0, args.d_model, device=device, dtype=R.dtype)
    sm_v = torch.zeros(kb_tokens.shape[0], 0, args.d_model, device=device, dtype=R.dtype)

    target_embed = model.embed(target_input)
    stream = model.streams["answer"]

    results = []
    R_prev = None
    for step in range(1, args.n_step_max + 1):
        R, sm_k, sm_v = model._step(R, sm_k, sm_v)

        norm_mean = R.flatten(1).norm(dim=1).mean().item()
        rank_mean = sum(effective_rank(R[b]) for b in range(R.shape[0])) / R.shape[0]
        cos_sim = None
        if R_prev is not None:
            cos_sim = F.cosine_similarity(R.flatten(1), R_prev.flatten(1), dim=1).mean().item()
        R_prev = R.clone()

        logit_entropy = None
        if step % 1 == 0:  # cheap enough to do every step; cross-attention over a short sm_k/sm_v
            out = stream(sm_k, sm_v, query_input=target_embed)
            first_pos_logits = out[:, 0]  # (B, vocab)
            probs = F.softmax(first_pos_logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
            logit_entropy = entropy

        results.append({"step": step, "R_norm": norm_mean, "R_effective_rank": rank_mean,
                         "R_cos_sim_prev": cos_sim, "answer_pos0_entropy_nats": logit_entropy})
        print(f"step={step:2d} ||R||={norm_mean:.3f} eff_rank={rank_mean:.3f} "
              f"cos(R,R_prev)={cos_sim if cos_sim is None else f'{cos_sim:.4f}'} "
              f"H(answer@0)={logit_entropy:.3f} nats (max={torch.log(torch.tensor(float(vocab_size))).item():.2f})",
              flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n_register": 8, "vocab_size": vocab_size, "max_entropy_nats": torch.log(torch.tensor(float(vocab_size))).item(),
                       "per_step": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
