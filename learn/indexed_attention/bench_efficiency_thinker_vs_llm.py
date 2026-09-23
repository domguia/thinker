"""Inference efficiency comparison (E2, thesis/paper/WRITING_PLAN.md): Thinker vs
Baseline C (dense GPT2, same scale) vs Qwen3.5-0.8B vs LFM2-350M, on the SAME
wikitext prompt/answer data (`wikitext/{train,val}_pr.jsonl`, reformatted by
make_prompt_response_from_realtext.py -- see experiment-agent, 2026-09-23),
swept over context length.

All three checkpoints being compared here (Thinker, Baseline C) were trained
via train_prompt_response.py --dataset_type reasoning on this SAME wikitext_pr
data -- so this script reuses ReasoningPromptDataset and the exact CE
convention from train_prompt_response.py's own evaluate() (F.cross_entropy on
streams["answer"] vs batch["answer_labels"], ignore_index=-100), instead of
reinventing a quality metric.

Context length is varied via --context_lengths -> ReasoningPromptDataset's
`n_ctx` (the padded PROMPT length; the real "problem" content is ~96 tokens
per the reformat, so n_ctx above that is mostly pad_id padding). To keep the
comparison fair as a function of context length specifically, every model
(Thinker AND the dense/HF baselines) processes the FULL n_ctx-length prompt
tensor, padding included -- this measures the actual compute cost of a context
of that length, not "how much real content helps" (that question belongs to
C1/E7, not here).

Measures per (model, context_length) point: parameter count, analytical
FLOPs/token, prefill latency, decode latency, tokens/sec, peak GPU memory, and
teacher-forced val CE/ppl (quality) -- feeds the Fig. 3 Pareto plot and Tab. 3.

Caveat (report honestly, don't hide): Thinker's generate_thinker_reasoning()
has NO KV-cache (one full forward() per generated token, re-attending to the
whole context every step), while the HF baselines use .generate() with their
native KV-cache. This is a real architectural difference worth reporting, not
an apples-to-apples inference-engine comparison.

Baseline C (dense GPT2, n_ctx=1024 at train time -- see
checkpoints/baselineC_wikitext/config/config.json) cannot be benched past 1024
tokens of context without retraining/position interpolation; context lengths
above that are silently skipped for it and reported as such in the log (not a
bug -- Baseline C's own trained n_ctx).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from data.prompt_response_dataset import ReasoningPromptDataset
from learn.indexed_attention.generate_qualitative_compare import generate_thinker_reasoning
from learn.indexed_attention.train_prompt_response import query_tokens_for


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def dense_transformer_flops_per_token(n_params: int, n_layer: int, d_model: int, n_ctx: int) -> float:
    """~2*N (dense matmuls) + 2*n_layer*n_ctx*d_model (quadratic self-attention term,
    dominant once n_ctx is a sizeable fraction of d_model*n_layer -- the regime this
    sweep probes). Standard Kaplan et al. 2020 convention."""
    return 2.0 * n_params + 2.0 * n_layer * n_ctx * d_model


def thinker_flops_per_token(n_params: int, n_step: int, n_ctx: int, d_model: int, n_register: int) -> float:
    """Thinker re-runs its shared recurrent block n_step times per generated token
    (no KV-cache), each step doing a cross-attention from the (small) register
    state over the full n_ctx-length context -- quadratic term is n_ctx*n_register
    cross-attention, not n_ctx^2 self-attention, but repeated n_step times per token."""
    dense_term = 2.0 * n_params
    cross_attn_term = 2.0 * n_ctx * n_register * d_model
    return n_step * (dense_term + cross_attn_term)


@torch.no_grad()
def bench_thinker(checkpoint: str, ds: ReasoningPromptDataset, tokenizer, device, indices,
                   d_model: int, n_head: int, n_step: int, block_size: int, n_ctx: int,
                   max_thinking_len: int, max_answer_len: int, use_ff: bool) -> dict:
    vocab_size = len(tokenizer)
    n_register = 8
    stream_dims = {"thinking": vocab_size, "answer": vocab_size}
    stream_sequence = {"thinking": True, "answer": True}
    stream_n_layers = {"thinking": 1, "answer": 1}
    model = Thinker(
        vocab_size=vocab_size, d_model=d_model, n_register=n_register,
        block_size=block_size, depth=0, n_slots=1, n_head=n_head,
        disable_kb=False, pool_n_head=1, k_dim=None,
        use_ff=use_ff, ff_hidden_mult=4,
        stream_dims=stream_dims, stream_sequence=stream_sequence,
        max_target_len=max(max_thinking_len, max_answer_len), stream_n_layers=stream_n_layers,
    ).to(device)
    # train_prompt_response.py's save_checkpoint saves the raw state_dict directly
    # (no "state_dict" wrapper key, unlike train_sft.py/Baseline C) -- confirmed
    # by experiment-agent 2026-09-23 for this checkpoint.
    sd = torch.load(checkpoint, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    n_params = count_params(model)

    # warmup
    generate_thinker_reasoning(model, ds, indices[:1], device, n_step=n_step,
                                max_answer_len=max_answer_len, tokenizer=tokenizer, temperature=0.0, seed=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    prefill_s, decode_s, total_tokens = 0.0, 0.0, 0
    for i in indices:
        t0 = time.perf_counter()
        generate_thinker_reasoning(model, ds, [i], device, n_step=n_step,
                                    max_answer_len=1, tokenizer=tokenizer, temperature=0.0, seed=0)
        torch.cuda.synchronize()
        prefill_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        answers = generate_thinker_reasoning(model, ds, [i], device, n_step=n_step,
                                              max_answer_len=max_answer_len, tokenizer=tokenizer,
                                              temperature=0.0, seed=0)
        torch.cuda.synchronize()
        decode_s += time.perf_counter() - t0
        n_tok = len(tokenizer(answers[0])["input_ids"])
        total_tokens += max(n_tok, 1)
    decode_s = max(decode_s - prefill_s, 1e-9)  # second call re-includes prefill -- subtract it back out
    peak_mem = torch.cuda.max_memory_allocated(device)

    # quality: exact same CE convention as train_prompt_response.py's evaluate()
    total_nll, total_tok = 0.0, 0
    for i in indices:
        it = ds[i]
        batch = {k: v.unsqueeze(0).to(device) for k, v in it.items() if isinstance(v, torch.Tensor)}
        query_tokens = query_tokens_for("reasoning", batch, block_size)
        target_input = {"answer": batch["answer_target_input"], "thinking": batch["thinking_target_input"]}
        _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, n_step,
                            kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)
        labels = batch["answer_labels"]
        nll = F.cross_entropy(streams["answer"].transpose(1, 2), labels, ignore_index=-100, reduction="sum")
        n = (labels != -100).sum().item()
        total_nll += nll.item()
        total_tok += n
    ce = (total_nll / total_tok) if total_tok else float("nan")
    flops_tok = thinker_flops_per_token(n_params, n_step, n_ctx, d_model, n_register)

    return {"n_params": n_params, "n_ctx": n_ctx, "n_examples": len(indices),
            "total_tokens": total_tokens, "prefill_seconds_total": prefill_s,
            "decode_seconds_total": decode_s, "tokens_per_sec": total_tokens / (prefill_s + decode_s),
            "sec_per_example": (prefill_s + decode_s) / len(indices),
            "peak_memory_bytes": peak_mem, "flops_per_token": flops_tok,
            "val_ce": ce, "val_ppl": (float("nan") if ce != ce else math.exp(ce))}


@torch.no_grad()
def bench_causal_lm(model, model_name: str, ds: ReasoningPromptDataset, tokenizer, device, indices,
                     n_layer: int, d_model: int, n_ctx: int, max_answer_len: int) -> dict:
    """Shared bench body for any AutoModelForCausalLM-compatible model (HF hub
    checkpoints AND the locally-trained Baseline C GPT2) -- prompt is the FULL
    n_ctx-length `kb_tokens` tensor (padding included, see module docstring),
    answer is teacher-forced from `answer_labels` for the CE/ppl quality number."""
    model.eval()
    n_params = count_params(model)

    def _prompt_ids(i):
        return ds[i]["kb_tokens"].unsqueeze(0).to(device)

    model.generate(_prompt_ids(indices[0]), max_new_tokens=max_answer_len, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    prefill_s, decode_s, total_tokens = 0.0, 0.0, 0
    for i in indices:
        input_ids = _prompt_ids(i)
        t0 = time.perf_counter()
        model(input_ids, use_cache=True)  # prefill only
        torch.cuda.synchronize()
        prefill_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=max_answer_len, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id, use_cache=True)
        torch.cuda.synchronize()
        decode_s += time.perf_counter() - t0
        total_tokens += max(out.shape[1] - input_ids.shape[1], 1)
    decode_s = max(decode_s - prefill_s, 1e-9)  # generate() redoes the prefill internally
    peak_mem = torch.cuda.max_memory_allocated(device)

    total_nll, total_tok = 0.0, 0
    for i in indices:
        it = ds[i]
        prompt_ids = it["kb_tokens"].unsqueeze(0).to(device)
        answer_labels = it["answer_labels"]
        valid = answer_labels != -100
        if not valid.any():
            continue
        answer_ids = answer_labels.clone()
        answer_ids[~valid] = tokenizer.pad_token_id or 0
        full = torch.cat([prompt_ids, answer_ids.unsqueeze(0).to(device)], dim=1)
        out = model(full)
        answer_logits = out.logits[0, prompt_ids.shape[1] - 1:-1]
        labels = answer_labels.to(device)
        nll = F.cross_entropy(answer_logits, labels, ignore_index=-100, reduction="sum")
        total_nll += nll.item()
        total_tok += valid.sum().item()
    ce = (total_nll / total_tok) if total_tok else float("nan")
    flops_tok = dense_transformer_flops_per_token(n_params, n_layer, d_model, n_ctx)

    return {"n_params": n_params, "n_ctx": n_ctx, "n_examples": len(indices),
            "total_tokens": total_tokens, "prefill_seconds_total": prefill_s,
            "decode_seconds_total": decode_s, "tokens_per_sec": total_tokens / (prefill_s + decode_s),
            "sec_per_example": (prefill_s + decode_s) / len(indices),
            "peak_memory_bytes": peak_mem, "flops_per_token": flops_tok,
            "val_ce": ce, "val_ppl": (float("nan") if ce != ce else math.exp(ce))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thinker_checkpoint", required=True)
    ap.add_argument("--baseline_c_checkpoint", default=None)
    ap.add_argument("--baseline_c_n_layer", type=int, default=6)
    ap.add_argument("--llm_models", default="qwen35,lfm2", help="comma-separated model_families aliases")
    ap.add_argument("--val_data", required=True, help="wikitext/val_pr.jsonl (prompt/answer schema)")
    ap.add_argument("--tokenizer", default="qwen35")
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_step", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--max_thinking_len", type=int, default=8)
    ap.add_argument("--max_answer_len", type=int, default=48)
    ap.add_argument("--context_lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--use_ff", action="store_true", default=True)
    ap.add_argument("--llm_dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.llm_dtype]

    baseline_c_cfg = None
    if args.baseline_c_checkpoint:
        baseline_c_cfg = GPT2Config.from_pretrained(os.path.join(os.path.dirname(args.baseline_c_checkpoint), "config"))

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    results = []
    for ctx in context_lengths:
        print(f"\n########## context_length={ctx} ##########", flush=True)
        ds = ReasoningPromptDataset(path=args.val_data, tokenizer=tokenizer, n_ctx=ctx,
                                     max_thinking_len=args.max_thinking_len, max_answer_len=args.max_answer_len,
                                     pad_id=tokenizer.pad_token_id)
        indices = list(range(min(args.n_examples, len(ds))))
        if not indices:
            continue

        print(f"=== Thinker ({args.thinker_checkpoint}) ===", flush=True)
        r = bench_thinker(args.thinker_checkpoint, ds, tokenizer, device, indices,
                           args.d_model, args.n_head, args.n_step, args.block_size, ctx,
                           args.max_thinking_len, args.max_answer_len, args.use_ff)
        r.update(model="thinker", context_length=ctx)
        print(json.dumps(r, indent=2), flush=True)
        results.append(r)

        if baseline_c_cfg is not None:
            # prompt (ctx tokens) + teacher-forced answer (up to max_answer_len tokens) are
            # concatenated for the CE/ppl pass -- both must fit under n_positions, not just ctx
            # alone (GPT2's absolute position embedding table is sized exactly n_positions and
            # overflowing it is a silent CUDA-side embedding-gather OOB, not a clean Python error).
            if ctx + args.max_answer_len > baseline_c_cfg.n_positions:
                print(f"skip Baseline C: ctx+max_answer_len={ctx + args.max_answer_len} > "
                      f"its trained n_ctx={baseline_c_cfg.n_positions}", flush=True)
            else:
                print(f"=== Baseline C ({args.baseline_c_checkpoint}) ===", flush=True)
                bc_model = GPT2LMHeadModel(baseline_c_cfg).to(device).to(torch_dtype)
                sd = torch.load(args.baseline_c_checkpoint, map_location=device)
                sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
                bc_model.load_state_dict(sd)
                r = bench_causal_lm(bc_model, "baseline_c", ds, tokenizer, device, indices,
                                     args.baseline_c_n_layer, args.d_model, ctx, args.max_answer_len)
                r.update(model="baseline_c", context_length=ctx)
                print(json.dumps(r, indent=2), flush=True)
                results.append(r)
                del bc_model
                torch.cuda.empty_cache()

        for llm_name in args.llm_models.split(","):
            # each model needs ITS OWN tokenizer/vocab -- ids from `tokenizer` (qwen35, used
            # for Thinker/Baseline C which were trained with it) are out-of-vocab garbage for
            # a model with a different vocab (e.g. LFM2-350M) and crash with an illegal
            # embedding-table gather, not a clean error -- rebuild the dataset per model.
            llm_tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(llm_name))
            if llm_tokenizer.pad_token_id is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            llm_ds = ReasoningPromptDataset(path=args.val_data, tokenizer=llm_tokenizer, n_ctx=ctx,
                                             max_thinking_len=args.max_thinking_len,
                                             max_answer_len=args.max_answer_len, pad_id=llm_tokenizer.pad_token_id)
            llm_indices = list(range(min(args.n_examples, len(llm_ds))))
            if not llm_indices:
                continue
            hf_model = AutoModelForCausalLM.from_pretrained(resolve_model_name(llm_name), dtype=torch_dtype).to(device)
            n_ctx_max = getattr(hf_model.config, "max_position_embeddings", None) or \
                getattr(hf_model.config, "n_positions", None)
            if n_ctx_max is not None and ctx + args.max_answer_len > n_ctx_max:
                print(f"skip {llm_name}: ctx+max_answer_len={ctx + args.max_answer_len} > "
                      f"its max_position_embeddings={n_ctx_max}", flush=True)
                del hf_model
                torch.cuda.empty_cache()
                continue
            print(f"=== LLM baseline ({llm_name}) ===", flush=True)
            n_layer = getattr(hf_model.config, "num_hidden_layers", None) or getattr(hf_model.config, "n_layer")
            d_model = getattr(hf_model.config, "hidden_size", None) or getattr(hf_model.config, "n_embd")
            r = bench_causal_lm(hf_model, llm_name, llm_ds, llm_tokenizer, device, llm_indices, n_layer, d_model, ctx,
                                 args.max_answer_len)
            r.update(model=llm_name, context_length=ctx)
            print(json.dumps(r, indent=2), flush=True)
            results.append(r)
            del hf_model
            torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out} ({len(results)} (model, context_length) points)")


if __name__ == "__main__":
    main()
