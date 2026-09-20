"""
Trains Thinker (core/indexed_thinker_model.py) on prompt/thinking/answer
data (data/prompt_response_dataset.py) -- distinct from train_real_text.py's
continuous sliding-window scheme, per user request (2026-09-20): our model
does not ingest data the way a plain decoder-only LLM does (every position
learning on its whole context); it needs an explicit prompt (fed as KB),
and a response -- optionally split into a `thinking` stream (reasoning
trace) and an `answer` stream (final answer), each Thinker's own output
stream with its own teacher-forced target (core/indexed_thinker_model.py's
per-stream target_input, added for exactly this).

Two of the project's three data.distill/prepare_*_data.py sources have
this structure and are covered here (`--dataset_type`):
  - reasoning (learn/distill/prepare_reasoning_data.py, OpenR1-Math-220k):
    ReasoningPromptDataset, TWO streams (thinking + answer).
  - retrieval (learn/distill/prepare_retrieval_data.py, HotpotQA):
    RetrievalPromptDataset, ONE stream (answer only, no reasoning trace
    available in this dataset).
The third source (general, WikiText/TinyStories) has no prompt/response
structure -- stays on train_real_text.py's sliding-window pipeline,
unchanged, not duplicated here.

Each example is an independent episode (no cross-example document
continuity, unlike real-text's LockstepLaneBatcher/register carry-over) --
a plain shuffled DataLoader, register always starts fresh from
register_init. This is deliberately simpler than train_real_text.py.

KD (logit-level, --teacher_targets, 2026-09-20): reuses topk_kd_loss()
(learn/distill/train_sft.py) exactly like train_real_text.py, but alignment
is handled entirely by data/prompt_response_dataset.py's
PromptResponseTeacherTargets (char-to-token offset mapping against the
Teacher's precomputed per-document arrays, since response spans here are
independently-tokenized episodes, not fixed window positions) -- this
script only consumes the already-sliced `*_kd_indices/values/residual/mask`
tensors the dataset attaches per example. See that module's docstring for
the alignment scheme and its CE-only fallback for unaligned spans.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args
from data.prompt_response_dataset import ReasoningPromptDataset, RetrievalPromptDataset
from learn.distill.train_sft import topk_kd_loss


def build_dataset(dataset_type: str, path: str, tokenizer, args, teacher_targets: str = None):
    if dataset_type == "reasoning":
        return ReasoningPromptDataset(path, tokenizer, n_ctx=args.n_ctx,
                                       max_thinking_len=args.max_thinking_len,
                                       max_answer_len=args.max_answer_len, pad_id=tokenizer.pad_token_id,
                                       teacher_targets=teacher_targets, teacher_max_length=args.teacher_max_length)
    if dataset_type == "retrieval":
        # block_size/n_docs_max, NOT n_ctx/t_local (2026-09-20 redesign, user
        # decision: treat HotpotQA's multiple documents as distinct indexable
        # blocks, depth=1, rather than one flattened truncated window --
        # see RetrievalPromptDataset's docstring). --block_size here is the
        # SAME value as Thinker's own --block_size constructor arg (must
        # match for HierarchicalMemory's depth=1 to compress one node per
        # document correctly).
        return RetrievalPromptDataset(path, tokenizer, block_size=args.block_size, n_docs_max=args.n_docs_max,
                                       max_answer_len=args.max_answer_len, pad_id=tokenizer.pad_token_id,
                                       teacher_targets=teacher_targets, teacher_max_length=args.teacher_max_length)
    raise ValueError(f"unknown --dataset_type {dataset_type!r} (expected 'reasoning' or 'retrieval' -- "
                      f"'general' stays on train_real_text.py's sliding-window pipeline, not this script)")


def kd_losses(streams, batch, dataset_type: str):
    """topk_kd_loss() per stream, using the *_kd_indices/values/residual/mask
    tensors data/prompt_response_dataset.py's teacher_targets path attaches
    to every example (all-False mask = CE-only fallback, contributes 0)."""
    kd_answer = topk_kd_loss(streams["answer"], batch["answer_kd_indices"], batch["answer_kd_values"],
                              batch["answer_kd_residual"], batch["answer_kd_mask"])
    kd_thinking = None
    if dataset_type == "reasoning":
        kd_thinking = topk_kd_loss(streams["thinking"], batch["thinking_kd_indices"], batch["thinking_kd_values"],
                                    batch["thinking_kd_residual"], batch["thinking_kd_mask"])
    return kd_answer, kd_thinking


def ingest_n_step_for(doc_leaf_mask: torch.Tensor, step_size: int, n_step_min: int, n_step_max: int) -> int:
    """
    Spec §8ter "grand volume" discussion, option B (2026-09-20, user decision:
    implement this simple length-based rule now, keep the density-based
    version (§8ter's [OUVERT] "densité d'information") as a separate research
    thread, not blocking this). One n_step PER DOCUMENT SLOT (not per
    example) -- ingest() processes the whole batch for a slot in one call, so
    real length is reduced to the batch's max real length for that slot, a
    coarse approximation deliberately kept simple rather than padding/masking
    a per-example n_step (which ingest()'s fixed-iteration-count loop can't
    express without a much larger refactor).

    doc_leaf_mask: (B, block_size) bool for this one document slot.
    step_size: real tokens "worth" one ingestion iteration.
    """
    real_len = int(doc_leaf_mask.sum(dim=1).max().item())
    n = -(-real_len // step_size)  # ceil
    return max(n_step_min, min(n_step_max, n))


def ingest_documents(model, batch, n_docs_max: int, block_size: int, n_step_ingest: int,
                      use_checkpoint: bool = False, step_size: int = None,
                      n_step_min: int = 1, n_step_max: int = None) -> None:
    """
    Spec §8ter, MVP (intra-batch only, no inter-batch cache -- 2026-09-20 user
    decision to build the minimal version first): runs each document block of
    `batch["kb_tokens"]` (retrieval only -- see RetrievalPromptDataset's
    per-document block layout) through `model.ingest()` instead of letting
    `HierarchicalMemory.build()` project it directly via k_proj/v_proj, then
    injects the resulting (K, V) as one extra memory level per document via
    `add_static_level`.

    Ingested in document order, threading each document's own (sm_k, sm_v,
    mask) forward as the NEXT document's `prior_k`/`prior_v`/`prior_mask`
    (2026-09-20 checkpoint-safety fix -- ingest() no longer reads
    self.memory's mutable state, see Thinker.ingest's docstring) -- so
    document i's ingestion pass can attend to documents 0..i-1 already
    ingested earlier in this same call, not to documents after it, nor
    (in this MVP) to any other example's documents. `self.memory` itself is
    only populated with add_static_level AFTER all documents are ingested,
    for the QA forward pass to read.

    A document slot that is fully padding for every example in the batch
    (RetrievalPromptDataset masks out missing documents when an example has
    fewer than n_docs_max real ones) is skipped entirely -- nothing to
    ingest. A slot real for SOME but not all examples in the batch is still
    ingested (padding rows produce garbage KV for the examples missing it),
    but `add_static_level`'s mask marks those examples' entries as unreal so
    `attend()`'s softmax never reads them.

    step_size (None = disabled, use the fixed n_step_ingest for every slot):
    when set, each document slot's n_step is derived from its own real
    length instead of being fixed (see ingest_n_step_for) -- n_step_max
    defaults to n_step_ingest when step_size is set and n_step_max is None.
    """
    model.memory.clear()
    prior_k = prior_v = prior_mask = None
    for i in range(n_docs_max):
        start = i * block_size
        doc_tokens = batch["kb_tokens"][:, start:start + block_size]
        doc_leaf_mask = batch["kb_leaf_mask"][:, start:start + block_size]
        doc_present = doc_leaf_mask.any(dim=1)  # (B,)
        if not doc_present.any():
            continue
        n_step = n_step_ingest
        if step_size is not None:
            n_step = ingest_n_step_for(doc_leaf_mask, step_size, n_step_min, n_step_max or n_step_ingest)
        sm_k, sm_v = model.ingest(doc_tokens, n_step, prior_k=prior_k, prior_v=prior_v,
                                   prior_mask=prior_mask, use_checkpoint=use_checkpoint)
        mask = doc_present.unsqueeze(1).expand(-1, sm_k.shape[1])
        prior_k = sm_k if prior_k is None else torch.cat([prior_k, sm_k], dim=1)
        prior_v = sm_v if prior_v is None else torch.cat([prior_v, sm_v], dim=1)
        prior_mask = mask if prior_mask is None else torch.cat([prior_mask, mask], dim=1)
        model.memory.add_static_level(sm_k, sm_v, mask=mask)


def query_tokens_for(dataset_type: str, batch, block_size: int):
    """What seeds the register (spec: mean-pooled embedding added to
    register_init) -- real-text uses the last t_local context tokens
    (recency); here, the natural analogue per dataset:
      - retrieval: the question's own block (the LAST block_size positions
        of kb_tokens, see RetrievalPromptDataset -- always the question,
        regardless of n_docs_max).
      - reasoning: the WHOLE problem (no local/long-range split there --
        source_id is uniformly 0, see ReasoningPromptDataset)."""
    if dataset_type == "retrieval":
        return batch["kb_tokens"][:, -block_size:]
    return batch["kb_tokens"]


@torch.no_grad()
def evaluate(model, loader, device, dataset_type: str, n_step: int, block_size: int, n_batches: int = 20,
             teacher_enabled: bool = False, ingest_kb: bool = False, n_docs_max: int = 0,
             ingest_n_step: int = 3, ingest_checkpoint: bool = False, ingest_step_size: int = None,
             ingest_n_step_min: int = 1, ingest_n_step_max: int = None):
    model.eval()
    keys = ["answer", "thinking"] if dataset_type == "reasoning" else ["answer"]
    losses = {k: [] for k in keys}
    if teacher_enabled:
        losses.update({f"kd_{k}": [] for k in keys})
    # §8ter multi-hop protocol (2026-09-20, long-term-memory-builder): pooled val_answer mixes
    # trivial (1 doc suffices) and genuinely multi-hop (>=2 docs needed) questions -- an ingestion
    # advantage, if real, can only show up in the second group. Per-EXAMPLE loss (not just the
    # batch mean) is needed to split by num_hops, so this always accumulates sum/count separately
    # from the pooled `losses["answer"]` above rather than replacing it.
    hop_loss_sum = {"ge2": 0.0, "le1": 0.0}
    hop_loss_count = {"ge2": 0, "le1": 0}
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        query_tokens = query_tokens_for(dataset_type, batch, block_size)
        target_input = {"answer": batch["answer_target_input"]}
        if dataset_type == "reasoning":
            target_input["thinking"] = batch["thinking_target_input"]
        if ingest_kb:
            ingest_documents(model, batch, n_docs_max, block_size, ingest_n_step,
                              use_checkpoint=ingest_checkpoint, step_size=ingest_step_size,
                              n_step_min=ingest_n_step_min, n_step_max=ingest_n_step_max)
            _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, n_step,
                                target_input=target_input, kb_prebuilt=True)
        else:
            _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, n_step,
                                kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)
        ce_answer = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
        losses["answer"].append(ce_answer.item())
        if dataset_type == "retrieval" and "num_hops" in batch:
            per_tok = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"],
                                       ignore_index=-100, reduction="none")  # (B, T)
            valid = (batch["answer_labels"] != -100)
            n_valid = valid.sum(dim=1).clamp(min=1)
            per_example = per_tok.sum(dim=1) / n_valid  # (B,) mean CE per example
            has_answer = valid.any(dim=1)
            for group, sel in (("ge2", batch["num_hops"] >= 2), ("le1", (batch["num_hops"] >= 0) & (batch["num_hops"] <= 1))):
                sel = sel & has_answer
                if sel.any():
                    hop_loss_sum[group] += per_example[sel].sum().item()
                    hop_loss_count[group] += sel.sum().item()
        if dataset_type == "reasoning":
            ce_thinking = F.cross_entropy(streams["thinking"].transpose(1, 2), batch["thinking_labels"], ignore_index=-100)
            losses["thinking"].append(ce_thinking.item())
        if teacher_enabled:
            kd_answer, kd_thinking = kd_losses(streams, batch, dataset_type)
            losses["kd_answer"].append(kd_answer.item())
            if kd_thinking is not None:
                losses["kd_thinking"].append(kd_thinking.item())
    model.train()
    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in losses.items()}
    if dataset_type == "retrieval":
        for group in ("ge2", "le1"):
            out[f"answer_hops_{group}"] = (hop_loss_sum[group] / hop_loss_count[group]
                                            if hop_loss_count[group] else float("nan"))
            out[f"answer_hops_{group}_n"] = hop_loss_count[group]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_type", required=True, choices=["reasoning", "retrieval"])
    p.add_argument("--data", required=True)
    p.add_argument("--val_data", default=None)
    p.add_argument("--tokenizer", default="lfm2")
    p.add_argument("--n_ctx", type=int, default=256, help="reasoning only: prompt length (flat, depth=0)")
    p.add_argument("--n_docs_max", type=int, default=10,
                   help="retrieval only: max context documents per example, each its own block_size-token "
                        "block -- HotpotQA distractor config has ~10 (2 supporting + up to 8 distractors)")
    p.add_argument("--max_thinking_len", type=int, default=1024, help="reasoning only")
    p.add_argument("--max_answer_len", type=int, default=64)
    p.add_argument("--depth", type=int, default=0,
                   help="reasoning: 0 (flat, single prompt block, no multi-document structure to index). "
                        "retrieval: MUST be 1 -- see RetrievalPromptDataset, one summary node per document.")
    p.add_argument("--block_size", type=int, default=16,
                   help="reasoning: HierarchicalMemory's block_size (depth=0 -> unused). "
                        "retrieval: tokens per document block -- MUST match what RetrievalPromptDataset "
                        "used to build kb_tokens (same --block_size value drives both).")
    p.add_argument("--n_register", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=1)
    p.add_argument("--n_step", type=int, default=6)
    p.add_argument("--pool_n_head", type=int, default=1)
    p.add_argument("--k_dim", type=int, default=None)
    p.add_argument("--disable_kb", action="store_true")
    p.add_argument("--ingest_kb", action="store_true",
                    help="spec §8ter, retrieval only: build the KB by running each document through "
                         "model.ingest() (the model's own recurrent loop + sm_write_proj) instead of "
                         "HierarchicalMemory.build()'s direct k_proj/v_proj projection. MVP: intra-batch "
                         "only, no inter-batch cache -- every document is re-ingested every batch.")
    p.add_argument("--ingest_n_step", type=int, default=3,
                    help="--ingest_kb only: fixed number of recurrent iterations used to ingest EACH "
                         "document (independent of --n_step, the QA pass's iteration count). Also the "
                         "CAP used by --ingest_step_size when --ingest_n_step_max is not set.")
    p.add_argument("--ingest_step_size", type=int, default=None,
                    help="--ingest_kb only, spec §8ter option B (2026-09-20): when set, n_step for each "
                         "document slot is derived from that slot's real length instead of being fixed "
                         "to --ingest_n_step -- ceil(real_len / ingest_step_size), clamped to "
                         "[--ingest_n_step_min, --ingest_n_step_max]. Unset (default): every document "
                         "uses the fixed --ingest_n_step, regardless of length.")
    p.add_argument("--ingest_n_step_min", type=int, default=1)
    p.add_argument("--ingest_n_step_max", type=int, default=None,
                    help="defaults to --ingest_n_step when --ingest_step_size is set.")
    p.add_argument("--ingest_checkpoint", action="store_true",
                    help="--ingest_kb only, spec §8ter option A (2026-09-20): wraps each document's "
                         "model.ingest() call in torch.utils.checkpoint.checkpoint() to avoid keeping "
                         "every ingested document's activations resident for the rest of the batch's "
                         "forward pass -- trades memory for a backward-time recompute. Safe because "
                         "ingest() is a pure function of its explicit (doc_tokens, prior_k, prior_v, "
                         "prior_mask) arguments, not of self.memory's mutable state (see Thinker.ingest's "
                         "docstring). Numerically identical to the default, just different memory/compute "
                         "tradeoff -- use when --ingest_kb runs out of memory with many/long documents.")
    p.add_argument("--use_ff", action="store_true")
    p.add_argument("--ff_hidden_mult", type=int, default=4)
    p.add_argument("--thinking_weight", type=float, default=1.0,
                   help="reasoning only: loss = ce_answer + thinking_weight * ce_thinking")
    p.add_argument("--answer_n_layers", type=int, default=1,
                   help="OutputStream cross-attention depth for 'answer' (1-3, spec §11bis ablation, "
                        "see train_real_text.py's flag of the same name for the hypothesis).")
    p.add_argument("--thinking_n_layers", type=int, default=1, help="reasoning only, same as --answer_n_layers")
    p.add_argument("--teacher_targets", default=None,
                    help="path to a precompute_teacher_targets.py .npz computed on the SAME --data file's "
                         "'text' field (same row order/tokenizer) -- enables logit-level KD on the answer "
                         "stream (and thinking, for --dataset_type reasoning) via topk_kd_loss(). See "
                         "data/prompt_response_dataset.py's PromptResponseTeacherTargets for the alignment "
                         "scheme and its per-example CE-only fallback. Requires --kd_alpha > 0 to have any effect.")
    p.add_argument("--val_teacher_targets", default=None,
                    help="same as --teacher_targets, computed on --val_data instead, for held-out KD reporting.")
    p.add_argument("--teacher_max_length", type=int, default=4096,
                    help="MUST match the --max_length used for the precompute_teacher_targets.py run "
                         "(alignment re-tokenizes 'text' with the same truncation to land on the same rows).")
    p.add_argument("--kd_alpha", type=float, default=0.5,
                    help="loss = (1-kd_alpha)*ce + kd_alpha*kd, same convention as train_sft.py/"
                         "train_real_text.py. Ignored when --teacher_targets is not given (pure CE).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_warmup_init", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bf16", action="store_true",
                    help="run the forward pass (model + loss) under torch.autocast(dtype=bfloat16) -- ~1.68x "
                         "measured elsewhere on this project (train_sft.py's 500M-core sweep). Master weights/"
                         "optimizer state stay fp32; no GradScaler needed for bf16.")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--extrapolate_n_steps", default=None,
                   help="comma-separated n_step_test values, probed on held-out --val_data after training "
                        "-- does 'thinking longer' at inference help solve reasoning/retrieval examples "
                        "it was never trained with that many steps on?")
    p.add_argument("--save_checkpoint_path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_run_args(p)
    args = p.parse_args()
    logger = logger_from_args(args)

    if args.ingest_kb:
        assert args.dataset_type == "retrieval", "--ingest_kb is only meaningful for --dataset_type retrieval"

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)

    train_ds = build_dataset(args.dataset_type, args.data, tok, args, teacher_targets=args.teacher_targets)
    print(f"loaded {len(train_ds)} {args.dataset_type} examples from {args.data}", flush=True)
    if train_ds.teacher is not None:
        print(f"loaded Teacher targets from {args.teacher_targets}: K={train_ds.teacher.k} -- "
              f"KD enabled, kd_alpha={args.kd_alpha}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data:
        val_ds = build_dataset(args.dataset_type, args.val_data, tok, args, teacher_targets=args.val_teacher_targets)
        print(f"loaded held-out val: {len(val_ds)} examples from {args.val_data}", flush=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.dataset_type == "reasoning":
        stream_dims = {"thinking": vocab_size, "answer": vocab_size}
        stream_sequence = {"thinking": True, "answer": True}
        stream_n_layers = {"thinking": args.thinking_n_layers, "answer": args.answer_n_layers}
        max_target_len = max(args.max_thinking_len, args.max_answer_len)
    else:
        stream_dims = {"answer": vocab_size}
        stream_sequence = {"answer": True}
        stream_n_layers = {"answer": args.answer_n_layers}
        max_target_len = args.max_answer_len

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
        stream_n_layers=stream_n_layers, use_ingest_token=args.ingest_kb,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"dataset_type={args.dataset_type} d_model={args.d_model} n_step={args.n_step} "
          f"params={n_params/1e6:.2f}M device={device}", flush=True)

    lr_warmup_init = args.lr_warmup_init if args.lr_warmup_init is not None else args.lr / 10

    def lr_at(step: int) -> float:
        if args.lr_warmup_steps <= 0 or step >= args.lr_warmup_steps:
            return args.lr
        return lr_warmup_init + (args.lr - lr_warmup_init) * (step / args.lr_warmup_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=1e-2)
    model.train()

    step, loss_hist = 0, []
    start_time = time.time()
    done = False
    while not done:
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed > args.max_time_minutes * 60 or step >= args.max_steps:
                print(f"Budget reached at step {step}. Stopping.", flush=True)
                done = True
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            query_tokens = query_tokens_for(args.dataset_type, batch, args.block_size)
            target_input = {"answer": batch["answer_target_input"]}
            if args.dataset_type == "reasoning":
                target_input["thinking"] = batch["thinking_target_input"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                if args.ingest_kb:
                    ingest_documents(model, batch, args.n_docs_max, args.block_size, args.ingest_n_step,
                                      use_checkpoint=args.ingest_checkpoint, step_size=args.ingest_step_size,
                                      n_step_min=args.ingest_n_step_min, n_step_max=args.ingest_n_step_max)
                    _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, args.n_step,
                                       target_input=target_input, kb_prebuilt=True)
                else:
                    _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, args.n_step,
                                       kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input)

                ce_answer = F.cross_entropy(streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
                if args.dataset_type == "reasoning":
                    ce_thinking = F.cross_entropy(streams["thinking"].transpose(1, 2), batch["thinking_labels"], ignore_index=-100)
                    ce_loss = ce_answer + args.thinking_weight * ce_thinking
                else:
                    ce_thinking = None
                    ce_loss = ce_answer

                if train_ds.teacher is not None:
                    kd_answer, kd_thinking = kd_losses(streams, batch, args.dataset_type)
                    kd_loss = kd_answer + args.thinking_weight * kd_thinking if kd_thinking is not None else kd_answer
                    loss = (1 - args.kd_alpha) * ce_loss + args.kd_alpha * kd_loss
                else:
                    kd_answer = kd_thinking = None
                    loss = ce_loss

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_hist.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
                mean_loss = sum(loss_hist[-args.log_every:]) / len(loss_hist[-args.log_every:])
                extra = f" ce_thinking={ce_thinking.item():.4f}" if ce_thinking is not None else ""
                kd_extra = ""
                if kd_answer is not None:
                    kd_extra = f" kd_answer={kd_answer.item():.4f}"
                    if kd_thinking is not None:
                        kd_extra += f" kd_thinking={kd_thinking.item():.4f}"
                print(f"step={step:6d} elapsed={elapsed/60:.2f}m loss={mean_loss:.4f} "
                      f"ce_answer={ce_answer.item():.4f}{extra}{kd_extra} lr={lr_at(step):.2e}", flush=True)
                log_kwargs = {"kd_answer": kd_answer.item()} if kd_answer is not None else {}
                if kd_thinking is not None:
                    log_kwargs["kd_thinking"] = kd_thinking.item()
                logger.progress(step, loss=mean_loss, ce_answer=ce_answer.item(), lr=lr_at(step), **log_kwargs)
            if val_loader is not None and (step % args.val_every == 0 or step == 1):
                val_losses = evaluate(model, val_loader, device, args.dataset_type, args.n_step,
                                       args.block_size, n_batches=args.val_batches,
                                       teacher_enabled=val_ds.teacher is not None,
                                       ingest_kb=args.ingest_kb, n_docs_max=args.n_docs_max,
                                       ingest_n_step=args.ingest_n_step, ingest_checkpoint=args.ingest_checkpoint,
                                       ingest_step_size=args.ingest_step_size,
                                       ingest_n_step_min=args.ingest_n_step_min,
                                       ingest_n_step_max=args.ingest_n_step_max)
                print(f"step={step:6d} VAL {val_losses}", flush=True)
                logger.progress(step, **{f"val_{k}": v for k, v in val_losses.items()})

    elapsed = time.time() - start_time
    print("---", flush=True)
    print(f"final_loss: {sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1):.4f}", flush=True)
    print(f"num_steps: {step}", flush=True)
    print(f"training_seconds: {elapsed:.1f}", flush=True)

    if args.save_checkpoint_path:
        import os
        ckpt_path = args.save_checkpoint_path
        if ckpt_path.endswith("/") or os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_path = os.path.join(ckpt_path, f"{args.run_id or 'adhoc'}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint saved to {ckpt_path}", flush=True)

    extrapolation_results = {}
    if args.extrapolate_n_steps and val_loader is not None:
        print("--- extrapolation probe (n_step_test vs training n_step), held-out ---", flush=True)
        for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
            r = evaluate(model, val_loader, device, args.dataset_type, n_step_test, args.block_size,
                         n_batches=args.val_batches, teacher_enabled=val_ds.teacher is not None)
            extrapolation_results[n_step_test] = r
            marker = " <- training n_step" if n_step_test == args.n_step else ""
            print(f"  n_step_test={n_step_test:3d} {r}{marker}", flush=True)

    logger.finish(summary={
        "final_loss": sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1),
        "num_steps": step, "training_seconds": elapsed,
        "extrapolation": extrapolation_results or None,
    })


if __name__ == "__main__":
    main()
