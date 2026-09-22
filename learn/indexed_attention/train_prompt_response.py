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
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args
from data.prompt_response_dataset import ReasoningPromptDataset, RetrievalPromptDataset
from learn.distill.chunked_loss import chunked_ce_kd_loss
from learn.distill.train_sft import embedding_kd_loss, repr_cosine_loss, topk_kd_loss


def build_dataset(dataset_type: str, path: str, tokenizer, args, teacher_targets: str = None, repr_teacher_hidden: str = None):
    repr_kwargs = dict(repr_teacher_hidden=repr_teacher_hidden, repr_teacher_layer=args.repr_teacher_layer,
                        repr_proj_dim=args.repr_proj_dim, repr_seed=args.seed)
    if dataset_type == "reasoning":
        return ReasoningPromptDataset(path, tokenizer, n_ctx=args.n_ctx,
                                       max_thinking_len=args.max_thinking_len,
                                       max_answer_len=args.max_answer_len, pad_id=tokenizer.pad_token_id,
                                       teacher_targets=teacher_targets, teacher_max_length=args.teacher_max_length,
                                       teacher_name=args.teacher_name,
                                       **repr_kwargs)
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
                                       teacher_targets=teacher_targets, teacher_max_length=args.teacher_max_length,
                                       teacher_name=args.teacher_name,
                                       **repr_kwargs)
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
def evaluate(model, loader, device, dataset_type: str, n_step: int, block_size: int, n_batches: int = None,
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
        if n_batches is not None and i >= n_batches:
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
    p.add_argument("--level_dropout_p", type=float, default=0.0,
                    help="HierarchicalMemory's stochastic level dropout (train-time only, core/"
                         "indexed_memory.py's attend()) -- randomly drops whole compressed hierarchy "
                         "levels (never level 0/leaves), probability scaled by depth. Already implemented "
                         "but never wired to a CLI flag before; the one architecture-native regularizer "
                         "available for this model (no dropout anywhere else in it). Default 0.0 keeps "
                         "every existing run's behavior unchanged.")
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
    p.add_argument("--teacher_name", default=None,
                    help="which Teacher's data to read (e.g. qwen_big, lfm2_1_2b) -- only needed when "
                         "--teacher_targets/--repr_teacher_hidden point at a new-format storage-tree directory "
                         "(topk/<split>/ or embedding/<split>/layer_<L>/, has a manifest.json); ignored for an "
                         "old single-.npz/.npz-memmap-dir path, see data/prompt_response_dataset.py.")
    p.add_argument("--teacher_max_length", type=int, default=4096,
                    help="MUST match the --max_length used for the precompute_teacher_targets.py run "
                         "(alignment re-tokenizes 'text' with the same truncation to land on the same rows).")
    p.add_argument("--kd_alpha", type=float, default=0.5,
                    help="loss = (1-kd_alpha)*ce + kd_alpha*kd, same convention as train_sft.py/"
                         "train_real_text.py. Ignored when --teacher_targets is not given (pure CE).")
    p.add_argument("--loss_chunk_size", type=int, default=0,
                    help="Compute CE/KD per stream via chunked_ce_kd_loss() (learn/distill/chunked_loss.py, "
                         "spec §13.3) instead of materializing each stream's full (B,T,vocab) logits at once -- "
                         "memory lever only, same FLOPs. Uses Thinker.forward(return_hidden=True) to get the "
                         "pre-head state and applies each stream's own head chunk-by-chunk. 0 (default) keeps "
                         "the original unchunked code path unchanged.")
    p.add_argument("--embed_teacher_target", default=None,
                    help="extract_teacher_embed_init.py .npz output (embed_init key) -- adds a CONTINUOUS MSE "
                         "anchor pulling Thinker's self.embed toward this fixed Teacher-projected target every "
                         "step (spec §13.1/13.2's embed_init is a one-time copy-at-init only). Requires "
                         "--embed_kd_weight > 0. Row correspondence requires the Teacher's own tokenizer.")
    p.add_argument("--embed_kd_weight", type=float, default=0.0,
                    help="weight on the embedding-anchor MSE term, added UNWEIGHTED on top of the main loss.")
    p.add_argument("--repr_teacher_hidden", default=None,
                    help="precompute_teacher_targets.py --hidden_layers output .npz (a hidden_<layer> key) -- "
                         "adds a cosine representation-distillation loss (dev_notes/indexed_attention_experiment_plan.md "
                         "Q2) between an OutputStream's pre-head hidden state (at the SAME response-token "
                         "positions already aligned for logit-KD) and a fixed random-projected Teacher hidden "
                         "state. Requires --teacher_targets (reuses its span alignment), --repr_teacher_layer, "
                         "and --repr_kd_weight > 0.")
    p.add_argument("--val_repr_teacher_hidden", default=None,
                    help="same as --repr_teacher_hidden, computed on --val_data instead.")
    p.add_argument("--repr_teacher_layer", type=int, default=None,
                    help="which hidden_<layer> key to read -- must match the --hidden_layers index used at "
                         "precompute time (e.g. the Teacher's own num_hidden_layers for --hidden_layers last).")
    p.add_argument("--repr_proj_dim", type=int, default=64,
                    help="output dimension of the fixed random orthogonal projection applied to the Teacher's "
                         "raw hidden states -- a learned nn.Linear maps each stream's d_model to this same dim.")
    p.add_argument("--repr_kd_weight", type=float, default=0.0,
                    help="target weight on the representation-distillation cosine loss, ramped linearly from 0 "
                         "over --repr_kd_warmup_steps (spec Q2's 'montée en poids progressive').")
    p.add_argument("--repr_kd_warmup_steps", type=int, default=1000,
                    help="steps over which --repr_kd_weight ramps linearly from 0 -- avoids destabilizing early "
                         "training with a rigid representation-matching constraint (FitNets/MiniLM literature "
                         "risk, raw/Distill-reasonning-stream.md:446).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_warmup_init", type=float, default=None)
    p.add_argument("--lr_decay_to", type=float, default=None,
                    help="if set, enables a Warmup-Stable-Decay schedule (arxiv 2410.05192): lr stays "
                         "flat at --lr from the end of warmup through --lr_stable_frac of --max_steps, "
                         "then cosine-decays down to this floor over the remaining steps. Unset (default) "
                         "keeps lr flat after warmup for the whole run, unchanged from before this flag existed. "
                         "Prefer this over smearing the decay across the full horizon when the val optimum "
                         "appears early (observed here: step ~750/6000) -- WSD only pays the decay cost near "
                         "the end, not throughout training.")
    p.add_argument("--lr_stable_frac", type=float, default=0.7,
                    help="fraction of --max_steps (after warmup) to hold lr flat at --lr before WSD's decay "
                         "phase begins -- only used when --lr_decay_to is set")
    p.add_argument("--lr_decay_steps", type=int, default=None,
                    help="fixed length (in steps) of the WSD decay phase, starting right after the stable "
                         "phase -- if unset (default), decay spans everything from the end of the stable "
                         "phase to --max_steps (can be very slow if the optimum is much earlier than "
                         "max_steps, e.g. an early-stopping run cut short before a slow decay has any real "
                         "effect). Set this explicitly to a short window (e.g. a few hundred/thousand steps) "
                         "when combining WSD with --patience, so the anneal actually completes before "
                         "patience would otherwise stop training. lr stays at --lr_decay_to for any step "
                         "past the end of this window.")
    p.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience in number of --val_every evals without a new best "
                         "val_answer -- if set, training stops as soon as patience is exhausted instead of "
                         "always running to --max_steps. Complements the LR schedule rather than replacing "
                         "it (a schedule alone does not reliably stop the overfitting-after-the-optimum "
                         "pattern seen on this task).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bf16", action="store_true",
                    help="run the forward pass (model + loss) under torch.autocast(dtype=bfloat16) -- ~1.68x "
                         "measured elsewhere on this project (train_sft.py's 500M-core sweep). Master weights/"
                         "optimizer state stay fp32; no GradScaler needed for bf16.")
    p.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader worker processes -- 0 (default, unchanged behavior) does loading in the "
                         "main process. Only worth raising if the GPU sits idle waiting on data (unlikely at "
                         "this model's size, but free to check).")
    p.add_argument("--compile", action="store_true",
                    help="torch.compile(model) before training -- typically 1.3-2x on recent GPUs, effectively "
                         "free when it works. Opt-in: the model's dict-shaped target_input/stream_outputs and "
                         "--ingest_kb's per-document Python loop are dynamic-control-flow-heavy, which can make "
                         "compilation slow/fragile -- verify with a short smoke test before trusting it on a "
                         "long run, don't assume it just works.")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=None,
                    help="cap the training-time val_answer to this many batches instead of the full "
                         "--val_data set -- unset (default) evaluates every held-out example, so val_answer "
                         "is directly comparable to any other full-val-set number (e.g. eval_llm_baseline_"
                         "retrieval.py's reference-LLM baselines). Set explicitly (e.g. 20) only to speed up "
                         "a quick sweep at the cost of evaluating on a smaller, fixed (shuffle=False) subset "
                         "of --val_data -- was silently the default (20) before this flag existed, which made "
                         "every val_answer reported by this script incomparable to a full-val-set number "
                         "without noticing (see dev_notes/experiments/prompt_response_pipeline.md 2026-09-22).")
    p.add_argument("--extrapolate_n_steps", default=None,
                   help="comma-separated n_step_test values, probed on held-out --val_data after training "
                        "-- does 'thinking longer' at inference help solve reasoning/retrieval examples "
                        "it was never trained with that many steps on?")
    p.add_argument("--init_from_checkpoint", default=None,
                    help="warm-start: load model.state_dict() from this .pt before training -- weights only, "
                         "fresh optimizer/LR schedule/step counter (no real resume mechanism yet).")
    p.add_argument("--save_checkpoint_path", default=None)
    p.add_argument("--save_best_checkpoint_path", default=None,
                    help="save model.state_dict() here every time val_answer improves (not just at the end) "
                         "-- lets a long run be stopped early at its best point if overfitting sets back in, "
                         "per model-design's 2026-09-20 flagship-run protocol.")
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

    train_ds = build_dataset(args.dataset_type, args.data, tok, args, teacher_targets=args.teacher_targets,
                              repr_teacher_hidden=args.repr_teacher_hidden)
    print(f"loaded {len(train_ds)} {args.dataset_type} examples from {args.data}", flush=True)
    if train_ds.teacher is not None:
        print(f"loaded Teacher targets from {args.teacher_targets}: K={train_ds.teacher.k} -- "
              f"KD enabled, kd_alpha={args.kd_alpha}", flush=True)
    if train_ds.repr_teacher is not None:
        print(f"loaded repr-KD targets from {args.repr_teacher_hidden} (layer={args.repr_teacher_layer}, "
              f"proj_dim={args.repr_proj_dim}) -- weight={args.repr_kd_weight}, warmup={args.repr_kd_warmup_steps}", flush=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory)

    val_loader = None
    if args.val_data:
        val_ds = build_dataset(args.dataset_type, args.val_data, tok, args, teacher_targets=args.val_teacher_targets,
                                repr_teacher_hidden=args.val_repr_teacher_hidden)
        print(f"loaded held-out val: {len(val_ds)} examples from {args.val_data}", flush=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin_memory)

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
        level_dropout_p=args.level_dropout_p,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims=stream_dims, stream_sequence=stream_sequence, max_target_len=max_target_len,
        stream_n_layers=stream_n_layers, use_ingest_token=args.ingest_kb,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"dataset_type={args.dataset_type} d_model={args.d_model} n_step={args.n_step} "
          f"params={n_params/1e6:.2f}M device={device}", flush=True)

    raw_model = model  # unwrapped module -- state_dict() below always saves/loads THIS, so checkpoints stay
                        # compatible with eval_val_loss.py/etc. regardless of --compile (an OptimizedModule's
                        # own state_dict() has carried an "_orig_mod." key prefix on some torch versions).

    if args.init_from_checkpoint:
        # Warm-start (2026-09-20, model-design): load weights only, fresh optimizer/LR schedule/step
        # counter -- not a real resume (no optimizer state saved), but enough to not throw away
        # progress from a run interrupted by walltime expiry (e.g. reasoning at val_answer=3.317).
        init_sd = torch.load(args.init_from_checkpoint, map_location=device)
        raw_model.load_state_dict(init_sd)
        print(f"warm-started weights from {args.init_from_checkpoint} (fresh optimizer/step)", flush=True)

    embed_teacher_target = None
    if args.embed_teacher_target:
        assert args.embed_kd_weight > 0, "--embed_teacher_target has no effect without --embed_kd_weight > 0"
        embed_teacher_target = torch.from_numpy(
            np.load(args.embed_teacher_target)["embed_init"].astype(np.float32)
        ).to(device)
        print(f"Embedding-KD enabled: anchoring self.embed toward {args.embed_teacher_target} "
              f"(weight={args.embed_kd_weight}, shape={tuple(embed_teacher_target.shape)})", flush=True)

    repr_proj = None
    if train_ds.repr_teacher is not None:
        assert args.repr_kd_weight > 0, "--repr_teacher_hidden has no effect without --repr_kd_weight > 0"
        # One shared adapter across streams (thinking/answer both have the same d_model) -- simplest
        # option for this exploratory version; a per-stream adapter is a natural follow-up if useful.
        repr_proj = torch.nn.Linear(args.d_model, args.repr_proj_dim).to(device)
        print(f"Representation-KD enabled: repr_proj(d_model={args.d_model}->{args.repr_proj_dim}), "
              f"weight={args.repr_kd_weight}, warmup={args.repr_kd_warmup_steps} steps", flush=True)

    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled -- first steps will be slower (compilation), watch for graph breaks", flush=True)

    lr_warmup_init = args.lr_warmup_init if args.lr_warmup_init is not None else args.lr / 10

    def lr_at(step: int) -> float:
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return lr_warmup_init + (args.lr - lr_warmup_init) * (step / args.lr_warmup_steps)
        if args.lr_decay_to is None:
            return args.lr
        post_warmup_span = max(args.max_steps - args.lr_warmup_steps, 1)
        stable_steps = args.lr_warmup_steps + int(post_warmup_span * args.lr_stable_frac)
        if step < stable_steps:
            return args.lr
        decay_span = args.lr_decay_steps if args.lr_decay_steps is not None else max(args.max_steps - stable_steps, 1)
        progress = min(max(step - stable_steps, 0) / decay_span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_decay_to + (args.lr - args.lr_decay_to) * cosine

    opt_params = list(model.parameters())
    if repr_proj is not None:
        opt_params = opt_params + list(repr_proj.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=lr_at(0), weight_decay=1e-2,
                                   fused=(device.type == "cuda"))
    model.train()

    step, loss_hist = 0, []
    best_val_answer = None
    evals_since_best = 0
    start_time = time.time()
    done = False
    while not done:
        for batch in train_loader:
            elapsed = time.time() - start_time
            if elapsed > args.max_time_minutes * 60 or step >= args.max_steps:
                print(f"Budget reached at step {step}. Stopping.", flush=True)
                done = True
                break
            if args.patience is not None and evals_since_best >= args.patience:
                print(f"Early stopping at step {step}: no new best val_answer in "
                      f"{args.patience} evals.", flush=True)
                done = True
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            query_tokens = query_tokens_for(args.dataset_type, batch, args.block_size)
            target_input = {"answer": batch["answer_target_input"]}
            if args.dataset_type == "reasoning":
                target_input["thinking"] = batch["thinking_target_input"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                # want_hidden: either repr-KD needs the pre-head state directly, or chunked CE/KD
                # needs it to apply each stream's head chunk-by-chunk instead of all at once.
                want_hidden = repr_proj is not None or args.loss_chunk_size > 0
                if args.ingest_kb:
                    ingest_documents(model, batch, args.n_docs_max, args.block_size, args.ingest_n_step,
                                      use_checkpoint=args.ingest_checkpoint, step_size=args.ingest_step_size,
                                      n_step_min=args.ingest_n_step_min, n_step_max=args.ingest_n_step_max)
                    _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, args.n_step,
                                       target_input=target_input, kb_prebuilt=True, return_hidden=want_hidden)
                else:
                    _, streams = model(batch["kb_tokens"], batch["kb_source_ids"], query_tokens, args.n_step,
                                       kb_leaf_mask=batch["kb_leaf_mask"], target_input=target_input, return_hidden=want_hidden)
                # `streams` holds pre-head hidden states (B,T,d_model) when want_hidden, else logits
                # (B,T,vocab) exactly as before -- hidden_streams is kept around for the repr-KD term
                # below regardless of which of the two paths produced them.
                hidden_streams = streams if want_hidden else None

                if args.loss_chunk_size > 0:
                    def stream_ce_kd(name, labels_key):
                        hidden = streams[name]
                        flat_hidden = hidden.reshape(-1, hidden.size(-1))
                        flat_labels = batch[labels_key].reshape(-1)
                        head = raw_model.streams[name].head
                        kd_kwargs = {}
                        if train_ds.teacher is not None:
                            k = train_ds.teacher.k
                            kd_kwargs = dict(
                                teacher_indices=batch[f"{name}_kd_indices"].reshape(-1, k),
                                teacher_values=batch[f"{name}_kd_values"].reshape(-1, k),
                                teacher_residual=batch[f"{name}_kd_residual"].reshape(-1),
                                teacher_mask=batch[f"{name}_kd_mask"].reshape(-1),
                            )
                        return chunked_ce_kd_loss(flat_hidden, head, flat_labels, chunk_size=args.loss_chunk_size, **kd_kwargs)

                    ce_answer, kd_answer = stream_ce_kd("answer", "answer_labels")
                    if args.dataset_type == "reasoning":
                        ce_thinking, kd_thinking = stream_ce_kd("thinking", "thinking_labels")
                        ce_loss = ce_answer + args.thinking_weight * ce_thinking
                    else:
                        ce_thinking = kd_thinking = None
                        ce_loss = ce_answer
                    if train_ds.teacher is not None:
                        kd_loss = kd_answer + args.thinking_weight * kd_thinking if kd_thinking is not None else kd_answer
                        loss = (1 - args.kd_alpha) * ce_loss + args.kd_alpha * kd_loss
                    else:
                        loss = ce_loss
                else:
                    logit_streams = (
                        {name: raw_model.streams[name].head(h) for name, h in streams.items()}
                        if want_hidden else streams
                    )
                    ce_answer = F.cross_entropy(logit_streams["answer"].transpose(1, 2), batch["answer_labels"], ignore_index=-100)
                    if args.dataset_type == "reasoning":
                        ce_thinking = F.cross_entropy(logit_streams["thinking"].transpose(1, 2), batch["thinking_labels"], ignore_index=-100)
                        ce_loss = ce_answer + args.thinking_weight * ce_thinking
                    else:
                        ce_thinking = None
                        ce_loss = ce_answer

                    if train_ds.teacher is not None:
                        kd_answer, kd_thinking = kd_losses(logit_streams, batch, args.dataset_type)
                        kd_loss = kd_answer + args.thinking_weight * kd_thinking if kd_thinking is not None else kd_answer
                        loss = (1 - args.kd_alpha) * ce_loss + args.kd_alpha * kd_loss
                    else:
                        kd_answer = kd_thinking = None
                        loss = ce_loss

                embed_kd_value = None
                if embed_teacher_target is not None:
                    embed_kd_value = embedding_kd_loss(raw_model.embed.weight, embed_teacher_target)
                    loss = loss + args.embed_kd_weight * embed_kd_value

                repr_kd_value = None
                if repr_proj is not None:
                    repr_terms = []
                    for name in (["answer", "thinking"] if args.dataset_type == "reasoning" else ["answer"]):
                        repr_mask_key = f"{name}_repr_mask"
                        if repr_mask_key not in batch or not batch[repr_mask_key].any():
                            continue
                        student_proj = repr_proj(hidden_streams[name]).float()
                        repr_terms.append(repr_cosine_loss(student_proj, batch[f"{name}_repr_target"], batch[repr_mask_key]))
                    if repr_terms:
                        repr_kd_value = torch.stack(repr_terms).mean()
                        repr_kd_ramp = min(1.0, step / max(1, args.repr_kd_warmup_steps))
                        loss = loss + repr_kd_ramp * args.repr_kd_weight * repr_kd_value

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
                extra2 = ""
                if embed_kd_value is not None:
                    extra2 += f" embed_kd={embed_kd_value.item():.4f}"
                if repr_kd_value is not None:
                    extra2 += f" repr_kd={repr_kd_value.item():.4f}(w={repr_kd_ramp * args.repr_kd_weight:.4f})"
                print(f"step={step:6d} elapsed={elapsed/60:.2f}m loss={mean_loss:.4f} "
                      f"ce_answer={ce_answer.item():.4f}{extra}{kd_extra}{extra2} lr={lr_at(step):.2e}", flush=True)
                log_kwargs = {"kd_answer": kd_answer.item()} if kd_answer is not None else {}
                if kd_thinking is not None:
                    log_kwargs["kd_thinking"] = kd_thinking.item()
                if embed_kd_value is not None:
                    log_kwargs["embed_kd"] = embed_kd_value.item()
                if repr_kd_value is not None:
                    log_kwargs["repr_kd"] = repr_kd_value.item()
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
                cur_val = val_losses["answer"]
                if best_val_answer is None or cur_val < best_val_answer:
                    best_val_answer = cur_val
                    evals_since_best = 0
                    if args.save_best_checkpoint_path is not None:
                        torch.save(raw_model.state_dict(), args.save_best_checkpoint_path)
                        print(f"  new best val_answer={cur_val:.4f} -> checkpoint saved to "
                              f"{args.save_best_checkpoint_path}", flush=True)
                else:
                    evals_since_best += 1

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
        torch.save(raw_model.state_dict(), ckpt_path)
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
