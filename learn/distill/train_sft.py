"""Minimal from-scratch SFT/KD loop for distillation onboarding.

Trains a RANDOMLY INITIALIZED model (architecture only sourced from
--base_config, weights are never loaded) on the "text" field of a JSONL file
produced by any of the scripts/prepare_*_data.py scripts (reasoning /
general / retrieval all share that field, so this loop is dataset-agnostic).

Two modes:
- Plain Sequence-level SFT (default, no --teacher_targets): next-token
  cross-entropy only. See raw/Distill-getting-start.md for why we start here
  before introducing logit-level KD.
- Logit-level KD (--teacher_targets pointing to a precompute_teacher_targets.py
  .npz output): adds a KL term against the Teacher's Top-K logits, restricted
  to a (K+1)-way categorical over the Teacher's Top-K token indices plus one
  merged "everything else" bucket (reconstructed from the stored residual
  log-sum-exp) -- the only distribution shape recoverable from Top-K-only
  storage. Requires --tokenizer to be the SAME tokenizer used to produce
  --teacher_targets (so token positions and vocab indices line up), and
  --block_size to match (or be <=) the --max_length used for that precompute
  run, since alignment is purely positional (no re-tokenization/alignment
  step is done here).

Output format at the end matches program.md's convention (best_loss /
training_seconds / num_steps / num_params_M) for consistency with the rest
of the repo's experiment scripts.

Multi-GPU (2026-09-20, user request: "allow the code to do data parallel,
we'll see at run time if it's worth it even with a large batch"): opt-in
DistributedDataParallel, zero behavior change for the existing single-process
invocation. This project's own established pattern (dev_notes/compute_scheduling.md)
is to pack independent runs onto a GPU rather than DDP a single one -- at
this model size (tens to a few hundred M params) DDP is not expected to be
necessary, but the option is now there to test directly rather than assume.

    # single GPU, unchanged:
    python3 learn/distill/train_sft.py --train_file ... [...]
    # multi-GPU DDP, same script, same flags, no code path change needed:
    torchrun --nproc_per_node=N learn/distill/train_sft.py --train_file ... [...]

Only rank 0 prints/logs (wandb/mlflow)/checkpoints, to avoid N-way duplication.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data.real_text_windows import RealTextWindowDataset
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args
from learn.distill.chunked_loss import chunked_ce_kd_loss


class JsonlTextDataset(Dataset):
    """Tokenizes "text" and, if teacher_targets is given, attaches that
    example's precomputed Top-K indices/values/residual (sliced from the
    flat (total_tokens, ...) npz arrays via `offsets`). Truncates to the
    shorter of block_size and the precomputed token count so student/teacher
    sequences always line up position-for-position.
    """

    def __init__(self, path, tokenizer, block_size, teacher_targets=None,
                 repr_teacher_hidden=None, repr_teacher_layer=None, repr_proj_dim=64, repr_seed=0):
        self.examples = []
        teacher = None
        if teacher_targets is not None:
            npz = np.load(teacher_targets)
            teacher = {
                "indices": npz["indices"], "values": npz["values"],
                "residual": npz["residual"], "offsets": npz["offsets"],
            }
            self.k = int(npz["k"])

        # Q2 (dev_notes/indexed_attention_experiment_plan.md): representation
        # distillation from a precompute_teacher_targets.py --hidden_layers
        # output. The raw hidden_size vector is projected ONCE here via a
        # FIXED random orthogonal matrix (never fit/trained -- same primitive
        # as extract_teacher_embed_init.py's random_projection(), just not
        # imported from there to avoid a GPU-only-script import at CPU-loop
        # dataset-construction time) down to repr_proj_dim, so the stored
        # per-example slices are already small and the projection cost is
        # paid once for the whole file, not per epoch.
        repr_teacher = None
        self.repr_proj_dim = None
        if repr_teacher_hidden is not None:
            assert repr_teacher_layer is not None, "--repr_teacher_layer is required with --repr_teacher_hidden"
            rnpz = np.load(repr_teacher_hidden)
            key = f"hidden_{repr_teacher_layer}"
            assert key in rnpz, f"{repr_teacher_hidden!r} has no {key!r} -- available: {list(rnpz.keys())}"
            raw_hidden = torch.from_numpy(rnpz[key].astype(np.float32))  # (total_tokens, teacher_hidden_size)
            g = torch.Generator().manual_seed(repr_seed)
            P = torch.empty(raw_hidden.shape[1], repr_proj_dim, dtype=torch.float32)
            torch.nn.init.orthogonal_(P, generator=g)
            projected = (raw_hidden @ P).to(torch.float16).numpy()  # (total_tokens, repr_proj_dim)
            repr_teacher = {"projected": projected, "offsets": rnpz["offsets"]}
            self.repr_proj_dim = repr_proj_dim

        with open(path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                ids = tokenizer(row["text"], truncation=True, max_length=block_size)["input_ids"]
                n_cap = len(ids)
                if teacher is not None:
                    n_cap = min(n_cap, teacher["offsets"][i + 1] - teacher["offsets"][i])
                if repr_teacher is not None:
                    n_cap = min(n_cap, repr_teacher["offsets"][i + 1] - repr_teacher["offsets"][i])
                if (teacher is not None or repr_teacher is not None) and n_cap < 2:
                    continue
                ids = ids[:n_cap]
                target = None
                if teacher is not None or repr_teacher is not None:
                    target = {}
                    if teacher is not None:
                        start = teacher["offsets"][i]
                        target.update(
                            indices=teacher["indices"][start:start + n_cap],
                            values=teacher["values"][start:start + n_cap],
                            residual=teacher["residual"][start:start + n_cap],
                        )
                    if repr_teacher is not None:
                        rstart = repr_teacher["offsets"][i]
                        target["repr"] = repr_teacher["projected"][rstart:rstart + n_cap]
                if len(ids) >= 2:
                    self.examples.append((ids, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id, k=None, repr_dim=None):
    ids_batch = [ids for ids, _ in batch]
    max_len = max(len(x) for x in ids_batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        labels[i, :n] = torch.tensor(ids)
        attention_mask[i, :n] = 1
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    if k is not None:
        teacher_indices = torch.zeros((len(batch), max_len, k), dtype=torch.long)
        teacher_values = torch.zeros((len(batch), max_len, k), dtype=torch.float32)
        teacher_residual = torch.zeros((len(batch), max_len), dtype=torch.float32)
        teacher_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, (_, target) in enumerate(batch):
            n = len(target["values"])
            teacher_indices[i, :n] = torch.from_numpy(target["indices"].astype(np.int64))
            teacher_values[i, :n] = torch.from_numpy(target["values"].astype(np.float32))
            teacher_residual[i, :n] = torch.from_numpy(target["residual"].astype(np.float32))
            teacher_mask[i, :n] = True
        out.update(
            teacher_indices=teacher_indices, teacher_values=teacher_values,
            teacher_residual=teacher_residual, teacher_mask=teacher_mask,
        )

    if repr_dim is not None:
        repr_target = torch.zeros((len(batch), max_len, repr_dim), dtype=torch.float32)
        repr_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, (_, target) in enumerate(batch):
            n = len(target["repr"])
            repr_target[i, :n] = torch.from_numpy(target["repr"].astype(np.float32))
            repr_mask[i, :n] = True
        out.update(repr_target=repr_target, repr_mask=repr_mask)
    return out


class RealTextWindowsSFT(Dataset):
    """Baseline C (spec Sec 9): wraps `data.real_text_windows.RealTextWindowDataset`
    so a plain (non-Thinker) causal LM is supervised on the EXACT same windows
    -- same context, same target span -- as `train_real_text.py`'s Thinker runs,
    instead of `JsonlTextDataset`'s whole-document supervision (which is not a
    loss-comparable task: full self-attention over a mostly-trivial-to-predict
    in-block context, vs. a held-out continuation conditioned only on a
    compressed KB/register representation).

    Per window: `input_ids = cat([kb_tokens, target_input])` (length
    `n_ctx + t_tgt`); `target_input` already IS the teacher-forced, shifted-by-
    one input for the target span (see `real_text_windows.py`'s docstring), so
    concatenating it straight after `kb_tokens` reproduces the same
    autoregressive alignment a standard HF causal LM expects -- no separate
    shift needed here, `labels` is set at the SAME index as the token each
    position should predict (masked to -100 over the KB region) and handled by
    the existing `model(**batch).loss` path's automatic internal shift.
    `attention_mask` reuses `kb_leaf_mask` to hide the padded prefix of short
    documents' context, exactly like Thinker does.

    Every window is a fixed, equal length -- no ragged-batch padding logic
    needed (unlike `JsonlTextDataset`/`collate`), a plain stack collate
    suffices (see `collate_real_text_windows`).

    Does NOT carry `R`/register state across windows (`is_first_window` is
    ignored) -- a plain transformer has no such state, each window is an
    independent example. This means the Thinker run being compared against can
    see information beyond its own `n_ctx` tokens (carried over from earlier
    windows in the same document) that this baseline never can -- a known,
    deliberate asymmetry (not a bug to fix here), see
    dev_notes/experiment.log.md's Phase 3 Baselines A/B/C entry.
    """

    def __init__(self, path, tokenizer, n_ctx, t_local, t_tgt, stride=None):
        self.ds = RealTextWindowDataset(path, tokenizer, n_ctx=n_ctx, t_local=t_local, t_tgt=t_tgt, stride=stride)
        self.n_ctx = n_ctx
        self.t_tgt = t_tgt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        w = self.ds[idx]
        input_ids = torch.cat([w["kb_tokens"], w["target_input"]])
        attention_mask = torch.cat([w["kb_leaf_mask"].long(), torch.ones(self.t_tgt, dtype=torch.long)])
        labels = torch.cat([torch.full((self.n_ctx,), -100, dtype=torch.long), w["labels"]])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_real_text_windows(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def is_mup_hidden_weight(name):
    """True for the "hidden" matmul weights muP treats as width-scaling:
    attention/MLP projections inside each transformer block. False for
    embeddings, the (untied) readout, biases, and LayerNorm params -- those
    keep the base init/LR regardless of width, per the muP table.

    GPT-2-arch-specific name matching (transformer.h.{i}.attn.c_attn/c_proj,
    transformer.h.{i}.mlp.c_fc/c_proj) -- would need adjusting for a
    different --base_config architecture.
    """
    return name.endswith(".weight") and any(s in name for s in (".attn.c_attn", ".attn.c_proj", ".mlp.c_fc", ".mlp.c_proj"))


def apply_mup_init(model, width_mult, base_std=0.02, tied_head=True):
    """Re-initializes a freshly-constructed model per the muP recipe (Yang et
    al., Tensor Programs V): hidden matmul weights get variance scaled down
    by width_mult (std = base_std / sqrt(width_mult)) so their contribution
    to the forward pass stays width-independent in expectation. Embeddings,
    LayerNorm, and biases are left at the framework's default init --
    input-like weights don't scale with width under muP.

    tied_head=False (canonical muP): the readout (lm_head, untied from the
    embedding) is zero-init, since its output is separately rescaled by
    1/width_mult at the loss (see main()'s logits scaling).
    tied_head=True (project's deliberate compromise, see README.md's
    student-size notes): keeps the embedding/lm_head SHARED, at its default
    init, to avoid doubling the vocab-sized table when the vocabulary is
    large relative to the core (this project's Teacher-aligned vocab is
    248,077 tokens -- untying would make the head dominate the model at
    small-to-mid core sizes, defeating the point of tracking core size
    separately from vocab size). The 1/width_mult logit rescaling at the
    loss is still applied either way -- that's the part of muP's readout
    treatment that actually controls update dynamics; skipping only the
    zero-init/untie is a narrower deviation from the paper than it might
    look, but is still a deviation, not canonical muP -- revisit if
    transfer doesn't hold up empirically with tying.

    Note: this covers init + LR scaling, the two components of muP with the
    largest empirical effect on hyperparameter transfer in the original
    paper's ablations. It does NOT patch attention's 1/sqrt(d) logit scaling
    to muP's 1/d convention (that requires reaching into the specific
    attention implementation's internals in a version-fragile way) -- treat
    this as "muP-lite"; revisit the attention scaling if transfer doesn't
    hold up empirically across the widths actually used.
    """
    hidden_std = base_std / (width_mult ** 0.5)
    for name, p in model.named_parameters():
        if is_mup_hidden_weight(name):
            torch.nn.init.normal_(p, mean=0.0, std=hidden_std)
    if not tied_head:
        readout = model.get_output_embeddings()
        torch.nn.init.zeros_(readout.weight)


def build_mup_param_groups(model, base_lr, width_mult):
    """Adam-under-muP LR rule: hidden matmul weights get base_lr / width_mult,
    everything else (embeddings, readout, biases, LayerNorm) keeps base_lr.
    """
    hidden, other = [], []
    for name, p in model.named_parameters():
        (hidden if is_mup_hidden_weight(name) else other).append(p)
    return [
        {"params": hidden, "lr": base_lr / width_mult},
        {"params": other, "lr": base_lr},
    ]


def apply_depth_mup_scaling(model, depth_mult):
    """Depth-muP-lite: scale each residual branch's OUTPUT (attention and
    MLP, before it's added back to the residual stream) by 1/sqrt(depth_mult).

    Motivation (found empirically 2026-09-04, see experiment.log.md): this
    project's model-size tiers scale n_layer *and* n_embd together (e.g.
    4->12->25 layers alongside the width jump), but width-only muP
    (apply_mup_init/build_mup_param_groups above) says nothing about depth --
    a deeper residual stream accumulates more per-layer contributions
    regardless of width, and empirically the LR that transferred fine at a
    fixed depth broke badly once depth also grew (lr=0.01, optimal at
    4-layer/40M-core, gave a 6x WORSE loss at 12-layer/150M-core than the
    untuned lr=0.003 baseline). This is the standard fix from the Depth-muP
    line of work (Tensor Programs VI / Bordelon-Noci-Pehlevan-style residual
    branch scaling): multiplying each block's branch output by 1/sqrt(L)
    (relative to a --mup_base_depth reference, so depth_mult=1 at the base
    tier leaves behavior unchanged) keeps the residual stream's per-layer
    update magnitude comparable as more layers are stacked.

    Implemented via forward hooks on each block's .attn/.mlp submodules
    rather than monkeypatching GPT2Block.forward -- GPT2Attention returns
    (attn_output, present) and GPT2MLP returns a plain tensor (verified
    against the installed transformers version), so hooks can rescale the
    branch output without touching the block's internal residual-add logic,
    which is less likely to break across transformers versions than copying
    and patching the block's forward source.

    "Lite" because it does NOT also rescale the branch's own weight init or
    add a corresponding LR term the way width-muP does for width -- this is
    the single largest-effect piece (matching how muP-lite above only did
    width init+LR, not the attention 1/d patch); revisit if this alone
    doesn't fully restore transfer.
    """
    scale = depth_mult ** -0.5

    def scale_attn_output(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] * scale,) + output[1:]
        return output * scale

    def scale_mlp_output(module, inputs, output):
        return output * scale

    for block in model.transformer.h:
        block.attn.register_forward_hook(scale_attn_output)
        block.mlp.register_forward_hook(scale_mlp_output)


def save_checkpoint(path, model, optimizer, step, losses, args, width_mult, depth_mult):
    """Saves everything needed to exactly resume training (state_dict +
    optimizer state, so Adam's momentum/variance isn't reset) or to later
    evaluate the model (args + width_mult/depth_mult, as eval_agreement.py
    needs to reconstruct the architecture and reapply depth-muP's hooks).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "step": step, "losses": losses, "args": vars(args),
            "width_mult": width_mult, "depth_mult": depth_mult,
        },
        path,
    )


def topk_kd_loss(student_logits, teacher_indices, teacher_values, teacher_residual, teacher_mask):
    """KL(teacher || student) over the (K+1)-way categorical formed by the
    Teacher's Top-K token indices plus one merged "everything else" bucket.

    This is the only distribution exactly recoverable from Top-K-only
    storage (see precompute_teacher_targets.py's topk_with_residual): the
    residual is the log-sum-exp of every non-Top-K teacher logit, so
    logsumexp(cat([values, residual])) reconstructs the true full-vocab
    normalizer. The student's matching "everything else" mass is computed
    the same way (mask the student logits at the teacher's chosen indices,
    logsumexp what's left), so the comparison is apples-to-apples even
    though the two models don't share the same Top-K token set.

    student_logits: (B, T, V). teacher_*: (B, T, K) / (B, T) as built by
    collate(). teacher_mask: (B, T) bool, True where a target exists (i.e.
    not padding, and within the precomputed sequence length).
    """
    student_topk = torch.gather(student_logits, dim=-1, index=teacher_indices)  # (B,T,K)
    scatter_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    scatter_mask.scatter_(-1, teacher_indices, True)
    student_masked = student_logits.masked_fill(scatter_mask, float("-inf"))
    student_residual = torch.logsumexp(student_masked, dim=-1)  # (B,T)

    student_logZ = torch.logsumexp(student_logits, dim=-1)  # (B,T)
    teacher_logZ = torch.logsumexp(
        torch.cat([teacher_values, teacher_residual.unsqueeze(-1)], dim=-1), dim=-1
    )  # (B,T)

    teacher_logp_topk = teacher_values - teacher_logZ.unsqueeze(-1)  # (B,T,K)
    teacher_logp_bucket = teacher_residual - teacher_logZ  # (B,T)
    student_logp_topk = student_topk - student_logZ.unsqueeze(-1)  # (B,T,K)
    student_logp_bucket = student_residual - student_logZ  # (B,T)

    teacher_p_topk = teacher_logp_topk.exp()
    teacher_p_bucket = teacher_logp_bucket.exp()
    per_token_kl = (
        (teacher_p_topk * (teacher_logp_topk - student_logp_topk)).sum(dim=-1)
        + teacher_p_bucket * (teacher_logp_bucket - student_logp_bucket)
    )  # (B,T)

    per_token_kl = per_token_kl.masked_fill(~teacher_mask, 0.0)
    denom = teacher_mask.sum().clamp(min=1)
    return per_token_kl.sum() / denom


def embedding_kd_loss(student_weight: torch.Tensor, teacher_weight: torch.Tensor) -> torch.Tensor:
    """MSE between the student's embedding table and a fixed Teacher-projected
    target (learn/distill/extract_teacher_embed_init.py) -- a CONTINUOUS anchor
    toward the Teacher's embedding space throughout training, complementing
    embed_init's one-time copy-at-init (core/indexed_thinker_model.py spec
    §13.1/13.2): without this, Adam is free to drift the embedding away from
    the Teacher's space after the very first step. Symmetric with
    topk_kd_loss (logit-level KD) but at the parameter level, not per-example.

    student_weight may have more rows than teacher_weight (e.g. Thinker's
    optional ingest-token row, spec §8ter) -- only the first
    teacher_weight.shape[0] rows are compared, the rest are ignored.

    2026-09-20 fix (experiment-manager, found via a real crash on LFM2):
    the reverse can also happen -- LFM2's tokenizer `len()` (64400, the actual
    vocabulary) is SMALLER than its model config's `vocab_size` (65536, some
    reserved/unused slots) -- so a Teacher embedding extracted at the
    model-config vocab size has MORE rows than the student's embedding table
    built from the tokenizer. Slice to min() of both, not just teacher's
    size, so this works regardless of which side is larger.
    """
    n = min(student_weight.shape[0], teacher_weight.shape[0])
    return F.mse_loss(student_weight[:n], teacher_weight[:n])


def repr_cosine_loss(student_repr: torch.Tensor, teacher_repr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, averaged over valid (masked) token positions --
    representation-distillation loss (dev_notes/indexed_attention_experiment_plan.md
    Q2): cosine rather than raw MSE per spec §11bis's flagged instability risk
    with an unnormalized MSE between latents of very different natural scales.

    student_repr/teacher_repr: (B,T,D) (D must already match -- the caller is
    responsible for projecting the student's hidden size to the Teacher's
    projected D, e.g. via a small trainable nn.Linear). mask: (B,T) bool.
    """
    cos = F.cosine_similarity(student_repr, teacher_repr, dim=-1)  # (B,T)
    per_tok = (1 - cos).masked_fill(~mask, 0.0)
    denom = mask.sum().clamp(min=1)
    return per_tok.sum() / denom


def evaluate_val(model, val_loader, device, args, width_mult):
    """Full pass over --val_file/--val_teacher_targets, CE+KD averaged over
    batches (no grad, eval mode) -- the train-vs-val curve this gives (called
    every --val_every steps) is what shows WHEN KD memorization starts,
    rather than just a single post-hoc number (see experiment.log.md).
    """
    model.eval()
    total_ce, total_kd, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {kk: v.to(device) for kk, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                labels = batch.pop("labels")
                teacher_indices = batch.pop("teacher_indices")
                teacher_values = batch.pop("teacher_values")
                teacher_residual = batch.pop("teacher_residual")
                teacher_mask = batch.pop("teacher_mask")
                logits = model(**batch).logits
                if args.mup:
                    logits = logits / width_mult
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                kd = topk_kd_loss(logits, teacher_indices, teacher_values, teacher_residual, teacher_mask)
            total_ce += ce.item()
            total_kd += kd.item()
            n_batches += 1
    model.train()
    val_ce, val_kd = total_ce / n_batches, total_kd / n_batches
    return val_ce, val_kd, (1 - args.kd_alpha) * val_ce + args.kd_alpha * val_kd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--tokenizer", default="lfm2", help="HF repo id, or a family alias from core/model_families.py (lfm2/olmo/qwen)")
    parser.add_argument("--base_config", default="gpt2", help="HF model id to source the architecture config from")
    parser.add_argument("--n_layer", type=int, default=None, help="override config n_layer (smaller/faster smoke-test model)")
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_time_minutes", type=float, default=10.0)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher_targets", default=None,
        help="precompute_teacher_targets.py .npz output -- enables KD (KL on Top-K logits) "
             "on top of the plain CE loss. --tokenizer must match the one used to produce it.",
    )
    parser.add_argument("--kd_alpha", type=float, default=0.5, help="weight on the KD/KL term; (1-alpha) on CE")
    parser.add_argument(
        "--loss_chunk_size", type=int, default=0,
        help="Compute CE/KD via chunked_ce_kd_loss() (learn/distill/chunked_loss.py, spec §13.3) instead of "
             "materializing the full (B,T,vocab) logits tensor at once -- a memory lever only (same FLOPs, "
             "trades peak memory for a recompute at backward via torch.utils.checkpoint), meant to unlock a "
             "bigger --batch_size at the large Teacher-aligned vocab head. 0 (default) keeps the original, "
             "unchunked code path byte-for-byte unchanged.",
    )
    parser.add_argument(
        "--embed_teacher_target", default=None,
        help="extract_teacher_embed_init.py .npz output (embed_init key) -- adds a CONTINUOUS MSE anchor "
             "pulling the student's input embedding table toward this fixed Teacher-projected target every "
             "step, on top of whatever the main CE/KD loss already is (spec §13.1/13.2's embed_init is a "
             "one-time copy-at-init only; Adam is then free to drift away from it -- this keeps pulling it "
             "back throughout training instead). Requires --embed_kd_weight > 0 to have any effect. Row "
             "correspondence requires the SAME tokenizer as the Teacher used to build this target (§13.1).",
    )
    parser.add_argument(
        "--embed_kd_weight", type=float, default=0.0,
        help="weight on the embedding-anchor MSE term, added UNWEIGHTED on top of the main loss (not part "
             "of the (1-kd_alpha)/kd_alpha split against CE/logit-KD).",
    )
    parser.add_argument(
        "--val_repr_teacher_hidden", default=None,
        help="same as --repr_teacher_hidden, computed on --val_file instead, for held-out repr-KD loss reporting.",
    )
    parser.add_argument(
        "--repr_teacher_hidden", default=None,
        help="precompute_teacher_targets.py --hidden_layers output .npz (a hidden_<layer> key) -- adds a "
             "cosine-similarity representation-distillation loss (dev_notes/indexed_attention_experiment_plan.md "
             "Q2) between a chosen student transformer block's hidden state and a FIXED random-orthogonal "
             "projection of the Teacher's raw hidden state at the same token positions (never fit/trained, "
             "just a fixed dimensionality reduction -- see --repr_proj_dim). Requires --repr_teacher_layer and "
             "--repr_kd_weight > 0 to have any effect.",
    )
    parser.add_argument(
        "--repr_teacher_layer", type=int, default=None,
        help="which hidden_<layer> key to read from --repr_teacher_hidden -- must match the --hidden_layers "
             "index used at precompute time (e.g. the Teacher's own num_hidden_layers for --hidden_layers last).",
    )
    parser.add_argument(
        "--repr_student_layer", type=int, default=-1,
        help="which entry of output_hidden_states to align (0 = embedding output, -1 = final block's output, "
             "the default) -- indexes the SAME tuple convention parse_hidden_layers() uses at precompute time.",
    )
    parser.add_argument(
        "--repr_proj_dim", type=int, default=64,
        help="output dimension of the fixed random orthogonal projection applied to the Teacher's raw hidden "
             "states before comparison (a learned nn.Linear maps the student's own hidden size to this same "
             "dimension) -- keeps the comparison space small without needing an SVD fit pass over the corpus.",
    )
    parser.add_argument(
        "--repr_kd_weight", type=float, default=0.0,
        help="target weight on the representation-distillation cosine loss, ramped linearly from 0 over "
             "--repr_kd_warmup_steps (spec Q2's 'montée en poids progressive').",
    )
    parser.add_argument(
        "--repr_kd_warmup_steps", type=int, default=1000,
        help="steps over which --repr_kd_weight ramps linearly from 0 -- avoids destabilizing early training "
             "with a rigid representation-matching constraint before the student has learned anything useful "
             "(FitNets/MiniLM literature risk on overly rigid latent geometry, raw/Distill-reasonning-stream.md:446).",
    )
    parser.add_argument(
        "--mup", action="store_true",
        help="use muP (Yang et al., Tensor Programs V) init + Adam-LR scaling by width, so --lr tuned "
             "at --mup_base_width transfers to a wider --n_embd without retuning. Requires --n_embd. "
             "By default keeps the LM head TIED to the embedding (see --mup_untie_head to use canonical "
             "muP instead) -- a deliberate project compromise since untying doubles an already-large "
             "vocab-sized table; see apply_mup_init()'s docstring.",
    )
    parser.add_argument("--mup_base_width", type=int, default=64, help="reference n_embd the LR/init multipliers are computed against")
    parser.add_argument("--bf16", action="store_true",
                         help="run the forward pass (model + loss) under torch.autocast(dtype=bfloat16). "
                              "Halves activation/logit memory (unlocks larger --batch_size, especially with "
                              "the large Teacher-aligned tied vocab head) and lets matmuls use Tensor Cores, "
                              "making achieved-vs-peak MFU comparisons meaningful (the peak FLOPS figures used "
                              "in README.md are bf16 Tensor Core numbers -- an fp32 run isn't measuring the same "
                              "thing). Master weights/optimizer state stay fp32; no GradScaler needed for bf16 "
                              "(unlike fp16) since it has fp32-like dynamic range.")
    parser.add_argument(
        "--mup_untie_head", action="store_true",
        help="use canonical muP (untied LM head, zero-init) instead of this project's tied-head compromise",
    )
    parser.add_argument(
        "--depth_mup", action="store_true",
        help="also scale each block's residual branch output by 1/sqrt(n_layer / mup_base_depth) -- "
             "Depth-muP-lite, addresses depth (not covered by width-only --mup) when tiers scale n_layer "
             "alongside n_embd. Requires --mup and --n_layer. See apply_depth_mup_scaling()'s docstring.",
    )
    parser.add_argument("--mup_base_depth", type=int, default=4, help="reference n_layer the depth-muP scaling is computed against")
    parser.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
    parser.add_argument("--wandb_project", default="thinker-distill")
    parser.add_argument("--mlflow", action="store_true", help="log this run to a local/file-based MLflow tracking store")
    parser.add_argument(
        "--mlflow_tracking_uri", default="sqlite:////home/jdomguia/thinker/mlflow.db",
        help="MLflow tracking URI -- defaults to a local SQLite DB on the Grid'5000 home (plain file:// store is "
             "deprecated/maintenance-mode in current MLflow). No server process needed; consult later with "
             "`mlflow ui --backend-store-uri <this>` run interactively on the frontend.",
    )
    parser.add_argument("--mlflow_experiment", default="thinker-distill")
    parser.add_argument("--run_name", default=None, help="shared run name for W&B/MLflow; defaults to an auto-generated one")
    parser.add_argument("--wandb_group", default=None, help="W&B group tag, e.g. to cluster a sweep's runs together on the dashboard")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (was previously hardcoded to the torch default, not sweepable)")
    parser.add_argument(
        "--save_dir", default=None,
        help="if set, save the final model (state_dict + optimizer state + reconstruction args) to "
             "<save_dir>/checkpoint.pt, for later evaluation (see eval_agreement.py) or resuming (--resume_from).",
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=0,
        help="if set (and --save_dir is set), also save a checkpoint every N steps, overwriting the same "
             "<save_dir>/checkpoint.pt -- besteffort jobs on Grid'5000 can be preempted with no warning at any "
             "time (observed within ~25 minutes in this project), so a real training run of any real length "
             "should set this rather than relying on the final-only save.",
    )
    parser.add_argument(
        "--resume_from", default=None,
        help="path to a checkpoint.pt (from --save_dir/--checkpoint_every) to resume training from -- restores "
             "model + optimizer state and the step counter/loss history, so muP's Adam momentum isn't reset "
             "and best_loss/num_steps in the final report reflect the FULL run, not just this invocation. "
             "Does NOT restore the DataLoader's exact shuffle position (a new epoch/shuffle order starts on "
             "resume) -- a known simplification, fine for the short validation-slice runs this project uses "
             "so far, revisit if resuming mid-epoch on a full-size dataset ever matters.",
    )
    parser.add_argument(
        "--val_file", default=None,
        help="held-out JSONL (same 'text' field format as --train_file) to periodically evaluate CE/KD loss "
             "against during training -- a train-vs-val curve over steps, not just a single post-hoc number, "
             "to see WHEN (if at all) the KD term starts overfitting (see experiment.log.md's memorization "
             "finding). Requires --val_teacher_targets when --teacher_targets is set (same K/tokenizer).",
    )
    parser.add_argument("--val_teacher_targets", default=None, help="precompute_teacher_targets.py .npz output for --val_file")
    parser.add_argument("--val_every", type=int, default=0, help="if set (and --val_file is set), evaluate on --val_file every N steps")
    parser.add_argument(
        "--real_text_windows", action="store_true",
        help="Baseline C (spec Sec 9, dev_notes/indexed_attention_experiment_plan.md Phase 3): supervise the "
             "SAME context/target windows as a train_real_text.py Thinker run (via "
             "data.real_text_windows.RealTextWindowDataset) instead of JsonlTextDataset's whole-document "
             "supervision -- required for a loss-comparable baseline (see RealTextWindowsSFT's docstring). "
             "--block_size is then derived as --n_ctx + --t_tgt, not read from --block_size directly. Not "
             "compatible with --teacher_targets/--val_file yet.",
    )
    parser.add_argument("--n_ctx", type=int, default=256, help="--real_text_windows only; must match the Thinker run being compared against")
    parser.add_argument("--t_local", type=int, default=32, help="--real_text_windows only; passed through to RealTextWindowDataset, not otherwise used by the plain-transformer path")
    parser.add_argument("--t_tgt", type=int, default=32, help="--real_text_windows only; must match the Thinker run being compared against")
    parser.add_argument("--stride", type=int, default=None, help="--real_text_windows only; defaults to --t_tgt like RealTextWindowDataset itself")
    add_run_args(parser)
    args = parser.parse_args()
    logger = logger_from_args(args)
    if args.real_text_windows:
        assert not args.teacher_targets, "--real_text_windows doesn't support --teacher_targets yet"
        assert not args.val_file, "--real_text_windows doesn't support --val_file yet"
        args.block_size = args.n_ctx + args.t_tgt  # derived, not read from a user-passed --block_size

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.base_config)
    for attr, val in [("n_layer", args.n_layer), ("n_embd", args.n_embd), ("n_head", args.n_head)]:
        if val is not None:
            setattr(config, attr, val)
    config.vocab_size = len(tokenizer)  # base_config's own vocab size is wrong whenever --tokenizer differs
    width_mult = 1.0
    if args.mup:
        if args.n_embd is None:
            raise ValueError("--mup requires --n_embd (the width multiplier is computed from it)")
        if args.mup_untie_head:
            config.tie_word_embeddings = False  # canonical muP: readout scaled independently of the input embedding
        width_mult = args.n_embd / args.mup_base_width
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank, world_size = 0, 1
    is_main = rank == 0
    model = AutoModelForCausalLM.from_config(config)  # random init: architecture only, no pretrained weights
    core_params = sum(p.numel() for n, p in model.named_parameters() if "wte" not in n and "lm_head" not in n)
    head_params = sum(p.numel() for n, p in model.named_parameters() if "wte" in n or "lm_head" in n)
    if args.mup:
        apply_mup_init(model, width_mult, tied_head=not args.mup_untie_head)
        head_desc = "untied+zero-init" if args.mup_untie_head else "TIED (project compromise, not canonical muP -- see --help)"
        print(f"muP enabled: width_mult={width_mult:.2f} (n_embd={args.n_embd} / base {args.mup_base_width}), LM head {head_desc}")
    depth_mult = 1.0
    if args.depth_mup:
        if not args.mup or args.n_layer is None:
            raise ValueError("--depth_mup requires --mup and --n_layer")
        depth_mult = args.n_layer / args.mup_base_depth
        apply_depth_mup_scaling(model, depth_mult)
        print(f"Depth-muP enabled: depth_mult={depth_mult:.2f} (n_layer={args.n_layer} / base {args.mup_base_depth}), residual branches scaled by {depth_mult ** -0.5:.3f}")
    model.to(device)

    embed_teacher_target = None
    if args.embed_teacher_target:
        assert args.embed_kd_weight > 0, "--embed_teacher_target has no effect without --embed_kd_weight > 0"
        embed_teacher_target = torch.from_numpy(
            np.load(args.embed_teacher_target)["embed_init"].astype(np.float32)
        ).to(device)
        print(f"Embedding-KD enabled: anchoring wte toward {args.embed_teacher_target} "
              f"(weight={args.embed_kd_weight}, shape={tuple(embed_teacher_target.shape)})", flush=True)

    repr_proj = None
    if args.repr_teacher_hidden:
        assert args.repr_kd_weight > 0, "--repr_teacher_hidden has no effect without --repr_kd_weight > 0"
        assert args.repr_teacher_layer is not None, "--repr_teacher_hidden requires --repr_teacher_layer"
        # A small TRAINABLE adapter, not part of the fixed random projection applied to the Teacher's
        # side (JsonlTextDataset already did that) -- lets the student's own hidden size differ freely
        # from --repr_proj_dim without needing to know it ahead of the precompute run.
        repr_proj = torch.nn.Linear(config.n_embd, args.repr_proj_dim).to(device)
        print(f"Representation-KD enabled: student block {args.repr_student_layer} -> repr_proj({config.n_embd}->"
              f"{args.repr_proj_dim}) vs {args.repr_teacher_hidden} (weight={args.repr_kd_weight}, "
              f"warmup={args.repr_kd_warmup_steps} steps)", flush=True)

    num_params = sum(p.numel() for p in model.parameters())
    # Core (transformer blocks + positional embedding) vs. head (vocab-sized embedding/lm_head) --
    # the vocab-driven head can dominate at small core sizes; report both so tier comparisons stay
    # meaningful (see README.md's "Student size" notes on why this split matters more at small scale).
    print(f"Model: {num_params / 1e6:.1f}M params total = {core_params / 1e6:.1f}M core + {head_params / 1e6:.1f}M head (random init, arch={args.base_config}), device={device}")

    run_config = {
        "base_config": args.base_config, "n_layer": config.n_layer, "n_embd": config.n_embd, "n_head": config.n_head,
        "block_size": args.block_size, "batch_size": args.batch_size, "lr": args.lr, "max_steps": args.max_steps,
        "kd_alpha": args.kd_alpha if args.teacher_targets else None, "mup": args.mup, "mup_base_width": args.mup_base_width if args.mup else None,
        "width_mult": width_mult, "num_params_M": num_params / 1e6, "core_params_M": core_params / 1e6, "head_params_M": head_params / 1e6,
        "depth_mup": args.depth_mup, "mup_base_depth": args.mup_base_depth if args.depth_mup else None, "depth_mult": depth_mult,
        "weight_decay": args.weight_decay,
    }
    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, group=args.wandb_group, config=run_config)
    if args.mlflow and is_main:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_params(run_config)

    if args.real_text_windows:
        assert not args.repr_teacher_hidden, "--real_text_windows doesn't support --repr_teacher_hidden yet"
        train_ds = RealTextWindowsSFT(args.train_file, tokenizer, args.n_ctx, args.t_local, args.t_tgt, args.stride)
        print(f"Train windows: {len(train_ds)} (real_text_windows mode, n_ctx={args.n_ctx} t_tgt={args.t_tgt} "
              f"stride={args.stride if args.stride is not None else args.t_tgt} -> block_size={args.block_size})")
        k = None
        repr_dim = None
        train_collate = collate_real_text_windows
    else:
        train_ds = JsonlTextDataset(
            args.train_file, tokenizer, args.block_size, teacher_targets=args.teacher_targets,
            repr_teacher_hidden=args.repr_teacher_hidden, repr_teacher_layer=args.repr_teacher_layer,
            repr_proj_dim=args.repr_proj_dim, repr_seed=args.seed,
        )
        print(f"Train examples: {len(train_ds)}" + (f" (KD against {args.teacher_targets}, K={train_ds.k})" if args.teacher_targets else "")
              + (f" (repr-KD against {args.repr_teacher_hidden}, dim={train_ds.repr_proj_dim})" if args.repr_teacher_hidden else ""))
        k = train_ds.k if args.teacher_targets else None
        repr_dim = train_ds.repr_proj_dim
        train_collate = lambda b: collate(b, tokenizer.pad_token_id, k=k, repr_dim=repr_dim)

    # DistributedSampler shards+shuffles the dataset itself (each rank sees a
    # disjoint 1/world_size slice per epoch) -- DataLoader's own shuffle=True
    # must NOT also be set when a sampler is given (mutually exclusive in
    # torch's API). --batch_size stays PER-PROCESS, so the effective global
    # batch size under DDP is batch_size * world_size -- report both so a
    # scaled run isn't silently compared against a single-GPU one at a
    # different effective batch size.
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                               sampler=train_sampler, collate_fn=train_collate)
    if is_distributed and is_main:
        print(f"DDP: world_size={world_size}, per-process batch_size={args.batch_size}, "
              f"effective global batch_size={args.batch_size * world_size}", flush=True)

    val_loader = None
    if args.val_file and args.val_every:
        val_ds = JsonlTextDataset(
            args.val_file, tokenizer, args.block_size, teacher_targets=args.val_teacher_targets,
            repr_teacher_hidden=args.val_repr_teacher_hidden, repr_teacher_layer=args.repr_teacher_layer,
            repr_proj_dim=args.repr_proj_dim, repr_seed=args.seed,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id, k=k, repr_dim=val_ds.repr_proj_dim),
        )
        print(f"Val examples: {len(val_ds)} (evaluated every {args.val_every} steps)")

    # Optimizer built from the RAW (pre-DDP) model: build_mup_param_groups
    # matches on parameter NAMES (is_mup_hidden_weight), which DDP's wrapper
    # would prefix with "module." -- DDP only wraps forward/backward, the
    # underlying Parameter tensors are unchanged, so building the optimizer
    # first and wrapping in DDP after is safe (same objects, no name lookup
    # needed post-wrap).
    if args.mup:
        param_groups = build_mup_param_groups(model, args.lr, width_mult)
    else:
        param_groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if repr_proj is not None:
        # repr_proj is a small standalone adapter (spec Q2), not part of the muP width-scaling
        # scheme -- always at the base (unscaled) LR, like embeddings/LayerNorm/biases. Must be its
        # own param GROUP (not appended to a flat param list) -- AdamW rejects a mix of bare
        # Parameters and group dicts in the same list.
        param_groups = param_groups + [{"params": list(repr_proj.parameters()), "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    model.train()

    step, losses = 0, []
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        # Checkpoints always store the UNWRAPPED module's state_dict (see
        # save_checkpoint's call sites below) -- DDP's own .state_dict() would
        # add a "module." prefix to every key, which a plain (non-DDP) load
        # elsewhere couldn't consume. Load into .module explicitly under DDP
        # so checkpoints stay interchangeable between single-GPU and DDP runs.
        (model.module if is_distributed else model).load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        losses = ckpt["losses"]
        if is_main:
            print(f"Resumed from {args.resume_from} at step {step} (best_loss so far: {min(losses):.6f})")

    start = time.time()
    done = False
    epoch = 0
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # reshuffles differently each epoch under DDP
        epoch += 1
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                repr_target = batch.pop("repr_target", None)
                repr_mask = batch.pop("repr_mask", None)
                want_hidden = repr_target is not None
                raw_model = model.module if is_distributed else model
                student_hidden_for_repr = None

                if args.loss_chunk_size > 0:
                    labels = batch.pop("labels")
                    teacher_indices = batch.pop("teacher_indices", None)
                    teacher_values = batch.pop("teacher_values", None)
                    teacher_residual = batch.pop("teacher_residual", None)
                    teacher_mask = batch.pop("teacher_mask", None)
                    # Bypasses the LM head entirely here -- the whole point of chunking is to never
                    # materialize a full (B,T,vocab) logits tensor, which model(**batch).logits would.
                    base_out = raw_model.transformer(**batch, output_hidden_states=want_hidden)
                    hidden = base_out.last_hidden_state
                    flat_hidden = hidden.reshape(-1, hidden.size(-1))
                    flat_labels = labels.reshape(-1)

                    def head_fn(h, _m=raw_model, _wm=width_mult, _mup=args.mup):
                        out_logits = _m.lm_head(h)
                        return out_logits / _wm if _mup else out_logits

                    kd_kwargs = {}
                    if teacher_indices is not None:
                        kdim = teacher_indices.shape[-1]
                        kd_kwargs = dict(
                            teacher_indices=teacher_indices.reshape(-1, kdim), teacher_values=teacher_values.reshape(-1, kdim),
                            teacher_residual=teacher_residual.reshape(-1), teacher_mask=teacher_mask.reshape(-1),
                        )
                    ce, kd = chunked_ce_kd_loss(flat_hidden, head_fn, flat_labels, chunk_size=args.loss_chunk_size, **kd_kwargs)
                    loss = (1 - args.kd_alpha) * ce + args.kd_alpha * kd if teacher_indices is not None else ce
                    if want_hidden:
                        student_hidden_for_repr = base_out.hidden_states[args.repr_student_layer]
                elif args.teacher_targets:
                    labels = batch.pop("labels")
                    teacher_indices = batch.pop("teacher_indices")
                    teacher_values = batch.pop("teacher_values")
                    teacher_residual = batch.pop("teacher_residual")
                    teacher_mask = batch.pop("teacher_mask")
                    out = model(**batch, output_hidden_states=want_hidden)
                    logits = out.logits
                    if args.mup:
                        logits = logits / width_mult  # muP readout output scaling
                    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                    kd = topk_kd_loss(logits, teacher_indices, teacher_values, teacher_residual, teacher_mask)
                    loss = (1 - args.kd_alpha) * ce + args.kd_alpha * kd
                    if want_hidden:
                        student_hidden_for_repr = out.hidden_states[args.repr_student_layer]
                elif args.mup:
                    labels = batch.pop("labels")
                    out = model(**batch, output_hidden_states=want_hidden)
                    logits = out.logits / width_mult  # muP readout output scaling
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                    if want_hidden:
                        student_hidden_for_repr = out.hidden_states[args.repr_student_layer]
                else:
                    out = model(**batch, output_hidden_states=want_hidden)
                    loss = out.loss
                    if want_hidden:
                        student_hidden_for_repr = out.hidden_states[args.repr_student_layer]

                embed_kd_value = None
                if embed_teacher_target is not None:
                    embed_kd_value = embedding_kd_loss(raw_model.transformer.wte.weight, embed_teacher_target)
                    loss = loss + args.embed_kd_weight * embed_kd_value

                repr_kd_value = None
                if repr_target is not None:
                    student_proj = repr_proj(student_hidden_for_repr).float()
                    repr_kd_value = repr_cosine_loss(student_proj, repr_target, repr_mask)
                    repr_kd_ramp = min(1.0, step / max(1, args.repr_kd_warmup_steps))
                    loss = loss + repr_kd_ramp * args.repr_kd_weight * repr_kd_value
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if is_main and (step % args.log_every == 0 or step == 1):
                step_metrics = {"loss": loss.item()}
                extra = ""
                if args.teacher_targets:
                    step_metrics["ce"] = ce.item()
                    step_metrics["kd"] = kd.item()
                    extra += f" (ce {ce.item():.4f} kd {kd.item():.4f})"
                elif args.loss_chunk_size > 0:
                    step_metrics["ce"] = ce.item()
                    extra += f" (ce {ce.item():.4f}"
                    if teacher_indices is not None:
                        step_metrics["kd"] = kd.item()
                        extra += f" kd {kd.item():.4f}"
                    extra += ")"
                if embed_kd_value is not None:
                    step_metrics["embed_kd"] = embed_kd_value.item()
                    extra += f" embed_kd {embed_kd_value.item():.4f}"
                if repr_kd_value is not None:
                    step_metrics["repr_kd"] = repr_kd_value.item()
                    step_metrics["repr_kd_weight"] = repr_kd_ramp * args.repr_kd_weight
                    extra += f" repr_kd {repr_kd_value.item():.4f} (w={repr_kd_ramp * args.repr_kd_weight:.4f})"
                print(f"step {step} loss {loss.item():.4f}{extra}")
                logger.progress(step, **step_metrics)
                if args.wandb:
                    wandb.log(step_metrics, step=step)
                if args.mlflow:
                    mlflow.log_metrics(step_metrics, step=step)
            if is_main and args.save_dir and args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(os.path.join(args.save_dir, "checkpoint.pt"),
                                 model.module if is_distributed else model, optimizer, step, losses, args, width_mult, depth_mult)
            if is_main and val_loader is not None and (step % args.val_every == 0 or step == 1):
                val_ce, val_kd, val_loss = evaluate_val(model, val_loader, device, args, width_mult)
                print(f"step {step} VAL loss {val_loss:.4f} (ce {val_ce:.4f} kd {val_kd:.4f})")
                if args.wandb:
                    wandb.log({"val_loss": val_loss, "val_ce": val_ce, "val_kd": val_kd}, step=step)
                if args.mlflow:
                    mlflow.log_metrics({"val_loss": val_loss, "val_ce": val_ce, "val_kd": val_kd}, step=step)
            elapsed_min = (time.time() - start) / 60
            if step >= args.max_steps or elapsed_min >= args.max_time_minutes:
                done = True
                break

    elapsed = time.time() - start
    if is_main:
        print("---")
        print(f"best_loss:        {min(losses):.6f}")
        print(f"training_seconds: {elapsed:.1f}")
        print(f"num_steps:        {step}")
        print(f"num_params_M:     {num_params / 1e6:.2f}")

        final_metrics = {"best_loss": min(losses), "training_seconds": elapsed, "num_steps": step}
        # Objectif de distillation : pertes uniquement, pas d'accuracy — donc pas
        # de contrôles triviaux exigibles.
        logger.finish(summary=final_metrics)
        if args.wandb:
            wandb.log(final_metrics)
            wandb.finish()
        if args.mlflow:
            mlflow.log_metrics(final_metrics)
            mlflow.end_run()

        if args.save_dir:
            ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
            save_checkpoint(ckpt_path, model.module if is_distributed else model, optimizer, step, losses, args, width_mult, depth_mult)
            print(f"Saved checkpoint to {ckpt_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
