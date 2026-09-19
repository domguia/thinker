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
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data.real_text_windows import RealTextWindowDataset
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args


class JsonlTextDataset(Dataset):
    """Tokenizes "text" and, if teacher_targets is given, attaches that
    example's precomputed Top-K indices/values/residual (sliced from the
    flat (total_tokens, ...) npz arrays via `offsets`). Truncates to the
    shorter of block_size and the precomputed token count so student/teacher
    sequences always line up position-for-position.
    """

    def __init__(self, path, tokenizer, block_size, teacher_targets=None):
        self.examples = []
        teacher = None
        if teacher_targets is not None:
            npz = np.load(teacher_targets)
            teacher = {
                "indices": npz["indices"], "values": npz["values"],
                "residual": npz["residual"], "offsets": npz["offsets"],
            }
            self.k = int(npz["k"])

        with open(path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                ids = tokenizer(row["text"], truncation=True, max_length=block_size)["input_ids"]
                target = None
                if teacher is not None:
                    start, end = teacher["offsets"][i], teacher["offsets"][i + 1]
                    n = min(len(ids), end - start)
                    if n < 2:
                        continue
                    ids = ids[:n]
                    target = {
                        "indices": teacher["indices"][start:start + n],
                        "values": teacher["values"][start:start + n],
                        "residual": teacher["residual"][start:start + n],
                    }
                if len(ids) >= 2:
                    self.examples.append((ids, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch, pad_id, k=None):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, group=args.wandb_group, config=run_config)
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=args.run_name)
        mlflow.log_params(run_config)

    if args.real_text_windows:
        train_ds = RealTextWindowsSFT(args.train_file, tokenizer, args.n_ctx, args.t_local, args.t_tgt, args.stride)
        print(f"Train windows: {len(train_ds)} (real_text_windows mode, n_ctx={args.n_ctx} t_tgt={args.t_tgt} "
              f"stride={args.stride if args.stride is not None else args.t_tgt} -> block_size={args.block_size})")
        k = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_real_text_windows)
    else:
        train_ds = JsonlTextDataset(args.train_file, tokenizer, args.block_size, teacher_targets=args.teacher_targets)
        print(f"Train examples: {len(train_ds)}" + (f" (KD against {args.teacher_targets}, K={train_ds.k})" if args.teacher_targets else ""))
        k = train_ds.k if args.teacher_targets else None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id, k=k),
        )

    val_loader = None
    if args.val_file and args.val_every:
        val_ds = JsonlTextDataset(args.val_file, tokenizer, args.block_size, teacher_targets=args.val_teacher_targets)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id, k=k),
        )
        print(f"Val examples: {len(val_ds)} (evaluated every {args.val_every} steps)")

    if args.mup:
        optimizer = torch.optim.AdamW(build_mup_param_groups(model, args.lr, width_mult), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    step, losses = 0, []
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        losses = ckpt["losses"]
        print(f"Resumed from {args.resume_from} at step {step} (best_loss so far: {min(losses):.6f})")

    start = time.time()
    done = False
    while not done:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                if args.teacher_targets:
                    labels = batch.pop("labels")
                    teacher_indices = batch.pop("teacher_indices")
                    teacher_values = batch.pop("teacher_values")
                    teacher_residual = batch.pop("teacher_residual")
                    teacher_mask = batch.pop("teacher_mask")
                    logits = model(**batch).logits
                    if args.mup:
                        logits = logits / width_mult  # muP readout output scaling
                    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                    kd = topk_kd_loss(logits, teacher_indices, teacher_values, teacher_residual, teacher_mask)
                    loss = (1 - args.kd_alpha) * ce + args.kd_alpha * kd
                elif args.mup:
                    labels = batch.pop("labels")
                    logits = model(**batch).logits / width_mult  # muP readout output scaling
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                else:
                    loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0 or step == 1:
                if args.teacher_targets:
                    print(f"step {step} loss {loss.item():.4f} (ce {ce.item():.4f} kd {kd.item():.4f})")
                    step_metrics = {"loss": loss.item(), "ce": ce.item(), "kd": kd.item()}
                else:
                    print(f"step {step} loss {loss.item():.4f}")
                    step_metrics = {"loss": loss.item()}
                logger.progress(step, **step_metrics)
                if args.wandb:
                    wandb.log(step_metrics, step=step)
                if args.mlflow:
                    mlflow.log_metrics(step_metrics, step=step)
            if args.save_dir and args.checkpoint_every and step % args.checkpoint_every == 0:
                save_checkpoint(os.path.join(args.save_dir, "checkpoint.pt"), model, optimizer, step, losses, args, width_mult, depth_mult)
            if val_loader is not None and (step % args.val_every == 0 or step == 1):
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
        save_checkpoint(ckpt_path, model, optimizer, step, losses, args, width_mult, depth_mult)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
