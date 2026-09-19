"""
Phase 3 (dev_notes/indexed_attention_experiment_plan.md): first real-text LM
training run for `Thinker` (core/indexed_thinker_model.py), at `depth=1`
("no index" -- decoupled compressor, single compression level, no
multi-level hierarchy). Default `depth=1` follows the 2026-09-14 decision:
Phase 0bis found depth=1 matches native (deeper) depth at n_facts=16/64/256
on the synthetic task once a curriculum is used, so hierarchical indexing
isn't needed to validate the mechanism on real text -- it's motivated by
memory *size*, to be introduced later once depth=1 is working end-to-end.

Builds on already-existing, already-tested infrastructure that had no
training loop wired to it yet:
  - data/real_text_windows.py::RealTextWindowDataset -- sliding-window
    loader (spec §14.1), already handles the context/target split and
    teacher-forcing shift.
  - Thinker.forward(register_init_override=..., target_input=...) -- spec
    §14.2/§14.3, register carry-over and sequence-mode streams, already
    wired into the model (see core/indexed_thinker_model.py's docstring).

Two design decisions made here, not specified by the spec -- tagged
[DEFAUT] per this project's convention, open to revision:

1. `query_tokens` (Thinker.forward requires it, to seed R via a mean-pooled
   embedding on top of register_init/the carried-over R -- spec never
   specifies what it should be for real text, unlike the synthetic tasks
   where it's the natural query). Default: the window's own LOCAL
   (recency) region, i.e. kb_tokens[:, -t_local:] -- "seed the register
   with a summary of what we're currently continuing from." Simple, uses
   only content already in the window, no new mechanism.

2. Register carry-over across windows is batched in LOCKSTEP across N
   parallel "lanes": each of the N batch positions tracks one document's
   windows in order (R carried, detached, across windows within a lane's
   document; reset to the learned register_init on is_first_window). When
   a lane's document runs out of windows, it's refilled with the next
   unseen document. This is an MVP choice, not the only valid one -- a
   simpler batch_size=1 (one document at a time) would also be correct but
   far less GPU-efficient; a fully shuffled batch (windows from arbitrary
   documents mixed every step) would break the carry-over invariant
   entirely and was not considered.

Baselines A/B/C (spec §9) are NOT implemented here yet -- this script's
job is the first working depth=1 run; baseline comparisons are a
documented follow-up (dev_notes/indexed_attention_experiment_plan.md,
Phase 3), not blocking.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
from core.model_families import resolve_model_name
from core.run_logging import add_run_args, logger_from_args
from learn.distill.train_sft import topk_kd_loss


class TeacherTargets:
    """Loads a precompute_teacher_targets.py .npz and slices per-window Top-K
    targets aligned to RealTextWindowDataset windows.

    Row-alignment convention (dev_notes/experiments/real_text_baselines.md,
    2026-09-20 design note -- derived, not assumed): the Teacher's row q
    (0-indexed within a document) is `out.logits[0][q]`, the standard causal-LM
    convention of predicting token q+1 having seen tokens 0..q. A Thinker
    window starting at `window_pos` feeds `target_input[t] = ids[window_pos+t-1]`
    as its query at step t and predicts `labels[t] = ids[window_pos+t]` -- so
    the matching Teacher row for step t is `q = window_pos + t - 1`.
    """

    def __init__(self, path: str):
        npz = np.load(path)
        self.indices = npz["indices"]      # (N, K) int32
        self.values = npz["values"]        # (N, K) fp16
        self.residual = npz["residual"]    # (N,) fp16
        self.offsets = npz["offsets"]      # (n_docs+1,) int64
        self.k = int(npz["k"])

    def slice_window(self, doc_id: int, window_pos: int, t_tgt: int):
        """Returns (indices, values, residual, mask), each length t_tgt --
        rows outside the Teacher's precomputed range for this document (e.g.
        --max_length at precompute time was shorter than window_pos+t_tgt-1)
        are zero-filled and masked out, not treated as an error: a partially-
        covered window still contributes KD loss on its covered positions."""
        doc_start, doc_end = int(self.offsets[doc_id]), int(self.offsets[doc_id + 1])
        n_doc = doc_end - doc_start
        idx = torch.zeros(t_tgt, self.k, dtype=torch.long)
        val = torch.zeros(t_tgt, self.k, dtype=torch.float32)
        res = torch.zeros(t_tgt, dtype=torch.float32)
        mask = torch.zeros(t_tgt, dtype=torch.bool)
        for t in range(t_tgt):
            q = window_pos + t - 1
            if 0 <= q < n_doc:
                idx[t] = torch.from_numpy(self.indices[doc_start + q].astype(np.int64))
                val[t] = torch.from_numpy(self.values[doc_start + q].astype(np.float32))
                res[t] = float(self.residual[doc_start + q])
                mask[t] = True
        return idx, val, res, mask

    def build_batch_targets(self, doc_ids, window_positions, t_tgt: int, lane_valid):
        """doc_ids/window_positions: (B,) LongTensor (e.g. the batcher's
        lane_doc and collate_lane_batch's window_pos). lane_valid: (B,) bool,
        an already-invalid lane (exhausted/padding) is masked out entirely
        regardless of what slice_window would return for doc_id=0."""
        B = doc_ids.shape[0]
        idx_b = torch.zeros(B, t_tgt, self.k, dtype=torch.long)
        val_b = torch.zeros(B, t_tgt, self.k, dtype=torch.float32)
        res_b = torch.zeros(B, t_tgt, dtype=torch.float32)
        mask_b = torch.zeros(B, t_tgt, dtype=torch.bool)
        for b in range(B):
            if not bool(lane_valid[b]):
                continue
            idx, val, res, mask = self.slice_window(int(doc_ids[b]), int(window_positions[b]), t_tgt)
            idx_b[b], val_b[b], res_b[b], mask_b[b] = idx, val, res, mask
        return idx_b, val_b, res_b, mask_b
from data.real_text_windows import RealTextWindowDataset


class LockstepLaneBatcher:
    """N parallel "lanes", each stepping through one document's windows in
    order (see module docstring, design decision 2). Not a torch Sampler --
    plain Python generator, since lane refill (assigning a new document once
    one is exhausted) needs stateful bookkeeping a Sampler protocol doesn't
    fit cleanly.

    Yields dicts with the same keys as RealTextWindowDataset's __getitem__,
    batched over N lanes (B=n_lanes always, no ragged batches) plus
    `lane_doc_id: LongTensor(N)` for the caller to detect a lane's document
    changing (reset R for that lane) beyond just `is_first_window` (which
    only tells you position-in-document, not lane identity).
    """

    def __init__(self, dataset: RealTextWindowDataset, n_lanes: int, seed: int = 0):
        self.ds = dataset
        self.n_lanes = n_lanes
        # windows already grouped by doc_id, in order (RealTextWindowDataset
        # appends a document's windows contiguously) -- bucket them here so
        # a lane can be handed one document's window list at a time.
        self.by_doc = {}
        for w_idx, (doc_id, p, is_first) in enumerate(self.ds.windows):
            self.by_doc.setdefault(doc_id, []).append(w_idx)
        self._doc_ids = list(self.by_doc.keys())
        assert self._doc_ids, "no document produced any window -- corpus too short/empty for these n_ctx/t_tgt settings"
        self.seed = seed

    def _shuffled_doc_order(self, epoch: int):
        # deterministic-but-different order per epoch (seed+epoch), not the
        # same fixed permutation replayed forever
        rng = torch.Generator().manual_seed(self.seed + epoch)
        perm = torch.randperm(len(self._doc_ids), generator=rng).tolist()
        return [self._doc_ids[i] for i in perm]

    def __iter__(self):
        # experiment-manager (2026-09-14): the original version stopped
        # after exactly one pass over the corpus (refill() returning False
        # once doc_order was exhausted) regardless of --max_time_minutes/
        # --max_steps -- e.g. 2700 TinyStories+WikiText docs exhausted at
        # step ~1537 no matter how generous the requested budget. Fixed to
        # wrap around into a new (reshuffled) epoch instead of stopping,
        # matching every other training script in this project (the corpus
        # is meant to be a stream, not a single fixed-length pass) --
        # --max_time_minutes/--max_steps are the only real stopping
        # conditions the caller should rely on.
        epoch = 0
        doc_order = self._shuffled_doc_order(epoch)
        next_doc_ptr = 0
        lane_queues = [[] for _ in range(self.n_lanes)]  # list of window indices left, per lane
        lane_doc_id = [-1] * self.n_lanes

        def refill(lane):
            nonlocal next_doc_ptr, doc_order, epoch
            if next_doc_ptr >= len(doc_order):
                epoch += 1
                doc_order = self._shuffled_doc_order(epoch)
                next_doc_ptr = 0
                print(f"[LockstepLaneBatcher] corpus exhausted, starting epoch {epoch}", flush=True)
            doc_id = doc_order[next_doc_ptr]
            next_doc_ptr += 1
            lane_queues[lane] = list(self.by_doc[doc_id])
            lane_doc_id[lane] = doc_id
            return True

        for lane in range(self.n_lanes):
            refill(lane)

        while True:
            batch_items = []
            batch_lane_doc = []
            active = False
            for lane in range(self.n_lanes):
                while not lane_queues[lane]:
                    if not refill(lane):
                        break
                if not lane_queues[lane]:
                    # Unreachable in normal operation since the epoch-wraparound
                    # fix above (refill() always succeeds -- self._doc_ids is
                    # asserted non-empty at construction). Left as a defensive
                    # fallback rather than removed, so a future change that
                    # reintroduces a genuinely-exhaustible lane degrades safely
                    # (masked-out dummy) instead of crashing/hanging.
                    batch_items.append(None)
                    batch_lane_doc.append(-1)
                    continue
                active = True
                w_idx = lane_queues[lane].pop(0)
                item = self.ds[w_idx]
                batch_items.append(item)
                batch_lane_doc.append(lane_doc_id[lane])
            if not active:
                return
            yield batch_items, torch.tensor(batch_lane_doc, dtype=torch.long)


def collate_lane_batch(batch_items, pad_id: int):
    """batch_items: list of dicts or None (exhausted lane, per
    LockstepLaneBatcher). None entries get a fully-masked dummy window so
    the batch stays rectangular; the caller must zero their loss
    contribution via the returned `lane_valid` mask."""
    keys = next(item for item in batch_items if item is not None).keys()
    lane_valid = torch.tensor([item is not None for item in batch_items], dtype=torch.bool)
    template = next(item for item in batch_items if item is not None)
    out = {}
    for k in keys:
        if k in ("doc_id", "is_first_window", "window_pos"):
            continue
        vals = [item[k] if item is not None else torch.zeros_like(template[k]) for item in batch_items]
        out[k] = torch.stack(vals, dim=0)
    out["is_first_window"] = torch.tensor(
        [item["is_first_window"] if item is not None else True for item in batch_items], dtype=torch.bool
    )
    # window_pos: plain python int per window (not a tensor in the source dict, see
    # data/real_text_windows.py), needed to align a window against a precomputed
    # Teacher's per-document targets (--teacher_targets) -- invalid lanes get 0,
    # harmless since lane_valid already excludes them from any loss.
    out["window_pos"] = torch.tensor(
        [item["window_pos"] if item is not None else 0 for item in batch_items], dtype=torch.long
    )
    out["lane_valid"] = lane_valid
    return out


@torch.no_grad()
def evaluate_at_nstep(model, ds, pad_id: int, n_lanes: int, t_local: int, seed: int, device, n_step_test: int,
                       n_eval_batches: int = 20) -> float:
    """Mean per-window LM loss at a given n_step, over a freshly-seeded pass
    over `ds` -- NOT a held-out split (this project has none for real text
    yet, see --extrapolate_n_steps' docstring), so this measures extrapolation
    behavior (does the trained model do better/worse with more/fewer loop
    iterations than it was trained with), not generalization to unseen text.
    Register carry-over/reset logic mirrors main()'s training loop exactly,
    just without the backward pass."""
    model.eval()
    eval_batcher = LockstepLaneBatcher(ds, n_lanes=n_lanes, seed=seed + 999_999)
    lane_R = [None] * n_lanes
    losses = []
    for i, (batch_items, lane_doc) in enumerate(eval_batcher):
        if i >= n_eval_batches:
            break
        batch = collate_lane_batch(batch_items, pad_id)
        kb_tokens = batch["kb_tokens"].to(device)
        kb_source_ids = batch["kb_source_ids"].to(device)
        kb_leaf_mask = batch["kb_leaf_mask"].to(device)
        target_input = batch["target_input"].to(device)
        labels = batch["labels"].to(device)
        is_first = batch["is_first_window"]
        lane_valid = batch["lane_valid"].to(device)
        query_tokens = kb_tokens[:, -t_local:]  # [DEFAUT], same convention as main()'s training loop

        register_override = None
        if any(not f for f in is_first.tolist()):
            base = model.register_init.unsqueeze(0).expand(kb_tokens.shape[0], -1, -1).clone()
            for lane in range(n_lanes):
                if not is_first[lane] and lane_R[lane] is not None:
                    base[lane] = lane_R[lane]
            register_override = base

        R, streams = model(kb_tokens, kb_source_ids, query_tokens, n_step_test,
                           kb_leaf_mask=kb_leaf_mask, register_init_override=register_override,
                           target_input=target_input)
        logits = streams["answer"]
        per_pos_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        per_lane_loss = per_pos_loss.mean(dim=1)
        valid_f = lane_valid.float()
        loss = (per_lane_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        losses.append(loss.item())

        for lane in range(n_lanes):
            lane_R[lane] = R[lane].detach() if lane_doc[lane].item() >= 0 else None

    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="JSONL path, one {'text': ...} per line")
    p.add_argument("--val_data", default=None,
                    help="optional separate JSONL path (same format as --data) for a genuine "
                         "held-out evaluation -- this project had NO held-out mechanism for real "
                         "text before this flag (training loss/ppl only). Required for any fair "
                         "loss/ppl comparison against a reference LLM (both must be scored on data "
                         "neither saw), and used as the extrapolation probe's data when given "
                         "(falls back to --data, NOT held-out, if omitted -- see "
                         "--extrapolate_n_steps' own docstring).")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=20)
    p.add_argument("--tokenizer", default="lfm2", help="HF repo id, or a family alias from core/model_families.py (lfm2/olmo/qwen)")
    p.add_argument("--depth", type=int, default=1, help="0 = Baseline C (flat); 1 = the 2026-09-14 default (no index)")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--n_ctx", type=int, default=256, help="must equal block_size**depth")
    p.add_argument("--t_local", type=int, default=32, help="how many of n_ctx's trailing positions count as 'local' (source_id=0)")
    p.add_argument("--t_tgt", type=int, default=32)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--n_lanes", type=int, default=8, help="parallel document streams, see LockstepLaneBatcher")
    p.add_argument("--n_register", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=1)
    p.add_argument("--n_step", type=int, default=6, help="spec §9 Baseline A = --n_step 1 (single-pass, no loop), zero new code")
    p.add_argument("--pool_n_head", type=int, default=1,
                   help="model-design (2026-09-14): multi-head compressor pooling, confirmed a clean "
                        "win (99.6-99.9%%, 5/5 seeds) on the synthetic n_hops task at its own tuned LR "
                        "(4e-4, see indexed_attention_spec.md Sec 5.1bis) -- untested on real text yet. "
                        "1 (default) reproduces prior behavior exactly.")
    p.add_argument("--k_dim", type=int, default=None,
                   help="asymmetric K narrower than V (Sec 5.4) -- None (default) reproduces prior "
                        "symmetric behavior.")
    p.add_argument("--disable_kb", action="store_true",
                   help="spec §9 Baseline B: loop still runs n_step times, but external-memory (KB) access is "
                        "disabled -- isolates whether any gain comes from the loop itself or the memory. "
                        "Also skips HierarchicalMemory.build() entirely (cheaper, not just architecturally different).")
    p.add_argument("--answer_n_layers", type=int, default=1,
                   help="OutputStream cross-attention depth for the 'answer' stream (1-3, "
                        "core/indexed_thinker_model.py's stream_n_layers) -- spec §11bis/§-1's own "
                        "prediction is that a stream should stay a lightweight reader of SM (most "
                        "of the work already done by the recurrent loop/memory), so depth>1 is an "
                        "ablation testing that prediction, not an expected win. 1 (default) "
                        "reproduces prior behavior exactly.")
    p.add_argument("--use_ff", action="store_true",
                   help="Phase 1bis variant (spec §-1's own predicted interpretation): reintroduces a "
                        "2-layer GELU MLP in the register-update fusion only. Tested on the synthetic "
                        "n_hops task (I1: no detectable effect there) but NEVER on real text -- directly "
                        "relevant now that Piste A/C found C (looped) losing to A (flat) on real text: "
                        "per the spec, a use_ff=True win here would mean 'missing composition/computation "
                        "capacity on real text', not 'the loop premise is wrong' (see "
                        "dev_notes/indexed_attention_spec.md's own guidance on how to read this variant).")
    p.add_argument("--ff_hidden_mult", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4, help="target/peak LR, reached at the end of warmup "
                   "(or from step 0 if --lr_warmup_steps=0)")
    p.add_argument("--lr_warmup_steps", type=int, default=0,
                    help="linear LR warmup FROM --lr_warmup_init TO --lr over this many steps, then held "
                         "constant (0 = no warmup, exact prior behavior). Same convention as "
                         "learn/toy_memory/train_toy_memory.py's --lr_warmup_steps -- Piste A's real-text "
                         "architectures (A: single-pass, C: looped) were only ever LR-swept without warmup; "
                         "C is the novel, less-understood-by-classical-practice architecture, so its "
                         "reported disadvantage vs A could reflect an under-tuned optimization recipe "
                         "rather than an architectural ceiling -- untested until now.")
    p.add_argument("--lr_warmup_init", type=float, default=None,
                   help="LR at step 0 when --lr_warmup_steps > 0 (linearly ramped up to --lr). Defaults to "
                        "--lr / 10 if not set. Ignored when --lr_warmup_steps=0.")
    p.add_argument("--teacher_targets", default=None,
                    help="path to a precompute_teacher_targets.py .npz computed on the SAME --data file "
                         "(same document order/tokenizer) -- enables logit-level KD via topk_kd_loss() "
                         "(learn/distill/train_sft.py, reused as-is). See TeacherTargets' docstring for "
                         "the row-alignment convention. Requires --kd_alpha > 0 to have any effect.")
    p.add_argument("--val_teacher_targets", default=None,
                    help="same as --teacher_targets, computed on --val_data instead, for held-out KD loss reporting.")
    p.add_argument("--kd_alpha", type=float, default=0.5,
                    help="loss = (1-kd_alpha)*ce + kd_alpha*kd, same convention as train_sft.py. Ignored "
                         "when --teacher_targets is not given (pure CE, prior behavior).")
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_checkpoint_path", default=None,
                    help="save model.state_dict() here after training -- same convention as "
                         "learn/indexed_attention/train_kb_chain.py's flag of the same name. A "
                         "directory (grid-friendly) or a literal file path both work.")
    p.add_argument("--extrapolate_n_steps", default=None,
                    help="comma-separated N_step_test values to probe in-memory after training "
                         "finishes, no checkpoint reload needed (e.g. '8,12,24') -- the direct test "
                         "of whether 'thinking longer' (more loop iterations at inference than "
                         "--n_step used at training) helps or hurts on real text, same convention "
                         "as train_kb_chain.py's flag of the same name. Evaluated on a freshly-seeded "
                         "pass over --data (NOT a held-out split -- this project has none for real "
                         "text yet -- so this measures extrapolation behavior, not generalization).")
    add_run_args(p)
    args = p.parse_args()
    logger = logger_from_args(args)

    # HierarchicalMemory.build() only requires N % block_size**depth == 0
    # (divisibility -- a "forest" of multiple top-level nodes is fine, see
    # core/indexed_memory.py, fixed 2026-09-13 for exactly this reason: the
    # OLD strict equality forced a single root, making "one node per chunk,
    # no further hierarchy" -- e.g. n_ctx=256/block_size=16/depth=1 -> 16
    # top-level nodes -- inexpressible for n_ctx > block_size**depth). This
    # script re-imposed the stricter equality by mistake; fixed to match.
    divisor = args.block_size ** args.depth if args.depth > 0 else 1
    assert args.n_ctx % divisor == 0, (
        f"--n_ctx={args.n_ctx} must be a multiple of block_size**depth={divisor} when depth>0"
    )
    assert 1 <= args.t_local <= args.n_ctx

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_model_name(args.tokenizer))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # tok.vocab_size is the BASE vocab size, excluding added/special tokens --
    # some tokenizers (e.g. Qwen3) assign ids above it to added tokens, so
    # sizing the embedding table from vocab_size alone crashes with
    # `IndexError: index out of range in self` the first time such a token
    # appears in the data. len(tok) is the true total vocab size.
    vocab_size = len(tok)
    ds = RealTextWindowDataset(args.data, tok, n_ctx=args.n_ctx, t_local=args.t_local,
                               t_tgt=args.t_tgt, stride=args.stride, pad_id=pad_id)
    print(f"loaded {len(ds.docs)} docs, {len(ds.windows)} windows, vocab_size={vocab_size}", flush=True)
    batcher = LockstepLaneBatcher(ds, n_lanes=args.n_lanes, seed=args.seed)

    val_ds = None
    if args.val_data:
        val_ds = RealTextWindowDataset(args.val_data, tok, n_ctx=args.n_ctx, t_local=args.t_local,
                                        t_tgt=args.t_tgt, stride=args.stride, pad_id=pad_id)
        print(f"loaded held-out val: {len(val_ds.docs)} docs, {len(val_ds.windows)} windows "
              f"from {args.val_data}", flush=True)

    teacher = TeacherTargets(args.teacher_targets) if args.teacher_targets else None
    if teacher is not None:
        print(f"loaded Teacher targets from {args.teacher_targets}: K={teacher.k}, "
              f"{len(teacher.offsets) - 1} docs -- KD enabled, kd_alpha={args.kd_alpha}", flush=True)
        assert len(teacher.offsets) - 1 == len(ds.docs), (
            f"--teacher_targets has {len(teacher.offsets) - 1} documents but --data has {len(ds.docs)} -- "
            f"must be precomputed on the exact same file (same order) for doc_id alignment to be valid"
        )
    val_teacher = TeacherTargets(args.val_teacher_targets) if args.val_teacher_targets else None

    model = Thinker(
        vocab_size=vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb, pool_n_head=args.pool_n_head, k_dim=args.k_dim,
        use_ff=args.use_ff, ff_hidden_mult=args.ff_hidden_mult,
        stream_dims={"answer": vocab_size},
        stream_sequence={"answer": True}, max_target_len=args.t_tgt,
        stream_n_layers={"answer": args.answer_n_layers},
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"depth={args.depth} block_size={args.block_size} n_ctx={args.n_ctx} t_local={args.t_local} "
          f"t_tgt={args.t_tgt} n_lanes={args.n_lanes} pool_n_head={args.pool_n_head} k_dim={args.k_dim} "
          f"params={n_params/1e6:.2f}M device={device}", flush=True)

    lr_warmup_init = args.lr_warmup_init if args.lr_warmup_init is not None else args.lr / 10
    if args.lr_warmup_steps > 0:
        print(f"lr_warmup: {lr_warmup_init:.2e} -> {args.lr:.2e} over {args.lr_warmup_steps} steps, "
              f"then held constant at {args.lr:.2e}", flush=True)

    def lr_at(step: int) -> float:
        if args.lr_warmup_steps <= 0 or step >= args.lr_warmup_steps:
            return args.lr
        return lr_warmup_init + (args.lr - lr_warmup_init) * (step / args.lr_warmup_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=1e-2)

    lane_R = [None] * args.n_lanes  # carried register state per lane, detached across windows

    start_time = time.time()
    max_time_seconds = args.max_time_minutes * 60
    step = 0
    loss_hist = []
    for batch_items, lane_doc in batcher:
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds or step >= args.max_steps:
            print(f"Budget reached at step {step}. Stopping.", flush=True)
            break

        batch = collate_lane_batch(batch_items, pad_id)
        kb_tokens = batch["kb_tokens"].to(device)
        kb_source_ids = batch["kb_source_ids"].to(device)
        kb_leaf_mask = batch["kb_leaf_mask"].to(device)
        target_input = batch["target_input"].to(device)
        labels = batch["labels"].to(device)
        is_first = batch["is_first_window"]
        lane_valid = batch["lane_valid"].to(device)

        query_tokens = kb_tokens[:, -args.t_local:]  # [DEFAUT], see module docstring #1

        register_override = None
        if any(not f for f in is_first.tolist()):
            # build a (B, n_register, d_model) tensor from whichever lanes carry
            # a previous R; first-window lanes fall back to the learned init by
            # leaving their slot as None-equivalent (Thinker adds register_init
            # itself when override is None for the WHOLE batch, but here it's
            # per-lane -- so construct the override explicitly for every lane,
            # using the model's own register_init for first-window/fresh lanes).
            base = model.register_init.unsqueeze(0).expand(kb_tokens.shape[0], -1, -1).clone()
            for lane in range(args.n_lanes):
                if not is_first[lane] and lane_R[lane] is not None:
                    base[lane] = lane_R[lane]
            register_override = base

        R, streams = model(kb_tokens, kb_source_ids, query_tokens, args.n_step,
                           kb_leaf_mask=kb_leaf_mask, register_init_override=register_override,
                           target_input=target_input)

        logits = streams["answer"]  # (B, t_tgt, vocab)
        per_pos_loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")  # (B, t_tgt)
        per_lane_loss = per_pos_loss.mean(dim=1)  # (B,)
        valid_f = lane_valid.float()
        ce_loss = (per_lane_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        if teacher is not None:
            t_idx, t_val, t_res, t_mask = teacher.build_batch_targets(
                lane_doc, batch["window_pos"], args.t_tgt, batch["lane_valid"]
            )
            kd_loss = topk_kd_loss(logits, t_idx.to(device), t_val.to(device), t_res.to(device), t_mask.to(device))
            loss = (1 - args.kd_alpha) * ce_loss + args.kd_alpha * kd_loss
        else:
            loss = ce_loss

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # spec §14.2 default: carry R forward stop-gradient'd (truncated BPTT
        # across window boundaries -- otherwise the graph grows with every
        # window of a document, unbounded memory over a long document).
        R_detached = R.detach()
        for lane in range(args.n_lanes):
            lane_R[lane] = R_detached[lane] if lane_doc[lane].item() >= 0 else None

        loss_hist.append(loss.item())
        if step % args.log_every == 0:
            mean_loss = sum(loss_hist[-args.log_every:]) / len(loss_hist[-args.log_every:])
            ppl = torch.exp(torch.tensor(mean_loss)).item()
            kd_str = f" ce={ce_loss.item():.4f} kd={kd_loss.item():.4f}" if teacher is not None else ""
            print(f"step={step:6d} elapsed={elapsed/60:.2f}m loss={mean_loss:.4f} ppl={ppl:.2f}{kd_str} "
                  f"lr={lr_at(step):.2e} active_lanes={int(valid_f.sum().item())}/{args.n_lanes}", flush=True)
            log_kwargs = {"ce": ce_loss.item(), "kd": kd_loss.item()} if teacher is not None else {}
            logger.progress(step, loss=mean_loss, ppl=ppl, lr=lr_at(step),
                            active_lanes=int(valid_f.sum().item()), **log_kwargs)
        if val_ds is not None and step % args.val_every == 0 and step > 0:
            val_loss = evaluate_at_nstep(model, val_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                          device, args.n_step, n_eval_batches=args.val_batches)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            print(f"step={step:6d} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} (held-out)", flush=True)
            logger.progress(step, val_loss=val_loss, val_ppl=val_ppl)
        step += 1

    print("\n---", flush=True)
    print(f"final_loss: {sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1):.4f}", flush=True)
    print(f"num_steps: {step}", flush=True)
    print(f"training_seconds: {time.time() - start_time:.1f}", flush=True)

    if args.save_checkpoint_path:
        # Same convention as learn/indexed_attention/train_kb_chain.py's flag
        # of the same name -- a directory (grid-friendly, one fixed value per
        # cell) or a literal file path both work.
        import os
        ckpt_path = args.save_checkpoint_path
        if ckpt_path.endswith("/") or os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
            ckpt_path = os.path.join(ckpt_path, f"{args.run_id or 'adhoc'}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint saved to {ckpt_path}", flush=True)

    final_val_loss = None
    if val_ds is not None:
        final_val_loss = evaluate_at_nstep(model, val_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                            device, args.n_step, n_eval_batches=max(args.val_batches, 20))
        print(f"final_val_loss: {final_val_loss:.4f} (held-out, n_step={args.n_step})", flush=True)

    extrapolation_results = {}
    if args.extrapolate_n_steps:
        probe_ds = val_ds if val_ds is not None else ds
        print(f"--- extrapolation probe (n_step_test vs training n_step), "
              f"{'held-out' if val_ds is not None else 'NOT held-out, see flag docstring'} ---", flush=True)
        for n_step_test in [int(x) for x in args.extrapolate_n_steps.split(",")]:
            probe_loss = evaluate_at_nstep(model, probe_ds, pad_id, args.n_lanes, args.t_local, args.seed,
                                            device, n_step_test, n_eval_batches=20)
            probe_ppl = torch.exp(torch.tensor(probe_loss)).item()
            extrapolation_results[n_step_test] = probe_loss
            print(f"n_step_test={n_step_test:3d} loss={probe_loss:.4f} ppl={probe_ppl:.2f} "
                  f"(training n_step={args.n_step})", flush=True)

    # Pas de métrique d'accuracy ici : ce script optimise une perplexité de
    # langage, donc aucun contrôle trivial n'est exigible (finish() ne les
    # réclame que pour les métriques de type accuracy).
    logger.finish(summary={
        "final_loss": sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1),
        "final_val_loss": final_val_loss,
        "num_steps": step,
        "training_seconds": time.time() - start_time,
        "extrapolation": extrapolation_results or None,
        "extrapolation_held_out": val_ds is not None,
    })


if __name__ == "__main__":
    main()
