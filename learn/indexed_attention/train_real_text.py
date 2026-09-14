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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.indexed_thinker_model import Thinker
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
        if k in ("doc_id", "is_first_window"):
            continue
        vals = [item[k] if item is not None else torch.zeros_like(template[k]) for item in batch_items]
        out[k] = torch.stack(vals, dim=0)
    out["is_first_window"] = torch.tensor(
        [item["is_first_window"] if item is not None else True for item in batch_items], dtype=torch.bool
    )
    out["lane_valid"] = lane_valid
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="JSONL path, one {'text': ...} per line")
    p.add_argument("--tokenizer", default="gpt2")
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
    p.add_argument("--disable_kb", action="store_true",
                   help="spec §9 Baseline B: loop still runs n_step times, but external-memory (KB) access is "
                        "disabled -- isolates whether any gain comes from the loop itself or the memory. "
                        "Also skips HierarchicalMemory.build() entirely (cheaper, not just architecturally different).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--max_time_minutes", type=float, default=15.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

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
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    ds = RealTextWindowDataset(args.data, tok, n_ctx=args.n_ctx, t_local=args.t_local,
                               t_tgt=args.t_tgt, stride=args.stride, pad_id=pad_id)
    print(f"loaded {len(ds.docs)} docs, {len(ds.windows)} windows, vocab_size={tok.vocab_size}", flush=True)
    batcher = LockstepLaneBatcher(ds, n_lanes=args.n_lanes, seed=args.seed)

    model = Thinker(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_register=args.n_register,
        block_size=args.block_size, depth=args.depth, n_slots=args.n_slots, n_head=args.n_head,
        disable_kb=args.disable_kb,
        stream_dims={"answer": tok.vocab_size},
        stream_sequence={"answer": True}, max_target_len=args.t_tgt,
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"depth={args.depth} block_size={args.block_size} n_ctx={args.n_ctx} t_local={args.t_local} "
          f"t_tgt={args.t_tgt} n_lanes={args.n_lanes} params={n_params/1e6:.2f}M device={device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

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
        loss = (per_lane_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)

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
            print(f"step={step:6d} elapsed={elapsed/60:.2f}m loss={mean_loss:.4f} ppl={ppl:.2f} "
                  f"active_lanes={int(valid_f.sum().item())}/{args.n_lanes}", flush=True)
        step += 1

    print("\n---", flush=True)
    print(f"final_loss: {sum(loss_hist[-50:]) / max(len(loss_hist[-50:]), 1):.4f}", flush=True)
    print(f"num_steps: {step}", flush=True)
    print(f"training_seconds: {time.time() - start_time:.1f}", flush=True)


if __name__ == "__main__":
    main()
