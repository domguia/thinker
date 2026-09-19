"""
I6 (relayed 2026-09-19/20 night, model-design/user): the spec's non-knowledge-
of-t rule (spec Sec.7.2, decided tonight -- step t is never an input, weight,
or embedding index anywhere in Thinker) makes what distinguishes iteration 1
from iteration 5 PURELY the register state R_t. If the loop does real work,
R_t should carry a phase signature a linear probe can decode -- with nothing
in the architecture handing it t directly.

Reuses the same healthy (post-decouple_kv-fix, ~99.6-100% task accuracy)
n_hops=2 checkpoint I4 regenerated (/tmp/diagnose_n_hops2_no_ff_checkpoint.pt)
-- CPU only, no training, same "checkpoint-provenance" caveat as I4 applies
here (this is NOT the historical plateaued checkpoint).

Three trivial controls, none of which the headline result is meaningful
without (same discipline as I4's soft-retrieval controls):
  (a) shuffled t-labels -- must fall to chance, else the probe overfits.
  (b) R_0 frozen and repeated N_STEP times -- must fall to chance too
      (rules out the probe just fitting some fixed per-example signature).
  (c) probe on ||R_t|| alone (a scalar) -- if this alone predicts t, "phase"
      is just a scale drift, not real structure.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import learn.indexed_attention.diagnose_no_ff_composition as base

CHECKPOINT = "/tmp/diagnose_n_hops2_no_ff_checkpoint.pt"
N_PROBE_SEEDS = 5


@torch.no_grad()
def collect_register_trajectory(model, ds, n_eval: int) -> np.ndarray:
    """Returns R_t as (n_eval, N_STEP, d_model), the mean-pooled register
    state at the START of each iteration (same convention as probe_curve)."""
    leaves, source_ids, mask, query, label = ds.sample_batch(n_eval)
    B = leaves.shape[0]
    leaf_emb = model.embed(leaves)
    model.memory.build(leaf_emb, source_ids, leaf_mask=mask)
    q_emb = model.embed(query).mean(dim=1, keepdim=True)
    R = model.register_init.unsqueeze(0).expand(B, -1, -1) + q_emb
    sm_k = torch.zeros(B, 0, model.d_model)
    sm_v = torch.zeros(B, 0, model.d_model)

    traj = []
    for t in range(base.N_STEP):
        traj.append(R.mean(dim=1).clone())  # (B, d) -- state BEFORE this iteration's update
        o_kb = model.memory.attend(R)
        if sm_k.shape[1] > 0:
            q_sm = model.sm_q_proj(R)
            o_sm = F.scaled_dot_product_attention(q_sm, sm_k, sm_v)
        else:
            o_sm = torch.zeros_like(R)
        fused = torch.cat([o_kb, o_sm, R], dim=-1)
        delta = model.fuse_proj(model.fuse_norm(fused))  # use_ff=False for this checkpoint
        R = R + delta
        new_k, new_v = model.sm_write_proj(R).chunk(2, dim=-1)
        sm_k = torch.cat([sm_k, new_k], dim=1)
        sm_v = torch.cat([sm_v, new_v], dim=1)
    return torch.stack(traj, dim=1).numpy()  # (n_eval, N_STEP, d)


def probe_once(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """X: (n_samples, d), y: (n_samples,) step index. Episode-level split
    is implicit here since X/y are already flattened per-episode-per-step;
    the caller passes a train/test split done at the EPISODE level."""
    clf = LogisticRegression(max_iter=2000, random_state=seed)  # multinomial by default (sklearn>=1.5)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    cm = confusion_matrix(yte, clf.predict(Xte), labels=sorted(set(y)))
    return acc, cm


def main() -> None:
    torch.manual_seed(1)
    ds, model = base.make_model(n_hops=2)
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()

    n_eval = 512
    traj = collect_register_trajectory(model, ds, n_eval)  # (n_eval, N_STEP, d)
    N_STEP = traj.shape[1]
    d = traj.shape[2]

    # main dataset: one (R_t, t) sample per (episode, step)
    X_main = traj.reshape(n_eval * N_STEP, d)
    y_main = np.tile(np.arange(N_STEP), n_eval)

    # control (b): R_0 frozen, repeated N_STEP times per episode
    R0 = traj[:, 0, :]  # (n_eval, d)
    X_frozen = np.repeat(R0, N_STEP, axis=0)
    y_frozen = y_main  # same labels, features carry no t-information by construction

    # control (c): scalar norm only
    X_norm = np.linalg.norm(X_main, axis=-1, keepdims=True)

    chance = 1.0 / N_STEP
    print(f"n_eval={n_eval} N_STEP={N_STEP} d_model={d} chance_level={chance:.4f}")

    for label, X, y, shuffle_labels in [
        ("main (R_t -> t)", X_main, y_main, False),
        ("control_a (R_t -> shuffled t)", X_main, y_main, True),
        ("control_b (frozen R_0 -> t)", X_frozen, y_frozen, False),
        ("control_c (||R_t|| -> t)", X_norm, y_main, False),
    ]:
        accs = []
        for seed in range(N_PROBE_SEEDS):
            yy = y.copy()
            if shuffle_labels:
                rng = np.random.default_rng(seed)
                rng.shuffle(yy)
            acc, cm = probe_once(X, yy, seed)
            accs.append(acc)
            if seed == 0:
                last_cm = cm
        accs = np.array(accs)
        print(f"\n=== {label} ===")
        print(f"accuracy over {N_PROBE_SEEDS} probe seeds: mean={accs.mean():.4f} "
              f"std={accs.std():.4f} min={accs.min():.4f} max={accs.max():.4f} "
              f"(chance={chance:.4f})")
        if label == "main (R_t -> t)":
            print("confusion matrix (seed 0), rows=true t, cols=pred t:")
            print(last_cm)
            off_by_one = sum(last_cm[i, i + 1] + last_cm[i + 1, i] for i in range(N_STEP - 1))
            total = last_cm.sum()
            print(f"mass on |true-pred|==1 (adjacent-step confusion): "
                  f"{off_by_one}/{total} = {off_by_one/total:.4f}")


if __name__ == "__main__":
    main()
