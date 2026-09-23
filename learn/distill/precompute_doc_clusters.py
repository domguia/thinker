"""
Precompute per-document embeddings and TWO candidate hierarchical-depth
labelings (dev_notes/indexed_attention_spec.md §8quater) for the retrieval
data's context documents (learn/distill/prepare_retrieval_data.py's
"context_docs" field, one JSONL row per QA example).

This is a RESEARCH PROBE, not a validated production heuristic: §8quater's
own conclusion is that the link between any density/depth measure and how
much ingestion capacity (n_step, §8ter option b) a document actually needs
is an unverified hypothesis, flagged by both the user and `model-design`.
This script exists to let the two candidate methods' BEHAVIOR be inspected
(depth distributions, agreement between methods, correlation with plain
document length) before any such measure is wired into training decisions.

Embeddings: mean-free sentence embeddings from a small pretrained
sentence-transformers model (--embedding_model, default a lightweight
MiniLM), NOT the Teacher used for KD elsewhere in this project and NOT
Thinker itself (which isn't trained yet -- the same poule/oeuf problem noted
in §8bis/§8ter). Deliberately a static, architecture-agnostic embedding
space, distinct from (and not meant to stand in for) whatever Thinker's own
sm_write_proj eventually learns to encode during ingestion.

Two recursive splitting methods, run over the SAME embedding matrix so their
outputs are directly comparable (same document set, same input space):

- --method gmm: recursively fits a 1-component vs 2-component GMM at each
  node (spec §8quater "2. Scission statistiquement justifiee") and splits
  only when the 2-component fit's BIC is lower by more than --bic_margin --
  a cluster of near-duplicate points is NOT split just because it has many
  members.
- --method hdbscan: recursively re-clusters each node with
  sklearn.cluster.HDBSCAN (spec §8quater "3. HDBSCAN") -- a node that
  HDBSCAN treats as one persistent cluster (or as unclustered "noise") stops
  there instead of being forced to split.

Both stop recursing at --max_depth or once a node no longer contains more
than --min_node_size points. depth[i] is the number of successful splits an
ancestor chain performed before document i's leaf node stopped splitting --
NOT the number of points in that leaf, which is exactly the distinction
§8quater is checking for.

Output: a single .npz, one row per (row_idx, doc_idx) document instance --
NOT deduplicated across rows (a document repeated as a distractor in several
rows gets one row per occurrence here, matching how RetrievalPromptDataset
addresses documents: by (batch row, block slot), not by document identity).
Flat arrays + an `offsets` array marking each JSONL row's document span,
same ragged-array convention as precompute_teacher_targets.py.
"""
import argparse
import json
import time

import numpy as np


def load_documents(path: str, max_rows: int = None):
    """Returns (docs: list[str], offsets: np.ndarray) -- offsets[r]:offsets[r+1]
    are the doc indices belonging to JSONL row r (mirrors
    RetrievalPromptDataset's own reading of "context_docs")."""
    docs = []
    offsets = [0]
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            row = json.loads(line)
            row_docs = row.get("context_docs")
            if row_docs is None:
                row_docs = row["context"].split("\n")
            docs.extend(row_docs)
            offsets.append(len(docs))
    return docs, np.array(offsets, dtype=np.int64)


def embed_documents(docs: list, model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(docs, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)


def reduce_dim(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    PCA reduction applied ONCE, globally, before either clustering method
    (2026-09-20 fix, first empirical pass had over-splitting near-duplicate
    documents with GMM). Two independent reasons this matters, not one:
    (1) curse of dimensionality -- distances/BIC in the embedding model's
    native dimension (384+) concentrate and become poorly discriminative
    with the small node sizes this recursion deals with (min_node_size is
    typically single/double digits) -- exactly the calibration problem
    flagged in spec §8quater's method-1 discussion, and it turns out to
    also affect methods 2/3 in practice, not just a raw-radius approach;
    (2) GMM's parameter count -- even with covariance_type='diag' (linear in
    dimension, not quadratic), fitting in 384 dims with ~10-30 points per
    node is underdetermined, so BIC comparisons were dominated by noise
    rather than real structure (observed directly: bic_margin=5 accepted
    splits inside a synthetic near-duplicate group that should never split).
    n_components is capped at min(n_components, n_samples - 1, embedding
    dim) -- PCA can't produce more components than samples.
    """
    from sklearn.decomposition import PCA
    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
    return PCA(n_components=n_components, random_state=0).fit_transform(embeddings)


def gmm_recursive_depth(embeddings: np.ndarray, max_depth: int, min_node_size: int,
                         bic_margin: float, seed: int = 0) -> np.ndarray:
    """§8quater method 2: split a node only when a 2-component GMM's BIC beats
    the 1-component fit's by more than bic_margin -- a node with many points
    but no real sub-structure (BIC doesn't improve enough) stops here."""
    from sklearn.mixture import GaussianMixture

    depth = np.zeros(embeddings.shape[0], dtype=np.int64)

    def recurse(idx: np.ndarray, cur_depth: int):
        if cur_depth >= max_depth or len(idx) < max(min_node_size, 4):
            return
        X = embeddings[idx]
        # covariance_type='diag': a 'full' covariance is still wildly
        # underdetermined at these node sizes even after reduce_dim's PCA
        # step -- 'diag' keeps the parameter count linear in dimension
        # instead of quadratic, at the cost of assuming axis-aligned
        # covariance (an approximation, not exact multivariate structure).
        gmm1 = GaussianMixture(n_components=1, covariance_type="diag", random_state=seed).fit(X)
        gmm2 = GaussianMixture(n_components=2, covariance_type="diag", random_state=seed).fit(X)
        if gmm1.bic(X) - gmm2.bic(X) <= bic_margin:
            return  # not a statistically justified split -- stop, regardless of len(idx)
        labels = gmm2.predict(X)
        depth[idx] += 1
        for c in (0, 1):
            recurse(idx[labels == c], cur_depth + 1)

    recurse(np.arange(embeddings.shape[0]), 0)
    return depth


def hdbscan_recursive_depth(embeddings: np.ndarray, max_depth: int, min_node_size: int,
                             min_cluster_size: int):
    """
    §8quater method 3: recursively re-cluster each node with HDBSCAN. A node
    HDBSCAN keeps as one cluster (or calls entirely noise, label -1) stops
    there -- persistence-based, not a forced binary split.

    Returns (depth_splits, depth_weighted):
    - depth_splits: raw count of recursion levels (comparable in SHAPE to
      gmm_recursive_depth's output, but NOT in scale -- see below).
    - depth_weighted (float): each split adds log2(n_clusters_found) instead
      of a flat +1. 2026-09-20 fix: HDBSCAN doesn't produce a binary tree
      like the GMM method above -- on the synthetic test set it found 9
      clusters directly at the ROOT (not 2), so depth_splits alone made it
      look "shallow" (constant depth 1 for every document, recursion never
      going deeper because each of those 9 already-tight clusters had no
      further internal structure) while it had actually already extracted
      much more discriminative information in that single step than one
      GMM binary split. log2(9) ~= 3.17 vs GMM's log2(2) = 1 per split makes
      the two methods comparable on the same "bits of distinction" scale
      instead of comparing tree depths of algorithms with different
      branching factors.
    """
    from sklearn.cluster import HDBSCAN

    depth_splits = np.zeros(embeddings.shape[0], dtype=np.int64)
    depth_weighted = np.zeros(embeddings.shape[0], dtype=np.float64)

    def recurse(idx: np.ndarray, cur_depth: int):
        if cur_depth >= max_depth or len(idx) < max(min_node_size, min_cluster_size * 2):
            return
        X = embeddings[idx]
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
        real_labels = sorted(set(labels) - {-1})
        if len(real_labels) < 2:
            return  # one persistent cluster (or all noise) -- not a justified split
        depth_splits[idx] += 1
        depth_weighted[idx] += np.log2(len(real_labels))
        for c in real_labels:
            recurse(idx[labels == c], cur_depth + 1)
        # noise points (label -1) stay at this node's post-increment depth (no further split)

    recurse(np.arange(embeddings.shape[0]), 0)
    return depth_splits, depth_weighted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="JSONL with context_docs (or context), e.g. a "
                                                   "prepare_retrieval_data.py output")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max_rows", type=int, default=None, help="cap JSONL rows read, for a quick probe")
    p.add_argument("--max_depth", type=int, default=4)
    p.add_argument("--min_node_size", type=int, default=8)
    p.add_argument("--bic_margin", type=float, default=10.0, help="gmm method only")
    p.add_argument("--hdbscan_min_cluster_size", type=int, default=5)
    p.add_argument("--pca_dim", type=int, default=32,
                    help="global PCA reduction applied before EITHER method (see reduce_dim's "
                         "docstring for why: curse of dimensionality + GMM's BIC degenerating at "
                         "the embedding model's native dimension with small node sizes). "
                         "0 disables it (use the raw embedding space).")
    p.add_argument("--save_embeddings", action="store_true",
                    help="also store the raw (pre-PCA) embeddings (fp16) in the .npz for later reuse")
    args = p.parse_args()

    t0 = time.time()
    docs, offsets = load_documents(args.data, max_rows=args.max_rows)
    print(f"loaded {len(docs)} document instances from {len(offsets) - 1} rows in {args.data}", flush=True)

    embeddings = embed_documents(docs, args.embedding_model)
    print(f"embedded in {time.time() - t0:.1f}s, shape={embeddings.shape}", flush=True)

    clustering_space = embeddings
    if args.pca_dim > 0:
        clustering_space = reduce_dim(embeddings, args.pca_dim)
        print(f"reduced to {clustering_space.shape[1]} dims for clustering (PCA)", flush=True)

    t1 = time.time()
    depth_gmm = gmm_recursive_depth(clustering_space, args.max_depth, args.min_node_size, args.bic_margin)
    print(f"gmm depth computed in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    depth_hdbscan_splits, depth_hdbscan_weighted = hdbscan_recursive_depth(
        clustering_space, args.max_depth, args.min_node_size, args.hdbscan_min_cluster_size)
    print(f"hdbscan depth computed in {time.time() - t2:.1f}s", flush=True)

    doc_len = np.array([len(d) for d in docs], dtype=np.int64)

    save_kwargs = dict(offsets=offsets, depth_gmm=depth_gmm, depth_hdbscan_splits=depth_hdbscan_splits,
                        depth_hdbscan_weighted=depth_hdbscan_weighted, doc_len=doc_len)
    if args.save_embeddings:
        save_kwargs["embeddings"] = embeddings.astype(np.float16)
    np.savez_compressed(args.out, **save_kwargs)
    print(f"saved to {args.out}", flush=True)

    print("---", flush=True)
    print("depth distribution (gmm, binary splits):        ", np.bincount(depth_gmm).tolist(), flush=True)
    print("depth distribution (hdbscan, raw split count):  ", np.bincount(depth_hdbscan_splits).tolist(), flush=True)
    print(f"depth_hdbscan_weighted (log2(n_clusters) per split): "
          f"min={depth_hdbscan_weighted.min():.2f} max={depth_hdbscan_weighted.max():.2f} "
          f"mean={depth_hdbscan_weighted.mean():.2f}", flush=True)
    print("(depth_gmm and depth_hdbscan_weighted are the comparable pair -- both roughly "
          "'bits of distinction accumulated', not depth_gmm vs depth_hdbscan_splits which "
          "compares tree depths of algorithms with different branching factors)", flush=True)

    def safe_corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"corr(depth_gmm, doc_len)              = {safe_corr(depth_gmm, doc_len):.3f}", flush=True)
    print(f"corr(depth_hdbscan_weighted, doc_len) = {safe_corr(depth_hdbscan_weighted, doc_len):.3f}", flush=True)
    print(f"corr(depth_gmm, depth_hdbscan_weighted) = {safe_corr(depth_gmm, depth_hdbscan_weighted):.3f}", flush=True)
    print("(low correlation with doc_len is the point -- depth should NOT just reduce to "
          "'longer document', see spec §8quater's motivation)", flush=True)


if __name__ == "__main__":
    main()
