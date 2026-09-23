"""
Spec dev_notes/indexed_attention_spec.md §8quater, 2026-09-20 revision: token-level
(NO pooling) embedding collection and clustering, across MULTIPLE datasets
(retrieval/reasoning/general) at once.

Why this replaces precompute_doc_clusters.py's approach rather than extending
it: that script (and the whole §8quater discussion up to this point) reduced
each document to ONE mean-pooled embedding vector before clustering --
confirmed directly by inspecting sentence-transformers/all-MiniLM-L6-v2's
pipeline (`Pooling(pooling_mode='mean')`). The user's objection is correct
and goes beyond a calibration detail: mean pooling (a) flattens a document's
internal heterogeneity into one point (a document mixing two sub-topics
looks identical to one uniformly about their midpoint), and (b) that
model's max_seq_length=256 silently truncates longer documents -- lost
content, not just a blurred average. Both losses matter directly for what
this whole line of research wants to measure ("does this document have rich
internal structure").

This script never reduces a document to a single vector. Every kept TOKEN
of every document gets its own embedding row; clustering (the same two
recursive methods as precompute_doc_clusters.py, §8quater methods 2/3) runs
directly on the full token cloud across ALL documents and ALL datasets at
once -- a cluster or a recursive split can mix tokens from different
documents (even different datasets) freely, exactly what "cluster all
embeddings simultaneously, track which token belongs to which document"
means. Per-document numbers reported at the end (mean token depth, number
of distinct leaf clusters a document's tokens land in, intra-document
token-cloud dispersion) are READOUTS computed FROM the individual token
embeddings after the fact -- not a preprocessing step the clustering itself
depends on, unlike pooling.

Embedding model: a bidirectional sentence-encoder backbone (BAAI/bge-large-
en-v1.5 by default, 1024-dim, max_seq_length 512 -- half MiniLM's truncation
loss, and higher-quality representations), used via its raw HF AutoModel
(NOT the sentence-transformers wrapper, which would pool) to keep
`last_hidden_state` token-by-token. This is deliberately NOT one of this
project's own causal Teacher families (core/model_families.py's lfm2/olmo/
qwen) by default -- those are decoder-only, so a token's hidden state never
sees the tokens after it (causal masking); a bidirectional encoder gives
every token a representation informed by the WHOLE document, which is what
a topic/density measure needs. --embedding_backend causal is available to
test the user's counter-hypothesis that these models' own representations
are richer for this project's data than a generic embedding model --
unverified, see embed_tokens' docstring for the mechanical caveats that
still apply (position-asymmetric representations, last-layer anisotropy).

Datasets: pass one or more --data DATASET_TYPE:path triples (dataset_type
in retrieval/reasoning/general), each read by its own field convention:
- retrieval: "context_docs" (list of documents per row, prepare_retrieval_data.py)
- reasoning: "problem" field, one document per row (prepare_reasoning_data.py)
- general: "text" field, one document per row (prepare_general_data.py)

Meant to run on a GPU node (embedding extraction) with enough CPU/RAM for
the clustering step at real scale (thousands of documents x tens/hundreds
of tokens each = potentially 100K+ points) -- NOT sized for a laptop-scale
smoke test beyond a few dozen documents (see this repo's --max_docs_per_dataset
for keeping a quick correctness check cheap).

Output: a single .npz --
- token_embeddings (N_tokens, pca_dim) float16 -- POST-PCA (see reduce_dim
  in precompute_doc_clusters.py for why PCA is applied before clustering;
  reused here unchanged), everything needed to inspect/re-cluster later.
- token_doc_id (N_tokens,) int64 -- which document (index into doc_meta) each
  token row belongs to. This IS the "mechanism to track which embedding
  belongs to which document" the user asked for, kept explicit rather than
  implied by array order.
- depth_gmm, depth_hdbscan_splits, depth_hdbscan_weighted (N_tokens,) --
  same three outputs as precompute_doc_clusters.py's methods 2/3, but
  PER TOKEN now, not per document.
- doc_meta: structured array, one row per document -- dataset_source,
  row_idx, doc_idx, n_tokens_kept, intra_doc_dispersion (mean pairwise
  cosine distance among that document's OWN token embeddings, computed
  directly from the per-token vectors, never from a pooled representative).
"""
import argparse
import json
import time

import numpy as np
import torch


def load_documents(spec: str):
    """spec: 'dataset_type:path'. Returns list of (dataset_source, row_idx, doc_idx, text)."""
    dataset_type, path = spec.split(":", 1)
    assert dataset_type in ("retrieval", "reasoning", "general"), (
        f"unknown dataset_type {dataset_type!r} in --data {spec!r} (expected retrieval/reasoning/general)"
    )
    out = []
    with open(path) as f:
        for row_idx, line in enumerate(f):
            row = json.loads(line)
            if dataset_type == "retrieval":
                docs = row.get("context_docs")
                if docs is None:
                    docs = row["context"].split("\n")
                for doc_idx, d in enumerate(docs):
                    out.append((dataset_type, row_idx, doc_idx, d))
            elif dataset_type == "reasoning":
                out.append((dataset_type, row_idx, 0, row["problem"]))
            else:  # general
                out.append((dataset_type, row_idx, 0, row["text"]))
    return out


def embed_tokens(texts: list, model_name: str, device: str, max_length: int, batch_size: int = 16,
                  backend: str = "bidirectional", layer: int = -1):
    """
    Returns (embeddings: (N_tokens, hidden_dim) float32, doc_id_of_token: (N_tokens,) int64,
    n_tokens_per_doc: (len(texts),) int64). NO pooling layer applied in either backend.
    Special/padding tokens are dropped via attention_mask before storing -- only real content
    tokens are kept.

    backend="bidirectional" (default): raw AutoModel forward, last_hidden_state -- the
    project's original choice (BAAI/bge-large-en-v1.5), see module docstring for why a
    bidirectional encoder was preferred over the project's own causal Teachers.

    backend="causal": AutoModelForCausalLM on one of this project's own downloaded families
    (core/model_families.py -- lfm2/olmo/qwen), output_hidden_states=True, reading
    hidden_states[layer] (default -1 = last). Added on the user's request (2026-09-20) to
    test their hypothesis that these models' representations are richer/more capable than a
    dedicated embedding model for this project's own data distribution -- an empirical
    question, not settled by the argument above. The caveat that argument raised still
    applies mechanically: causal masking means an early token's hidden state never sees the
    tokens after it (representation asymmetric by position within a document, unlike
    bidirectional), and a decoder's LAST layer is trained purely for next-token prediction,
    a known source of anisotropic/degenerate similarity geometry in the literature -- which
    is why `layer` defaults to -1 but is exposed so a middle layer (often more linearly
    semantic in decoder-only models) can be tried instead, e.g. --embedding_layer 12.
    """
    from transformers import AutoTokenizer

    if backend == "causal":
        from bench_teacher import load_model_and_tokenizer
        from core.model_families import resolve_model_name
        model_name = resolve_model_name(model_name)
        # reuses precompute_teacher_targets.py's own loader (device_map/max_memory/quantization,
        # flash_attention_2->sdpa fallback) instead of a naive from_pretrained -- needed for
        # multi-GPU/FP8 Teachers (e.g. qwen_big), not just the small lfm2/olmo checkpoints.
        model, tokenizer = load_model_and_tokenizer(model_name, dtype=torch.bfloat16)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        device = next(model.parameters()).device  # device_map="auto" decides placement, not `device`
    else:
        from transformers import AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeds = []
    doc_id_of_token = []
    n_tokens_per_doc = np.zeros(len(texts), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(device)
            if backend == "causal":
                out = model(**enc, output_hidden_states=True).hidden_states[layer]  # (B, T, H), no pooling
            else:
                out = model(**enc).last_hidden_state  # (B, T, H) -- no pooling
            mask = enc["attention_mask"].bool()
            for i in range(len(batch_texts)):
                doc_idx_global = start + i
                kept = out[i][mask[i]]  # (n_real_tokens, H) -- drop pad (and BOS/EOS stay, deliberately:
                                          # they're real forward-pass positions, not padding)
                all_embeds.append(kept.cpu().float().numpy())
                doc_id_of_token.extend([doc_idx_global] * kept.shape[0])
                n_tokens_per_doc[doc_idx_global] = kept.shape[0]
            if (start // batch_size) % 10 == 0:
                print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)} documents", flush=True)

    embeddings = np.concatenate(all_embeds, axis=0)
    return embeddings, np.array(doc_id_of_token, dtype=np.int64), n_tokens_per_doc


def intra_doc_dispersion(embeddings: np.ndarray, token_doc_id: np.ndarray, n_docs: int,
                          max_tokens_for_stat: int = 200) -> np.ndarray:
    """
    Mean pairwise cosine distance among a document's OWN token embeddings --
    computed directly from the individual (un-pooled) vectors, never from a
    single representative. This is the "density of information per document"
    readout, distinct from (and not required by) the cross-document
    clustering below. A document with 0-1 real tokens gets NaN (undefined).
    max_tokens_for_stat subsamples very long documents for this O(n^2)
    pairwise computation only -- does not affect the embeddings/clustering
    stored elsewhere.
    """
    disp = np.full(n_docs, np.nan, dtype=np.float64)
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    for d in range(n_docs):
        idx = np.where(token_doc_id == d)[0]
        if len(idx) < 2:
            continue
        if len(idx) > max_tokens_for_stat:
            idx = np.random.default_rng(0).choice(idx, size=max_tokens_for_stat, replace=False)
        X = norm[idx]
        sims = X @ X.T
        n = X.shape[0]
        mean_cos_dist = 1.0 - (sims.sum() - n) / (n * (n - 1))  # exclude diagonal (self-similarity = 1)
        disp[d] = mean_cos_dist
    return disp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True,
                    help="dataset_type:path, repeatable (dataset_type in retrieval/reasoning/general)")
    p.add_argument("--out", required=True)
    p.add_argument("--embedding_model", default="BAAI/bge-large-en-v1.5",
                    help="HF model id, or lfm2/olmo/qwen when --embedding_backend causal")
    p.add_argument("--embedding_backend", choices=["bidirectional", "causal"], default="bidirectional")
    p.add_argument("--embedding_layer", type=int, default=-1,
                    help="hidden_states index to read, causal backend only (-1=last; try a middle "
                         "layer, e.g. 12, if last-layer geometry looks degenerate)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_length", type=int, default=512, help="tokenizer truncation length per document")
    p.add_argument("--max_docs_per_dataset", type=int, default=None,
                    help="cap documents PER --data entry, for a quick local smoke test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--min_node_size", type=int, default=20)
    p.add_argument("--bic_margin", type=float, default=10.0)
    p.add_argument("--hdbscan_min_cluster_size", type=int, default=10)
    p.add_argument("--pca_dim", type=int, default=32)
    args = p.parse_args()

    from precompute_doc_clusters import reduce_dim, gmm_recursive_depth, hdbscan_recursive_depth

    t0 = time.time()
    doc_records = []
    for spec in args.data:
        recs = load_documents(spec)
        if args.max_docs_per_dataset is not None:
            recs = recs[:args.max_docs_per_dataset]
        doc_records.extend(recs)
        print(f"{spec}: {len(recs)} document instances", flush=True)
    texts = [r[3] for r in doc_records]
    print(f"total: {len(texts)} documents across {len(args.data)} dataset(s)", flush=True)

    embeddings, token_doc_id, n_tokens_per_doc = embed_tokens(
        texts, args.embedding_model, args.device, args.max_length, batch_size=args.batch_size,
        backend=args.embedding_backend, layer=args.embedding_layer)
    print(f"embedded {embeddings.shape[0]} tokens (dim={embeddings.shape[1]}) "
          f"in {time.time() - t0:.1f}s, device={args.device}", flush=True)

    t1 = time.time()
    disp = intra_doc_dispersion(embeddings, token_doc_id, len(doc_records))
    print(f"intra-doc dispersion computed in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    clustering_space = reduce_dim(embeddings, args.pca_dim) if args.pca_dim > 0 else embeddings
    print(f"reduced to {clustering_space.shape[1]} dims for clustering (PCA) in {time.time() - t2:.1f}s",
          flush=True)

    t3 = time.time()
    depth_gmm = gmm_recursive_depth(clustering_space, args.max_depth, args.min_node_size, args.bic_margin)
    print(f"gmm depth (per TOKEN, {len(depth_gmm)} tokens) computed in {time.time() - t3:.1f}s", flush=True)

    t4 = time.time()
    depth_hdbscan_splits, depth_hdbscan_weighted = hdbscan_recursive_depth(
        clustering_space, args.max_depth, args.min_node_size, args.hdbscan_min_cluster_size)
    print(f"hdbscan depth (per TOKEN) computed in {time.time() - t4:.1f}s", flush=True)

    # per-document readouts, computed FROM the per-token results (not fed back into clustering)
    n_docs = len(doc_records)
    doc_dataset = np.array([r[0] for r in doc_records])
    doc_row_idx = np.array([r[1] for r in doc_records], dtype=np.int64)
    doc_doc_idx = np.array([r[2] for r in doc_records], dtype=np.int64)
    mean_depth_gmm_per_doc = np.full(n_docs, np.nan)
    n_distinct_leaf_gmm_per_doc = np.zeros(n_docs, dtype=np.int64)
    for d in range(n_docs):
        idx = np.where(token_doc_id == d)[0]
        if len(idx) == 0:
            continue
        mean_depth_gmm_per_doc[d] = depth_gmm[idx].mean()
        n_distinct_leaf_gmm_per_doc[d] = len(set(depth_gmm[idx].tolist()))

    np.savez_compressed(
        args.out,
        token_embeddings=clustering_space.astype(np.float16),
        token_doc_id=token_doc_id,
        depth_gmm=depth_gmm,
        depth_hdbscan_splits=depth_hdbscan_splits,
        depth_hdbscan_weighted=depth_hdbscan_weighted,
        doc_dataset=doc_dataset,
        doc_row_idx=doc_row_idx,
        doc_doc_idx=doc_doc_idx,
        n_tokens_per_doc=n_tokens_per_doc,
        intra_doc_dispersion=disp,
        mean_depth_gmm_per_doc=mean_depth_gmm_per_doc,
        n_distinct_leaf_gmm_per_doc=n_distinct_leaf_gmm_per_doc,
    )
    print(f"saved to {args.out}", flush=True)

    print("---", flush=True)
    for ds in sorted(set(doc_dataset.tolist())):
        sel = doc_dataset == ds
        print(f"[{ds}] n_docs={sel.sum()} mean_intra_doc_dispersion={np.nanmean(disp[sel]):.4f} "
              f"mean_depth_gmm={np.nanmean(mean_depth_gmm_per_doc[sel]):.2f} "
              f"mean_n_distinct_leaf={n_distinct_leaf_gmm_per_doc[sel].mean():.2f}", flush=True)
    print(f"total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
