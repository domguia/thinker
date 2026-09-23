"""One-off adoption of an ALREADY-MERGED legacy precompute artifact (a single
final .npz for topk, and either a single .npz or a hidden_<L>.npy+offsets.npy
memmap-dir for hidden states -- i.e. NOT the shard_*.npz format that
migrate_legacy_shards.py handles) into the new named-subset storage tree,
WITHOUT regenerating or copying the (potentially huge, tens-to-hundreds-of-GB)
hidden-states array: the raw arrays are hardlinked (os.link, instant, no
extra disk use, falls back to a real copy only if hardlinking isn't possible
-- e.g. across filesystems) rather than copied, so the legacy path keeps
working unchanged for anything else still pointing at it.

Assumes -- verify this holds for the specific legacy artifact before running,
see finalize_topk/finalize_hidden's own format for comparison -- that the
merged .npz/offsets already use the "offsets[i]:offsets[i+1] is doc i's
token span" convention, IDENTITY-indexed against <split>.jsonl (row i of the
jsonl <-> row i of the array, len(offsets)-1 == number of jsonl lines), with
no sparse/gap rows and no separate remapping file. If that's not true for a
given legacy artifact, this script will silently produce a wrong subset --
don't use it without confirming identity indexing first (e.g. against the
producer's own journal entry, like this project's dev_notes/grid5000_usage.log.md).

Example (hotpotqa train_repr10k_ab, confirmed identity-indexed against
train_repr10k_ab.jsonl, 9500 lines):
    python learn/distill/adopt_legacy_merged.py \\
      --legacy_topk_npz .../retrieval1_backup/train_repr10k_ab.npz \\
      --legacy_hidden_dir .../retrieval1_backup/train_repr10k_ab._hidden \\
      --hidden_layers 64 \\
      --dataset_root data/distill/hotpotqa_full --split train \\
      --subset_name repr10k_n9500 --n_examples 9500 \\
      --teacher_name qwen_big --top_k 32 \\
      --dtype auto --checkpoint Qwen3.8-27B-FP8 --max_length 4096 \\
      --attn_implementation flash_attention_2 --quantization none
"""
import argparse
import os

import numpy as np

from precompute_teacher_targets import _load_manifest, _save_manifest, _check_no_collision


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
        print(f"Hardlinked {dst} -> {src} (same inode, no extra disk use)", flush=True)
    except OSError as e:
        import shutil
        print(f"Hardlink failed ({e}) -- falling back to a real copy (this will take a while for a large file)", flush=True)
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy_topk_npz", required=True, help="already-merged final .npz (indices/values/residual/offsets/k)")
    p.add_argument("--legacy_hidden_dir", default=None, help="dir with hidden_<L>.npy + offsets.npy (mutually exclusive with --legacy_hidden_npz)")
    p.add_argument("--legacy_hidden_npz", default=None, help="single merged .npz with hidden_<L> key(s) + offsets (mutually exclusive with --legacy_hidden_dir)")
    p.add_argument("--hidden_layers", default="", help="comma-separated hidden_<L> indices present in the legacy hidden artifact, e.g. '64' -- empty if no hidden data")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--subset_name", required=True)
    p.add_argument("--n_examples", type=int, required=True,
                    help="asserted against the legacy artifact's own offsets length -- identity subset (0..n-1) is assumed, see module docstring")
    p.add_argument("--teacher_name", required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_length", type=int, required=True)
    p.add_argument("--attn_implementation", default="unknown")
    p.add_argument("--quantization", default="unknown")
    args = p.parse_args()

    if args.legacy_hidden_dir and args.legacy_hidden_npz:
        raise SystemExit("--legacy_hidden_dir and --legacy_hidden_npz are mutually exclusive")

    subset_dir = os.path.join(args.dataset_root, "subsets", args.split)
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = os.path.join(subset_dir, f"{args.subset_name}.indices.npy")
    if os.path.exists(subset_path):
        raise SystemExit(f"{subset_path} already exists -- pick a different --subset_name, or remove it first if this is genuinely a re-adoption")

    teacher_cfg = {
        "dtype": args.dtype, "checkpoint": args.checkpoint, "max_length": args.max_length,
        "attn_implementation": args.attn_implementation, "quantization": args.quantization,
    }

    # --- topk ---
    topk_npz = np.load(args.legacy_topk_npz)
    offsets = topk_npz["offsets"]
    n_docs = len(offsets) - 1
    assert n_docs == args.n_examples, f"{args.legacy_topk_npz}: offsets imply {n_docs} docs, expected --n_examples {args.n_examples}"
    k = int(topk_npz["k"]) if "k" in topk_npz.files else args.top_k
    assert k == args.top_k, f"{args.legacy_topk_npz}: npz's own k={k} != --top_k {args.top_k}"
    topk_npz.close()

    indices = np.arange(args.n_examples, dtype=np.int64)
    np.save(subset_path, indices)
    print(f"Wrote {subset_path} ({len(indices)} examples, identity)", flush=True)

    out_dir = os.path.join(args.dataset_root, "topk", args.split)
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{args.subset_name}.{args.teacher_name}.npz"
    out_path = os.path.join(out_dir, out_name)
    _link_or_copy(os.path.abspath(args.legacy_topk_npz), out_path)

    manifest = _load_manifest(out_dir)
    entry = manifest.setdefault(args.subset_name, {"subset_file": f"subsets/{args.split}/{args.subset_name}.indices.npy",
                                                     "teachers": {}})
    new_cfg = {**teacher_cfg, "top_k": k, "file": out_name}
    _check_no_collision(entry["teachers"].get(args.teacher_name), new_cfg, f"{out_dir}/manifest.json[{args.subset_name}][{args.teacher_name}]")
    entry["teachers"][args.teacher_name] = new_cfg
    _save_manifest(out_dir, manifest)
    print(f"Adopted topk -> {out_path} (hardlink, {int(offsets[-1])} tokens across {n_docs} docs)", flush=True)

    # --- hidden (optional) ---
    hidden_layer_indices = [int(x) for x in args.hidden_layers.split(",") if x != ""]
    if hidden_layer_indices:
        short_name = f"n{args.n_examples}"
        for layer in hidden_layer_indices:
            if args.legacy_hidden_dir:
                raw_src = os.path.join(args.legacy_hidden_dir, f"hidden_{layer}.npy")
                off_src = os.path.join(args.legacy_hidden_dir, "offsets.npy")
            else:
                # single merged .npz -- can't hardlink a key out of an npz, must extract to a plain .npy once
                hd = np.load(args.legacy_hidden_npz, mmap_mode="r")
                raw_src = None  # handled below
                off_src = None

            layer_dir = os.path.join(args.dataset_root, "embedding", args.split, f"layer_{layer}")
            os.makedirs(layer_dir, exist_ok=True)
            dst_raw = os.path.join(layer_dir, f"{short_name}.{args.teacher_name}.npy")
            dst_off = os.path.join(layer_dir, f"{short_name}.{args.teacher_name}.offsets.npy")

            if raw_src is not None:
                h_offsets = np.load(off_src)
                assert len(h_offsets) - 1 == args.n_examples, f"{off_src}: implies {len(h_offsets) - 1} docs, expected {args.n_examples}"
                _link_or_copy(os.path.abspath(raw_src), dst_raw)
                _link_or_copy(os.path.abspath(off_src), dst_off)
            else:
                arr = hd[f"hidden_{layer}"]
                h_offsets = hd["offsets"]
                assert len(h_offsets) - 1 == args.n_examples, f"{args.legacy_hidden_npz}: implies {len(h_offsets) - 1} docs, expected {args.n_examples}"
                np.save(dst_raw, np.asarray(arr))
                np.save(dst_off, h_offsets)
                print(f"Extracted hidden_{layer} from merged npz to {dst_raw} (real copy, npz doesn't support hardlinking a single key)", flush=True)

            manifest = _load_manifest(layer_dir)
            entry = manifest.setdefault(short_name, {"subset": args.subset_name, "teachers": {}})
            new_cfg = {**teacher_cfg, "file": f"{short_name}.{args.teacher_name}.npy"}
            _check_no_collision(entry["teachers"].get(args.teacher_name), new_cfg,
                                 f"{layer_dir}/manifest.json[{short_name}][{args.teacher_name}]")
            entry["teachers"][args.teacher_name] = new_cfg
            _save_manifest(layer_dir, manifest)
            print(f"Adopted hidden layer {layer} -> {dst_raw}", flush=True)

    print("MARKER: ADOPTION_COMPLETE -- legacy artifact left untouched at its original path.", flush=True)


if __name__ == "__main__":
    main()
