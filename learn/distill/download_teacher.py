"""Download a Teacher model snapshot from the Hub (resumable).

Meant to run on a Grid'5000 CPU reservation (no GPU needed just to download).
huggingface_hub resumes partial downloads automatically, so re-running this
after a walltime cutoff or a network drop just continues where it left off.

Example (Qwen/Qwen3.8-27B-FP8, ~30.9 GB, the FP8-quantized version of a
Qwen3.5 vision-language model used here purely as a text Teacher -- see
learn/distill/README.md):

    python learn/distill/download_teacher.py \
      --repo_id Qwen/Qwen3.8-27B-FP8 --local_dir /tmp/teachers/Qwen3.8-27B-FP8
"""
import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"Downloading {args.repo_id} -> {args.local_dir} (resumable) ...")
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir, revision=args.revision)

    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )
    print(f"Done: {path}")
    print(f"Total size on disk: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
