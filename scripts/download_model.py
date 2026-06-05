r"""Phase 1 / Route A: download the pre-converted Gemma 4 E4B INT4 OpenVINO IR
from Hugging Face into models/<repo-name>/.

Usage:
    .\.venv\Scripts\python.exe scripts\download_model.py
    .\.venv\Scripts\python.exe scripts\download_model.py --repo OpenVINO/gemma-4-E4B-it-int8-ov

The repo is public (not gated), so an HF login is not required. If you do have
HF_TOKEN set or `hf auth login` has been run, the token is used automatically.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from huggingface_hub import snapshot_download


DEFAULT_REPO = "OpenVINO/gemma-4-E4B-it-int4-ov"
DEFAULT_TARGET_ROOT = pathlib.Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_TARGET_ROOT,
        help="Local root directory; the model goes into <root>/<repo-name>",
    )
    args = parser.parse_args()

    target = args.root / args.repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {target}")

    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(target),
    )
    print(f"\nDownloaded to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
