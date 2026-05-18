#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "a8cheng/navila-llama3-8b-8f"


def default_output_dir(model_id: str) -> Path:
    vrrobo_isaaclab_dir = Path(__file__).resolve().parents[1]
    return vrrobo_isaaclab_dir / "vla_weights" / model_id.split("/")[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NaVILA model weights from Hugging Face.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to download.")
    parser.add_argument("--output_dir", default=None, help="Local directory for the downloaded snapshot.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Optional Hugging Face token.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("huggingface_hub is not installed. Install it with: python -m pip install huggingface_hub")
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        token=args.token,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.model_id} to {snapshot_path}")
    print(f"Use with play_gs.py: --vla_model_path {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
