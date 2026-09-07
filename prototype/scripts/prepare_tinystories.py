"""Create a small, reproducible TinyStories dataset for Phase 0 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# The execution environment may forbid writes to the user's home directory.
# Configure Hub caches before importing Hugging Face packages.
DEFAULT_CACHE_DIR = Path("data/.hf-cache").resolve()
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_CACHE_DIR / "datasets"))

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "main"
MIB = 1024 * 1024


def write_subset(
    split: str, output_path: Path, target_bytes: int, cache_dir: Path
) -> dict[str, Any]:
    """Stream one split and write one complete story per JSON Lines record."""
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        streaming=True,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    written_bytes = 0
    story_count = 0
    digest = hashlib.sha256()

    with output_path.open("wb") as handle, tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        desc=f"Writing {split}",
    ) as progress:
        for record in dataset:
            story = record["text"].strip()
            if not story:
                continue

            encoded = (json.dumps({"text": story}, ensure_ascii=False) + "\n").encode("utf-8")
            if written_bytes + len(encoded) > target_bytes:
                break

            handle.write(encoded)
            digest.update(encoded)
            written_bytes += len(encoded)
            story_count += 1
            progress.update(len(encoded))

    if story_count == 0:
        raise RuntimeError(f"No text was written for the {split!r} split.")

    return {
        "source_split": split,
        "file": str(output_path.name),
        "target_bytes": target_bytes,
        "written_bytes": written_bytes,
        "story_count": story_count,
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-mib", type=int, default=100)
    parser.add_argument("--validation-mib", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories-v2"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Project-local Hugging Face cache; avoids writing to a user home directory.",
    )
    args = parser.parse_args()
    if args.train_mib <= 0 or args.validation_mib <= 0:
        parser.error("--train-mib and --validation-mib must both be positive.")
    return args


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    if train_path.exists() or validation_path.exists():
        raise FileExistsError(
            f"{output_dir} already contains data. Choose a new --output-dir or remove it explicitly."
        )

    manifest = {
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": "CDLA-Sharing-1.0; review the upstream dataset card before use.",
        },
        "sampling": "first complete stories from each streamed official split; deterministic for a fixed revision",
        "splits": {
            "train": write_subset(
                "train", train_path, args.train_mib * MIB, args.cache_dir
            ),
            "validation": write_subset(
                "validation", validation_path, args.validation_mib * MIB, args.cache_dir
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Dataset manifest written to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
