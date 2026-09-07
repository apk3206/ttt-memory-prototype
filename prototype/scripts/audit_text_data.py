"""Audit train/validation text splits before language-model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean


WHITESPACE = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().casefold()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def inspect_file(path: Path) -> tuple[dict[str, object], set[str]]:
    exact_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    normalized_counts: Counter[str] = Counter()
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    empty_records = 0
    control_characters = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record["text"]
            if not isinstance(text, str):
                raise TypeError(f"Expected string text in {path}.")
            normalized = canonicalize(text)
            if not normalized:
                empty_records += 1
                continue

            exact_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            normalized_hashes.add(normalized_hash)
            normalized_counts[normalized_hash] += 1
            char_lengths.append(len(text))
            word_lengths.append(len(normalized.split()))
            control_characters += sum(ord(character) < 32 and character not in "\t\n\r" for character in text)

    records = len(char_lengths) + empty_records
    duplicates = records - empty_records - len(normalized_hashes)
    return {
        "path": str(path),
        "records": records,
        "empty_records": empty_records,
        "exact_unique_records": len(exact_hashes),
        "normalized_duplicate_records": duplicates,
        "normalized_duplicate_rate": round(duplicates / max(records - empty_records, 1), 8),
        "control_characters": control_characters,
        "characters": {
            "min": min(char_lengths),
            "mean": round(mean(char_lengths), 2),
            "p50": percentile(char_lengths, 0.50),
            "p95": percentile(char_lengths, 0.95),
            "max": max(char_lengths),
        },
        "words": {
            "min": min(word_lengths),
            "mean": round(mean(word_lengths), 2),
            "p50": percentile(word_lengths, 0.50),
            "p95": percentile(word_lengths, 0.95),
            "max": max(word_lengths),
        },
    }, normalized_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/tinystories-v2"))
    args = parser.parse_args()
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Expected train.jsonl and validation.jsonl in --data-dir.")

    train, train_hashes = inspect_file(train_path)
    validation, validation_hashes = inspect_file(validation_path)
    overlap = len(train_hashes & validation_hashes)
    report = {
        "task": "causal_language_modeling",
        "checks": {
            "train_validation_exact_normalized_overlap": overlap,
            "train_validation_overlap_rate": round(overlap / max(len(validation_hashes), 1), 8),
            "note": "This catches exact duplicates after whitespace/case normalization, not semantic near-duplicates.",
        },
        "splits": {"train": train, "validation": validation},
        "modeling_guidance": {
            "use": ["next-token cross-entropy", "validation perplexity", "early stopping", "weight decay", "gradient clipping"],
            "do_not_use_for_this_task": ["SMOTE", "class weights", "focal loss", "ROC-AUC", "confusion matrix", "tabular feature scaling"],
        },
    }
    report_path = args.data_dir / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit report written to {report_path}")


if __name__ == "__main__":
    main()
