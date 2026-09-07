"""Encode TinyStories JSONL to a 1-D uint16 token stream."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eos-id", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids: list[int] = []
    import json

    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            encoded = tokenizer.encode(text).ids
            ids.extend(encoded)
            ids.append(args.eos_id)
    array = np.asarray(ids, dtype=np.uint16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    print(f"Wrote {args.output} tokens={array.size} unique_max={int(array.max())}")


if __name__ == "__main__":
    main()
