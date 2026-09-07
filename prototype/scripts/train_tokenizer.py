"""Train a BPE tokenizer on TinyStories-v2 JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/tinystories-v2/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    return parser.parse_args()


def text_iter(path: Path):
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing {args.input}. Run scripts/prepare_tinystories.py first.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(text_iter(args.input), trainer=trainer)
    tokenizer.save(str(args.output))
    print(f"Wrote {args.output} vocab={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
