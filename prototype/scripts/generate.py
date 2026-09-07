"""Sample text from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer

from tttmem.adapt import adapt
from tttmem.config import ModelConfig
from tttmem.model import CausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--adapt-steps", type=int, default=0, help="E2E TTT inner steps on the prompt before sampling")
    parser.add_argument("--adapt-lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


@torch.no_grad()
def sample(model: CausalLM, ids: torch.Tensor, n: int, temperature: float = 0.8) -> torch.Tensor:
    model.eval()
    for _ in range(n):
        logits = model(ids)[:, -1]
        next_id = torch.multinomial(torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
        if ids.size(1) >= model.cfg.max_seq_len:
            ids = ids[:, -model.cfg.max_seq_len :]
    return ids


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg: ModelConfig = blob["config"]
    model = CausalLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt_ids = tokenizer.encode(args.prompt).ids
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    if args.adapt_steps:
        model = adapt(model, ids, lr=args.adapt_lr, steps=args.adapt_steps)
    out = sample(model, ids, args.tokens, temperature=args.temperature)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
