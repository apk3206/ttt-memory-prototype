"""CPU smoke test: one forward/backward for attn and TTT-Linear mixers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tttmem.config import ModelConfig
from tttmem.model import CausalLM


def run(mixer: str) -> None:
    cfg = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, mixer=mixer, ttt_mini_batch=8, max_seq_len=64)
    model = CausalLM(cfg)
    tokens = torch.randint(0, 128, (2, 24))
    loss = model.loss(tokens)
    loss.backward()
    print(f"{mixer}: params={model.param_count()} loss={loss.item():.4f}")


if __name__ == "__main__":
    run("attn")
    run("ttt_linear")
    full = CausalLM(ModelConfig(mixer="ttt_linear"))
    print(f"ttt_linear 30M config params={full.param_count():,}")
    if full.param_count() < 30_000_000:
        raise SystemExit("30M TTT-Linear config is under the matched-size gate")
    print("ok")
