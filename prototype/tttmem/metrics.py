"""Language-model diagnostics. This task is next-token prediction, not classification."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import CausalLM


@torch.no_grad()
def batch_report(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    vocab = logits.size(-1)
    loss = F.cross_entropy(logits.float(), targets)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    pred = logits.argmax(dim=-1)
    unique = int(pred.unique().numel())
    top1_mass = probs.max(dim=-1).values.mean().item()
    token_match = (pred == targets).float().mean().item()
    # Argmax can sit on a few frequent tokens while softmax is still spread.
    # Collapse = peaked distribution, not "few unique argmax ids" early in training.
    collapsed = entropy < 1.2 and (top1_mass > 0.9 or token_match > 0.85)
    return {
        "loss": float(loss.item()),
        "ppl": float(math.exp(min(loss.item(), 20))),
        "entropy": float(entropy),
        "unique_preds": unique,
        "unique_frac": unique / vocab,
        "top1_mass": float(top1_mass),
        "token_match": float(token_match),
        "collapsed": float(collapsed),
    }


@torch.no_grad()
def evaluate(model: CausalLM, loader: DataLoader, device: torch.device, batches: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        batch = batch.to(device)
        logits = model(batch)[:, :-1].reshape(-1, model.cfg.vocab_size)
        targets = batch[:, 1:].reshape(-1)
        report = batch_report(logits, targets)
        for key, value in report.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    model.train()
    if count == 0:
        raise RuntimeError("evaluate() received no batches")
    out = {key: value / count for key, value in totals.items()}
    out["collapsed"] = 1.0 if out["collapsed"] >= 0.5 else 0.0
    return out
