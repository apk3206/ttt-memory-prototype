from torch.optim import Adam, AdamW


def build_optimizer(
    model,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
):
    """AdamW with decoupled L2 on matrices only (bias/norm/1D skipped)."""
    decay, no_decay = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name.endswith("bias") or "norm" in param_name.lower() or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = name.lower()
    kwargs = {"lr": lr, "betas": betas, "eps": eps}
    if name == "adamw":
        return AdamW(groups, **kwargs)
    if name == "adam":
        return Adam(groups, **kwargs)
    raise ValueError(f"Unknown optimizer {name!r}; use adamw or adam")
