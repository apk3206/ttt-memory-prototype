"""Epoch-based pretrain / meta-train for the TTT memory prototype."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
print("train.py starting", flush=True)

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tttmem.adapt import meta_loss
from tttmem.ckpt import load_ckpt
from tttmem.config import ModelConfig, TrainConfig
from tttmem.data import EpochChunks
from tttmem.metrics import evaluate
from tttmem.model import CausalLM
from tttmem.optim import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tinystories-v2/train.npy"))
    parser.add_argument("--val-data", type=Path, default=Path("data/tinystories-v2/validation.npy"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mixer", choices=("attn", "ttt_linear"), default="attn")
    parser.add_argument("--mode", choices=("pretrain", "meta"), default="pretrain")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Cosine floor as a fraction of peak LR so late epochs still update.",
    )
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 (0.95 is GPT-style).")
    parser.add_argument(
        "--weight-decay",
        "--l2",
        dest="weight_decay",
        type=float,
        default=0.1,
        help="Decoupled L2 via AdamW on matrices (not bias/norm). Do not raise above 0.1 while underfitting.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="0.1 from scratch. Use 0.05 when resuming a plateaued 30M run to fit more.",
    )
    parser.add_argument(
        "--save-epoch-ckpts",
        action="store_true",
        help="Also write epoch_XX.pt (large). Default is last.pt only.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore --resume. Required for Phase 1 TTT-Linear (do not load attention weights).",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto")
    parser.add_argument(
        "--min-params",
        type=int,
        default=30_000_000,
        help="Refuse to start if the model is smaller than this (matched 30M comparison). Use 0 to skip.",
    )
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def require_cuda_kernels(device: torch.device) -> None:
    """Kaggle's current PyTorch has no Pascal (P100 sm_60) kernels."""
    if device.type != "cuda":
        return
    major, minor = torch.cuda.get_device_capability(device)
    if major < 7:
        name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"{name} is sm_{major}{minor}. This PyTorch only runs sm_70+ (T4, V100, A100, L4). "
            "Kaggle: Settings → Accelerator → GPU T4, then Restart session. Do not use P100."
        )


def pick_amp(device: torch.device, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "off"
    # GTX 1650 Ti is Turing (7.5): real speed is fp16, not bf16.
    major, _minor = torch.cuda.get_device_capability(device)
    if major >= 8:
        return "bf16"
    return "fp16"


def cosine_lr(step: int, total: int, warmup: int, peak: float, min_ratio: float = 0.1) -> float:
    floor = peak * min_ratio
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(progress, 1.0)))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def log_fit(
    model,
    train_loader,
    val_loader,
    device: torch.device,
    batches: int,
    log_path: Path,
    epoch: int,
    step: int,
) -> dict:
    """Eval-mode CE on train and val (dropout off). Gap diagnoses overfit."""
    val_report = evaluate(model, val_loader, device, batches)
    append_jsonl(log_path, {"epoch": epoch, "step": step, "split": "val", **val_report})
    train_report = evaluate(model, train_loader, device, batches)
    gap = train_report["loss"] - val_report["loss"]
    append_jsonl(
        log_path,
        {"epoch": epoch, "step": step, "split": "train_eval", "gap_train_minus_val": gap, **train_report},
    )
    flag = " COLLAPSE" if val_report["collapsed"] else ""
    print(
        f"eval step={step:06d} val_loss={val_report['loss']:.4f} val_ppl={val_report['ppl']:.2f} "
        f"train_eval_loss={train_report['loss']:.4f} gap={gap:+.3f} "
        f"entropy={val_report['entropy']:.3f} token_match={val_report['token_match']:.3f}{flag}",
        flush=True,
    )
    return val_report


def save_ckpt(path: Path, model: CausalLM, optimizer: torch.optim.Optimizer, step: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": model.cfg,
            **extra,
        },
        path,
    )
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.is_dir() and path.name == "last.pt":
        shutil.copy2(path, kaggle_root / "last.pt")
        print("wrote /kaggle/working/last.pt for Kaggle Output download", flush=True)


def main() -> None:
    args = parse_args()
    if args.out is None:
        args.out = Path("checkpoints/ttt-linear-30m" if args.mixer == "ttt_linear" else "checkpoints/baseline-30m")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    require_cuda_kernels(device)
    amp = pick_amp(device, args.amp)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp, torch.float32)

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        mixer=args.mixer,
        max_seq_len=max(args.seq_len, 512),
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        seed=args.seed,
        mode=args.mode,
        amp=amp,
        optimizer=args.optimizer,
        eval_batches=args.eval_batches,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        betas=(0.9, args.beta2),
    )
    model = CausalLM(model_cfg).to(device)
    optimizer = build_optimizer(
        model,
        train_cfg.optimizer,
        train_cfg.lr,
        train_cfg.weight_decay,
        betas=train_cfg.betas,
    )
    scaler = GradScaler("cuda", enabled=amp == "fp16")

    train_set = EpochChunks(args.data, train_cfg.seq_len)
    steps_per_epoch = max(len(train_set) // train_cfg.batch_size, 1)
    opt_steps_per_epoch = max(steps_per_epoch // train_cfg.grad_accum, 1)
    total_opt_steps = opt_steps_per_epoch * train_cfg.epochs
    warmup = max(int(total_opt_steps * train_cfg.warmup_ratio), 50)

    start_step = 0
    start_epoch = 0
    resume_path = None if args.from_scratch else args.resume
    if args.from_scratch and args.resume:
        print(f"--from-scratch set; ignoring --resume {args.resume}", flush=True)
    if resume_path and resume_path.exists():
        resume_s = str(resume_path).replace("\\", "/").lower()
        if args.mixer == "ttt_linear" and "baseline-30m" in resume_s:
            raise RuntimeError(
                f"Refusing to resume {resume_path} into TTT-Linear. "
                "Phase 1 trains from scratch. Attention weights stay in checkpoints/baseline-30m."
            )
        blob = load_ckpt(resume_path, map_location=device)
        saved_cfg = blob["config"]
        if getattr(saved_cfg, "mixer", model_cfg.mixer) != model_cfg.mixer:
            raise RuntimeError(
                f"Checkpoint mixer={getattr(saved_cfg, 'mixer', '?')} cannot resume into mixer={model_cfg.mixer}. "
                "Attention weights go in checkpoints/baseline-30m; TTT-Linear uses checkpoints/ttt-linear-30m. "
                "Phase 1: pass --from-scratch (do not load attn last.pt)."
            )
        if (
            saved_cfg.hidden_size != model_cfg.hidden_size
            or saved_cfg.num_layers != model_cfg.num_layers
            or saved_cfg.vocab_size != model_cfg.vocab_size
        ):
            raise RuntimeError(
                f"Checkpoint {resume_path} is a different model size. "
                "30M training writes to checkpoints/baseline-30m; do not resume the old 17M run."
            )
        model.load_state_dict(blob["model"])
        try:
            optimizer.load_state_dict(blob["optimizer"])
        except (ValueError, KeyError) as exc:
            print(f"Optimizer state not reused ({exc}); continuing with fresh AdamW moments.", flush=True)
        start_step = int(blob["step"]) + 1
        start_epoch = int(blob.get("epoch", 0))
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0.0) > 0.0:
                group["weight_decay"] = train_cfg.weight_decay
        print(
            f"Resumed {resume_path} at step {start_step} epoch={start_epoch} "
            f"weight_decay={train_cfg.weight_decay} dropout={train_cfg.dropout} "
            f"betas={train_cfg.betas}"
        )

    val_loader = None
    train_eval_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    if args.val_data.exists():
        val_set = EpochChunks(args.val_data, train_cfg.seq_len)
        val_loader = DataLoader(val_set, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "metrics.jsonl"
    run_path = args.out / "run.json"
    run_info = {
        "task": "causal_language_modeling",
        "not_classification": True,
        "mixer": args.mixer,
        "mode": args.mode,
        "params": model.param_count(),
        "device": str(device),
        "amp": amp,
        "optimizer": train_cfg.optimizer,
        "betas": list(train_cfg.betas),
        "weight_decay": train_cfg.weight_decay,
        "l2": "adamw_decoupled_weight_decay",
        "dropout": train_cfg.dropout,
        "min_lr_ratio": train_cfg.min_lr_ratio,
        "epochs": train_cfg.epochs,
        "seq_len": train_cfg.seq_len,
        "batch_size": train_cfg.batch_size,
        "grad_accum": train_cfg.grad_accum,
        "train_chunks": len(train_set),
        "opt_steps_per_epoch": opt_steps_per_epoch,
        "metrics": ["loss", "perplexity", "entropy", "unique_preds", "token_match"],
        "do_not_use": ["class accuracy", "SMOTE", "ROC-AUC"],
        "collapse_rule": "flag only if entropy<1.2 and (top1_mass>0.9 or token_match>0.85)",
        "resumed_from": str(resume_path) if resume_path else None,
        "start_step": start_step,
        "from_scratch": bool(args.from_scratch) or resume_path is None,
        "comparison_baseline": {
            "mixer": "attn",
            "val_ppl": 5.23,
            "note": "epoch-7 first-run snapshot; not epoch-19 last.pt (PPL 7.00)",
        },
    }
    if not (resume_path and run_path.exists()):
        run_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    if args.from_scratch and log_path.exists() and start_step == 0:
        bak = args.out / "metrics.prev.jsonl"
        log_path.replace(bak)
        print(f"Phase 1 from scratch: moved previous metrics to {bak}", flush=True)
    print(
        f"params={model.param_count():,} mixer={args.mixer} opt={train_cfg.optimizer} "
        f"wd={train_cfg.weight_decay} dropout={train_cfg.dropout} "
        f"device={device} amp={amp} epochs={train_cfg.epochs} out={args.out} "
        f"chunks/epoch={len(train_set)} opt_steps/epoch={opt_steps_per_epoch}"
    )
    if args.mixer == "ttt_linear" and start_step == 0:
        print(
            "Phase 1: TTT-Linear 30M from scratch. Report vs attention val PPL 5.23 "
            "(old snapshot), not vs epoch-19 last.pt.",
            flush=True,
        )
    if args.min_params and model.param_count() < args.min_params:
        raise RuntimeError(
            f"Model is {model.param_count()} params; expected at least {args.min_params} "
            f"for a matched 30M {args.mixer} comparison. Pass --min-params 0 only for debug runs."
        )

    model.train()
    global_step = start_step
    t0 = time.time()
    for epoch in range(start_epoch, train_cfg.epochs):
        loader = DataLoader(
            train_set,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in loader:
            batch = batch.to(device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp != "off"):
                if train_cfg.mode == "meta":
                    split = train_cfg.seq_len // 2
                    loss = meta_loss(
                        model,
                        batch[:, :split],
                        batch[:, split - 1 :],
                        lr=train_cfg.ttt_lr,
                        steps=train_cfg.ttt_steps,
                        parameter_prefixes=train_cfg.adapt_prefixes,
                    )
                else:
                    loss = model.loss(batch)
                loss = loss / train_cfg.grad_accum
            if amp == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_loss += loss.item() * train_cfg.grad_accum
            epoch_batches += 1
            accum += 1
            if accum < train_cfg.grad_accum:
                continue

            lr = cosine_lr(
                global_step,
                total_opt_steps,
                warmup,
                train_cfg.lr,
                min_ratio=train_cfg.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            if amp == "fp16":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0

            if global_step % 20 == 0:
                tokens = train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum * 20
                elapsed = max(time.time() - t0, 1e-6)
                print(
                    f"epoch={epoch:02d} step={global_step:06d} loss={loss.item() * train_cfg.grad_accum:.4f} "
                    f"lr={lr:.2e} tok/s={tokens / elapsed:.0f}"
                )
                t0 = time.time()
            if val_loader is not None and global_step > 0 and global_step % 250 == 0:
                report = log_fit(
                    model,
                    train_eval_loader,
                    val_loader,
                    device,
                    train_cfg.eval_batches,
                    log_path,
                    epoch,
                    global_step,
                )
                save_ckpt(args.out / "last.pt", model, optimizer, global_step, {"epoch": epoch, "val": report})
                if report["collapsed"]:
                    print("Training stopped: prediction distribution collapsed.", flush=True)
                    return
            global_step += 1

        if val_loader is not None:
            report = log_fit(
                model,
                train_eval_loader,
                val_loader,
                device,
                train_cfg.eval_batches,
                log_path,
                epoch,
                global_step,
            )
            save_ckpt(
                args.out / "last.pt",
                model,
                optimizer,
                global_step - 1,
                {"epoch": epoch + 1, "val": report},
            )
            if args.save_epoch_ckpts:
                save_ckpt(
                    args.out / f"epoch_{epoch:02d}.pt",
                    model,
                    optimizer,
                    global_step - 1,
                    {"epoch": epoch + 1, "val": report},
                )
            if report["collapsed"]:
                print("Training stopped: prediction distribution collapsed. This is not a 99% accuracy win.")
                return
        mean_train = epoch_loss / max(epoch_batches, 1)
        append_jsonl(log_path, {"epoch": epoch, "step": global_step, "split": "train", "loss": mean_train})
        print(f"epoch {epoch} done train_loss~{mean_train:.4f}")

    save_ckpt(args.out / "last.pt", model, optimizer, global_step - 1, {"epoch": train_cfg.epochs})
    print(f"Finished {train_cfg.epochs} epochs. Checkpoint {args.out / 'last.pt'}")


if __name__ == "__main__":
    main()
