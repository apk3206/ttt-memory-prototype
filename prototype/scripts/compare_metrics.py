"""Print latest val rows from one or two training logs (attention vs TTT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_val(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == "val":
            rows.append(row)
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {"name": name, "n": 0}
    best = min(rows, key=lambda r: r["loss"])
    last = rows[-1]
    return {
        "name": name,
        "n": len(rows),
        "last_step": last["step"],
        "last_epoch": last["epoch"],
        "last_loss": last["loss"],
        "last_ppl": last["ppl"],
        "last_entropy": last["entropy"],
        "last_match": last["token_match"],
        "collapsed": last.get("collapsed", 0),
        "best_loss": best["loss"],
        "best_ppl": best["ppl"],
        "best_step": best["step"],
    }


def fmt(row: dict) -> str:
    if row.get("n", 0) == 0:
        return f"{row['name']}: no val rows yet"
    return (
        f"{row['name']}: last epoch={row['last_epoch']} step={row['last_step']} "
        f"loss={row['last_loss']:.3f} ppl={row['last_ppl']:.2f} entropy={row['last_entropy']:.2f} "
        f"token_match={row['last_match']:.3f} collapsed={int(row['collapsed'])} "
        f"| best ppl={row['best_ppl']:.2f} @ step {row['best_step']}"
    )


LOCKED_ATTN_PPL = 5.23  # epoch-7 first-run snapshot; do not use epoch-19 last.pt (PPL 7.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attn",
        type=Path,
        default=Path("checkpoints/baseline-30m/metrics.jsonl"),
    )
    parser.add_argument(
        "--ttt",
        type=Path,
        default=Path("checkpoints/ttt-linear-30m/metrics.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attn = summarize("attn", load_val(args.attn))
    ttt = summarize("ttt_linear", load_val(args.ttt))
    print(fmt(attn))
    print(fmt(ttt))
    print(f"locked attn report target: val PPL {LOCKED_ATTN_PPL:.2f} (epoch-7 snapshot, not continue-fit last.pt)")
    if ttt.get("n"):
        print(f"ttt last ppl vs locked 5.23: {ttt['last_ppl'] - LOCKED_ATTN_PPL:+.2f}")
        print(f"ttt best ppl vs locked 5.23: {ttt['best_ppl'] - LOCKED_ATTN_PPL:+.2f}")
    if attn.get("n") and ttt.get("n"):
        delta = ttt["last_ppl"] - attn["last_ppl"]
        print(f"last ppl delta vs metrics.jsonl attn (ttt - attn) = {delta:+.2f}  [ignore if attn log is the continue-fit]")


if __name__ == "__main__":
    main()
