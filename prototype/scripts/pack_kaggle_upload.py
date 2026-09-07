"""Build colab_upload.zip for Phase 1 Kaggle: code + TinyStories npy, no checkpoints."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

SKIP_DIR_NAMES = {
    ".venv-data",
    ".venv",
    "__pycache__",
    ".hf-cache",
    "checkpoints",
    ".git",
    "kaggle_upload",
    "node_modules",
}

SKIP_SUFFIXES = {".pt", ".pyc", ".zip"}
SKIP_FILE_NAMES = {"train.jsonl", "validation.jsonl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="prototype/ directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Zip path. Default: prototype/kaggle_upload/ttt-phase1-src.zip",
    )
    return parser.parse_args()


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    return False


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = (args.out or (root / "kaggle_upload" / "ttt-phase1-src.zip")).resolve()
    train_npy = root / "data" / "tinystories-v2" / "train.npy"
    val_npy = root / "data" / "tinystories-v2" / "validation.npy"
    tok = root / "data" / "tokenizer" / "tokenizer.json"
    train_py = root / "scripts" / "train.py"
    missing = [p for p in (train_npy, val_npy, tok, train_py) if not p.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(str(p) for p in missing))

    include_dirs = [
        root / "tttmem",
        root / "scripts",
        root / "notebooks",
        root / "data" / "tinystories-v2",
        root / "data" / "tokenizer",
    ]
    extra_files = [root / "README.md", root / "RESULTS.md", root / "app.py"]
    files: list[Path] = []
    for folder in include_dirs:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or should_skip(path, root):
                continue
            if path.parent.name == "notebooks" and path.name != "kaggle_train_ttt_30m.ipynb":
                continue
            files.append(path)
    for path in extra_files:
        if path.is_file():
            files.append(path)

    out.parent.mkdir(parents=True, exist_ok=True)
    # Flatten to zip root. A top-level `prototype/` folder makes Kaggle fail with
    # "Directory already exists: prototype" if the dataset title is also prototype.
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arc = path.relative_to(root).as_posix()
            if arc.split("/")[0] in {"prototype", "src30m", "colab_upload"}:
                raise RuntimeError(f"refusing zip path {arc}")
            zf.write(path, arc)

    names = zf_namelist(out)
    assert any(n == "scripts/train.py" for n in names), "zip missing scripts/train.py"
    assert any(n == "data/tinystories-v2/train.npy" for n in names), "zip missing train.npy"
    assert not any(n.endswith(".pt") for n in names), "zip must not include checkpoints"
    assert not any(n.split("/")[0] in {"prototype", "src30m"} for n in names)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files)")
    print("Kaggle: New Dataset, title ttt-phase1-src (not 'prototype'). Upload only this zip.")


def zf_namelist(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


if __name__ == "__main__":
    main()
