"""Load last.pt or an unzipped PyTorch checkpoint folder (Windows-safe)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch


def resolve_ckpt(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if (path / "data.pkl").is_file():
        return path
    nested = path / "last"
    if (nested / "data.pkl").is_file():
        return nested
    pt = path / "last.pt"
    if pt.is_file():
        return pt
    raise FileNotFoundError(
        f"{path} is a folder but has no data.pkl / last.pt. "
        "If you extracted last.pt, pass the inner folder that contains data.pkl."
    )


def pack_unpacked_ckpt(src: Path, dest: Path) -> Path:
    """Zip an extracted torch.save directory into a loadable last.pt."""
    src = resolve_ckpt(src)
    if src.is_file():
        return src
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prefix = dest.stem
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        for file in src.rglob("*"):
            if not file.is_file():
                continue
            rel = file.relative_to(src)
            if any(part.startswith(".") for part in rel.parts):
                continue
            zf.write(file, f"{prefix}/{rel.as_posix()}")
    return dest


def load_ckpt(path: Path, map_location):
    src = resolve_ckpt(path)
    if src.is_dir():
        packed = src.parent / "last.pt"
        if not packed.is_file():
            print(f"Packing unzipped checkpoint {src} -> {packed}", flush=True)
            pack_unpacked_ckpt(src, packed)
        src = packed
    return torch.load(src, map_location=map_location, weights_only=False)


def find_run_ckpt(ckpt_dir: Path) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    pt = ckpt_dir / "last.pt"
    if pt.is_file():
        return pt
    folder = ckpt_dir / "last"
    if folder.exists():
        try:
            resolved = resolve_ckpt(folder)
        except FileNotFoundError:
            return None
        if resolved.is_file():
            return resolved
        packed = ckpt_dir / "last.pt"
        pack_unpacked_ckpt(resolved, packed)
        return packed
    return None
