"""Remove near-duplicate images so the same photo never lands in train AND test."""
from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from .config import DEDUP_HAMMING, IMG_EXT


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def dedupe(paths: list[Path], max_distance: int = DEDUP_HAMMING) -> tuple[list[Path], int]:
    """Keep one image per cluster of perceptually similar images."""
    kept: list[tuple[Path, imagehash.ImageHash]] = []
    for p in paths:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            continue  # unreadable / corrupt file
        if all(h - other > max_distance for _, other in kept):
            kept.append((p, h))
    return [p for p, _ in kept], len(paths) - len(kept)
