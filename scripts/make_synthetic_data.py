"""Create a small synthetic surface-defect dataset for tests and CI.

Each class has a distinct texture so a tiny model can learn it in one epoch.
It also plants one corrupt file, one tiny file and one exact duplicate,
so the data-quality checks have something to catch.

This is NOT the real dataset. Real results come from NEU-CLS (see README).

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic --n 40
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def make(cls: str, rng: np.random.Generator, size: int = 200) -> Image.Image:
    base = rng.normal(128, 12, (size, size)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, "L")
    d = ImageDraw.Draw(img)
    if cls == "scratches":
        for _ in range(3):
            x = int(rng.integers(0, size))
            d.line([(x, 0), (x + int(rng.integers(-20, 20)), size)], fill=230, width=2)
    elif cls == "pitted_surface":
        for _ in range(40):
            x, y = rng.integers(0, size, 2)
            d.ellipse([x, y, x + 4, y + 4], fill=40)
    elif cls == "patches":
        for _ in range(4):
            x, y = rng.integers(0, size - 50, 2)
            d.rectangle([x, y, x + 50, y + 40], fill=60)
    elif cls == "inclusion":
        for _ in range(6):
            x, y = rng.integers(0, size - 30, 2)
            d.line([(x, y), (x + 30, y + 4)], fill=20, width=4)
    elif cls == "crazing":
        for _ in range(25):
            x, y = rng.integers(0, size, 2)
            d.line([(x, y), (x + int(rng.integers(-15, 15)), y + int(rng.integers(-15, 15)))],
                   fill=90, width=1)
    elif cls == "rolled-in_scale":
        for _ in range(12):
            x, y = rng.integers(0, size - 20, 2)
            d.polygon([(x, y), (x + 18, y + 5), (x + 8, y + 16)], fill=200)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=40, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(args.seed)
    for c in CLASSES:
        (out / c).mkdir(parents=True)
        for i in range(args.n):
            make(c, rng).save(out / c / f"{c}_{i:03d}.bmp")
    # Planted quality problems.
    (out / "scratches" / "corrupt.bmp").write_bytes(b"not an image")
    Image.new("L", (10, 10)).save(out / "patches" / "tiny.bmp")
    shutil.copy(out / "crazing" / "crazing_000.bmp", out / "crazing" / "crazing_dup.bmp")
    print(f"Wrote {len(CLASSES) * args.n} images + 3 planted faults to {out}")


if __name__ == "__main__":
    main()
