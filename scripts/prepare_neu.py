"""Arrange a downloaded copy of NEU-CLS into data/raw/<class>/.

Handles the layouts in circulation:
  1. Original NEU-CLS: one flat folder of files named Cr_1.bmp, In_1.bmp, ...
  2. Kaggle "NEU Metal Surface Defects Data": train/ valid/ test/ with class folders
  3. Files named by full class name, e.g. crazing_1.jpg, rolled-in scale (1).jpg

Any pre-made split is merged. ingest.py re-splits by content hash, so the split
is deterministic and duplicates cannot leak across splits.

Usage:
    python scripts/prepare_neu.py --src ~/Downloads/NEU-CLS --dst data/raw
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

CANON = {
    "cr": "crazing", "crazing": "crazing",
    "in": "inclusion", "inclusion": "inclusion",
    "pa": "patches", "patches": "patches",
    "ps": "pitted_surface", "pitted": "pitted_surface", "pitted_surface": "pitted_surface",
    "rs": "rolled-in_scale", "rolled": "rolled-in_scale", "rolled-in": "rolled-in_scale", "rolled-in_scale": "rolled-in_scale",
    "sc": "scratches", "scratches": "scratches",
}
EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def label_for(path: Path, src: Path) -> str | None:
    # Folder name first (layout 2), then the leading letters of the file name.
    for part in path.relative_to(src).parts[:-1][::-1]:
        if part.lower() in CANON:
            return CANON[part.lower()]
    prefix = re.match(r"[A-Za-z]*", path.stem).group(0).lower()
    return CANON.get(prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/raw")
    args = ap.parse_args()
    src, dst = Path(args.src).expanduser(), Path(args.dst)
    counts: dict[str, int] = {}
    skipped = 0
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        label = label_for(p, src)
        if label is None:
            skipped += 1
            continue
        out = dst / label
        out.mkdir(parents=True, exist_ok=True)
        # Prefix with the relative path so same-named files from different splits do not collide.
        name = "__".join(p.relative_to(src).parts)
        shutil.copy2(p, out / name)
        counts[label] = counts.get(label, 0) + 1
    for k in sorted(counts):
        print(f"{k:18s} {counts[k]}")
    print(f"total {sum(counts.values())}, skipped (no label) {skipped}")


if __name__ == "__main__":
    main()