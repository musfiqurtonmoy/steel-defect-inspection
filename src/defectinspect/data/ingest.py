"""Ingest raw images, run data-quality checks, split, and write a versioned manifest.

Expected raw layout (one folder per class, any nesting below is ignored):

    data/raw/<class_name>/*.{bmp,jpg,jpeg,png}

Checks per file: decodable, minimum side length, exact duplicate (SHA-256),
and the same image filed under two labels (label conflict).
The split is stratified and keyed on the content hash, so it is deterministic
and a duplicate can never land in two splits.

Usage:
    python -m defectinspect.data.ingest --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import delete, select

from defectinspect.config import load_config
from defectinspect.db import DatasetVersion, ImageRecord, make_session

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(raw_dir: Path) -> list[tuple[Path, str]]:
    files = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for p in sorted(class_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                files.append((p, class_dir.name))
    return files


def check_image(path: Path, min_side: int) -> tuple[dict, str | None]:
    try:
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:
            im.load()  # full decode
            w, h, mode = im.width, im.height, im.mode
    except (UnidentifiedImageError, OSError, SyntaxError):
        return {}, "corrupt"
    if min(w, h) < min_side:
        return {"width": w, "height": h, "mode": mode}, "too_small"
    return {"width": w, "height": h, "mode": mode}, None


def stratified_split(
    items: list[tuple[str, str]], val_frac: float, test_frac: float, seed: int
) -> dict[str, str]:
    """items: (sha256, label). Returns sha256 -> split."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for sha, label in items:
        by_label[label].append(sha)
    rng = random.Random(seed)
    out = {}
    for label in sorted(by_label):
        shas = sorted(by_label[label])
        rng.shuffle(shas)
        n = len(shas)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        for i, sha in enumerate(shas):
            out[sha] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def build_manifest(raw_dir: Path, min_side: int, val_frac: float, test_frac: float, seed: int):
    files = scan(raw_dir)
    if not files:
        raise SystemExit(f"No images found under {raw_dir}/<class>/")

    records = []
    first_seen: dict[str, tuple[str, int]] = {}  # sha -> (label, record index)
    for path, label in files:
        meta, reason = check_image(path, min_side)
        sha = sha256_file(path)
        if reason is None and sha in first_seen:
            prev_label, prev_idx = first_seen[sha]
            if prev_label != label:
                reason = "label_conflict"
                records[prev_idx]["valid"] = False
                records[prev_idx]["reject_reason"] = "label_conflict"
            else:
                reason = "duplicate"
        elif reason is None:
            first_seen[sha] = (label, len(records))
        records.append(
            {
                "path": str(path),
                "sha256": sha,
                "label": label,
                **meta,
                "valid": reason is None,
                "reject_reason": reason,
            }
        )

    valid = [(r["sha256"], r["label"]) for r in records if r["valid"]]
    splits = stratified_split(valid, val_frac, test_frac, seed)
    for r in records:
        r["split"] = splits.get(r["sha256"]) if r["valid"] else None

    # Version id = hash of the sorted (sha, label, split) triples.
    key = "\n".join(sorted(f"{s}|{l}|{splits[s]}" for s, l in valid))
    version_id = hashlib.sha256(key.encode()).hexdigest()[:12]

    classes = sorted({l for _, l in valid})
    report = {
        "version_id": version_id,
        "files_scanned": len(records),
        "valid": len(valid),
        "rejected": Counter(r["reject_reason"] for r in records if not r["valid"]),
        "per_class": {
            c: Counter(r["split"] for r in records if r["valid"] and r["label"] == c)
            for c in classes
        },
        "modes": Counter(r.get("mode") for r in records if r["valid"]),
        "sizes": Counter(f'{r.get("width")}x{r.get("height")}' for r in records if r["valid"]),
    }
    # Leakage guard: each hash appears in exactly one split.
    assert len({s for s, _ in valid}) == len(valid), "duplicate hash survived dedup"
    return version_id, classes, records, json.loads(json.dumps(report))


def write_manifest(db_url: str, raw_dir: Path, version_id, classes, records, report):
    Session = make_session(db_url)
    with Session() as s:
        if s.get(DatasetVersion, version_id) is not None:
            s.execute(delete(ImageRecord).where(ImageRecord.version_id == version_id))
            s.execute(delete(DatasetVersion).where(DatasetVersion.id == version_id))
        s.add(
            DatasetVersion(
                id=version_id,
                raw_dir=str(raw_dir),
                n_images=report["valid"],
                n_rejected=report["files_scanned"] - report["valid"],
                classes=",".join(classes),
                report_json=json.dumps(report),
            )
        )
        s.add_all(ImageRecord(version_id=version_id, **r) for r in records)
        s.commit()


def load_split(db_url: str, version_id: str, split: str) -> list[tuple[str, str]]:
    Session = make_session(db_url)
    with Session() as s:
        rows = s.execute(
            select(ImageRecord.path, ImageRecord.label)
            .where(ImageRecord.version_id == version_id)
            .where(ImageRecord.split == split)
            .where(ImageRecord.valid.is_(True))
            .order_by(ImageRecord.sha256)
        ).all()
    return [(p, l) for p, l in rows]


def get_version(db_url: str, version_id: str | None = None) -> DatasetVersion:
    Session = make_session(db_url)
    with Session() as s:
        if version_id:
            v = s.get(DatasetVersion, version_id)
        else:
            v = s.execute(
                select(DatasetVersion).order_by(DatasetVersion.created_at.desc())
            ).scalars().first()
    if v is None:
        raise SystemExit("No dataset version found. Run ingest first.")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config).data
    raw = Path(cfg.raw_dir)
    vid, classes, records, report = build_manifest(
        raw, cfg.min_side, cfg.val_frac, cfg.test_frac, cfg.seed
    )
    write_manifest(cfg.db_url, raw, vid, classes, records, report)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / f"data_report_{vid}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nDataset version {vid} written to {cfg.db_url}")


if __name__ == "__main__":
    main()
