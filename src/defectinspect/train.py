"""Train a defect classifier from a versioned manifest.

Every run records the dataset version, config, git commit and metrics, so any
result can be traced back to the exact data and code that produced it.

Usage:
    python -m defectinspect.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from defectinspect.config import load_config
from defectinspect.data.dataset import Augment, ManifestDataset
from defectinspect.data.ingest import get_version, load_split
from defectinspect.evaluate import evaluate_loader
from defectinspect.models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train(cfg, version_id: str | None = None) -> Path:
    tc, dc = cfg.train, cfg.data
    set_seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    version = get_version(dc.db_url, version_id)
    classes = version.classes.split(",")
    train_items = load_split(dc.db_url, version.id, "train")
    val_items = load_split(dc.db_url, version.id, "val")

    train_ds = ManifestDataset(train_items, classes, dc.image_size, Augment())
    val_ds = ManifestDataset(val_items, classes, dc.image_size)
    g = torch.Generator().manual_seed(tc.seed)
    train_dl = DataLoader(
        train_ds, tc.batch_size, shuffle=True, num_workers=tc.num_workers,
        generator=g, pin_memory=device.type == "cuda",
    )
    val_dl = DataLoader(val_ds, tc.batch_size, num_workers=tc.num_workers)

    model = build_model(tc.arch, len(classes), tc.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tc.epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"  # mixed precision fits a 4 GB card
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = Path(tc.out_dir) / f"{tc.arch}_{version.id}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    t0 = time.time()
    for epoch in range(tc.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()

        val = evaluate_loader(model, val_dl, classes, device)
        row = {"epoch": epoch + 1, "train_loss": total / max(n, 1),
               "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"]}
        history.append(row)
        print(json.dumps(row))

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["macro_f1"], epoch + 1, 0
            torch.save({"state_dict": model.state_dict(), "arch": tc.arch,
                        "classes": classes, "image_size": dc.image_size}, run_dir / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

    meta = {
        "dataset_version": version.id,
        "classes": classes,
        "git_commit": git_commit(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_seconds": round(time.time() - t0, 1),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "config": cfg.to_dict(),
        "history": history,
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest val macro-F1 {best_f1:.4f} at epoch {best_epoch}. Run saved to {run_dir}")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--version", default=None, help="dataset version id (default: latest)")
    args = ap.parse_args(argv)
    train(load_config(args.config), args.version)


if __name__ == "__main__":
    main()
