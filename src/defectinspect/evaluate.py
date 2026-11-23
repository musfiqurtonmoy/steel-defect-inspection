"""Evaluate a trained model on the held-out test split and write an error analysis.

Outputs in the run folder:
    test_metrics.json   accuracy, macro-F1, per-class precision/recall/F1
    confusion_matrix.png
    errors.csv          every misclassified test image, with confidence

Usage:
    python -m defectinspect.evaluate --run artifacts/<run_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from defectinspect.config import load_config


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy()
        probs.append(p)
        ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


def evaluate_loader(model, loader, classes, device) -> dict:
    probs, y = predict_loader(model, loader, device)
    pred = probs.argmax(1)
    return {
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(classes)),
                                   zero_division=0)) if len(y) else 0.0,
    }


def plot_confusion(cm, classes, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    from defectinspect.data.dataset import ManifestDataset
    from defectinspect.data.ingest import load_split
    from defectinspect.models import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    run = Path(args.run)
    cfg = load_config(args.config)
    meta = json.loads((run / "run.json").read_text())
    ckpt = torch.load(run / "model.pt", map_location="cpu")
    classes = ckpt["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    items = load_split(cfg.data.db_url, meta["dataset_version"], "test")
    ds = ManifestDataset(items, classes, ckpt["image_size"])
    probs, y = predict_loader(model, DataLoader(ds, 64), device)
    pred = probs.argmax(1)

    report = classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    metrics = {
        "dataset_version": meta["dataset_version"],
        "n_test": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {c: {k: round(float(report[c][k]), 4)
                          for k in ("precision", "recall", "f1-score")} for c in classes},
        "confusion_matrix": cm.tolist(),
    }
    (run / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, classes, run / "confusion_matrix.png")

    with (run / "errors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence"])
        for (path, _), t, p, pr in zip(items, y, pred, probs):
            if t != p:
                w.writerow([path, classes[t], classes[p], f"{pr[p]:.3f}"])

    print(json.dumps({k: metrics[k] for k in ("n_test", "accuracy", "macro_f1")}, indent=2))
    print(f"Errors: {int((pred != y).sum())} written to {run / 'errors.csv'}")


if __name__ == "__main__":
    main()
