# Steel surface defect inspection: from raw images to a production API

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

An end-to-end machine vision pipeline for industrial surface inspection. It takes raw
camera images, checks and versions the data, trains and evaluates a PyTorch classifier,
exports it to ONNX, and serves it through a containerised FastAPI service that logs every
prediction to SQL for monitoring.

The aim is the full engineering path, not only a model: reproducible data, traceable
results, a fast inference artefact, and a service an automation line could call.

```
raw images ──► ingest + quality checks ──► SQL manifest (versioned split)
                                               │
                         PyTorch training ◄────┘   (AMP on GPU, early stopping)
                               │
                 test evaluation + error analysis
                               │
                 ONNX export ──► parity check ──► CPU latency benchmark
                               │
          FastAPI + ONNX Runtime in Docker ──► SQL prediction log ──► /metrics
```

## Dataset

NEU-CLS (Northeastern University surface defect database): 1,800 grayscale images of
hot-rolled steel strip, 300 per class, 200 x 200 px. Six classes: crazing, inclusion,
patches, pitted surface, rolled-in scale, scratches.
Source: [NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm).
Citation: K. Song and Y. Yan, "A noise robust method based on completed local binary
patterns for hot-rolled steel strip surface defects", *Applied Surface Science*, 2013.

NEU has no official test split, so published numbers are hard to compare. This project
builds its own split, stratified and keyed on image content hash, and records it as a
dataset version.

## What each stage does

| Stage | File | Engineering point |
|---|---|---|
| Ingest and validate | `data/ingest.py` | Decode check, minimum size, exact duplicates (SHA-256), same image under two labels. Rejections are kept in the manifest with a reason, not silently dropped. |
| Version and split | `data/ingest.py`, `db.py` | Deterministic stratified split. Version id = hash of (image, label, split). A duplicate can never sit in two splits. SQLite by default; any SQLAlchemy URL (PostgreSQL) works. |
| Preprocess | `data/preprocess.py` | One NumPy/PIL function used by training and serving, so there is no train/serve skew. |
| Train | `train.py` | ResNet-18 (accuracy) or MobileNetV3-Small (edge). AdamW, cosine schedule, mixed precision on GPU, early stopping on val macro-F1. Each run saves dataset version, git commit, config and history. |
| Evaluate | `evaluate.py` | Held-out test accuracy, macro-F1, per-class precision/recall, confusion matrix, and `errors.csv` listing every misclassified image for error analysis. |
| Export | `export.py` | ONNX export with dynamic batch, numerical parity check against PyTorch, batch-1 CPU latency (p50/p95) for PyTorch and ONNX Runtime. |
| Serve | `api/app.py` | FastAPI: `/predict`, `/health`, `/predictions`, `/metrics`. Rejects bad uploads, flags low-confidence results for human review, logs each request (hash, label, confidence, latency, model version). |
| Ship | `Dockerfile`, `docker-compose.yml` | Serving image carries ONNX Runtime only, no PyTorch. Non-root user, health check, persistent prediction volume. |
| Test | `tests/`, `.github/workflows/ci.yml` | Synthetic data with planted faults, then a full ingest, train, export and API smoke test on every push. |

## Quick start

```bash
# 1. Install (GPU users: install the CUDA build of torch first, from pytorch.org)
pip install -e ".[train,dev]"
pytest -q                                   # runs on synthetic data, about 10 s

# 2. Data: download NEU-CLS, then arrange it into data/raw/<class>/
python scripts/prepare_neu.py --src ~/Downloads/NEU-CLS --dst data/raw
make ingest                                 # quality report -> reports/

# 3. Train, evaluate, export
make train                                  # configs/default.yaml (ResNet-18)
make evaluate export
python -m defectinspect.train --config configs/edge.yaml   # MobileNetV3 for comparison

# 4. Serve
make serve                                  # http://localhost:8000/docs
make docker                                 # same service in a container
curl -F "file=@data/raw/scratches/<some_image>.bmp" http://localhost:8000/predict
```

## Results

Filled from `scripts/results_table.py` after training on NEU-CLS.

<!-- RESULTS -->
_Run `python scripts/results_table.py artifacts/*/` and paste the table here._
<!-- /RESULTS -->

## Design choices

- **Classification first.** NEU-CLS gives one defect per image, which suits a
  pass/fail or defect-type station. Localisation (NEU-DET with a YOLO-family detector)
  is the next step and reuses the same ingest, versioning and serving layers.
- **Two backbones.** ResNet-18 for a GPU server, MobileNetV3-Small for CPU or an edge
  box. The latency benchmark shows the trade-off on the target hardware.
- **ONNX Runtime in production.** Smaller image and the same artefact can move to
  TensorRT on an NVIDIA Jetson.
- **Low-confidence flag.** On a real line a wrong "pass" costs more than a manual check.
  `LOW_CONF` routes uncertain parts to an operator.

## Roadmap

- NEU-DET object detection with a YOLO-family model behind the same API
- TensorRT engine and Jetson benchmark
- MLflow experiment tracking and a model registry
- Drift alert on `/metrics` (class mix and confidence against the training baseline)

## Author

Md. Musfiqur Rahman, BSc Mechatronics Engineering, RUET.
musfiqurtonmoy@gmail.com
