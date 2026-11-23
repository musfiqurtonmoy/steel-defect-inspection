"""FastAPI inference service.

Serves the ONNX model with ONNX Runtime (no PyTorch in the container) and logs
every prediction to SQL for monitoring.

Environment variables:
    MODEL_DIR   folder holding model.onnx and labels.json   (default: model)
    DB_URL      SQLAlchemy URL for the prediction log        (default: sqlite:///data/predictions.db)
    LOW_CONF    confidence below which a result is flagged for human review (default: 0.6)

Run locally:
    uvicorn defectinspect.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from sqlalchemy import func, select

from defectinspect.data.preprocess import preprocess, softmax
from defectinspect.db import Prediction, make_session

MAX_BYTES = 10 * 1024 * 1024


class Predictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "labels.json").read_text())
        self.classes = meta["classes"]
        self.image_size = meta["image_size"]
        self.version = meta.get("model_version", "unknown")
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=ort.get_available_providers()
        )

    def __call__(self, img: Image.Image) -> tuple[str, float, dict[str, float]]:
        x = preprocess(img, self.image_size)[None]
        probs = softmax(self.session.run(None, {"input": x})[0])[0]
        i = int(probs.argmax())
        return self.classes[i], float(probs[i]), {
            c: round(float(p), 4) for c, p in zip(self.classes, probs)
        }


class PredictionOut(BaseModel):
    label: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor(os.getenv("MODEL_DIR", "model"))
    state["Session"] = make_session(os.getenv("DB_URL", "sqlite:///data/predictions.db"))
    state["low_conf"] = float(os.getenv("LOW_CONF", "0.6"))
    yield
    state.clear()


app = FastAPI(title="Steel surface defect inspection", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    p = state["predictor"]
    return {"status": "ok", "model_version": p.version, "classes": p.classes}


@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a readable image")

    p = state["predictor"]
    t = time.perf_counter()
    label, conf, probs = p(img)
    ms = (time.perf_counter() - t) * 1000

    with state["Session"]() as s:
        s.add(Prediction(model_version=p.version, filename=file.filename,
                         sha256=hashlib.sha256(data).hexdigest(), label=label,
                         confidence=conf, latency_ms=ms))
        s.commit()

    return PredictionOut(label=label, confidence=round(conf, 4),
                         needs_review=conf < state["low_conf"], probabilities=probs,
                         latency_ms=round(ms, 2), model_version=p.version)


@app.get("/predictions")
def recent(limit: int = Query(20, ge=1, le=500)):
    with state["Session"]() as s:
        rows = s.execute(
            select(Prediction).order_by(Prediction.id.desc()).limit(limit)
        ).scalars().all()
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "filename": r.filename,
             "label": r.label, "confidence": round(r.confidence, 4),
             "latency_ms": round(r.latency_ms, 2), "model_version": r.model_version}
            for r in rows]


@app.get("/metrics")
def metrics():
    """Simple production monitoring: volume, class mix, confidence and latency."""
    low = state["low_conf"]
    with state["Session"]() as s:
        n = s.scalar(select(func.count(Prediction.id))) or 0
        by_label = dict(s.execute(
            select(Prediction.label, func.count()).group_by(Prediction.label)).all())
        avg_conf = s.scalar(select(func.avg(Prediction.confidence)))
        low_n = s.scalar(select(func.count()).where(Prediction.confidence < low)) or 0
        lat = [r for (r,) in s.execute(select(Prediction.latency_ms)).all()]
    return {
        "n_predictions": n,
        "by_label": by_label,
        "mean_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "low_confidence_rate": round(low_n / n, 4) if n else None,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2) if lat else None,
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 2) if lat else None,
    }
