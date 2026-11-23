"""Smoke test of the full path: ingest -> train -> export -> serve."""

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from defectinspect.config import Config
from defectinspect.data.ingest import build_manifest, write_manifest
from defectinspect.data.preprocess import preprocess


@pytest.fixture(scope="module")
def trained_run(synthetic_dir, tmp_path_factory):
    from defectinspect.export import export
    from defectinspect.train import train

    tmp = tmp_path_factory.mktemp("run")
    cfg = Config()
    cfg.data.db_url = f"sqlite:///{tmp}/manifest.db"
    cfg.data.image_size = 64
    cfg.train.arch = "mobilenet_v3_small"
    cfg.train.pretrained = False
    cfg.train.epochs = 1
    cfg.train.batch_size = 16
    cfg.train.num_workers = 0
    cfg.train.out_dir = str(tmp / "artifacts")
    vid, classes, records, report = build_manifest(synthetic_dir, 32, 0.15, 0.15, 42)
    write_manifest(cfg.data.db_url, synthetic_dir, vid, classes, records, report)
    run = train(cfg)
    rep = export(run, n_bench=5)
    return run, rep


def test_preprocess_shape_and_dtype():
    x = preprocess(Image.new("L", (200, 200), 128), 224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32


def test_train_writes_traceable_run(trained_run):
    run, _ = trained_run
    meta = json.loads((run / "run.json").read_text())
    assert (run / "model.pt").exists()
    assert len(meta["dataset_version"]) == 12
    assert meta["n_train"] > 0 and meta["n_val"] > 0


def test_onnx_parity(trained_run):
    _, rep = trained_run
    assert rep["parity_argmax_match"]
    assert rep["parity_max_abs_logit_diff"] < 1e-3


def test_api_end_to_end(trained_run, tmp_path, monkeypatch, synthetic_dir):
    run, _ = trained_run
    monkeypatch.setenv("MODEL_DIR", str(run))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/pred.db")
    from defectinspect.api.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        img = next((synthetic_dir / "scratches").glob("scratches_*.bmp"))
        r = client.post("/predict", files={"file": (img.name, img.read_bytes(), "image/bmp")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in body["probabilities"]
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-3

        bad = client.post("/predict", files={"file": ("x.png", b"junk", "image/png")})
        assert bad.status_code == 400

        buf = io.BytesIO()
        Image.new("RGB", (120, 90)).save(buf, "PNG")
        assert client.post("/predict", files={"file": ("c.png", buf.getvalue())}).status_code == 200

        assert len(client.get("/predictions").json()) == 2
        m = client.get("/metrics").json()
        assert m["n_predictions"] == 2 and m["latency_ms_p95"] is not None
