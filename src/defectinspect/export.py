"""Export a trained checkpoint to ONNX, check numerical parity, and benchmark latency.

Writes to <run>/:
    model.onnx, labels.json, export_report.json

Usage:
    python -m defectinspect.export --run artifacts/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from defectinspect.models import build_model


def latency_ms(fn, n_warmup=10, n=100) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2)}


def export(run: Path, n_bench: int = 100) -> dict:
    ckpt = torch.load(run / "model.pt", map_location="cpu")
    classes, size = ckpt["classes"], ckpt["image_size"]
    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    onnx_path = run / "model.onnx"
    torch.onnx.export(
        model, dummy, onnx_path, input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
        dynamo=False,
    )
    (run / "labels.json").write_text(json.dumps(
        {"classes": classes, "image_size": size, "arch": ckpt["arch"],
         "model_version": run.name}, indent=2))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(8, 3, size, size)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"input": x.numpy()})[0]
    max_abs = float(np.abs(ref - out).max())

    x1 = dummy.numpy()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        t_torch = latency_ms(lambda: model(dummy), n=n_bench)
    t_onnx = latency_ms(lambda: sess.run(None, {"input": x1}), n=n_bench)

    report = {
        "onnx_file_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "parity_max_abs_logit_diff": max_abs,
        "parity_argmax_match": bool((ref.argmax(1) == out.argmax(1)).all()),
        "cpu": platform.processor() or platform.machine(),
        "batch1_latency_ms_pytorch_cpu": t_torch,
        "batch1_latency_ms_onnxruntime_cpu": t_onnx,
    }
    (run / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-bench", type=int, default=100)
    args = ap.parse_args(argv)
    print(json.dumps(export(Path(args.run), args.n_bench), indent=2))


if __name__ == "__main__":
    main()
