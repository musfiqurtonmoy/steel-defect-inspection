"""Print a Markdown results table from one or more finished runs, for the README.

Usage:
    python scripts/results_table.py artifacts/resnet18_* artifacts/mobilenet_v3_small_*
"""

import json
import sys
from pathlib import Path

rows = ["| Model | Dataset version | Test accuracy | Test macro-F1 | ONNX size (MB) | CPU latency p50 / p95 (ms) | Train time (s) | GPU |",
        "|---|---|---|---|---|---|---|---|"]
for d in map(Path, sys.argv[1:]):
    run = json.loads((d / "run.json").read_text())
    test = json.loads((d / "test_metrics.json").read_text())
    exp = json.loads((d / "export_report.json").read_text())
    lat = exp["batch1_latency_ms_onnxruntime_cpu"]
    rows.append(
        f"| {run['config']['train']['arch']} | `{run['dataset_version']}` | "
        f"{test['accuracy']:.4f} | {test['macro_f1']:.4f} | {exp['onnx_file_mb']} | "
        f"{lat['p50']} / {lat['p95']} | {run['train_seconds']} | {run['gpu'] or 'CPU'} |"
    )
print("\n".join(rows))
