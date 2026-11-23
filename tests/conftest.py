import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data") / "raw"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
         "--out", str(out), "--n", "12"],
        check=True,
    )
    return out
