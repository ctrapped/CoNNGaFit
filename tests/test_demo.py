"""Smoke test: examples/demo.py trains on the committed demo data for two epochs and writes its figure."""

import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
TIMEOUT_S = 600


def test_demo_trains_and_plots(tmp_path):
    result = subprocess.run(
        [sys.executable, DEMO, "--epochs", "2", "--output", tmp_path],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]  # demo exits cleanly
    assert (tmp_path / "demo.png").is_file()  # and writes its figure
    assert (tmp_path / "NetworkWeights" / "demo.pt").is_file()  # from a checkpoint it trained
