"""Smoke tests: examples/demo.py trains on the committed demo data for two epochs and writes its figure, and
examples/demo_inference.py runs the committed demo network on the test galaxy."""

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DEMO = EXAMPLES / "demo.py"
INFERENCE_DEMO = EXAMPLES / "demo_inference.py"
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


def test_inference_demo_runs_committed_network(tmp_path):
    result = subprocess.run(
        [sys.executable, INFERENCE_DEMO, "--output", tmp_path],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]  # demo exits cleanly
    # and writes a plot and an hdf5 of the predicted rotation curve for each of the test galaxy's 4 views
    assert len(list(tmp_path.glob("*_radialPlot.png"))) == len(list(tmp_path.glob("*_radialPlot.hdf5"))) == 4
