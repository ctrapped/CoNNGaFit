"""Accuracy tests on the demo data: examples/demo.py trains the network on every demo galaxy at its default settings,
and the training must converge and predict the unseen test galaxy's rotation curve accurately. The network committed
in examples/demo_model/ must still load and predict as accurately with the current code.

These train for about a minute, so they're marked slow: run them with `pytest`, skip them with `pytest -m "not slow"`.
"""

import h5py
import numpy as np
import pytest

import demo  # examples/demo.py, on pytest's pythonpath
from conngafit_datasets import radial_profile
from use_model import load_model, predict

pytestmark = pytest.mark.slow

INFERENCE_DEMO = demo.REPO_ROOT / "examples" / "demo_inference.py"
COMMITTED_MODEL = demo.REPO_ROOT / "examples" / "demo_model" / "demo"
TEST_IMAGES = sorted((demo.DATA_DIR / demo.SPLITS["testing"][0]).rglob("*.hdf5"))

# Limits for training and validation loss and rotation curve RMS error. Demo tests should be at least this accurate to pass.
MIN_TRAINING_LOSS_DROP = 20  # first epoch's training loss / last epoch's
MIN_VALIDATION_LOSS_DROP = 5  # first epoch's validation loss / best epoch's
MAX_RMS_KMS = 10  # rotation curve RMS error at each of the test galaxy's position angles


def true_curve(path):
    """The rotation curve the network is trained to predict for the image at path."""
    with h5py.File(path, "r") as hf:
        return radial_profile(hf, demo.DATASETS["target"], demo.DATASETS["radial_bins"])


def rms_errors(predictions):
    """RMS difference in km/s between each predicted and true rotation curve, keyed by image name."""
    truths = {path.stem: true_curve(path) for path in TEST_IMAGES}
    return {name: float(np.sqrt(np.mean((curve - truths[name]) ** 2))) for name, curve in predictions.items()}


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory, run_script):
    """Run examples/demo.py once at its default settings; returns its output folder."""
    output = tmp_path_factory.mktemp("demo")
    run_script(demo.REPO_ROOT / "examples" / "demo.py", "--output", output)
    return output


@pytest.fixture(scope="module")
def losses(demo_run):
    path = demo_run / "DiagnosticOutputs" / "diagnostics" / f"{demo.RUN_NAME}_LossesAndAccuracies.hdf5"
    with h5py.File(path, "r") as hf:
        return hf["training_losses"][()], hf["validation_losses"][()]


def test_demo_writes_figure(demo_run):
    assert (demo_run / "demo.png").is_file()


def test_training_loss_decreases(losses):
    training, _ = losses
    assert np.isfinite(training).all()  # no NaN/inf at any epoch
    assert training[0] / training[-1] >= MIN_TRAINING_LOSS_DROP  # and the network fits its training set


def test_validation_loss_decreases(losses):
    _, validation = losses
    assert np.isfinite(validation).all()  # no NaN/inf at any epoch
    assert validation[0] / validation.min() >= MIN_VALIDATION_LOSS_DROP  # and it generalizes to an unseen galaxy


def test_trained_network_predicts_test_galaxy(demo_run):
    model, metadata = load_model(str(demo_run / "NetworkWeights" / demo.RUN_NAME))
    errors = rms_errors(dict(predict(model, metadata, [str(p) for p in TEST_IMAGES])))
    assert max(errors.values()) < MAX_RMS_KMS, errors


def test_committed_network_predicts_test_galaxy(tmp_path, run_script):
    run_script(INFERENCE_DEMO, "--model-path", COMMITTED_MODEL, "--output", tmp_path)
    predictions = {}
    for path in TEST_IMAGES:
        with h5py.File(tmp_path / f"{path.stem}_radialPlot.hdf5", "r") as hf:
            predictions[path.stem] = hf["prediction"][()]
    errors = rms_errors(predictions)
    assert max(errors.values()) < MAX_RMS_KMS, errors
