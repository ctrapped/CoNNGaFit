"""Accuracy tests on the demo data: examples/demo.py trains the network on every demo galaxy at its default settings,
and the training must converge and predict the unseen test galaxy's rotation curve accurately, with calibrated 1-sigma
uncertainties. The network committed in examples/demo_model/ must still load and predict as accurately with the current
code.

These train for a few minutes, so they're marked slow: run them with `pytest`, skip them with `pytest -m "not slow"`.
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
MIN_TRAINING_LOSS_DROP = 20  # first epoch's training MSE of the mean / last epoch's
MIN_VALIDATION_LOSS_DROP = 5  # first epoch's validation MSE of the mean / best epoch's
MAX_RMS_KMS = 10  # rotation curve RMS error at each of the test galaxy's position angles

# Limits for the predicted 1-sigma uncertainties of the test galaxy's rotation curve
MIN_WITHIN_1_SIGMA = 0.5  # fraction of rings whose prediction is within 1 sigma of the truth; 0.68 if calibrated
MIN_WITHIN_2_SIGMA = 0.85  # fraction within 2 sigma; 0.95 if calibrated
MAX_MEDIAN_SIGMA_KMS = 20  # so the uncertainties can't pass by being uselessly wide


def true_curve(path):
    """The rotation curve the network is trained to predict for the image at path."""
    with h5py.File(path, "r") as hf:
        return radial_profile(hf, demo.DATASETS["target"], demo.DATASETS["radial_bins"])


def rms_errors(predictions):
    """RMS difference in km/s between each predicted and true rotation curve, keyed by image name."""
    truths = {path.stem: true_curve(path) for path in TEST_IMAGES}
    return {name: float(np.sqrt(np.mean((curve - truths[name]) ** 2))) for name, curve in predictions.items()}


def check_calibration(predictions, uncertainties):
    """Assert the predicted 1-sigma uncertainties (keyed by image name, like the predicted curves) cover the true
    rotation curves about as often as they should, without being uselessly wide."""
    truths = {path.stem: true_curve(path) for path in TEST_IMAGES}
    sigmas = np.concatenate([uncertainties[name] for name in predictions])
    errors = np.concatenate([np.abs(predictions[name] - truths[name]) for name in predictions])
    assert np.mean(errors < sigmas) >= MIN_WITHIN_1_SIGMA, np.mean(errors < sigmas)  # not overconfident
    assert np.mean(errors < 2 * sigmas) >= MIN_WITHIN_2_SIGMA, np.mean(errors < 2 * sigmas)
    assert np.median(sigmas) <= MAX_MEDIAN_SIGMA_KMS, np.median(sigmas)  # and not uselessly wide


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory, run_script):
    """Run examples/demo.py once at its default settings; returns its output folder."""
    output = tmp_path_factory.mktemp("demo")
    run_script(demo.REPO_ROOT / "examples" / "demo.py", "--output", output)
    return output


@pytest.fixture(scope="module")
def losses(demo_run):
    """Each epoch's Gaussian NLL (the training objective) and MSE of the mean, keyed by their names in the hdf5."""
    path = demo_run / "DiagnosticOutputs" / "diagnostics" / f"{demo.RUN_NAME}_LossesAndAccuracies.hdf5"
    with h5py.File(path, "r") as hf:
        return {
            key: hf[key][()] for key in ("training_losses", "validation_losses", "training_mses", "validation_mses")
        }


@pytest.fixture(scope="module")
def trained_outputs(demo_run):
    """The newly trained network's predicted rotation curves and their 1-sigma uncertainties, keyed by image name."""
    model, metadata = load_model(str(demo_run / "NetworkWeights" / demo.RUN_NAME))
    predictions, uncertainties = {}, {}
    for name, curve, sigma in predict(model, metadata, [str(p) for p in TEST_IMAGES], return_std=True):
        predictions[name] = curve
        uncertainties[name] = sigma
    return predictions, uncertainties


@pytest.fixture(scope="module")
def committed_outputs(tmp_path_factory, run_script):
    """The committed network's predicted rotation curves and their 1-sigma uncertainties, keyed by image name, as
    written by examples/demo_inference.py."""
    output = tmp_path_factory.mktemp("committed")
    run_script(INFERENCE_DEMO, "--model-path", COMMITTED_MODEL, "--output", output)
    predictions, uncertainties = {}, {}
    for path in TEST_IMAGES:
        with h5py.File(output / f"{path.stem}_radialPlot.hdf5", "r") as hf:
            predictions[path.stem] = hf["prediction"][()]
            uncertainties[path.stem] = hf["uncertainty"][()]
    return predictions, uncertainties


def test_demo_writes_figure(demo_run):
    assert (demo_run / "demo.png").is_file()


def test_training_loss_decreases(losses):
    assert np.isfinite(losses["training_losses"]).all()  # no NaN/inf Gaussian NLL at any epoch
    mse = losses["training_mses"]
    assert np.isfinite(mse).all()  # no NaN/inf MSE at any epoch
    assert mse[0] / mse[-1] >= MIN_TRAINING_LOSS_DROP  # and the network fits its training set


def test_validation_loss_decreases(losses):
    assert np.isfinite(losses["validation_losses"]).all()  # no NaN/inf Gaussian NLL at any epoch
    mse = losses["validation_mses"]
    assert np.isfinite(mse).all()  # no NaN/inf MSE at any epoch
    assert mse[0] / mse.min() >= MIN_VALIDATION_LOSS_DROP  # and it generalizes to an unseen galaxy


def test_trained_network_predicts_test_galaxy(trained_outputs):
    predictions, _ = trained_outputs
    errors = rms_errors(predictions)
    assert max(errors.values()) < MAX_RMS_KMS, errors


def test_trained_network_is_calibrated(trained_outputs):
    check_calibration(*trained_outputs)


def test_committed_network_predicts_test_galaxy(committed_outputs):
    predictions, _ = committed_outputs
    errors = rms_errors(predictions)
    assert max(errors.values()) < MAX_RMS_KMS, errors


def test_committed_network_is_calibrated(committed_outputs):
    check_calibration(*committed_outputs)
