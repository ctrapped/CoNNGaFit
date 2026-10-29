"""Smoke tests of the training and inference pipeline: every output configuration of the ResBlock U-Net, with and
without predict_variance, trains on a single VOF image for two epochs, writes a valid checkpoint, and predicts outputs
(and, with predict_variance, their 1-sigma uncertainties) of the expected size, both when training ends and when the
checkpoint is reloaded by use_model.py and inference_template.py."""

import json
import shutil
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest

from use_model import load_model, predict

REPO = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO / "train_model_unet.py"
INFERENCE_SCRIPT = REPO / "inference_template.py"
DEFAULT_HYPERPARAMS = REPO / "configs" / "hyperparams.toml"

# One committed VOF image (examples/demo_data/, made by examples/make_demo_data.py), used as the training,
# validation and testing set alike
GALAXY = "vflat220"
INCLINATION = 45
SNAPSHOT = 600
IMAGE_NAME = f"{GALAXY}_i{INCLINATION}_pa0_{SNAPSHOT}_image_fullSpectra.hdf5"
IMAGE = REPO / "examples" / "demo_data" / GALAXY / f"i{INCLINATION}" / IMAGE_NAME
TARGET = "rotational_velocity"  # The demo images carry mass and rotational velocity annotations

# Overrides that make the default hyperparameters small enough to train in seconds on a CPU
SMALL_HYPERPARAMS = dict(epochs=2, n_fc=8, n_filt0=2, k0=3, k1=3, label_scaling="one-sided")
RADIAL_BINS = 10
PREDICTION_RTOL, PREDICTION_ATOL = 1e-4, 1e-6

# (run name, output head, radial bins, predict_variance): a map from every head, plus a radial curve from the fc
# head, each with and without predict_variance (every head predicts its log-variance differently)
OUTPUTS = [("fc", "fc", 0), ("conv", "conv", 0), ("conv_halfres", "conv_halfres", 0), ("fc_radial", "fc", RADIAL_BINS)]
RUNS = [(name, head, bins, False) for name, head, bins in OUTPUTS] + [
    (f"{name}_variance", head, bins, True) for name, head, bins in OUTPUTS
]


def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings, numbers and lists)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(folder, head, radial_bins, predict_variance):
    with open(DEFAULT_HYPERPARAMS, "rb") as f:
        hyperparams = {
            **tomllib.load(f),
            **SMALL_HYPERPARAMS,
            "output_head": head,
            "predict_variance": predict_variance,
        }
    (folder / "hyperparams.toml").write_text("\n".join(toml_lines(hyperparams)) + "\n")

    datasets = dict(
        target=TARGET, augment="none", inclinations=[INCLINATION], radial_bins=radial_bins, radial_projection="sky"
    )
    lines = toml_lines(datasets)
    for split in ("training", "validation", "testing"):
        lines += [f"[{split}]", f'"vof/{GALAXY}" = {SNAPSHOT}']
    (folder / "datasets.toml").write_text("\n".join(lines) + "\n")


def image_shape():
    """IMAGE's datacube shape as VOF writes it: (n_y, n_x, n_spec)."""
    with h5py.File(IMAGE, "r") as hf:
        return hf["spectra"].shape


def expected_shape(radial_bins):
    """A map has one value per sky pixel of the cube; a curve has one value per ring."""
    n_y, n_x, _ = image_shape()
    return (radial_bins,) if radial_bins else (n_y, n_x)


@pytest.fixture(scope="module", params=RUNS, ids=[name for name, *_ in RUNS])
def trained(request, tmp_path_factory, run_script):
    """Train once per configuration on IMAGE; returns (run name, head, radial bins, predict_variance, run folder)."""
    name, head, radial_bins, predict_variance = request.param
    run_dir = tmp_path_factory.mktemp(name)
    image_dir = run_dir / "vof" / GALAXY / f"i{INCLINATION}"
    image_dir.mkdir(parents=True)
    shutil.copy(IMAGE, image_dir)
    write_configs(run_dir, head, radial_bins, predict_variance)
    run_script(
        TRAIN_SCRIPT,
        "--hyperparams", "hyperparams.toml",
        "--datasets", "datasets.toml",
        "--run-name", name,
        "--network-dir", "nets",
        "--output-dir", "diag",
        cwd=run_dir,
    )  # fmt: skip
    return name, head, radial_bins, predict_variance, run_dir


def saved_test_outputs(run_dir, name):
    """The prediction and its 1-sigma uncertainty (None without predict_variance) the training script saved for IMAGE
    at the end of training."""
    saved = next((run_dir / "diag" / "images").glob(f"{name}_test_{IMAGE.stem}_Epoch*.hdf5"))
    with h5py.File(saved, "r") as hf:
        return hf["prediction"][()], (hf["uncertainty"][()] if "uncertainty" in hf else None)


def test_training_writes_valid_checkpoint(trained):
    name, head, radial_bins, predict_variance, run_dir = trained
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        assert hf["output_head"][()].decode() == head  # records the head it was trained with
        assert bool(hf["predict_variance"][()]) == predict_variance  # and whether it predicts its uncertainty
        n_y, n_x, n_spec = image_shape()
        assert tuple(hf["input_shape"][()]) == (n_spec, n_y, n_x)  # built for the cube, spectral axis first
        # the saved copy of the datasets file round-trips
        assert tomllib.loads(hf["datasets_text"][()].decode())["radial_bins"] == radial_bins
        # image normalization and label scaling needed for inference are finite
        assert np.isfinite(hf["image_stats"][()]).all() and np.isfinite(hf["label_stats"][()]).all()
    with h5py.File(run_dir / "diag" / "diagnostics" / f"{name}_LossesAndAccuracies.hdf5", "r") as hf:
        for history in ("training_losses", "validation_losses", "training_mses", "validation_mses"):
            assert len(hf[history][()]) == SMALL_HYPERPARAMS["epochs"], history  # one value per epoch
            assert np.isfinite(hf[history][()]).all(), history  # every loss (and MSE of the mean) is finite
        assert np.isfinite(hf["final_test_loss"][()])  # the final test loss is finite
        assert np.isfinite(hf["final_test_mse"][()])  # and so is the final test MSE


def test_saved_prediction_has_expected_size(trained):
    name, _, radial_bins, predict_variance, run_dir = trained
    prediction, uncertainty = saved_test_outputs(run_dir, name)
    assert np.squeeze(prediction).shape == expected_shape(radial_bins)  # a whole map, or one value per ring
    assert (uncertainty is not None) == predict_variance  # an uncertainty is saved only with predict_variance
    if predict_variance:
        assert uncertainty.shape == prediction.shape  # one 1-sigma per predicted value
        assert np.isfinite(uncertainty).all() and (uncertainty > 0).all()  # each a positive, finite spread


def test_checkpoint_reproduces_saved_prediction(trained):
    name, _, radial_bins, predict_variance, run_dir = trained
    model, metadata = load_model(str(run_dir / "nets" / name))
    [(_, prediction, std)] = predict(model, metadata, [str(IMAGE)], return_std=True)
    assert prediction.shape == expected_shape(radial_bins)  # predict returns a map or a curve of the expected size
    # and the same values the training script saved for this image
    saved_prediction, saved_uncertainty = saved_test_outputs(run_dir, name)
    np.testing.assert_allclose(prediction, np.squeeze(saved_prediction), rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
    if predict_variance:  # the same 1-sigma too
        assert std.shape == prediction.shape
        np.testing.assert_allclose(std, np.squeeze(saved_uncertainty), rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)
    else:
        assert std is None  # a network without predict_variance has no uncertainty to give


def test_inference_script_writes_expected_outputs(trained, run_script):
    name, _, radial_bins, predict_variance, run_dir = trained
    image_list = run_dir / "images.csv"
    image_list.write_text(f"{IMAGE}\n")
    run_script(
        INFERENCE_SCRIPT,
        "--image-list", image_list,
        "--output-prefix", run_dir / "inference_",
        "--model-path", run_dir / "nets" / name,
        cwd=run_dir,
    )  # fmt: skip
    output, dataset = ("radialPlot", "prediction") if radial_bins else ("projectionMap", "image")
    output = run_dir / f"inference_{IMAGE.stem}_{output}"
    assert output.with_suffix(".png").is_file()  # a plot of the prediction
    with h5py.File(output.with_suffix(".hdf5"), "r") as hf:
        assert hf[dataset].shape == expected_shape(radial_bins)  # and an hdf5 of it, of the expected size
        assert ("uncertainty" in hf) == predict_variance  # with its 1-sigma only with predict_variance
        if predict_variance:
            assert hf["uncertainty"].shape == expected_shape(radial_bins)
