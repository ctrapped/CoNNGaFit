"""Smoke tests of the training and inference pipeline: every output configuration of the ResBlock U-Net, and one with
several targets, trains on a single VOF image for two epochs, writes a valid checkpoint, and predicts outputs (and,
with predict_variance, their 1-sigma uncertainties) of the expected size, both when training ends and when the
checkpoint is reloaded by use_model.py. run_inference.py runs the committed demo network."""

import tomllib

import h5py
import numpy as np
import pytest
import torch
from conftest import (
    CONFIG_DIR,
    GALAXY,
    IMAGE,
    IMAGE_SHAPE,
    INCLINATION,
    REPO,
    SNAPSHOT,
    TARGET,
    expected_shape,
    saved_test_outputs,
)

from conngafit.train_model import train
from conngafit.use_model import load_model, predict, run_inferences

INFERENCE_SCRIPT = REPO / "scripts" / "run_inference.py"
DEMO_MODEL = REPO / "examples" / "demo_model" / "demo"  # a committed checkpoint, for run_inference.py

# Overrides that make the default hyperparameters small enough to train in seconds on a CPU
SMALL_HYPERPARAMS = dict(epochs=2, n_fc=8, n_filt0=2, k0=3, k1=3, label_scaling="one-sided")
# The runs with predict_variance also weight their loss by mass, covering the weighted Gaussian NLL for maps and
# radial curves; the run without it trains unweighted, like the default configs
MASS_WEIGHT_POWER = 0.5
RADIAL_BINS = 10
PREDICTION_RTOL, PREDICTION_ATOL = 1e-4, 1e-6
# Annotations the demo images carry, predicted together by one run
TARGETS = ["mass", "rotational_momentum"]

# (run name, output head, radial bins, predict_variance, targets): a map from every head and a radial curve from the
# fc head, each with predict_variance, plus the default head without it, and maps of several targets. Without
# predict_variance only the training loss and saved outputs change, not the head, and the unit tests run every head
# without it.
RUNS = [
    ("fc", "fc", 0, True, [TARGET]),
    ("conv", "conv", 0, True, [TARGET]),
    ("conv_halfres", "conv_halfres", 0, True, [TARGET]),
    ("fc_radial", "fc", RADIAL_BINS, True, [TARGET]),
    ("conv_halfres_no_variance", "conv_halfres", 0, False, [TARGET]),
    ("conv_halfres_two_targets", "conv_halfres", 0, True, TARGETS),
]


@pytest.fixture(scope="module", params=RUNS, ids=[name for name, *_ in RUNS])
def trained(request, train_on_image):
    """Train once per configuration on IMAGE; returns (run name, head, radial bins, predict_variance, targets, run
    folder)."""
    name, head, radial_bins, predict_variance, targets = request.param
    with open(CONFIG_DIR / "hyperparams_resblock.toml", "rb") as f:
        default_hyperparams = tomllib.load(f)
    hyperparams = {
        **default_hyperparams,
        **SMALL_HYPERPARAMS,
        "output_head": head,
        "predict_variance": predict_variance,
        "mass_weight_power": MASS_WEIGHT_POWER if predict_variance else 0,
    }
    run_dir = train_on_image(name, hyperparams, radial_bins, targets)
    return name, head, radial_bins, predict_variance, targets, run_dir


def test_training_writes_valid_checkpoint(trained):
    name, head, radial_bins, predict_variance, targets, run_dir = trained
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        assert hf["output_head"][()].decode() == head  # records the head it was trained with
        assert [t.decode() for t in hf["target"][()]] == targets  # and its targets, in order
        assert bool(hf["predict_variance"][()]) == predict_variance  # and whether it predicts its uncertainty
        n_y, n_x, n_spec = IMAGE_SHAPE
        assert tuple(hf["input_shape"][()]) == (n_spec, n_y, n_x)  # built for the cube, spectral axis first
        # the saved copy of the datasets file round-trips
        assert tomllib.loads(hf["datasets_text"][()].decode())["radial_bins"] == radial_bins
        # image normalization and label scaling needed for inference are finite
        assert np.isfinite(hf["image_stats"][()]).all() and np.isfinite(hf["label_stats"][()]).all()
        assert hf["label_stats"].shape == (2, len(targets))  # an offset and a scale for each target
    with h5py.File(run_dir / "diag" / "diagnostics" / f"{name}_LossesAndAccuracies.hdf5", "r") as hf:
        for history in ("training_losses", "validation_losses", "training_mses", "validation_mses"):
            assert len(hf[history][()]) == SMALL_HYPERPARAMS["epochs"], history  # one value per epoch
            assert np.isfinite(hf[history][()]).all(), history  # every loss (and MSE of the mean) is finite
        assert np.isfinite(hf["final_test_loss"][()])  # the final test loss is finite
        assert np.isfinite(hf["final_test_mse"][()])  # and so is the final test MSE


def test_checkpoint_reproduces_saved_prediction(trained):
    name, _, radial_bins, predict_variance, targets, run_dir = trained
    model, metadata = load_model(str(run_dir / "nets" / name))
    [(_, prediction, std)] = predict(model, metadata, [str(IMAGE)], return_std=True)
    # predict returns a map or a curve of the expected size for each target
    assert prediction.shape == expected_shape(radial_bins, len(targets))
    if not predict_variance:
        assert std is None  # a network without predict_variance has no uncertainty to give
    for m, target in enumerate(targets):  # and the same values the training script saved for this image
        saved_prediction, saved_uncertainty = saved_test_outputs(run_dir, name, target)
        np.testing.assert_allclose(
            prediction[m], np.squeeze(saved_prediction), rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL
        )
        assert (saved_uncertainty is not None) == predict_variance  # an uncertainty is saved only with predict_variance
        if predict_variance:  # the same 1-sigma too, each a positive, finite spread
            assert np.isfinite(saved_uncertainty).all() and (saved_uncertainty > 0).all()
            np.testing.assert_allclose(
                std[m], np.squeeze(saved_uncertainty), rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL
            )


def test_run_inferences_writes_expected_outputs(trained):
    name, _, radial_bins, predict_variance, targets, run_dir = trained
    run_inferences(str(run_dir / "nets" / name), [str(IMAGE)], str(run_dir / "inference_"))
    assert torch.is_grad_enabled()  # inference leaves gradients on for the caller
    output, dataset = ("radialPlot", "prediction") if radial_bins else ("projectionMap", "image")
    for target in targets:
        target_output = run_dir / f"inference_{IMAGE.stem}_{target}_{output}"
        assert target_output.with_suffix(".png").is_file()  # a plot of each target's prediction
        with h5py.File(target_output.with_suffix(".hdf5"), "r") as hf:
            assert hf[dataset].shape == expected_shape(radial_bins)  # and an hdf5 of it, of the expected size
            assert ("uncertainty" in hf) == predict_variance  # with its 1-sigma only with predict_variance
            if predict_variance:
                assert hf["uncertainty"].shape == expected_shape(radial_bins)


def test_inference_script_runs_committed_network(tmp_path, run_script):
    image_list = tmp_path / "images.csv"
    image_list.write_text(f"{IMAGE}\n")
    run_script(
        INFERENCE_SCRIPT,
        "--image-list", image_list,
        "--output-prefix", tmp_path / "inference_",
        "--model-path", DEMO_MODEL,
        cwd=tmp_path,
    )  # fmt: skip
    # the demo network predicts radial curves of one target
    assert (tmp_path / f"inference_{IMAGE.stem}_{TARGET}_radialPlot.hdf5").is_file()


def test_train_from_python(tmp_path):
    """train() runs with the parameters as dicts, as examples/demo.py calls it, returns the network's path and loss
    history, and with save_diagnostics=False saves only the network and the loss history."""
    with open(CONFIG_DIR / "hyperparams_resblock.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS, "output_head": "fc"}
    galaxy = {GALAXY: SNAPSHOT}
    datasets = {
        "target": TARGET,
        "augment": "none",
        "inclinations": [INCLINATION],
        "radial_bins": 0,
        "radial_projection": "sky",
        "data_dir": IMAGE.parents[2],  # The folder of the galaxy's folder
        "training": galaxy,
        "validation": galaxy,
        "testing": galaxy,
    }
    result = train(
        hyperparams,
        datasets,
        run_name="python",
        network_dir=tmp_path / "nets",
        output_dir=tmp_path / "diag",
        save_diagnostics=False,
    )
    assert (tmp_path / "nets" / "python.pt").is_file() and result.model_path == str(tmp_path / "nets" / "python")
    for history in (result.training_losses, result.validation_losses, result.training_mses, result.validation_mses):
        assert len(history) == SMALL_HYPERPARAMS["epochs"] and np.isfinite(history).all()
    assert np.isfinite(result.final_test_loss)
    assert sorted(p.name for p in (tmp_path / "diag").rglob("*") if p.is_file()) == ["python_LossesAndAccuracies.hdf5"]
    _, metadata = load_model(result.model_path)  # the checkpoint's generated parameter text loads back
    assert tomllib.loads(metadata["hyperparams_text"]) == hyperparams
