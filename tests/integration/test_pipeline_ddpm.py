"""Smoke tests of the training and inference pipeline for the DDPM: it trains on a single VOF image for three
epochs, for one target and for two, writes a valid checkpoint, predicts every predict_every epochs, and gives posterior
mean and standard deviation maps of the expected size for each target, both when training ends and when the
checkpoint is reloaded by use_model.py."""

import tomllib

import h5py
import numpy as np
import pytest
import torch
from conftest import CONFIG_DIR, IMAGE, IMAGE_SHAPE, TARGET, expected_shape, saved_test_outputs

from use_model import load_model, predict, run_inferences

# (run name, targets): one target, and two the demo images carry, predicted together
RUNS = [("ddpm", [TARGET]), ("ddpm_two_targets", ["mass", "rotational_momentum"])]
# Overrides that make configs/hyperparams_ddpm.toml small enough to train and sample in seconds on a CPU
SMALL_HYPERPARAMS = dict(
    epochs=3, n_filt0=2, k0=3, k1=3, label_scaling="one-sided", n_timesteps=10, n_samples=2, predict_every=2
)
# With 3 epochs and predict_every = 2, predictions are made on epoch 2 and the final epoch
PREDICTED_EPOCHS = [False, True, True]


@pytest.fixture(scope="module", params=RUNS, ids=[name for name, _ in RUNS])
def trained(request, train_on_image):
    """Train once per run on IMAGE; returns (run name, targets, run folder)."""
    name, targets = request.param
    with open(CONFIG_DIR / "hyperparams_ddpm.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS}
    return name, targets, train_on_image(name, hyperparams, target=targets)


def test_training_writes_valid_checkpoint(trained):
    name, targets, run_dir = trained
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        assert hf["network"][()].decode() == "ddpm"  # records the network
        assert [t.decode() for t in hf["target"][()]] == targets  # and its targets, in order
        assert hf["n_samples"][()] == SMALL_HYPERPARAMS["n_samples"]  # and its own hyperparameters
        assert "output_head" not in hf  # but none of the resblock network's
        n_y, n_x, n_spec = IMAGE_SHAPE
        assert tuple(hf["input_shape"][()]) == (n_spec, n_y, n_x)  # built for the cube, spectral axis first
        # image normalization and label scaling needed for inference are finite
        assert np.isfinite(hf["image_stats"][()]).all() and np.isfinite(hf["label_stats"][()]).all()
    with h5py.File(run_dir / "diag" / "diagnostics" / f"{name}_LossesAndAccuracies.hdf5", "r") as hf:
        for history in ("training_losses", "validation_losses"):
            assert len(hf[history][()]) == SMALL_HYPERPARAMS["epochs"], history  # one value per epoch
            assert np.isfinite(hf[history][()]).all(), history  # every denoising loss is finite
        assert np.isfinite(hf["final_test_loss"][()]) and np.isfinite(hf["final_test_mse"][()])  # and the test's


def test_saved_prediction_has_posterior_mean_and_std(trained):
    name, targets, run_dir = trained
    for target in targets:
        prediction, uncertainty = saved_test_outputs(run_dir, name, target)
        assert np.squeeze(prediction).shape == expected_shape()  # a map of each target's posterior mean
        assert uncertainty is not None and np.squeeze(uncertainty).shape == expected_shape()  # and of its spread
        assert np.isfinite(uncertainty).all() and (uncertainty >= 0).all()  # a valid standard deviation everywhere


def test_checkpoint_predicts_posterior_mean_and_std(trained):
    name, targets, run_dir = trained
    model, metadata = load_model(str(run_dir / "nets" / name))
    [(_, mean, std)] = predict(model, metadata, [str(IMAGE)], return_std=True)
    # maps of each target's posterior mean and its spread
    assert mean.shape == std.shape == expected_shape(n_maps=len(targets))
    assert np.isfinite(std).all() and (std >= 0).all()  # a valid standard deviation at every pixel


def test_checkpoint_returns_joint_posterior_samples(trained):
    name, targets, run_dir = trained
    model, metadata = load_model(str(run_dir / "nets" / name))
    [(_, samples)] = predict(model, metadata, [str(IMAGE)], return_samples=True)
    # n_samples maps of every target, in physical units
    assert samples.shape == (SMALL_HYPERPARAMS["n_samples"], *expected_shape(n_maps=len(targets)))
    assert np.isfinite(samples).all()


def test_run_inferences_writes_mean_and_std_map(trained):
    name, targets, run_dir = trained
    run_inferences(str(run_dir / "nets" / name), [str(IMAGE)], str(run_dir / "inference_"))
    assert torch.is_grad_enabled()  # inference leaves gradients on for the caller
    for target in targets:
        output = run_dir / f"inference_{IMAGE.stem}_{target}_projectionMap"
        assert output.with_suffix(".png").is_file()  # a plot of each target's prediction and its 1-sigma
        with h5py.File(output.with_suffix(".hdf5"), "r") as hf:
            assert hf["image"].shape == expected_shape()  # and an hdf5 of the posterior mean
            assert hf["uncertainty"].shape == expected_shape()  # with its standard deviation
