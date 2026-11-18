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
    # check model files are saved
    assert (tmp_path / "nets" / "python.pt").is_file() and result.model_path == str(tmp_path / "nets" / "python")
    for history in (result.training_losses, result.validation_losses, result.training_mses, result.validation_mses):
        assert len(history) == SMALL_HYPERPARAMS["epochs"] and np.isfinite(history).all()  # make sure losses are finite
    assert np.isfinite(result.final_test_loss)  # ensure last test loss is finite
    assert sorted(p.name for p in (tmp_path / "diag").rglob("*") if p.is_file()) == ["python_LossesAndAccuracies.hdf5"]
    _, metadata = load_model(result.model_path)  # the checkpoint's generated parameter text loads back
    assert tomllib.loads(metadata["hyperparams_text"]) == hyperparams


def test_snapshots_during_training(tmp_path):
    """snapshot_every saves a comparison image and hdf5 of a few training and validation images every snapshot_every epochs, even
    with save_diagnostics=False, for radial curves as for maps."""
    with open(CONFIG_DIR / "hyperparams_resblock_rc.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS, "output_head": "fc", "epochs": 3, "predict_every": 3}
    galaxy = {GALAXY: SNAPSHOT}
    datasets = {
        "target": TARGET,
        "augment": "none",
        "inclinations": [INCLINATION],
        "radial_bins": RADIAL_BINS,
        "radial_projection": "sky",
        "data_dir": IMAGE.parents[2],
        "training": galaxy,
        "validation": galaxy,
        "testing": galaxy,
    }
    train(
        hyperparams,
        datasets,
        run_name="snap",
        network_dir=tmp_path / "nets",
        output_dir=tmp_path / "diag",
        save_diagnostics=False,
        snapshot_every=2,
        snapshot_examples=2,
    )
    saved = sorted(p.name for p in (tmp_path / "diag" / "images").iterdir())
    # Only epoch 2 (index 1) of 3 is a snapshot epoch: an hdf5 and a png of each of the 2 examples of each set
    assert all(name.endswith(("_Epoch1.hdf5", "_Epoch1.png")) for name in saved)
    for split in ("training", "validation"):
        assert len([name for name in saved if name.startswith(f"snap_{split}_")]) == 4, split


def test_loading_workers_match_main_process(tmp_path):
    """Loading the data in worker processes only reads and transforms samples, in the order the main process shuffles
    them: the first epoch matches loading in the main process exactly, and any number of workers trains the same
    network. (Later epochs differ slightly from num_workers=0, which starts a new DataLoader iterator, drawing a seed
    from the global RNG, every epoch; persistent workers draw it once, so the shuffles and dropout that follow differ.)"""
    with open(CONFIG_DIR / "hyperparams_resblock.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS, "output_head": "fc"}
    galaxy = {GALAXY: SNAPSHOT}
    datasets = {
        "target": TARGET,
        "augment": "major_axis",  # 4 samples, so the training set is shuffled across 2 batches
        "inclinations": [INCLINATION],
        "radial_bins": 0,
        "radial_projection": "sky",
        "data_dir": IMAGE.parents[2],
        "training": galaxy,
        "validation": galaxy,
        "testing": galaxy,
    }
    losses = {}
    for num_workers in (0, 1, 3):
        result = train(
            hyperparams,
            datasets,
            run_name=f"workers{num_workers}",
            network_dir=tmp_path / "nets",
            output_dir=tmp_path / "diag",
            save_diagnostics=False,
            batch_size=2,
            num_workers=num_workers,
        )
        losses[num_workers] = np.concatenate(
            [result.training_losses, result.validation_losses, [result.final_test_loss]]
        )
    np.testing.assert_array_equal(losses[3], losses[1])
    # The first epoch's training and validation losses
    n_epochs = SMALL_HYPERPARAMS["epochs"]
    np.testing.assert_array_equal(losses[1][[0, n_epochs]], losses[0][[0, n_epochs]])
    np.testing.assert_allclose(losses[1], losses[0], rtol=1e-3)
