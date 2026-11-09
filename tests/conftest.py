"""Shared fixtures: small, CPU-sized models and input cubes for the network tests, and a script runner, a
single-image training run and their helpers for the end-to-end tests."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import torch

from conngafit.models import unet3d_dilated_resnet, unet3d_resblock

INPUT_SHAPE = (16, 16, 16)  # (nSpec, nX, nY)
BATCH_SIZE = 2
SEED = 0
SCRIPT_TIMEOUT_S = 600

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "scripts" / "train_model_unet.py"
CONFIG_DIR = REPO / "configs"

# One committed VOF image (examples/demo_data/, made by examples/make_demo_data.py), used as the training,
# validation and testing set of the pipeline tests
GALAXY = "vflat220"
INCLINATION = 45
SNAPSHOT = 600
IMAGE_NAME = f"{GALAXY}_i{INCLINATION}_pa0_{SNAPSHOT}_image_fullSpectra.hdf5"
IMAGE = REPO / "examples" / "demo_data" / GALAXY / f"i{INCLINATION}" / IMAGE_NAME
TARGET = "rotational_velocity"  # The demo images carry mass and rotational velocity annotations
with h5py.File(IMAGE, "r") as hf:
    IMAGE_SHAPE = hf["spectra"].shape  # (n_y, n_x, n_spec), as VOF writes it

# Constructor arguments shared by every network, plus per-network arguments that keep it small
COMMON_KWARGS = dict(input_shape=INPUT_SHAPE, n_filt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), n_fc=8)
SMALL_KWARGS = {
    unet3d_resblock: dict(n_stages=3),
    unet3d_dilated_resnet: dict(n_blocks=1, dilation_rates=(1, 2, 1)),
}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


@pytest.fixture(params=list(SMALL_KWARGS), ids=lambda module: module.__name__)
def network(request):
    """Each network module in turn."""
    return request.param


@pytest.fixture
def make_model():
    """Factory for a small module.NeuralNetwork; keyword arguments override the defaults."""

    def make(module, **overrides):
        return module.NeuralNetwork(**{**COMMON_KWARGS, **SMALL_KWARGS[module], **overrides})

    return make


@pytest.fixture
def make_cube():
    """Factory for a random batch of datacubes of shape (BATCH_SIZE, *input_shape)."""

    def make(input_shape=INPUT_SHAPE):
        return torch.randn(BATCH_SIZE, *input_shape)

    return make


@pytest.fixture
def record_shapes():
    """Factory that hooks each of a list of modules; returns the list every forward pass then appends each module's
    output (channels, spatial shape) to, in execution order."""

    def record(modules):
        recorded = []

        def hook(module, inputs, out):
            out = out[0] if isinstance(out, tuple) else out  # e.g. a stem that also returns its half-resolution map
            recorded.append((out.shape[1], tuple(out.shape[2:])))

        for module in modules:
            module.register_forward_hook(hook)
        return recorded

    return record


@pytest.fixture(scope="session")
def run_script():
    """Run a Python script in a subprocess, as from the command line; fails the test with the end of its output
    if it exits with an error."""

    def run(script, *args, cwd=None):
        result = subprocess.run(
            [sys.executable, str(script), *(str(a) for a in args)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_S,
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        return result

    return run


def expected_shape(radial_bins=0, n_maps=None):
    """A map has one value per sky pixel of the cube; a curve has one value per ring. With n_maps, the shape of
    n_maps such maps or curves stacked, as predicted for that many targets."""
    n_y, n_x, _ = IMAGE_SHAPE
    shape = (radial_bins,) if radial_bins else (n_y, n_x)
    return shape if n_maps is None else (n_maps, *shape)


def saved_test_outputs(run_dir, name, target=TARGET):
    """The prediction of target and its 1-sigma uncertainty (None if the network gave none) the training script
    saved for IMAGE at the end of training."""
    saved = next((run_dir / "diag" / "images").glob(f"{name}_test_{IMAGE.stem}_{target}_Epoch*.hdf5"))
    with h5py.File(saved, "r") as hf:
        return hf["prediction"][()], (hf["uncertainty"][()] if "uncertainty" in hf else None)


@pytest.fixture(scope="session")
def train_on_image(tmp_path_factory, run_script):
    """Run the training script on IMAGE with a hyperparameters dict, in a new folder, and return the folder. The
    network is written to nets/<name>, the diagnostic outputs to diag/. target is a target name or a list of them."""

    def train(name, hyperparams, radial_bins=0, target=TARGET):
        run_dir = tmp_path_factory.mktemp(name)
        image_dir = run_dir / "vof" / GALAXY / f"i{INCLINATION}"
        image_dir.mkdir(parents=True)
        shutil.copy(IMAGE, image_dir)
        datasets = dict(
            target=target, augment="none", inclinations=[INCLINATION], radial_bins=radial_bins, radial_projection="sky"
        )
        # Flat key = value TOML lines: JSON literals are valid TOML for strings, numbers and lists
        hyperparams_lines = [f"{key} = {json.dumps(value)}" for key, value in hyperparams.items()]
        datasets_lines = [f"{key} = {json.dumps(value)}" for key, value in datasets.items()]
        for split in ("training", "validation", "testing"):
            datasets_lines += [f"[{split}]", f'"vof/{GALAXY}" = {SNAPSHOT}']
        (run_dir / "hyperparams.toml").write_text("\n".join(hyperparams_lines) + "\n")
        (run_dir / "datasets.toml").write_text("\n".join(datasets_lines) + "\n")

        run_script(
            TRAIN_SCRIPT,
            "--hyperparams", "hyperparams.toml",
            "--datasets", "datasets.toml",
            "--run-name", name,
            "--network-dir", "nets",
            "--output-dir", "diag",
            cwd=run_dir,
        )  # fmt: skip
        return run_dir

    return train
