"""End-to-end smoke tests: train train_model_unet.py for two epochs on synthetic VOF outputs, then
check the checkpoint it writes rebuilds the network and reproduces the predictions it saved."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np
import pytest

from use_model import load_model, predict

REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO / "train_model_unet.py"
INFERENCE_SCRIPT = REPO / "inference_template.py"
DEFAULT_HYPERPARAMS = REPO / "configs" / "hyperparams.toml"

CUBE_SHAPE = (32, 32, 32)  # (nX, nY, nSpec) as VOF writes it; 32 survives the default 4 encoder stages
FOV_KPC = 40.0
INCLINATION = 45
SNAPSHOT = 600
POSITION_ANGLES = (0, 90)
GALAXIES = {"training": ["m12a", "m12b"], "validation": ["m12c"], "testing": ["m12d"]}
# Overrides that make the default hyperparameters small enough to train in seconds on a CPU
SMALL_HYPERPARAMS = dict(epochs=2, n_fc=8, n_filt0=2, k0=3, k1=3, label_scaling="symmetric")
RADIAL_BINS = 8
TIMEOUT_S = 600
PREDICTION_RTOL, PREDICTION_ATOL = 1e-4, 1e-6

# (run name, output head, radial bins): maps for every head, plus radial curves for the fc head
RUNS = [("fc", "fc", 0), ("conv", "conv", 0), ("conv_halfres", "conv_halfres", 0), ("fc_radial", "fc", RADIAL_BINS)]


def write_vof_outputs(root):
    """Write random VOF-style datacubes and annotation maps, one folder per galaxy."""
    rng = np.random.default_rng(0)
    n_x, n_y, _ = CUBE_SHAPE
    for galaxy in sum(GALAXIES.values(), []):
        folder = root / galaxy / f"i{INCLINATION}"
        folder.mkdir(parents=True)
        for pa in POSITION_ANGLES:
            with h5py.File(folder / f"{galaxy}_i{INCLINATION}_pa{pa}_{SNAPSHOT}_image_fullSpectra.hdf5", "w") as hf:
                hf["spectra"] = rng.random(CUBE_SHAPE)
                hf["mass_annotation"] = rng.random((n_x, n_y)) * 1e6
                hf["radial_velocity_annotation"] = rng.normal(0, 50, (n_x, n_y))
                hf.attrs["fov_kpc"] = FOV_KPC


def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings and numbers)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(folder, head, radial_bins):
    with open(DEFAULT_HYPERPARAMS, "rb") as f:
        hyperparams = {**tomllib.load(f), **SMALL_HYPERPARAMS, "output_head": head}
    (folder / "hyperparams.toml").write_text("\n".join(toml_lines(hyperparams)) + "\n")

    datasets = dict(
        target="radial_mass_flux",
        augment="major_axis",
        inclinations=[INCLINATION],
        radial_bins=radial_bins,
        radial_projection="sky",
    )
    lines = toml_lines(datasets)
    for split, galaxies in GALAXIES.items():
        lines += [f"[{split}]"] + [f'"vof/{galaxy}" = {SNAPSHOT}' for galaxy in galaxies]
    (folder / "datasets.toml").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def vof_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    write_vof_outputs(root / "vof")
    return root


@pytest.fixture(scope="module", params=RUNS, ids=[name for name, _, _ in RUNS])
def trained(request, vof_root):
    """Run the training script once per configuration; returns (run name, head, radial bins, run folder)."""
    name, head, radial_bins = request.param
    run_dir = vof_root / name
    run_dir.mkdir()
    write_configs(vof_root, head, radial_bins)
    for config in ("hyperparams.toml", "datasets.toml"):
        (vof_root / config).rename(run_dir / config)
    (run_dir / "vof").symlink_to(vof_root / "vof")
    result = subprocess.run(
        [
            sys.executable,
            TRAIN_SCRIPT,
            "--hyperparams",
            "hyperparams.toml",
            "--datasets",
            "datasets.toml",
            "--run-name",
            name,
            "--network-dir",
            "nets",
            "--output-dir",
            "diag",
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return name, head, radial_bins, run_dir


def test_training_writes_checkpoint_and_diagnostics(trained):
    name, head, radial_bins, run_dir = trained
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        assert hf["output_head"][()].decode() == head  # checkpoint records the head it was trained with
        # the saved copy of the datasets file round-trips
        assert tomllib.loads(hf["datasets_text"][()].decode())["radial_bins"] == radial_bins
        # image normalization and label scaling needed for inference are finite
        assert np.isfinite(hf["image_stats"][()]).all() and np.isfinite(hf["label_stats"][()]).all()
    with h5py.File(run_dir / "diag" / "diagnostics" / f"{name}_LossesAndAccuracies.hdf5", "r") as hf:
        assert np.isfinite(hf["training_losses"][()]).all()  # every epoch's training loss is finite
        assert np.isfinite(hf["validation_losses"][()]).all()  # every epoch's validation loss is finite
        assert np.isfinite(hf["final_test_loss"][()])  # final test loss is finite
    n_test = len(GALAXIES["testing"]) * len(POSITION_ANGLES)
    # one prediction snapshot per test image
    assert len(list((run_dir / "diag" / "images").glob(f"{name}_test_*.hdf5"))) == n_test


def test_checkpoint_reproduces_saved_test_predictions(trained):
    name, _, radial_bins, run_dir = trained
    if radial_bins:
        pytest.skip("radial-curve snapshots save the whole batch rather than one sample")
    model, metadata = load_model(str(run_dir / "nets" / name))
    paths = sorted(str(p) for p in (run_dir / "vof").glob("m12d/*/*.hdf5"))
    for image_name, pred in predict(model, metadata, paths):
        saved = next((run_dir / "diag" / "images").glob(f"{name}_test_{image_name}_Epoch*.hdf5"))
        with h5py.File(saved, "r") as hf:
            np.testing.assert_allclose(pred, hf["prediction"][()], rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)


def test_inference_script_runs_on_trained_checkpoint(trained):
    name, _, radial_bins, run_dir = trained
    paths = sorted((run_dir / "vof").glob("m12d/*/*.hdf5"))
    image_list = run_dir / "images.csv"
    image_list.write_text("\n".join(str(p) for p in paths) + "\n")
    result = subprocess.run(
        [
            sys.executable,
            INFERENCE_SCRIPT,
            "--image-list",
            image_list,
            "--output-prefix",
            run_dir / "inference_",
            "--model-path",
            run_dir / "nets" / name,
        ],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]  # script exits cleanly
    output = "radialPlot" if radial_bins else "projectionMap"
    for path in paths:  # a plot and an hdf5 of every prediction
        for extension in (".png", ".hdf5"):
            assert (run_dir / f"inference_{path.stem}_{output}{extension}").is_file()
