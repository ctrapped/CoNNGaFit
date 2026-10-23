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
import torch

from unet3d_resblock import NeuralNetwork

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
SMALL_HYPERPARAMS = dict(epochs=2, nFC=8, nFilt0=2, k0=3, k1=3, label_scaling="symmetric")
RADIAL_BINS = 8
TIMEOUT_S = 600
PREDICTION_RTOL, PREDICTION_ATOL = 1e-4, 1e-6

# (run name, output head, radial bins): maps for every head, plus radial curves for the fc head
RUNS = [("fc", "fc", 0), ("conv", "conv", 0), ("conv_halfres", "conv_halfres", 0), ("fc_radial", "fc", RADIAL_BINS)]


def write_vof_outputs(root):
    """Write random VOF-style datacubes and annotation maps, one folder per galaxy."""
    rng = np.random.default_rng(0)
    nX, nY, _ = CUBE_SHAPE
    for galaxy in sum(GALAXIES.values(), []):
        folder = root / galaxy / f"i{INCLINATION}"
        folder.mkdir(parents=True)
        for pa in POSITION_ANGLES:
            with h5py.File(folder / f"{galaxy}_i{INCLINATION}_pa{pa}_{SNAPSHOT}_image_fullSpectra.hdf5", "w") as hf:
                hf["spectra"] = rng.random(CUBE_SHAPE)
                hf["mass_annotation"] = rng.random((nX, nY)) * 1e6
                hf["radial_velocity_annotation"] = rng.normal(0, 50, (nX, nY))
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


def load_checkpoint(run_dir, name):
    """Rebuild the trained network from its .hdf5 metadata and .pt weights; returns (model, metadata)."""
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        meta = {key: hf[key][()] for key in hf}
    hyperparams = tomllib.loads(meta["hyperparamsFile"].decode())
    radial_bins = int(meta["radialBins"]) if "radialBins" in meta else None
    k0, k1 = hyperparams["k0"], hyperparams["k1"]
    model = NeuralNetwork(
        tuple(int(n) for n in meta["inputShape"]),
        hyperparams["nFilt0"],
        (k0,) * 3,
        (k1,) * 3,
        hyperparams["nFC"],
        head=meta["outputHead"].decode(),
        output_size=radial_bins,
    )
    model.load_state_dict(torch.load(run_dir / "nets" / f"{name}.pt", map_location="cpu", weights_only=True))
    return model.eval(), meta


def test_training_writes_checkpoint_and_diagnostics(trained):
    name, head, radial_bins, run_dir = trained
    with h5py.File(run_dir / "nets" / f"{name}.hdf5", "r") as hf:
        assert hf["outputHead"][()].decode() == head  # checkpoint records the head it was trained with
        # the saved copy of the datasets file round-trips
        assert tomllib.loads(hf["datasetsFile"][()].decode())["radial_bins"] == radial_bins
        # image normalization and label scaling needed for inference are finite
        assert np.isfinite(hf["imageStats"][()]).all() and np.isfinite(hf["labelStats"][()]).all()
    with h5py.File(run_dir / "diag" / "diagnostics" / f"{name}_LossesAndAccuracies.hdf5", "r") as hf:
        assert np.isfinite(hf["trainingLoss"][()]).all()  # every epoch's training loss is finite
        assert np.isfinite(hf["validationLoss"][()]).all()  # every epoch's validation loss is finite
        assert np.isfinite(hf["testingLoss"][()])  # final test loss is finite
    n_test = len(GALAXIES["testing"]) * len(POSITION_ANGLES)
    # one prediction snapshot per test image
    assert len(list((run_dir / "diag" / "images").glob(f"{name}_test_*.hdf5"))) == n_test


def test_checkpoint_reproduces_saved_test_predictions(trained):
    name, _, radial_bins, run_dir = trained
    if radial_bins:
        pytest.skip("radial-curve snapshots save the whole batch rather than one sample")
    model, meta = load_checkpoint(run_dir, name)
    mean, stdv = meta["imageStats"]
    offset, scale = meta["labelStats"]
    for path in sorted((run_dir / "vof").glob("m12d/*/*.hdf5")):
        with h5py.File(path, "r") as hf:
            cube = torch.from_numpy(hf["spectra"][()].transpose(2, 0, 1))  # (nSpec, nX, nY), as ToTensor gives
        with torch.no_grad():
            pred = model(((cube - mean) / stdv).float()[None]).numpy() * scale + offset
        saved = next((run_dir / "diag" / "images").glob(f"{name}_test_{path.stem}_Epoch*.hdf5"))
        with h5py.File(saved, "r") as hf:
            np.testing.assert_allclose(
                pred.reshape(hf["prediction"].shape), hf["prediction"][()], rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL
            )


@pytest.mark.xfail(strict=True, reason="use_model.py still targets the pre-refactor network and dataset API")
def test_inference_script_runs_on_trained_checkpoint(trained):
    name, _, radial_bins, run_dir = trained
    if radial_bins:
        pytest.skip("inference plots are maps")
    image_list = run_dir / "images.csv"
    image_list.write_text("\n".join(str(p) for p in sorted((run_dir / "vof").glob("m12d/*/*.hdf5"))) + "\n")
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
    assert list(run_dir.glob("inference_*"))  # and writes its prediction outputs
