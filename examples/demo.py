"""Demo: train a small CoNNGaFit network on synthetic VOF observations, then predict a galaxy it hasn't seen.

Needs no simulation data and runs in about a minute on a laptop CPU. From the repository root:

    python examples/demo.py                 # writes examples/demo_output/demo.png
    python examples/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic
disk galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test
on. The network learns each galaxy's rotation curve (its rotation speed in 10 rings out to the image edge)
from the datacube's spectra, along with each ring's 1-sigma uncertainty (predict_variance).
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # So CoNNGaFit's modules import from examples/

from conngafit_datasets import VOF_VIEW, radial_profile  # noqa: E402 - after the sys.path line above
from use_model import load_model, predict, ring_radii_kpc  # noqa: E402

DATA_DIR = REPO_ROOT / "examples" / "demo_data"
RUN_NAME = "demo"
DATASETS = {
    "target": "rotational_velocity",
    "augment": "major_axis",
    "inclinations": [45],
    "radial_bins": 10,
    "radial_projection": "sky",
}
SPLITS = {
    "training": ["vflat120", "vflat160", "vflat240", "vflat280"],
    "validation": ["vflat180"],
    "testing": ["vflat220"],
}
SNAPSHOT = 600
# configs/hyperparams_rc.toml, made small enough to train in a few minutes on a CPU
HYPERPARAM_OVERRIDES = {
    "n_filt0": 4,
    "k0": 5,
    "output_head": "fc",  # The head for radial curves
    "n_fc": 64,
    "dropout_rate": 0.0,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
    "predict_variance": True,  # Also predict each ring's 1-sigma uncertainty
}
DEFAULT_EPOCHS = 200


def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings, numbers and lists)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(output, epochs):
    """Write the demo's hyperparameters and datasets files into output; returns their paths."""
    with open(REPO_ROOT / "configs" / "hyperparams_rc.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAM_OVERRIDES, "epochs": epochs}
    hyperparams_file = output / "hyperparams.toml"
    hyperparams_file.write_text("\n".join(toml_lines(hyperparams)) + "\n")

    # Relative to the datasets file, so the copy saved in the checkpoint doesn't hold this computer's paths
    lines = toml_lines({**DATASETS, "base_dir": os.path.relpath(DATA_DIR, output)})
    for split, galaxies in SPLITS.items():
        lines += [f"[{split}]"] + [f'"{galaxy}" = {SNAPSHOT}' for galaxy in galaxies]
    datasets_file = output / "datasets.toml"
    datasets_file.write_text("\n".join(lines) + "\n")
    return hyperparams_file, datasets_file


def train(output, epochs):
    """Run train_model_unet.py on the demo data, exactly as you would from the command line."""
    hyperparams_file, datasets_file = write_configs(output, epochs)
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "train_model_unet.py",
            "--hyperparams",
            hyperparams_file,
            "--datasets",
            datasets_file,
            "--run-name",
            RUN_NAME,
            "--network-dir",
            output / "NetworkWeights",
            "--output-dir",
            output / "DiagnosticOutputs",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output / "NetworkWeights" / RUN_NAME


def plot(model_path, output, figure_file):
    """Plot a test image's velocity maps, its true and predicted rotation curves (with their predicted 1-sigma) at
    every position angle, and the loss curves. Returns (the RMS difference between the predicted and true curves in
    km/s, the fraction of rings whose prediction is within its predicted 1-sigma of the truth)."""
    model, metadata = load_model(str(model_path))
    n_bins = int(metadata["radial_bins"])
    test_files = sorted(
        (str(p) for p in (DATA_DIR / SPLITS["testing"][0]).rglob("*.hdf5")),
        key=lambda path: float(VOF_VIEW.search(Path(path).name)["position_angle"]),
    )
    predictions = {name: (pred, std) for name, pred, std in predict(model, metadata, test_files, return_std=True)}
    radii = ring_radii_kpc(test_files[0], n_bins)
    with h5py.File(test_files[0]) as f:
        cube = f["spectra"][()]  # (y, x, velocity channel), in cm^-2 per channel
        channel_width = f.attrs["dnu_kmps"]
        half_width = f.attrs["fov_kpc"] / 2
    with h5py.File(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5") as f:
        training_losses = f["training_losses"][()]  # Gaussian NLL, the training objective
        validation_losses = f["validation_losses"][()]
        training_mses = f["training_mses"][()]  # MSE of the predicted mean
        validation_mses = f["validation_mses"][()]

    velocities = (np.arange(cube.shape[2]) - (cube.shape[2] - 1) / 2) * channel_width
    moment0 = cube.sum(axis=2)  # total HI column density
    moment1 = (cube * velocities).sum(axis=2) / moment0
    extent = [-half_width, half_width, -half_width, half_width]
    maps = [
        (
            "Input: HI column density [cm$^{-2}$]",
            np.clip(moment0, 1e19, None),
            {"norm": LogNorm(1e19), "cmap": "inferno"},
        ),
        ("Input: line-of-sight velocity [km/s]", moment1, {"vmin": -200, "vmax": 200, "cmap": "RdBu_r"}),
    ]

    # Inputs and the rotation curve on top; the residuals and both loss curves below
    fig, axes = plt.subplots(2, 3, figsize=(14.25, 9.2), constrained_layout=True)
    (ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_nll) = axes
    for ax, (label, image, style) in zip((ax_density, ax_velocity), maps):
        im = ax.imshow(image, origin="lower", extent=extent, **style)
        fig.colorbar(im, ax=ax, location="top", shrink=0.9, pad=0.02).set_label(label)
        ax.set(xlabel="x [kpc]", ylabel="y [kpc]")

    residuals, sigmas = [], []
    for i, path in enumerate(test_files):
        with h5py.File(path) as f:
            truth = radial_profile(f, DATASETS["target"], n_bins)  # The curve the network was trained to predict
        prediction, sigma = predictions[Path(path).stem]
        residual = prediction - truth
        residuals.append(residual)
        sigmas.append(sigma)
        position_angle = VOF_VIEW.search(Path(path).name)["position_angle"]
        ax_curve.plot(radii, truth, "k", lw=2, label="True" if i == 0 else None)
        (line,) = ax_curve.plot(radii, prediction, "--", marker="o", ms=3, label=f"Predicted, PA {position_angle}°")
        ax_curve.fill_between(radii, prediction - sigma, prediction + sigma, color=line.get_color(), alpha=0.15, lw=0)
        # The same bands around each residual: a ring is within its 1-sigma where its band crosses zero
        ax_residual.plot(radii, residual, "--", marker="o", ms=3, color=line.get_color(), label=f"PA {position_angle}°")
        ax_residual.fill_between(radii, residual - sigma, residual + sigma, color=line.get_color(), alpha=0.15, lw=0)
    ax_curve.set(xlabel="Radius [kpc]", ylabel="Rotational velocity [km/s]", title="Test galaxy's rotation curve")
    ax_curve.set_ylim(bottom=0)
    ax_residual.axhline(0, color="k", lw=2)
    ax_residual.set(xlabel="Radius [kpc]", ylabel="Predicted − true [km/s]", title="Residuals")
    # One legend entry for every line's band, in grey since the bands share their line's color
    sigma_band = Patch(color="grey", alpha=0.3, lw=0, label="Predicted ±1σ")
    for ax in (ax_curve, ax_residual):
        ax.legend(handles=[*ax.get_legend_handles_labels()[0], sigma_band], fontsize=8)

    epochs = np.arange(1, len(training_losses) + 1)
    for ax, (training, validation, title, label, scale) in zip(
        (ax_mse, ax_nll),
        [
            (training_mses, validation_mses, "MSE of the mean", "MSE (scaled labels)", "log"),
            (training_losses, validation_losses, "Gaussian NLL", "NLL (training objective)", "linear"),  # can be < 0
        ],
    ):
        ax.plot(epochs, training, label="Training")
        ax.plot(epochs, validation, label="Validation")
        ax.set(xlabel="Epoch", ylabel=label, yscale=scale, title=title)
        ax.legend()
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)
    residuals = np.array(residuals)
    return (
        np.sqrt(np.mean(np.square(residuals))),
        np.mean(np.abs(residuals) < np.array(sigmas)),
        np.mean(np.abs(residuals) < 3 * np.array(sigmas)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_path = train(output, args.epochs)
    figure_file = output / "demo.png"
    rms, within_sigma, within_3sigma = plot(model_path, output, figure_file)
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201


if __name__ == "__main__":
    main()
