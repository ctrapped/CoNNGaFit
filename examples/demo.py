"""Demo: train a small CoNNGaFit network on synthetic VOF observations, then predict a galaxy it hasn't seen.

Needs no simulation data and runs in about a minute on a laptop CPU. From the repository root:

    python examples/demo.py                 # writes examples/demo_output/demo.png
    python examples/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic
disk galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test
on. The network learns each galaxy's rotation curve (its rotation speed in 10 rings out to the image edge)
from the datacube's spectra.
"""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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
    "base_dir": str(DATA_DIR),
}
SPLITS = {
    "training": ["vflat120", "vflat160", "vflat240", "vflat280"],
    "validation": ["vflat180"],
    "testing": ["vflat220"],
}
SNAPSHOT = 600
# configs/hyperparams_rc.toml, made small enough to train in about a minute on a CPU
HYPERPARAM_OVERRIDES = {
    "n_filt0": 4,
    "k0": 5,
    "output_head": "fc",  # The head for radial curves
    "n_fc": 64,
    "dropout_rate": 0.0,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
}
DEFAULT_EPOCHS = 50
DETECTION = 1e20  # cm^-2: about 5 sigma for VOF's demo noise (3e18 cm^-2 in each of 48 channels)


def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings, numbers and lists)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(output, epochs):
    """Write the demo's hyperparameters and datasets files into output; returns their paths."""
    with open(REPO_ROOT / "configs" / "hyperparams_rc.toml", "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAM_OVERRIDES, "epochs": epochs}
    hyperparams_file = output / "hyperparams.toml"
    hyperparams_file.write_text("\n".join(toml_lines(hyperparams)) + "\n")

    lines = toml_lines(DATASETS)
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
    """Plot a test image's velocity maps, its true and predicted rotation curves at every position angle, and the
    loss curves. Returns the RMS difference between the predicted and true curves, in km/s."""
    model, metadata = load_model(str(model_path))
    n_bins = int(metadata["radial_bins"])
    test_files = sorted(
        (str(p) for p in (DATA_DIR / SPLITS["testing"][0]).rglob("*.hdf5")),
        key=lambda path: float(VOF_VIEW.search(Path(path).name)["position_angle"]),
    )
    predictions = dict(predict(model, metadata, test_files))
    radii = ring_radii_kpc(test_files[0], n_bins)
    with h5py.File(test_files[0]) as f:
        cube = f["spectra"][()]  # (y, x, velocity channel), in cm^-2 per channel
        channel_width = f.attrs["dnu_kmps"]
        half_width = f.attrs["fov_kpc"] / 2
    with h5py.File(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5") as f:
        training_losses = f["training_losses"][()]
        validation_losses = f["validation_losses"][()]

    velocities = (np.arange(cube.shape[2]) - (cube.shape[2] - 1) / 2) * channel_width
    moment0 = cube.sum(axis=2)  # total HI column density
    detected = moment0 > DETECTION
    moment1 = np.where(detected, (cube * velocities).sum(axis=2) / moment0, np.nan)
    extent = [-half_width, half_width, -half_width, half_width]
    maps = [
        (
            "Input: HI column density [cm$^{-2}$]",
            np.clip(moment0, 1e19, None),
            {"norm": LogNorm(1e19), "cmap": "inferno"},
        ),
        ("Input: line-of-sight velocity [km/s]", moment1, {"vmin": -200, "vmax": 200, "cmap": "RdBu_r"}),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), constrained_layout=True)
    for ax, (label, image, style) in zip(axes, maps):
        im = ax.imshow(image, origin="lower", extent=extent, **style)
        fig.colorbar(im, ax=ax, location="top", shrink=0.9, pad=0.02).set_label(label)
        ax.set(xlabel="x [kpc]", ylabel="y [kpc]")

    residuals = []
    for i, path in enumerate(test_files):
        with h5py.File(path) as f:
            truth = radial_profile(f, DATASETS["target"], n_bins)  # The curve the network was trained to predict
        prediction = predictions[Path(path).stem]
        residuals.append(prediction - truth)
        position_angle = VOF_VIEW.search(Path(path).name)["position_angle"]
        axes[2].plot(radii, truth, "k", lw=2, label="True" if i == 0 else None)
        axes[2].plot(radii, prediction, "--", marker="o", ms=3, label=f"Predicted, PA {position_angle}°")
    axes[2].set(xlabel="Radius [kpc]", ylabel="Rotational velocity [km/s]", title="Test galaxy's rotation curve")
    axes[2].set_ylim(bottom=0)
    axes[2].legend(fontsize=8)

    epochs = np.arange(1, len(training_losses) + 1)
    axes[3].plot(epochs, training_losses, label="Training")
    axes[3].plot(epochs, validation_losses, label="Validation")
    axes[3].set(xlabel="Epoch", ylabel="Loss (MSE, scaled labels)", yscale="log", title="Loss curves")
    axes[3].legend()
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return np.sqrt(np.mean(np.square(residuals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    model_path = train(output, args.epochs)
    figure_file = output / "demo.png"
    rms = plot(model_path, output, figure_file)
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201


if __name__ == "__main__":
    main()
