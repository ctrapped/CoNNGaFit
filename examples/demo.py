"""Demo: train a small CoNNGaFit network on synthetic VOF observations, then predict a galaxy it hasn't seen.

Needs no simulation data and runs in a few minutes on a laptop CPU. From the repository root:

    python examples/demo.py                 # writes examples/demo_output/demo.png
    python examples/demo.py --predict map   # writes examples/demo_output/demo_map.png
    python examples/demo.py --predict ddpm  # writes examples/demo_output/demo_ddpm.png
    python examples/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic
disk galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test
on. The network learns each galaxy's rotation curve (its rotation speed in 10 rings out to the image edge)
from the datacube's spectra, along with each ring's 1-sigma uncertainty (predict_variance). With --predict map
it learns the rotational velocity map instead, with each pixel's 1-sigma, and the map is binned into the
rotation curve afterwards. With --predict ddpm a conditional diffusion model samples the velocity map from its
posterior instead: the mean and spread of its samples give each pixel's value and 1-sigma, binned the same way.
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

from conngafit_datasets import TARGETS, VOF_VIEW, radial_profile  # noqa: E402 - after the sys.path line above
from use_model import load_model, predict, ring_radii_kpc  # noqa: E402

DATA_DIR = REPO_ROOT / "examples" / "demo_data"
RUN_NAME = "demo"
DATASETS = {
    "target": "rotational_velocity",
    "augment": "major_axis",
    "inclinations": [45],
    "radial_projection": "sky",
}
N_RINGS = 10  # Rings in each rotation curve
# fainter pixels are left out of the velocity map panels but still used for training
DETECTION = 1e20  # cm^-2: about 5 sigma for VOF's demo noise;
# km/s: color range of the residual velocity map, in which gas-poor pixels with true velocity near zero saturate
RESIDUAL_LIMIT = 50
SPLITS = {
    "training": ["vflat120", "vflat160", "vflat240", "vflat280"],
    "validation": ["vflat180"],
    "testing": ["vflat220"],
}
SNAPSHOT = 600
# Each mode's config file, made small enough to train in a few minutes on a CPU
HYPERPARAM_OVERRIDES = {
    "n_filt0": 4,
    "k0": 5,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
}
RESBLOCK_OVERRIDES = {
    "n_fc": 64,
    "dropout_rate": 0.0,
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty
}
# The loss curve's title, axis label and scale for each network's training objective
NLL_LOSS = ("Gaussian NLL", "NLL (training objective)", "linear")  # can be < 0
DENOISING_LOSS = ("Denoising MSE", "MSE of the predicted noise", "log")
# What the network predicts: the rotation curve directly, or the velocity map, binned into the curve afterwards
MODES = {
    "curve": dict(
        run_name=RUN_NAME,
        radial_bins=N_RINGS,
        config="hyperparams_resblock_rc.toml",
        hyperparams={**RESBLOCK_OVERRIDES, "output_head": "fc"},
        epochs=200,
        loss=NLL_LOSS,
    ),
    # Weighting the loss by mass, as the rotation curve weights each pixel, keeps it off pixels with little gas,
    # whose velocity is noise
    "map": dict(
        run_name=f"{RUN_NAME}_map",
        radial_bins=0,
        config="hyperparams_resblock_rc.toml",
        hyperparams={**RESBLOCK_OVERRIDES, "output_head": "conv", "mass_weight_power": 1},
        epochs=200,
        loss=NLL_LOSS,
    ),
    # Each prediction takes n_samples * n_timesteps denoiser passes per cube, so the training and validation sets
    # are predicted only every predict_every epochs
    "ddpm": dict(
        run_name=f"{RUN_NAME}_ddpm",
        radial_bins=0,
        config="hyperparams_ddpm.toml",
        hyperparams={"n_timesteps": 50, "n_samples": 5, "predict_every": 50},
        epochs=200,
        loss=DENOISING_LOSS,
    ),
}


def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings, numbers and lists)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(output, epochs, predict):
    """Write the demo's hyperparameters and datasets files into output, for predict ("curve" or "map"); returns
    their paths."""
    mode = MODES[predict]
    with open(REPO_ROOT / "configs" / mode["config"], "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAM_OVERRIDES, **mode["hyperparams"], "epochs": epochs}
    hyperparams_file = output / "hyperparams.toml"
    hyperparams_file.write_text("\n".join(toml_lines(hyperparams)) + "\n")

    # Relative to the datasets file, so the copy saved in the checkpoint doesn't hold this computer's paths
    lines = toml_lines({**DATASETS, "radial_bins": mode["radial_bins"], "base_dir": os.path.relpath(DATA_DIR, output)})
    for split, galaxies in SPLITS.items():
        lines += [f"[{split}]"] + [f'"{galaxy}" = {SNAPSHOT}' for galaxy in galaxies]
    datasets_file = output / "datasets.toml"
    datasets_file.write_text("\n".join(lines) + "\n")
    return hyperparams_file, datasets_file


def train(output, epochs, predict):
    """Run train_model_unet.py on the demo data, exactly as you would from the command line; returns the network's
    path."""
    run_name = MODES[predict]["run_name"]
    hyperparams_file, datasets_file = write_configs(output, epochs, predict)
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "train_model_unet.py",
            "--hyperparams",
            hyperparams_file,
            "--datasets",
            datasets_file,
            "--run-name",
            run_name,
            "--network-dir",
            output / "NetworkWeights",
            "--output-dir",
            output / "DiagnosticOutputs",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output / "NetworkWeights" / run_name


def binned_curve(path, prediction, sigma):
    """Bin a predicted velocity map of the image at path, and its 1-sigma map (or None), into the rotation curve
    and its 1-sigma, weighting each pixel by its mass as radial_profile does for the true curve. The ring 1-sigma
    assumes the pixels' errors are independent."""
    with h5py.File(path) as f:
        curve = radial_profile(f, DATASETS["target"], N_RINGS, values=prediction)
        if sigma is None:
            return curve, None
        # Variance of a mass-weighted mean, sum(m^2 sigma^2) / sum(m)^2: the mass-weighted mean of m sigma^2,
        # divided by the ring's mass
        mass = f["mass_annotation"][()]
        weighted_variance = radial_profile(f, DATASETS["target"], N_RINGS, values=mass * sigma**2)
        ring_mass = radial_profile(f, "mass", N_RINGS)
    ring_variance = np.divide(weighted_variance, ring_mass, out=np.zeros(N_RINGS), where=ring_mass > 0)
    return curve, np.sqrt(ring_variance)


def plot(model_path, output, figure_file, loss):
    """Plot a test image's velocity maps, its true and predicted rotation curves (with their predicted 1-sigma) at
    every position angle, and the loss curves (loss: the training objective's title, axis label and scale); for a
    network that predicts maps, also the test image's true and predicted velocity maps and its predicted 1-sigma
    map. Returns (the RMS difference between the predicted and true
    curves in km/s, the fractions of rings whose prediction is within its predicted 1-sigma and 3-sigma of the
    truth)."""
    model, metadata = load_model(str(model_path))
    predicts_maps = "radial_bins" not in metadata  # Networks that predict curves record their number of rings
    test_files = sorted(
        (str(p) for p in (DATA_DIR / SPLITS["testing"][0]).rglob("*.hdf5")),
        key=lambda path: float(VOF_VIEW.search(Path(path).name)["position_angle"]),
    )
    predictions, maps_predicted = {}, {}
    for path, (name, pred, std) in zip(test_files, predict(model, metadata, test_files, return_std=True)):
        if predicts_maps:
            maps_predicted[name] = (pred, std)
            predictions[name] = binned_curve(path, pred, std)
        else:
            predictions[name] = (pred, std)
    radii = ring_radii_kpc(test_files[0], N_RINGS)
    with h5py.File(test_files[0]) as f:
        cube = f["spectra"][()]  # (y, x, velocity channel), in cm^-2 per channel
        channel_width = f.attrs["dnu_kmps"]
        half_width = f.attrs["fov_kpc"] / 2
        true_map = f[TARGETS[DATASETS["target"]][-1]][()]
    run_name = metadata["run_name"]
    with h5py.File(output / "DiagnosticOutputs" / "diagnostics" / f"{run_name}_LossesAndAccuracies.hdf5") as f:
        training_losses = f["training_losses"][()]  # The training objective
        validation_losses = f["validation_losses"][()]
        training_mses = f["training_mses"][()]  # MSE of the predicted mean, NaN on epochs that aren't predicted
        validation_mses = f["validation_mses"][()]

    velocities = (np.arange(cube.shape[2]) - (cube.shape[2] - 1) / 2) * channel_width
    moment0 = cube.sum(axis=2)  # total HI column density
    moment1 = (cube * velocities).sum(axis=2) / moment0
    undetected = moment0 < DETECTION
    extent = [-half_width, half_width, -half_width, half_width]
    maps = [
        (
            "Input: HI column density [cm$^{-2}$]",
            np.clip(moment0, 1e19, None),
            {"norm": LogNorm(1e19), "cmap": "inferno"},
        ),
        ("Input: line-of-sight velocity [km/s]", moment1, {"vmin": -200, "vmax": 200, "cmap": "RdBu_r"}),
    ]

    # Inputs and the rotation curve on top; the residuals and both loss curves below; for a network that predicts
    # maps, the first test image's true and predicted velocity maps, their residual and the predicted 1-sigma at the
    # bottom
    n_rows = 3 if predicts_maps else 2
    fig = plt.figure(figsize=(14.25, 4.6 * n_rows), layout="constrained")
    if predicts_maps:
        curves_figure, maps_figure = fig.subfigures(2, 1, height_ratios=[2, 1])
    else:
        curves_figure = fig
    (ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss) = curves_figure.subplots(2, 3)
    map_axes = [ax_density, ax_velocity]
    if predicts_maps:
        predicted_map, sigma_map = maps_predicted[Path(test_files[0]).stem]
        residual_map = np.ma.masked_where(undetected, predicted_map - true_map)
        velocity_style = {"vmin": 0, "vmax": 300, "cmap": "magma"}
        maps += [
            ("True rotational velocity [km/s]", np.ma.masked_where(undetected, true_map), velocity_style),
            ("Predicted rotational velocity [km/s]", np.ma.masked_where(undetected, predicted_map), velocity_style),
            (
                "Predicted − true [km/s]",
                residual_map,
                {"vmin": -RESIDUAL_LIMIT, "vmax": RESIDUAL_LIMIT, "cmap": "RdBu_r"},
            ),
            ("Predicted 1σ [km/s]", np.ma.masked_where(undetected, sigma_map), {"cmap": "viridis"}),
        ]
        map_axes += list(maps_figure.subplots(1, 4))
    for ax, (label, image, style) in zip(map_axes, maps):
        im = ax.imshow(image, origin="lower", extent=extent, **style)
        ax.figure.colorbar(im, ax=ax, location="top", shrink=0.9, pad=0.02).set_label(label)
        ax.set(xlabel="x [kpc]", ylabel="y [kpc]")

    residuals, sigmas = [], []
    for i, path in enumerate(test_files):
        with h5py.File(path) as f:
            truth = radial_profile(f, DATASETS["target"], N_RINGS)  # The true rotation curve
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
        (ax_mse, ax_loss),
        [
            (training_mses, validation_mses, "MSE of the mean", "MSE (scaled labels)", "log"),
            (training_losses, validation_losses, *loss),
        ],
    ):
        predicted = np.isfinite(training)
        marker = "o"
        ax.plot(epochs[predicted], training[predicted], marker=marker, label="Training")
        ax.plot(epochs[predicted], validation[predicted], marker=marker, label="Validation")
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
    parser.add_argument("--epochs", type=int, help="Default: the mode's epochs in MODES")
    parser.add_argument(
        "--predict",
        choices=sorted(MODES),
        default="curve",
        help="Train the ResBlock network on the rotation curve, or on the velocity map binned into it (map), or the "
        "diffusion model on the velocity map (ddpm). Default: %(default)s",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    mode = MODES[args.predict]
    epochs = mode["epochs"] if args.epochs is None else args.epochs
    model_path = train(output, epochs, args.predict)
    figure_file = output / f"{mode['run_name']}.png"
    rms, within_sigma, within_3sigma = plot(model_path, output, figure_file, mode["loss"])
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201


if __name__ == "__main__":
    main()
