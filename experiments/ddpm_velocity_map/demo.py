"""Experiment: train a conditional diffusion model (DDPM) to sample a galaxy's rotational velocity map from its posterior
given its datacube, then sample one it hasn't seen and bin its samples into the rotation curve.

Needs no simulation data and runs in several minutes on a laptop CPU. From the repository root:

    python experiments/ddpm_velocity_map/demo.py    # writes experiments/ddpm_velocity_map/output/ddpm_velocity.png
    python experiments/ddpm_velocity_map/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic disk
galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test on. The model
learns to denoise the velocity map, and the mean and spread of its samples give each pixel's value and 1-sigma. Each
sample is binned into a rotation curve, each ring's mass-weighted mean velocity weighted by the true mass map as the
true curve is, and the mean and spread of the samples' curves give the curve and its 1-sigma, including the
correlation between the pixels' errors.
"""

import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))  # demo_common and demo_plots

from conngafit.datasets import TARGET_SETTINGS, ring_index, ring_radii  # noqa: E402
from conngafit.use_model import load_model, predict  # noqa: E402
from demo_common import (  # noqa: E402
    COLUMN_DENSITY_LIMITS,
    DATA_DIR,
    DATASETS,
    DETECTION,
    LOS_VELOCITY_LIMITS,
    MAP_PANELS,
    REPO_ROOT,
    SNAPSHOT,
    SNR_LIMITS,
    SPLITS,
    curve_errors,
    load_history,
    load_images,
    map_curve,
    map_errors,
    moment_maps,
    testing_images,
    train,
    write_configs,
)
from demo_plots import demo_figure, plot_curves, plot_history, plot_inputs, plot_map  # noqa: E402

RUN_NAME = "ddpm_velocity"
CURVE_TARGET = "rotational_velocity"  # The map the experiment predicts, and the rotation curve it is checked against
N_RINGS = 10  # Rings in the rotation curve
CONFIG = REPO_ROOT / "configs" / "hyperparams_ddpm.toml"
HYPERPARAMS = {
    # Small enough to train in a few minutes on a CPU
    "n_filt0": 4,
    "k0": 5,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
    "n_timesteps": 50,
    "n_samples": 5,
    # Each prediction takes n_samples * n_timesteps denoiser passes per cube, so the training and validation sets
    # are predicted only every predict_every epochs
    "predict_every": 50,
}
EPOCHS = 200
# The training objective's title, axis label and scale
LOSS = ("Denoising MSE", "MSE of the predicted noise", "log")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Train on the rotational velocity map
    with open(CONFIG, "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAMS, "epochs": args.epochs}
    datasets = {**DATASETS, "target": CURVE_TARGET, "radial_bins": 0}
    config_files = write_configs(output, hyperparams, datasets, SPLITS, SNAPSHOT, DATA_DIR)
    model_path = train(output, RUN_NAME, *config_files)

    # Sample the test galaxy's velocity map from its posterior at every position angle
    paths, position_angles = testing_images(DATA_DIR, SPLITS["testing"][0])
    images = load_images(paths, CURVE_TARGET, ["mass", CURVE_TARGET], N_RINGS)
    true_mass, true_velocity = images["true_maps"][:, :1], images["true_maps"][:, 1:]  # (n_images, 1, nY, nX)
    model, metadata = load_model(str(model_path))
    samples = np.array([image_samples for _, image_samples in predict(model, metadata, paths, return_samples=True)])
    maps = samples.mean(axis=1)  # (n_images, 1, nY, nX)
    map_sigmas = samples.std(axis=1, ddof=1)
    history = load_history(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5")

    # Each sample's rotation curve: the true mass times its velocity is the momentum, so their ring totals' ratio is
    # the mass-weighted mean velocity. The mean and spread of the samples' curves give each image's curve
    fov_kpc = images["fovs_kpc"][0]
    ring = ring_index(images["true_maps"].shape[2:], fov_kpc, N_RINGS)
    sample_curves = []
    for (mass,), image_samples in zip(true_mass, samples):
        sample_curves.append([map_curve(ring, mass, mass * velocity, N_RINGS)[0] for (velocity,) in image_samples])
    sample_curves = np.array(sample_curves)  # (n_images, n_samples, n_rings)
    curves = sample_curves.mean(axis=1)
    sigmas = sample_curves.std(axis=1, ddof=1)

    # The first test image's input and predicted maps, every position angle's curves, and the training histories
    fig, ((ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss)), map_axes = demo_figure(1)
    column_density, velocity = moment_maps(images["cubes"][0], images["channel_widths"][0])
    extent = [-fov_kpc / 2, fov_kpc / 2, -fov_kpc / 2, fov_kpc / 2]
    plot_inputs((ax_density, ax_velocity), column_density, velocity, extent, COLUMN_DENSITY_LIMITS, LOS_VELOCITY_LIMITS)
    radii = ring_radii(fov_kpc, N_RINGS)
    plot_curves(ax_curve, ax_residual, radii, images["true_curves"], curves, sigmas, position_angles)
    plot_history(
        ax_mse, history["training_mses"], history["validation_mses"], "MSE of the mean", "MSE (scaled labels)", "log"
    )
    plot_history(ax_loss, history["training_losses"], history["validation_losses"], *LOSS)
    label = f"{TARGET_SETTINGS[CURVE_TARGET][0]} {TARGET_SETTINGS[CURVE_TARGET][1]}"
    undetected = column_density < DETECTION
    plot_map(
        map_axes[0],
        label,
        true_velocity[0, 0],
        maps[0, 0],
        map_sigmas[0, 0],
        undetected,
        extent,
        SNR_LIMITS,
        **MAP_PANELS[CURVE_TARGET],
    )
    figure_file = output / f"{RUN_NAME}.png"
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)

    rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201
    [error] = map_errors(true_velocity, maps)
    print(f"{CURVE_TARGET} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
