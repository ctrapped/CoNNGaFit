"""Experiment: train the spectral 2-d ResNet to predict a galaxy's rotational velocity map from its datacube, then
predict one it hasn't seen and bin its map into the rotation curve.

Needs no simulation data and runs in a few minutes on a laptop CPU. From the repository root:

    python experiments/resnet_2d_velocity_map/demo.py    # writes output/resnet_2d_velocity.png in this folder
    python experiments/resnet_2d_velocity_map/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic disk
galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test on. The network
(resblock_2d: a spectral encoder, then a 2-d residual U-Net at full map resolution) learns the velocity map from the
datacube's spectra, along with each pixel's 1-sigma (predict_variance). The rotation curve is each ring's
mass-weighted mean velocity, weighted by the true mass map as the true curve is, with the map's 1-sigma propagated
into it assuming independent errors in every pixel.

Set TARGET to "mass" to learn the mass surface density map instead; it is checked against the true map only, with no
radial profile.
"""

import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
sys.path.append(str(EXAMPLES_DIR))  # demo_utils and demo_plots; after this folder, so its own demo.py is found

from conngafit.datasets import TARGET_SETTINGS, ring_index, ring_radii  # noqa: E402
from conngafit.use_model import load_model, predict  # noqa: E402
from demo_plots import demo_figure, plot_curves, plot_history, plot_inputs, plot_map  # noqa: E402
from demo_utils import (  # noqa: E402
    COLUMN_DENSITY_LIMITS,
    DATA_DIR,
    DATASETS,
    DETECTION,
    LOS_VELOCITY_LIMITS,
    MAP_PANELS,
    REPO_ROOT,
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
)

TARGET = "rotational_velocity"  # The map the experiment predicts: "rotational_velocity" or "mass"
CURVE_TARGET = "rotational_velocity"  # The target whose map is binned into the rotation curve
RUN_NAME = {"rotational_velocity": "resnet_2d_velocity", "mass": "resnet_2d_mass"}[TARGET]
N_RINGS = 10  # Rings in the rotation curve
CONFIG = REPO_ROOT / "configs" / "hyperparams_resblock_2d.toml"
HYPERPARAMS = {
    # Small enough to train in a few minutes on a CPU
    "n_filt0": 4,
    "k0": 5,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
    "dropout_rate": 0.0,
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty
    # Unweighted loss, as in experiments/resnet_maps
    # "mass_weight_power": 0.5,
    # "max_grad_norm": 0,  # No gradient clipping, or use a ~10 clipping
    "ema_decay": 0.99,  # Exponential moving average (EMA) of the weights for evaluation; 0 is no EMA
}
EPOCHS = 200
# The training objective's title, axis label and scale: the Gaussian NLL can be < 0
LOSS = ("Gaussian NLL", "NLL (training objective)", "linear")


def velocity_curves(ring, true_mass, maps, map_sigmas):
    """Each image's rotation curve and its 1-sigma from its predicted velocity map and that map's per-pixel 1-sigma:
    each ring's momentum (the true mass times the velocity) divided by its mass, the mass-weighted mean velocity.
    true_mass, maps, map_sigmas: (n_images, nY, nX). Returns two (n_images, N_RINGS) arrays."""
    curves, sigmas = [], []
    for mass, velocity, velocity_sigma in zip(true_mass, maps, map_sigmas):
        mass_var = np.zeros_like(mass)  # The true mass map has no error
        curve, sigma = map_curve(ring, mass, mass * velocity, N_RINGS, mass_var, (mass * velocity_sigma) ** 2)
        curves.append(curve)
        sigmas.append(sigma)
    return np.array(curves), np.array(sigmas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Train on TARGET's map
    with open(CONFIG, "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAMS, "epochs": args.epochs}
    datasets = {**DATASETS, "target": TARGET, "radial_bins": 0}
    model_path = train(output, RUN_NAME, hyperparams, datasets)

    # Predict the test galaxy's TARGET map, and its 1-sigma, at every position angle
    paths, position_angles = testing_images(DATA_DIR, SPLITS["testing"][0])
    map_targets = list(dict.fromkeys(["mass", TARGET]))  # The true mass map, for the rotation curve, and TARGET's
    images = load_images(paths, CURVE_TARGET, map_targets, N_RINGS)
    true_mass, true_maps = images["true_maps"][:, :1], images["true_maps"][:, -1:]  # (n_images, 1, nY, nX)
    model, metadata = load_model(str(model_path))
    predictions = list(predict(model, metadata, paths, return_std=True))
    maps = np.array([means for _, means, _ in predictions])  # (n_images, 1, nY, nX)
    map_sigmas = np.array([stds for _, _, stds in predictions])
    history = load_history(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5")

    # The first test image's input and predicted maps, every position angle's curves, and the training histories
    fig, ((ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss)), map_axes = demo_figure(1)
    column_density, velocity = moment_maps(images["cubes"][0], images["channel_widths"][0])
    fov_kpc = images["fovs_kpc"][0]
    extent = [-fov_kpc / 2, fov_kpc / 2, -fov_kpc / 2, fov_kpc / 2]
    plot_inputs((ax_density, ax_velocity), column_density, velocity, extent, COLUMN_DENSITY_LIMITS, LOS_VELOCITY_LIMITS)
    if TARGET == CURVE_TARGET:
        ring = ring_index(images["true_maps"].shape[2:], fov_kpc, N_RINGS)
        curves, sigmas = velocity_curves(ring, true_mass[:, 0], maps[:, 0], map_sigmas[:, 0])
        radii = ring_radii(fov_kpc, N_RINGS)
        plot_curves(ax_curve, ax_residual, radii, images["true_curves"], curves, sigmas, position_angles)
    else:
        # No radial profile of the mass map
        ax_curve.set_axis_off()
        ax_residual.set_axis_off()
    plot_history(
        ax_mse, history["training_mses"], history["validation_mses"], "MSE of the mean", "MSE (scaled labels)", "log"
    )
    plot_history(ax_loss, history["training_losses"], history["validation_losses"], *LOSS)
    label = f"{TARGET_SETTINGS[TARGET][0]} {TARGET_SETTINGS[TARGET][1]}"
    undetected = column_density < DETECTION
    plot_map(
        map_axes[0],
        label,
        true_maps[0, 0],
        maps[0, 0],
        map_sigmas[0, 0],
        undetected,
        extent,
        SNR_LIMITS,
        **MAP_PANELS[TARGET],
    )
    figure_file = output / f"{RUN_NAME}.png"
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"Model:  {model_path}.pt\nFigure: {figure_file}")  # noqa: T201
    if TARGET == CURVE_TARGET:
        rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
        print(f"Rotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
        print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201
    [error] = map_errors(true_maps, maps)
    print(f"{TARGET} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
