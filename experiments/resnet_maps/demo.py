"""Experiment: train the ResBlock U-Net to predict a galaxy's mass and rotational momentum maps from its datacube, then
predict one it hasn't seen and derive its rotation curve.

Needs no simulation data and runs in a few minutes on a laptop CPU. From the repository root:

    python experiments/resnet_maps/demo.py          # writes experiments/resnet_maps/output/resnet_map.png
    python experiments/resnet_maps/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic disk
galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test on. The network
learns the mass and rotational momentum maps from the datacube's spectra, along with each pixel's 1-sigma
(predict_variance). The rotation curve is each ring's total momentum divided by its total mass, with the maps'
1-sigma propagated into it.
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

RUN_NAME = "resnet_map"
CURVE_TARGET = "rotational_velocity"  # The rotation curve the experiment is checked against
# The maps the experiment predicts; the rotation curve is their ratio, summed in rings
MAP_TARGETS = ["mass", "rotational_momentum"]
N_RINGS = 10  # Rings in the rotation curve
CONFIG = REPO_ROOT / "configs" / "hyperparams_resblock_rc.toml"
HYPERPARAMS = {
    # Small enough to train in a few minutes on a CPU
    "n_filt0": 4,
    "k0": 5,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
    "n_fc": 64,
    "dropout_rate": 0.0,
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty
    "output_head": "conv",
    # Unweighted loss: the mass and momentum maps, like the rotation curve's sums, are dominated by massive pixels
    "mass_weight_power": 0,
}
EPOCHS = 200
# The training objective's title, axis label and scale: the Gaussian NLL can be < 0
LOSS = ("Gaussian NLL", "NLL (training objective)", "linear")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Train on the mass and rotational momentum maps
    with open(CONFIG, "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAMS, "epochs": args.epochs}
    datasets = {**DATASETS, "target": MAP_TARGETS, "radial_bins": 0}
    config_files = write_configs(output, hyperparams, datasets, SPLITS, SNAPSHOT, DATA_DIR)
    model_path = train(output, RUN_NAME, *config_files)

    # Predict the test galaxy's maps, and their 1-sigma, at every position angle
    paths, position_angles = testing_images(DATA_DIR, SPLITS["testing"][0])
    images = load_images(paths, CURVE_TARGET, MAP_TARGETS, N_RINGS)
    model, metadata = load_model(str(model_path))
    predictions = list(predict(model, metadata, paths, return_std=True))
    maps = np.array([means for _, means, _ in predictions])  # (n_images, n_targets, nY, nX)
    map_sigmas = np.array([stds for _, _, stds in predictions])
    history = load_history(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5")

    # Each image's rotation curve from its predicted maps, in MAP_TARGETS order
    fov_kpc = images["fovs_kpc"][0]
    ring = ring_index(images["true_maps"].shape[2:], fov_kpc, N_RINGS)
    curves, sigmas = [], []
    for (mass, momentum), (mass_sigma, momentum_sigma) in zip(maps, map_sigmas):
        curve, sigma = map_curve(ring, mass, momentum, N_RINGS, mass_sigma**2, momentum_sigma**2)
        curves.append(curve)
        sigmas.append(sigma)
    curves, sigmas = np.array(curves), np.array(sigmas)

    # The first test image's input and predicted maps, every position angle's curves, and the training histories
    fig, ((ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss)), map_axes = demo_figure(len(MAP_TARGETS))
    column_density, velocity = moment_maps(images["cubes"][0], images["channel_widths"][0])
    extent = [-fov_kpc / 2, fov_kpc / 2, -fov_kpc / 2, fov_kpc / 2]
    plot_inputs((ax_density, ax_velocity), column_density, velocity, extent, COLUMN_DENSITY_LIMITS, LOS_VELOCITY_LIMITS)
    radii = ring_radii(fov_kpc, N_RINGS)
    plot_curves(ax_curve, ax_residual, radii, images["true_curves"], curves, sigmas, position_angles)
    plot_history(
        ax_mse, history["training_mses"], history["validation_mses"], "MSE of the mean", "MSE (scaled labels)", "log"
    )
    plot_history(ax_loss, history["training_losses"], history["validation_losses"], *LOSS)
    undetected = column_density < DETECTION
    for row, target, true_map, predicted_map, sigma_map in zip(
        map_axes, MAP_TARGETS, images["true_maps"][0], maps[0], map_sigmas[0]
    ):
        label = f"{TARGET_SETTINGS[target][0]} {TARGET_SETTINGS[target][1]}"
        plot_map(row, label, true_map, predicted_map, sigma_map, undetected, extent, SNR_LIMITS, **MAP_PANELS[target])
    figure_file = output / f"{RUN_NAME}.png"
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)

    rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201
    for target, error in zip(MAP_TARGETS, map_errors(images["true_maps"], maps)):
        print(f"{target} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
