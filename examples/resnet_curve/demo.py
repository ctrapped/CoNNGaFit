"""Demo: train the ResBlock U-Net to predict a galaxy's rotation curve from its datacube, then predict one it hasn't
seen.

Needs no simulation data and runs in a few minutes on a laptop CPU. From the repository root:

    python examples/resnet_curve/demo.py         # writes examples/resnet_curve/output/demo_resnet_curve.png
    python examples/resnet_curve/demo.py --epochs 10 --output DIR

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic disk
galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test on. The network
learns each galaxy's rotation curve (its rotation speed in 10 rings out to the image edge) from the datacube's
spectra, along with each ring's 1-sigma uncertainty (predict_variance).
"""

import argparse
import sys
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXAMPLES_DIR), str(EXAMPLES_DIR.parent)]  # The shared demo modules, and CoNNGaFit's

from conngafit_datasets import ring_radii  # noqa: E402 - after sys.path
from demo_common import (  # noqa: E402
    COLUMN_DENSITY_LIMITS,
    DATA_DIR,
    DATASETS,
    LOS_VELOCITY_LIMITS,
    REPO_ROOT,
    SNAPSHOT,
    SPLITS,
    curve_errors,
    load_history,
    load_images,
    moment_maps,
    testing_images,
    train,
    write_configs,
)
from demo_plots import demo_figure, plot_curves, plot_history, plot_inputs  # noqa: E402
from use_model import load_model, predict  # noqa: E402

RUN_NAME = "demo_resnet_curve"
CURVE_TARGET = "rotational_velocity"  # The rotation curve the demo is checked against
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
    "output_head": "fc",
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

    # Train on the rotation curve
    with open(CONFIG, "rb") as f:
        hyperparams = {**tomllib.load(f), **HYPERPARAMS, "epochs": args.epochs}
    datasets = {**DATASETS, "target": CURVE_TARGET, "radial_bins": N_RINGS}
    config_files = write_configs(output, hyperparams, datasets, SPLITS, SNAPSHOT, DATA_DIR)
    model_path = train(output, RUN_NAME, *config_files)

    # Predict the test galaxy's rotation curve, and its 1-sigma, at every position angle
    paths, position_angles = testing_images(DATA_DIR, SPLITS["testing"][0])
    images = load_images(paths, CURVE_TARGET, [], N_RINGS)
    model, metadata = load_model(str(model_path))
    predictions = list(predict(model, metadata, paths, return_std=True))
    curves = np.array([means[0] for _, means, _ in predictions])  # The network predicts a single target
    sigmas = np.array([stds[0] for _, _, stds in predictions])
    history = load_history(output / "DiagnosticOutputs" / "diagnostics" / f"{RUN_NAME}_LossesAndAccuracies.hdf5")

    # The first test image's input maps, every position angle's curves, and the training histories
    fig, ((ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss)), _ = demo_figure()
    column_density, velocity = moment_maps(images["cubes"][0], images["channel_widths"][0])
    half_width = images["fovs_kpc"][0] / 2
    extent = [-half_width, half_width, -half_width, half_width]
    plot_inputs((ax_density, ax_velocity), column_density, velocity, extent, COLUMN_DENSITY_LIMITS, LOS_VELOCITY_LIMITS)
    radii = ring_radii(images["fovs_kpc"][0], N_RINGS)
    plot_curves(ax_curve, ax_residual, radii, images["true_curves"], curves, sigmas, position_angles)
    plot_history(
        ax_mse, history["training_mses"], history["validation_mses"], "MSE of the mean", "MSE (scaled labels)", "log"
    )
    plot_history(ax_loss, history["training_losses"], history["validation_losses"], *LOSS)
    figure_file = output / f"{RUN_NAME}.png"
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)

    rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
    print(f"Model:  {model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201


if __name__ == "__main__":
    main()
