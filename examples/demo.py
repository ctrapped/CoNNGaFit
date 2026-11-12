"""Demo: train a small CoNNGaFit network on synthetic VOF observations, then predict a galaxy it hasn't seen.

Needs no simulation data and runs in a few minutes on a laptop CPU. From the repository root:

    python examples/demo.py                 # writes examples/demo_output/demo.png
    python examples/demo.py --predict map   # writes examples/demo_output/demo_map.png
    python examples/demo.py --epochs 10 --output DIR
    python examples/demo.py --diagnostics   # also saves the training's snapshots and diagnostic plots

The data (examples/demo_data/, made by examples/make_demo_data.py) are VOF observations of six synthetic
disk galaxies, each rotating at a different speed: four to train on, one to validate on, and one to test
on. The network learns each galaxy's rotation curve (its rotation speed in 10 rings out to the image edge)
from the datacube's spectra, along with each ring's 1-sigma uncertainty (predict_variance). With --predict map
it learns the rotational velocity map instead, with each pixel's 1-sigma, and the map is binned into the
rotation curve afterwards.

The script walks through training a network:
    Step 1: Set the network and its hyperparameters
    Step 2: Set the datasets (training, validation, test)
    Step 3: Run the training with conngafit.train_model.train
    Step 4: Predicts the test galaxy and plot the results.
"""

import argparse
from pathlib import Path

import numpy as np

from conngafit.datasets import ring_index
from conngafit.logging_setup import configure_logging
from conngafit.train_model import train
from conngafit.use_model import load_model, predict
from demo_plots import save_demo_figure
from demo_utils import curve_errors, load_images, map_curve, testing_images

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "examples" / "demo_data"
RUN_NAME = "demo"
N_RINGS = 10  # Rings in each rotation curve

#### Step 1: Choose a network and set its hyperparameters ####
# Every key is required (see configs/hyperparams_resblock_rc.toml for more on each). These are small enough to train
# in a few minutes on a CPU.
HYPERPARAMS = {
    "network": "resblock",  # The residual U-Net in conngafit/models/unet3d_resblock.py
    "n_filt0": 4,  # Filters in the first encoder stage; doubles at each later stage
    "k0": 5,  # Kernel size of the first convolution
    "k1": 3,  # Kernel size of the residual blocks
    "output_head": "fc",  # A fully connected head, for radial curves and small images
    "n_fc": 64,  # Width of the fully connected head
    "dropout_rate": 0.0,  # Dropout in the output head
    "block_dropout_rate": 0.0,  # Channel-wise dropout in every residual block
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty, trained with a Gaussian NLL
    "label_scaling": "one-sided",  # Scale the labels' [min, max] to [-1, 1]
    "mass_weight_power": 0,  # Weight each value's loss by mass to this power; 0 is unweighted
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 200,
    "predict_every": 1,  # Score the training and validation predictions every epoch
    "seed": 0,  # Seeds weight initialization, dropout and batch order, so runs are repeatable
}

#### Step 2: Set the datasets ####
# What the network predicts, and which galaxies it trains, validates and tests on. The demo data are VOF
# observations of six synthetic galaxies, each rotating at a different speed, at snapshot 600.
DATASETS = {
    "target": "rotational_velocity",
    "radial_bins": N_RINGS,  # Predict the rotation curve in N_RINGS rings; 0 predicts a map
    "radial_projection": "sky",  # Rings in the plane of the sky
    "inclinations": [45],  # The inclinations to use, in degrees
    "augment": "major_axis",  # Add rotated and flipped copies of each training image
    "data_dir": DATA_DIR,  # The galaxies' folders below are in here
    "training": {
        "vflat120": 600,
        "vflat160": 600,
        "vflat240": 600,
        "vflat280": 600,
    },  # the galaxy name and their snapshot number(s)
    "validation": {"vflat180": 600},
    "testing": {"vflat220": 600},
}

# --predict map: changes to steps 1 and 2 to predict the rotational velocity map
MAP_HYPERPARAMS = {"output_head": "conv", "mass_weight_power": 1}  # change output head and add mass weighting
MAP_DATASETS = {"radial_bins": 0}  # predict a map


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predict",
        choices=["curve", "map"],
        default="curve",
        help="Train on the rotation curve, or on the velocity map binned into it (map). Default: %(default)s",
    )
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["epochs"], help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also save the training's final-epoch snapshots and diagnostic plots in OUTPUT/DiagnosticOutputs",
    )
    args = parser.parse_args()
    configure_logging()  # Show the training's progress
    output = args.output.resolve()

    # Steps 1 and 2, set hyperparameters and datasets to be used
    hyperparams = {**HYPERPARAMS, "epochs": args.epochs}
    datasets = dict(DATASETS)
    run_name = RUN_NAME
    if args.predict == "map":
        hyperparams.update(MAP_HYPERPARAMS)
        datasets.update(MAP_DATASETS)
        run_name = f"{RUN_NAME}_map"

    #### Step 3: train ####
    # Saves the network to OUTPUT/NetworkWeights/<run_name>.pt and .hdf5, and its loss history to
    # OUTPUT/DiagnosticOutputs/diagnostics/; returns the network's path and the loss history
    result = train(
        hyperparams,
        datasets,
        run_name=run_name,
        network_dir=output / "NetworkWeights",
        output_dir=output / "DiagnosticOutputs",
        save_diagnostics=args.diagnostics,
    )

    #### Step 4: predict the test galaxy, which the network hasn't seen, and plot ####
    target = datasets["target"]
    paths, position_angles = testing_images(DATA_DIR, next(iter(datasets["testing"])))
    images = load_images(paths, target, ["mass", target], N_RINGS)  # With their true curves and maps
    model, metadata = load_model(result.model_path)
    predictions = list(predict(model, metadata, paths, return_std=True))
    # (n_images, ...) predictions and their 1-sigma: the network predicts a single target
    means = np.array([image_means[0] for _, image_means, _ in predictions])
    stds = np.array([image_stds[0] for _, _, image_stds in predictions])

    if datasets["radial_bins"] == 0:
        # Bin each predicted velocity map into the rotation curve, weighting each pixel by its true mass as the true
        # curve is: mass * velocity is the momentum, and only the velocity carries an error
        true_mass, true_maps = images["true_maps"][:, 0], images["true_maps"][:, 1]
        ring = ring_index(true_mass.shape[1:], images["fovs_kpc"][0], N_RINGS)
        curves, sigmas = [], []
        for mass, velocity, velocity_sigma in zip(true_mass, means, stds):
            momentum_var = (mass * velocity_sigma) ** 2
            curve, sigma = map_curve(ring, mass, mass * velocity, N_RINGS, np.zeros_like(mass), momentum_var)
            curves.append(curve)
            sigmas.append(sigma)
        curves, sigmas = np.array(curves), np.array(sigmas)
        map_panels = {"true_maps": true_maps, "maps": means, "map_sigmas": stds}
    else:
        curves, sigmas, map_panels = means, stds, {}

    figure_file = output / f"{run_name}.png"
    save_demo_figure(figure_file, images, position_angles, curves, sigmas, result, target, **map_panels)
    rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
    print(f"Model:  {result.model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201


if __name__ == "__main__":
    main()
