"""Train a CoNNGaFit network on synthetic VOF observations, then predict a galaxy it hasn't seen.
    python examples/demo.py                 # writes examples/demo_output/demo.png
    python examples/demo.py --predict map   # writes examples/demo_output/demo_map.png
    python examples/demo.py --epochs 10 --output DIR
    python examples/demo.py --diagnostics   # also saves the training's snapshots and diagnostic plots

The script walks through training a network:
    Step 1: Set the network and its hyperparameters
    Step 2: Set the datasets (training, validation, test)
    Step 3: Run the training with conngafit.train_model.train
    Step 4: Predicts the test galaxy and plot the results.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from conngafit.datasets import find_vof_images, label_map, radial_profile, ring_index
from conngafit.logging_setup import configure_logging
from conngafit.plotting import save_test_map_summary, save_test_summary
from conngafit.train_model import train
from conngafit.use_model import load_model, predict
from demo_utils import curve_errors, map_curve

DATA_DIR = Path("/N/project/choban_group/CONGA_project/VOF_outputs/low_res/FIRE-2")
OUTPUT_DIR = Path("/N/slate/cwtrapp/connga_jobs/fire2_lowres_test")

RUN_NAME = "fire2_lowres_test"
N_RINGS = 15  # Rings in each rotation curve
# cm^-2: pixels with a lower HI column density are left out of the map figure's statistics and hidden on its maps.
# Set it for this data's noise
DETECTION = 5 * 4e19

#### Step 1: Choose a network and set its hyperparameters ####
# Every key is required (see configs/hyperparams_resblock_rc.toml for more on each).
HYPERPARAMS = {
    "network": "resblock",  # The residual U-Net in conngafit/models/unet3d_resblock.py
    "n_filt0": 6,  # Filters in the first encoder stage; doubles at each later stage
    "k0": 9,  # Kernel size of the first convolution
    "k1": 3,  # Kernel size of the residual blocks
    "output_head": "fc",  # A fully connected head, for radial curves and small images
    "n_fc": 256,  # Width of the fully connected head
    "dropout_rate": 0.3,  # Dropout in the output head
    "block_dropout_rate": 0.0,  # Channel-wise dropout in every residual block
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty, trained with a Gaussian NLL
    "label_scaling": "one-sided",  # Scale the labels' [min, max] to [-1, 1]
    "mass_weight_power": 0.5,  # Weight each value's loss by mass to this power; 0 is unweighted
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "ema_decay": 0.99,  # Exponential moving average (EMA) of the weights for evaluation; 0 is no EMA
    "max_grad_norm": 40,  # Clip only gradient spikes: about 2-3x the typical norm (5-20 at this learning rate)
    "epochs": 30,
    "predict_every": 5,  # Score the training and validation predictions every N epochs
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
        "core_md/m12b_res7100": [500, 600],
        "core_md/m12_elvis_RomeoJuliet_r3500": [500, 600],
        "core_md/m12_elvis_RomulusRemus_r4000": [500, 600],
        "core_md/m12_elvis_ThelmaLouise_r4000": [500, 600],
        "core_md/m12m_res7100": [500, 600],
        "core_md/m12i_res7100": [500, 600],
        "core_md/m12r_res7100": [500, 600],
        "core_md/m12f_res7100": [500, 600],
        "cosmic_ray/m12b_res7100": [500, 600],
        "cosmic_ray/m12f_res7100": [500, 600],
        "cosmic_ray/m12i_res7100": [500, 600],
        "cosmic_ray/m12m_res7100": [500, 600],
        "cosmic_ray/m12z_res4200": [500, 600],
        "mhd/m12f_res7100": [500, 600],
        "mhd/m12i_res7100": [500, 600],
        "mhd/m12m_res7100": [500, 600],
    },
    "validation": {"core_md/m12w_res7100": [500, 600]},
    "testing": {"core_md/m12c_res7100": [500, 600]},
}

# --predict map: changes to steps 1 and 2 to predict the rotational velocity map
MAP_HYPERPARAMS = {"output_head": "conv", "mass_weight_power": 1}  # change output head and add mass weighting
MAP_DATASETS = {"radial_bins": 0}  # predict a map


class Cubes:
    """Test datacubes, each read only when indexed to limit memory usage."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with h5py.File(self.paths[i], "r") as f:
            return f["spectra"][()]


def load_test_images(paths, target, n_rings, detection):
    """Like demo_utils.load_images, without keeping every cube in memory: each image's field of view in kpc, true
    curve of target in n_rings rings, true mass and target maps ("true_maps", (n_images, 2, nY, nX)) and detected
    pixels (column density above detection, in cm^-2), plus "cubes", which reads each cube when indexed."""
    images = {"fovs_kpc": [], "true_curves": [], "true_maps": [], "detected": []}
    for path in paths:
        with h5py.File(path, "r") as f:
            images["fovs_kpc"].append(f.attrs["fov_kpc"])
            images["true_curves"].append(radial_profile(f, target, n_rings))
            images["true_maps"].append([label_map(f, "mass"), label_map(f, target)])
            images["detected"].append(f["spectra"][()].sum(axis=2) > detection)
    images = {key: np.array(values) for key, values in images.items()}
    images["cubes"] = Cubes(paths)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predict",
        choices=["curve", "map"],
        default="curve",
        help="Train on the rotation curve, or on the velocity map binned into it (map). Default: %(default)s",
    )
    parser.add_argument("--epochs", type=int, default=HYPERPARAMS["epochs"], help="Default: %(default)s")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also save the training's final-epoch snapshots and diagnostic plots in OUTPUT/DiagnosticOutputs",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=5,
        help="Every this many epochs, save comparison images of a few training and validation images to "
        "OUTPUT/DiagnosticOutputs/images; 0 saves none. Default: %(default)s",
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
        batch_size=32,
        test_batch_size=None,
        snapshot_every=args.snapshot_every,
        snapshot_examples=6,  # Training and validation images in each snapshot, the same ones every time
    )

    #### Step 4: predict the test galaxy, which the network hasn't seen, and plot ####
    target = datasets["target"]
    # The testing set's images, chosen as training chooses them: its snapshots, at the inclinations used
    testing = {str(DATA_DIR / galaxy): snapshots for galaxy, snapshots in datasets["testing"].items()}
    paths = find_vof_images(testing, datasets["inclinations"])
    names = [Path(path).stem.removesuffix("_image_fullSpectra") for path in paths]
    images = load_test_images(paths, target, N_RINGS, DETECTION)  # With their true curves and maps
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
        map_figure = output / f"{run_name}_test_maps.png"
        save_test_map_summary(
            map_figure, images, names, true_maps, means, stds, target, true_mass, images["detected"], N_RINGS
        )
        print(f"Map figure: {map_figure}")  # noqa: T201
    else:
        curves, sigmas = means, stds

    figure_file = output / f"{run_name}_test_curves.png"
    save_test_summary(figure_file, images, names, curves, sigmas, result, target)
    rms, within_sigma, within_3sigma = curve_errors(images["true_curves"], curves, sigmas)
    print(f"Model:  {result.model_path}.pt\nFigure: {figure_file}\nRotation curve RMS error: {rms:.1f} km/s")  # noqa: T201
    print(f"Within predicted 1σ: {within_sigma:.0%} of rings ({within_3sigma:.0%} within 3σ)")  # noqa: T201


if __name__ == "__main__":
    main()
