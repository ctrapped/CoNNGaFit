"""Experiment: run a trained mass and rotational momentum map network on the demo data's test galaxy, with no training.

Runs the network trained by experiments/resnet_maps/demo.py in a few seconds. From the repository root:

    python experiments/resnet_maps/demo_inference.py   # writes experiments/resnet_maps/output/inference/
    python experiments/resnet_maps/demo_inference.py --model-path DIR/NetworkWeights/resnet_map

For each datacube, use_model.run_inferences writes each predicted map and its 1-sigma as a plot and an hdf5 file,
exactly as it would for a real observation converted with fits_to_conngafit.py. The rotation curve is then each
ring's total momentum divided by its total mass, its 1-sigma assuming independent errors in every pixel.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS_DIR))  # demo_common and demo_plots

from conngafit.datasets import ring_index  # noqa: E402
from conngafit.logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from conngafit.use_model import run_inferences  # noqa: E402
from demo import CURVE_TARGET, MAP_TARGETS, N_RINGS, RUN_NAME  # noqa: E402 - this folder's demo.py
from demo_common import (  # noqa: E402
    DATA_DIR,
    SPLITS,
    curve_errors,
    load_images,
    map_curve,
    map_errors,
    testing_images,
)

DEMO_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEMO_DIR / "output" / "NetworkWeights" / RUN_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    parser.add_argument("--output", type=Path, default=DEMO_DIR / "output" / "inference")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    paths, _ = testing_images(DATA_DIR, SPLITS["testing"][0])
    run_inferences(str(args.model_path), paths, str(args.output) + "/")

    # The saved maps, and the truths the synthetic galaxies come with, which a real observation wouldn't
    maps, map_sigmas = [], []
    for path in paths:
        image_maps, image_sigmas = [], []
        for target in MAP_TARGETS:
            with h5py.File(args.output / f"{Path(path).stem}_{target}_projectionMap.hdf5") as f:
                image_maps.append(f["image"][()])
                image_sigmas.append(f["uncertainty"][()])
        maps.append(image_maps)
        map_sigmas.append(image_sigmas)
    maps, map_sigmas = np.array(maps), np.array(map_sigmas)  # (n_images, n_targets, nY, nX)
    images = load_images(paths, CURVE_TARGET, MAP_TARGETS, N_RINGS)

    # Each image's rotation curve from its predicted maps, in MAP_TARGETS order
    ring = ring_index(maps.shape[2:], images["fovs_kpc"][0], N_RINGS)
    for i, path in enumerate(paths):
        (mass, momentum), (mass_sigma, momentum_sigma) = maps[i], map_sigmas[i]
        curve, sigma = map_curve(ring, mass, momentum, N_RINGS, mass_sigma**2, momentum_sigma**2)
        rms, within_sigma, _ = curve_errors(images["true_curves"][i], curve, sigma)
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    for target, error in zip(MAP_TARGETS, map_errors(images["true_maps"], maps)):
        print(f"{target} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
