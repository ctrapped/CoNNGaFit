"""Experiment: run a trained spectral 2-d ResNet on the demo data's test galaxy, with no training.

Runs the network trained by experiments/resnet_2d_velocity_map/demo.py in a few seconds. From the repository root:

    python experiments/resnet_2d_velocity_map/demo_inference.py
    python experiments/resnet_2d_velocity_map/demo_inference.py --model-path DIR/NetworkWeights/resnet_2d_velocity

use_model.predict(..., return_std=True) gives demo.py's TARGET map and its per-pixel 1-sigma for each datacube, as it
would for a real observation converted with fits_to_conngafit.py. For the rotational velocity, the rotation curve is
each ring's mass-weighted mean velocity, weighted by the true mass map, its 1-sigma assuming independent errors in
every pixel. The mass map is checked against the true map only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
sys.path.append(str(EXAMPLES_DIR))  # demo_utils and demo_plots; after this folder, so its own demo.py is found

from conngafit.datasets import ring_index  # noqa: E402
from conngafit.logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from conngafit.use_model import load_model, predict  # noqa: E402
from demo import CURVE_TARGET, N_RINGS, RUN_NAME, TARGET, velocity_curves  # noqa: E402 - this folder's demo.py
from demo_utils import DATA_DIR, SPLITS, curve_errors, load_images, map_errors, testing_images  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "output" / "NetworkWeights" / RUN_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Every image's predicted map and 1-sigma, and the truths the synthetic galaxies come with (a real observation
    # wouldn't)
    paths, _ = testing_images(DATA_DIR, SPLITS["testing"][0])
    model, metadata = load_model(str(args.model_path))
    predictions = list(predict(model, metadata, paths, return_std=True))
    maps = np.array([means for _, means, _ in predictions])  # (n_images, 1, nY, nX)
    map_sigmas = np.array([stds for _, _, stds in predictions])
    map_targets = list(dict.fromkeys(["mass", TARGET]))  # The true mass map, for the rotation curve, and TARGET's
    images = load_images(paths, CURVE_TARGET, map_targets, N_RINGS)
    true_mass, true_maps = images["true_maps"][:, :1], images["true_maps"][:, -1:]  # (n_images, 1, nY, nX)

    if TARGET == CURVE_TARGET:
        ring = ring_index(maps.shape[2:], images["fovs_kpc"][0], N_RINGS)
        curves, sigmas = velocity_curves(ring, true_mass[:, 0], maps[:, 0], map_sigmas[:, 0])
        for path, true_curve, curve, sigma in zip(paths, images["true_curves"], curves, sigmas):
            rms, within_sigma, _ = curve_errors(true_curve, curve, sigma)
            print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    [error] = map_errors(true_maps, maps)
    print(f"{TARGET} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
