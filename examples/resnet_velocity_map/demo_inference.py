"""Demo: run a trained rotational velocity map network on the demo's test galaxy, with no training.

Runs the network trained by examples/resnet_velocity_map/demo.py in a few seconds. From the repository root:

    python examples/resnet_velocity_map/demo_inference.py   # writes examples/resnet_velocity_map/output/inference/
    python examples/resnet_velocity_map/demo_inference.py --model-path DIR/NetworkWeights/demo_resnet_velocity

For each datacube, use_model.run_inferences writes the predicted velocity map and its 1-sigma as a plot and an hdf5
file, exactly as it would for a real observation converted with fits_to_conngafit.py. The rotation curve is then
each ring's mass-weighted mean velocity, weighted by the true mass map, its 1-sigma assuming independent errors in
every pixel.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXAMPLES_DIR), str(EXAMPLES_DIR.parent)]  # The shared demo modules, and CoNNGaFit's

from conngafit_datasets import ring_index  # noqa: E402 - after sys.path
from demo import CURVE_TARGET, N_RINGS, RUN_NAME  # noqa: E402 - this folder's demo.py
from demo_common import (  # noqa: E402
    DATA_DIR,
    SPLITS,
    curve_errors,
    load_images,
    map_curve,
    map_errors,
    testing_images,
)
from logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from use_model import run_inferences  # noqa: E402

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

    # The saved velocity maps, and the truths the synthetic galaxies come with, which a real observation wouldn't
    maps, map_sigmas = [], []
    for path in paths:
        with h5py.File(args.output / f"{Path(path).stem}_{CURVE_TARGET}_projectionMap.hdf5") as f:
            maps.append(f["image"][()])
            map_sigmas.append(f["uncertainty"][()])
    maps, map_sigmas = np.array(maps), np.array(map_sigmas)  # (n_images, nY, nX)
    images = load_images(paths, CURVE_TARGET, ["mass", CURVE_TARGET], N_RINGS)
    true_mass, true_velocity = images["true_maps"][:, 0], images["true_maps"][:, 1]

    # Each image's rotation curve: the true mass times the predicted velocity is the momentum, so their ring totals'
    # ratio is the mass-weighted mean velocity, and only the velocity carries an error
    ring = ring_index(maps.shape[1:], images["fovs_kpc"][0], N_RINGS)
    for i, path in enumerate(paths):
        mass, velocity, velocity_sigma = true_mass[i], maps[i], map_sigmas[i]
        mass_var = np.zeros_like(mass)
        curve, sigma = map_curve(ring, mass, mass * velocity, N_RINGS, mass_var, (mass * velocity_sigma) ** 2)
        rms, within_sigma, _ = curve_errors(images["true_curves"][i], curve, sigma)
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    [error] = map_errors(true_velocity[:, None], maps[:, None])  # map_errors takes (n_images, n_targets, nY, nX)
    print(f"{CURVE_TARGET} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
