"""Experiment: sample a trained diffusion model's rotational velocity map for the demo data's test galaxy, with no training.

Runs the model trained by experiments/ddpm_velocity_map/demo.py in under a minute. From the repository root:

    python experiments/ddpm_velocity_map/demo_inference.py
    python experiments/ddpm_velocity_map/demo_inference.py --model-path DIR/NetworkWeights/ddpm_velocity

use_model.predict(..., return_samples=True) draws n_samples velocity maps from their posterior for each datacube, as
it would for a real observation converted with fits_to_conngafit.py. Each sample's rotation curve is each ring's
mass-weighted mean velocity, weighted by the true mass map, and their mean and spread give the curve and its
1-sigma, including the correlation between the pixels' errors.
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
from demo import CURVE_TARGET, N_RINGS, RUN_NAME  # noqa: E402 - this folder's demo.py
from demo_utils import (  # noqa: E402
    DATA_DIR,
    SPLITS,
    curve_errors,
    load_images,
    map_curve,
    map_errors,
    testing_images,
)

MODEL_PATH = Path(__file__).resolve().parent / "output" / "NetworkWeights" / RUN_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Every image's posterior samples, and the truths the synthetic galaxies come with (a real observation wouldn't)
    paths, _ = testing_images(DATA_DIR, SPLITS["testing"][0])
    model, metadata = load_model(str(args.model_path))
    samples = np.array([image_samples for _, image_samples in predict(model, metadata, paths, return_samples=True)])
    images = load_images(paths, CURVE_TARGET, ["mass", CURVE_TARGET], N_RINGS)
    true_mass, true_velocity = images["true_maps"][:, :1], images["true_maps"][:, 1:]  # (n_images, 1, nY, nX)

    # Each sample's rotation curve: the true mass times its velocity is the momentum, so their ring totals' ratio is
    # the mass-weighted mean velocity. The mean and spread of the samples' curves give each image's curve
    ring = ring_index(samples.shape[3:], images["fovs_kpc"][0], N_RINGS)
    for i, path in enumerate(paths):
        mass = true_mass[i, 0]
        sample_curves = np.array([map_curve(ring, mass, mass * velocity, N_RINGS)[0] for (velocity,) in samples[i]])
        curve, sigma = sample_curves.mean(axis=0), sample_curves.std(axis=0, ddof=1)
        rms, within_sigma, _ = curve_errors(images["true_curves"][i], curve, sigma)
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    [error] = map_errors(true_velocity, samples.mean(axis=1))
    print(f"{CURVE_TARGET} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
