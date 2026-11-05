"""Demo: sample a trained diffusion model's mass and rotational momentum maps for the demo's test galaxy, with no
training.

Runs the model trained by examples/ddpm_maps/demo.py in under a minute. From the repository root:

    python examples/ddpm_maps/demo_inference.py
    python examples/ddpm_maps/demo_inference.py --model-path DIR/NetworkWeights/demo

use_model.predict(..., return_samples=True) draws n_samples maps of both targets from their joint posterior for each
datacube, as it would for a real observation converted with fits_to_conngafit.py. Each sample's rotation curve is
its ring totals of momentum divided by mass, and their mean and spread give the curve and its 1-sigma, including the
correlation between the mass and momentum errors.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXAMPLES_DIR), str(EXAMPLES_DIR.parent)]  # The shared demo modules, and CoNNGaFit's

from conngafit_datasets import ring_index  # noqa: E402 - after sys.path
from demo_common import (  # noqa: E402
    CURVE_TARGET,
    DATA_DIR,
    MAP_TARGETS,
    N_RINGS,
    SPLITS,
    curve_errors,
    load_images,
    map_curve,
    map_errors,
    testing_images,
)
from logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from use_model import load_model, predict  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "output" / "NetworkWeights" / "demo"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Every image's posterior samples, and the truths the synthetic galaxies come with, which a real observation wouldn't
    paths, _ = testing_images(DATA_DIR, SPLITS["testing"][0])
    model, metadata = load_model(str(args.model_path))
    samples = np.array([image_samples for _, image_samples in predict(model, metadata, paths, return_samples=True)])
    images = load_images(paths, CURVE_TARGET, MAP_TARGETS, N_RINGS)

    # Each sample's rotation curve from its maps, in MAP_TARGETS order; their mean and spread give each image's curve
    ring = ring_index(samples.shape[3:], images["fovs_kpc"][0], N_RINGS)
    for i, path in enumerate(paths):
        sample_curves = np.array([map_curve(ring, mass, momentum, N_RINGS)[0] for mass, momentum in samples[i]])
        curve, sigma = sample_curves.mean(axis=0), sample_curves.std(axis=0, ddof=1)
        rms, within_sigma, _ = curve_errors(images["true_curves"][i], curve, sigma)
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    for target, error in zip(MAP_TARGETS, map_errors(images["true_maps"], samples.mean(axis=1))):
        print(f"{target} map RMS error: {error:.0%} of the true map's RMS")  # noqa: T201


if __name__ == "__main__":
    main()
