"""Demo: run a trained CoNNGaFit network on new datacubes. This runs the demo's test galaxy, with no training.

Uses the pre-trained demo network in examples/demo_model/ and runs in a few seconds. From the repository root:

    python examples/demo_inference.py       # writes examples/demo_output/inference/
    python examples/demo_inference.py --model-path examples/demo_output/NetworkWeights/demo   # your own demo.py run
    python examples/demo_inference.py --model-path examples/demo_output/NetworkWeights/demo_map   # demo.py --predict map

examples/demo_model/ is a network trained by examples/demo.py (copied from examples/demo_output/NetworkWeights/).
For each datacube, use_model.run_inferences writes the predicted rotation curve, or for a network that predicts
maps the predicted velocity map, as a plot and an hdf5 file, exactly as it would for a real observation converted
with fits_to_conngafit.py. Predicted maps are binned into the rotation curve to compare with the truth.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from conngafit.datasets import ring_index
from conngafit.logging_setup import add_logging_arguments, configure_logging
from conngafit.use_model import run_inferences
from demo import DATA_DIR, DATASETS, N_RINGS  # examples/demo.py
from demo_utils import load_images, map_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "examples" / "demo_model" / "demo"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output" / "inference")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    image_paths = sorted(str(p) for p in (DATA_DIR / next(iter(DATASETS["testing"]))).rglob("*.hdf5"))
    run_inferences(str(args.model_path), image_paths, str(args.output) + "/")
    with h5py.File(f"{args.model_path}.hdf5") as f:
        predicts_maps = "radial_bins" not in f  # Networks that predict curves record their number of rings

    # The synthetic galaxies come with their true rotation curves and mass maps, which a real observation wouldn't
    target = DATASETS["target"]
    images = load_images(image_paths, target, ["mass"], N_RINGS)
    ring = ring_index(images["true_maps"].shape[2:], images["fovs_kpc"][0], N_RINGS)
    for path, truth, (mass,) in zip(image_paths, images["true_curves"], images["true_maps"]):
        name = Path(path).stem
        if predicts_maps:
            with h5py.File(args.output / f"{name}_{target}_projectionMap.hdf5") as f:
                velocity = f["image"][()]
                velocity_sigma = f["uncertainty"][()] if "uncertainty" in f else None  # Trained with predict_variance
            # Bin the velocity map into the rotation curve, weighting each pixel by its true mass as the true curve
            # is: mass * velocity is the momentum, and only the velocity carries an error
            if velocity_sigma is None:
                prediction, sigma = map_curve(ring, mass, mass * velocity, N_RINGS)
            else:
                momentum_var = (mass * velocity_sigma) ** 2
                prediction, sigma = map_curve(ring, mass, mass * velocity, N_RINGS, np.zeros_like(mass), momentum_var)
        else:
            with h5py.File(args.output / f"{name}_{target}_radialPlot.hdf5") as f:
                prediction = f["prediction"][()]
                sigma = f["uncertainty"][()] if "uncertainty" in f else None
        rms = np.sqrt(np.mean((prediction - truth) ** 2))
        within = "" if sigma is None else f", {np.mean(np.abs(prediction - truth) < sigma):.0%} of rings within 1σ"
        print(f"{name}: RMS error {rms:.1f} km/s{within}")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
