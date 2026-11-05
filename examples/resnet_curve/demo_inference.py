"""Demo: run a trained rotation curve network on the demo's test galaxy, with no training.

Uses the pre-trained network in examples/resnet_curve/model/ and runs in a few seconds. From the repository root:

    python examples/resnet_curve/demo_inference.py   # writes examples/resnet_curve/output/inference/
    python examples/resnet_curve/demo_inference.py --model-path examples/resnet_curve/output/NetworkWeights/demo

examples/resnet_curve/model/ is a network trained by examples/resnet_curve/demo.py. For each datacube,
use_model.run_inferences writes the predicted rotation curve and its 1-sigma as a plot and an hdf5 file, exactly as
it would for a real observation converted with fits_to_conngafit.py.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXAMPLES_DIR), str(EXAMPLES_DIR.parent)]  # The shared demo modules, and CoNNGaFit's

from demo_common import CURVE_TARGET, DATA_DIR, N_RINGS, SPLITS, curve_errors, load_images, testing_images  # noqa: E402
from logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from use_model import run_inferences  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEMO_DIR / "model" / "demo"


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

    # The saved predictions, and the true curves the synthetic galaxies come with, which a real observation wouldn't
    curves, sigmas = [], []
    for path in paths:
        with h5py.File(args.output / f"{Path(path).stem}_{CURVE_TARGET}_radialPlot.hdf5") as f:
            curves.append(f["prediction"][()])
            sigmas.append(f["uncertainty"][()])
    curves, sigmas = np.array(curves), np.array(sigmas)
    true_curves = load_images(paths, CURVE_TARGET, [], N_RINGS)["true_curves"]

    for i, path in enumerate(paths):
        rms, within_sigma, _ = curve_errors(true_curves[i], curves[i], sigmas[i])
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s, {within_sigma:.0%} of rings within 1σ")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
