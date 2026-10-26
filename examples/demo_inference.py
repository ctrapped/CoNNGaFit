"""Demo: run a trained CoNNGaFit network on new datacubes, here the demo's test galaxy, with no training.

Uses the pre-trained demo network in examples/demo_model/ and runs in a few seconds. From the repository root:

    python examples/demo_inference.py       # writes examples/demo_output/inference/
    python examples/demo_inference.py --model-path examples/demo_output/NetworkWeights/demo   # your own demo.py run

examples/demo_model/ is a network trained by examples/demo.py (copied from examples/demo_output/NetworkWeights/).
For each datacube, use_model.run_inferences writes the predicted rotation curve as a plot and an hdf5 file,
exactly as it would for a real observation converted with fits_to_conngafit.py.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # So CoNNGaFit's modules import from examples/

from demo import DATA_DIR, DATASETS, SPLITS  # noqa: E402 - after the sys.path line above; examples/demo.py

from conngafit_datasets import radial_profile  # noqa: E402
from logging_setup import add_logging_arguments, configure_logging  # noqa: E402
from use_model import run_inferences  # noqa: E402

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

    image_paths = sorted(str(p) for p in (DATA_DIR / SPLITS["testing"][0]).rglob("*.hdf5"))
    run_inferences(str(args.model_path), image_paths, str(args.output) + "/")

    # The synthetic galaxies come with their true rotation curves, which a real observation wouldn't
    for path in image_paths:
        with h5py.File(path) as f:
            truth = radial_profile(f, DATASETS["target"], DATASETS["radial_bins"])
        with h5py.File(args.output / f"{Path(path).stem}_radialPlot.hdf5") as f:
            prediction = f["prediction"][()]
        rms = np.sqrt(np.mean((prediction - truth) ** 2))
        print(f"{Path(path).stem}: RMS error {rms:.1f} km/s")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
