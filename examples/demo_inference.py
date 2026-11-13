"""Demo: run a trained CoNNGaFit network on new datacubes. This runs the demo's test galaxy, with no training.

Uses the pre-trained demo network in examples/demo_model/ and runs in a few seconds. From the repository root:

    python examples/demo_inference.py       # writes examples/demo_output/inference/
    python examples/demo_inference.py --model-path examples/demo_output/NetworkWeights/demo   # your own demo.py run
    python examples/demo_inference.py --model-path examples/demo_output/NetworkWeights/demo_map   # demo.py --predict map

The script walks through running a network:
Step 1: Choose the trained network
Step 2: Set the input datacubes
Step 3: runs it with conngafit.use_model.run_inferences, which writes each prediction (a rotation curve, or for a
network that predicts maps, a velocity map) as a plot and an hdf5 file.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from conngafit.datasets import ring_index
from conngafit.logging_setup import add_logging_arguments, configure_logging
from conngafit.use_model import run_inferences
from demo_utils import load_images, map_curve

REPO_ROOT = Path(__file__).resolve().parents[1]

#### Step 1: the trained network ####
# Its path without the .pt/.hdf5 extension: a network trained by examples/demo.py (copied from
# examples/demo_output/NetworkWeights/). Override it with --model-path.
MODEL_PATH = REPO_ROOT / "examples" / "demo_model" / "demo"

#### Step 2: the datacubes to run it on ####
# VOF *_fullSpectra.hdf5 outputs, or observations converted with fits_to_conngafit.py. These are the demo's test
# galaxy, which the network didn't train on, seen at four position angles.
DATA_DIR = REPO_ROOT / "examples" / "demo_data"
IMAGE_PATHS = [
    DATA_DIR / "vflat220" / "i45" / "vflat220_i45_pa0_600_image_fullSpectra.hdf5",
    DATA_DIR / "vflat220" / "i45" / "vflat220_i45_pa90_600_image_fullSpectra.hdf5",
    DATA_DIR / "vflat220" / "i45" / "vflat220_i45_pa180_600_image_fullSpectra.hdf5",
    DATA_DIR / "vflat220" / "i45" / "vflat220_i45_pa270_600_image_fullSpectra.hdf5",
]

# What the demo networks predict, and the rings their predicted maps are binned into for step 4
TARGET = "rotational_velocity"
N_RINGS = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path", type=Path, default=MODEL_PATH, help="Checkpoint without .pt/.hdf5. Default: %(default)s"
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "examples" / "demo_output" / "inference")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    #### Step 3: run the network on every datacube ####
    # Writes <name>_<target>_radialPlot.png/.hdf5 (rotation curves) or _projectionMap.png/.hdf5 (maps) into OUTPUT
    run_inferences(str(args.model_path), IMAGE_PATHS, f"{args.output}/")

    #### Step 4: compare with the truth (demo data only) ####
    # The synthetic galaxies come with their true rotation curves and mass maps
    with h5py.File(f"{args.model_path}.hdf5") as f:
        predicts_maps = "radial_bins" not in f  # Networks that predict curves record their number of rings
    images = load_images(IMAGE_PATHS, TARGET, ["mass"], N_RINGS)
    ring = ring_index(images["true_maps"].shape[2:], images["fovs_kpc"][0], N_RINGS)
    for path, truth, (mass,) in zip(IMAGE_PATHS, images["true_curves"], images["true_maps"]):
        name = path.stem
        if predicts_maps:
            with h5py.File(args.output / f"{name}_{TARGET}_projectionMap.hdf5") as f:
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
            with h5py.File(args.output / f"{name}_{TARGET}_radialPlot.hdf5") as f:
                prediction = f["prediction"][()]
                sigma = f["uncertainty"][()] if "uncertainty" in f else None
        rms = np.sqrt(np.mean((prediction - truth) ** 2))
        within = "" if sigma is None else f", {np.mean(np.abs(prediction - truth) < sigma):.0%} of rings within 1σ"
        print(f"{name}: RMS error {rms:.1f} km/s{within}")  # noqa: T201
    print(f"Outputs: {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
