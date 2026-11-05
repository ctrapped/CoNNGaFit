"""Settings, loaders and computations shared by the demos in examples/. The loaders are the only functions that read
or write files; the computations take and return arrays. The demos put examples/ and the repository root on
sys.path before importing it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from conngafit_datasets import VOF_VIEW, label_map, radial_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "examples" / "demo_data"
CURVE_TARGET = "rotational_velocity"  # The rotation curve every demo is checked against
# The maps the map demos predict; the rotation curve is their ratio, summed in rings
MAP_TARGETS = ["mass", "rotational_momentum"]
DATASETS = {
    "augment": "major_axis",
    "inclinations": [45],
    "radial_projection": "sky",
}
N_RINGS = 10  # Rings in each rotation curve
SPLITS = {
    "training": ["vflat120", "vflat160", "vflat240", "vflat280"],
    "validation": ["vflat180"],
    "testing": ["vflat220"],
}
SNAPSHOT = 600
# Every demo's config file, made small enough to train in a few minutes on a CPU
HYPERPARAM_OVERRIDES = {
    "n_filt0": 4,
    "k0": 5,
    "learning_rate": 1e-3,
    "label_scaling": "one-sided",
}
RESBLOCK_OVERRIDES = {
    "n_fc": 64,
    "dropout_rate": 0.0,
    "predict_variance": True,  # Also predict each value's 1-sigma uncertainty
}


#### Loaders ####
def toml_lines(params):
    """Flat key = value TOML lines (JSON literals are valid TOML for strings, numbers and lists)."""
    return [f"{key} = {json.dumps(value)}" for key, value in params.items()]


def write_configs(output, hyperparams, datasets, splits, snapshot, data_dir):
    """Write hyperparams to output/hyperparams.toml, and the datasets settings with each split's galaxies (all at
    snapshot) in data_dir to output/datasets.toml; returns their paths."""
    hyperparams_file = output / "hyperparams.toml"
    hyperparams_file.write_text("\n".join(toml_lines(hyperparams)) + "\n")

    # Relative to the datasets file, so the copy saved in the checkpoint doesn't hold this computer's paths
    lines = toml_lines({**datasets, "base_dir": os.path.relpath(data_dir, output)})
    for split, galaxies in splits.items():
        lines += [f"[{split}]"] + [f'"{galaxy}" = {snapshot}' for galaxy in galaxies]
    datasets_file = output / "datasets.toml"
    datasets_file.write_text("\n".join(lines) + "\n")
    return hyperparams_file, datasets_file


def train(output, run_name, hyperparams_file, datasets_file):
    """Run train_model_unet.py on the config files, exactly as you would from the command line, saving into output;
    returns the network's path."""
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "train_model_unet.py",
            "--hyperparams",
            hyperparams_file,
            "--datasets",
            datasets_file,
            "--run-name",
            run_name,
            "--network-dir",
            output / "NetworkWeights",
            "--output-dir",
            output / "DiagnosticOutputs",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output / "NetworkWeights" / run_name


def testing_images(data_dir, galaxy):
    """The paths of galaxy's images in data_dir and their position angles in degrees, in order of position angle."""
    paths = [str(p) for p in (data_dir / galaxy).rglob("*.hdf5")]
    position_angles = [float(VOF_VIEW.search(Path(path).name)["position_angle"]) for path in paths]
    order = np.argsort(position_angles)
    return [paths[i] for i in order], [position_angles[i] for i in order]


def load_images(paths, curve_target, map_targets, n_rings):
    """The images at paths, as a dict of arrays with one entry per image: their datacubes ("cubes", (nY, nX, nChannel)
    in cm^-2 per channel), channel widths in km/s, fields of view in kpc, true curves of curve_target in n_rings
    rings, and true maps of each of map_targets ("true_maps", (n_targets, nY, nX))."""
    images = {"cubes": [], "channel_widths": [], "fovs_kpc": [], "true_curves": [], "true_maps": []}
    for path in paths:
        with h5py.File(path) as f:
            images["cubes"].append(f["spectra"][()])
            images["channel_widths"].append(f.attrs["dnu_kmps"])
            images["fovs_kpc"].append(f.attrs["fov_kpc"])
            images["true_curves"].append(radial_profile(f, curve_target, n_rings))
            images["true_maps"].append([label_map(f, target) for target in map_targets])
    return {key: np.array(values) for key, values in images.items()}


def load_history(path):
    """The training and validation losses and MSEs of the predicted mean, per epoch, from train_model_unet.py's
    LossesAndAccuracies file at path."""
    keys = ["training_losses", "validation_losses", "training_mses", "validation_mses"]
    with h5py.File(path) as f:
        return {key: f[key][()] for key in keys}


#### Computations ####
def moment_maps(cube, channel_width):
    """The total HI column density (cm^-2) and mean line-of-sight velocity (km/s) maps of a datacube (nY, nX,
    nChannel) in cm^-2 per channel, whose channels are channel_width km/s wide and centered on zero velocity."""
    n_channels = cube.shape[2]
    velocities = (np.arange(n_channels) - (n_channels - 1) / 2) * channel_width
    column_density = cube.sum(axis=2)
    velocity = (cube * velocities).sum(axis=2) / column_density
    return column_density, velocity


def map_curve(ring, mass, momentum, n_rings, mass_var=None, momentum_var=None):
    """The velocity (rotational or radial) curve in n_rings rings from mass and momentum maps, given each pixel's ring (see
    conngafit_datasets.ring_index): each ring's total momentum divided by its total mass, the mass-weighted mean
    velocity. With the maps' per-pixel variances, also the curve's 1-sigma, propagated through the ratio assuming
    every pixel's errors are independent (None otherwise)."""
    inside = ring < n_rings  # Pixels in the corners, beyond the image edge, are left out
    ring_mass, ring_momentum = (np.bincount(ring[inside], m[inside], n_rings) for m in (mass, momentum))
    curve = np.divide(ring_momentum, ring_mass, out=np.zeros(n_rings), where=ring_mass > 0)
    if mass_var is None:
        return curve, None
    ring_mass_var, ring_momentum_var = (np.bincount(ring[inside], v[inside], n_rings) for v in (mass_var, momentum_var))
    # v = P / M, so var(v) = (var(P) + v^2 var(M)) / M^2
    curve_var = np.divide(
        ring_momentum_var + curve**2 * ring_mass_var, ring_mass**2, out=np.zeros(n_rings), where=ring_mass > 0
    )
    return curve, np.sqrt(curve_var)


def curve_errors(true_curves, curves, sigmas):
    """The RMS difference between the predicted curves and true_curves, and the fractions of rings whose prediction
    is within its predicted 1-sigma (sigmas) and 3-sigma of the truth; all are (n_images, n_rings)."""
    residuals = np.abs(curves - true_curves)
    return np.sqrt(np.mean(np.square(residuals))), np.mean(residuals < sigmas), np.mean(residuals < 3 * sigmas)


def map_errors(true_maps, maps):
    """Each target's RMS error over every pixel of the predicted maps, as a fraction of true_maps' RMS; both are
    (n_images, n_targets, nY, nX)."""
    pixels = (0, 2, 3)  # Every image's every pixel
    return np.sqrt(np.mean(np.square(maps - true_maps), axis=pixels) / np.mean(np.square(true_maps), axis=pixels))
