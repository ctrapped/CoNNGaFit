"""Settings, loaders and computations shared by examples/demo.py and the scripts in experiments/. The loaders are
the only functions that read or write files; the computations take and return arrays. The experiments put examples/
on sys.path before importing it."""

from pathlib import Path

import h5py
import numpy as np

from conngafit import train_model
from conngafit.datasets import VOF_VIEW, label_map, radial_profile
from conngafit.logging_setup import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "examples" / "demo_data"
DATASETS = {
    "augment": "major_axis",
    "inclinations": [45],
    "radial_projection": "sky",
}
SPLITS = {
    "training": ["vflat120", "vflat160", "vflat240", "vflat280"],
    "validation": ["vflat180"],
    "testing": ["vflat220"],
}
SNAPSHOT = 600
# cm^-2: about 5 sigma for VOF's demo noise; fainter pixels are hidden in the map panels but still used for training
DETECTION = 1e20
# Color limits of every demo's map panels, so each property is drawn on the same scale in every figure
COLUMN_DENSITY_LIMITS = (1e19, 1e23)  # cm^-2, on a log scale
LOS_VELOCITY_LIMITS = (-300, 300)  # km/s
SNR_LIMITS = (0, 10)  # Predicted / 1-sigma
# Each target map's panels: on a log or linear color scale, the values' limits, and the limit of their difference
# (log10(predicted / true) in dex on a log scale, predicted - true otherwise). Radial velocities and momenta change
# sign between inflow and outflow, so they're on linear scales centered on zero
MAP_PANELS = {
    "mass": {"log_scale": True, "value_limits": (1e5, 2e8), "difference_limit": 1.5},
    "rotational_velocity": {"log_scale": False, "value_limits": (-100, 300), "difference_limit": 100},
    "rotational_momentum": {"log_scale": True, "value_limits": (1e7, 2e10), "difference_limit": 1.5},
    "radial_velocity": {"log_scale": False, "value_limits": (-100, 100), "difference_limit": 50},
    "cylindrical_radial_velocity": {"log_scale": False, "value_limits": (-100, 100), "difference_limit": 50},
    "radial_momentum": {"log_scale": False, "value_limits": (-1e9, 1e9), "difference_limit": 5e8},
    "cylindrical_radial_momentum": {"log_scale": False, "value_limits": (-1e9, 1e9), "difference_limit": 5e8},
}


#### Loaders ####
def train(output, run_name, hyperparams, datasets):
    """Train on the demo data with conngafit.train_model.train, saving into output; returns the network's path.
    datasets needs no galaxies or data_dir: every split's galaxies in SPLITS, at SNAPSHOT, are added."""
    configure_logging()  # Show the training's progress
    splits = {split: {galaxy: SNAPSHOT for galaxy in galaxies} for split, galaxies in SPLITS.items()}
    result = train_model.train(
        hyperparams,
        {**datasets, **splits, "data_dir": DATA_DIR},
        run_name=run_name,
        network_dir=output / "NetworkWeights",
        output_dir=output / "DiagnosticOutputs",
    )
    return Path(result.model_path)


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
    """The training and validation losses and MSEs of the predicted mean, per epoch, from conngafit.train_model's
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
    conngafit.datasets.ring_index): each ring's total momentum divided by its total mass, the mass-weighted mean
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
