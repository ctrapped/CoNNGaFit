"""Regenerate examples/demo_data/: VOF observations of synthetic disk galaxies, for examples/demo.py.

You only need this to change the demo data, which is already in the repository. Needs VOF installed
and a clone of its repository (for its synthetic galaxy, tests/synthetic_galaxy.py). From this repository's root:

    python examples/make_demo_data.py --vof-repo ../VeryObservableFIRE

Each galaxy is VOF's demo galaxy (the same observation settings as VOF's examples/demo.py) with its own
rotation speed and random seed, observed at four position angles. Only what CoNNGaFit's demo uses is kept,
as compressed float32, to keep the repository small.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # So CoNNGaFit's modules import from examples/

from logging_setup import add_logging_arguments, configure_logging  # noqa: E402 - after the sys.path line above

logger = logging.getLogger(__name__)

OUTPUT = REPO_ROOT / "examples" / "demo_data"

# Asymptotic rotation speed (km/s) of each galaxy, by set: the network has to learn it from the spectra
GALAXIES = {
    "training": [120, 160, 240, 280],
    "validation": [180],
    "testing": [220],
}
INCLINATION = 45
POSITION_ANGLES = [0, 90, 180, 270]
# VOF outputs kept, and the attributes CoNNGaFit reads (fov_kpc) or the demo plots with
KEEP = ["spectra", "mass_annotation", "rotational_velocity_annotation"]

# As in VOF's examples/demo.py, in the same key = value format as VOF's configs/param_template.param
MAX_RADIUS = 15.0  # kpc
N_PIXELS = 48
RES_KM_S = 10.0
N_CHANNELS = 48
NOISE = 3e18  # noise per channel, cm^-2
PARAMS = """\
galName = {name}
minSnap = {snapshot}
maxSnap = {snapshot}
snapStep = 1
fileDir = {snapdir}/
statsDir = {stats_dir}
output = {output}/

observerDistance = 10000
observerVelocity = [0, 0, 0]
maxRadius = {max_radius}
maxHeight = 5
targetBeamSize = 20
Nsightlines1d = {n_pixels}
phiObs = 0
inclinations = [{inclination}]
position_angles = {position_angles}

speciesToRun = HI_21cm
res_km_s = {res_km_s}
Nchannels = {n_channels}
noiseAmplitude = {noise}

runBinfire = False
runVOF = True
savePng = False
onlySaveFirstPng = True
createMaskFromExistingStatsDir = False
num_cores = 1
projectGasProperties = True
runAsOpticallyThinProjection = True
"""


def galaxy_name(v_flat):
    return f"vflat{v_flat}"


def run_vof(synthetic_galaxy, v_flat, seed, work_dir, vof_repo):
    """Write a synthetic galaxy rotating at v_flat km/s and observe it with VOF; returns VOF's output directory."""
    synthetic_galaxy.V_FLAT = v_flat  # read by circular_velocity() when the disk is built
    snapshot = synthetic_galaxy.SNAPSHOT
    galaxy = synthetic_galaxy.write_synthetic_snapshot(work_dir / "snapshots" / f"snapdir_{snapshot}", seed=seed)
    synthetic_galaxy.write_stats_file(galaxy, MAX_RADIUS)

    param_file = work_dir / "demo.param"
    param_file.write_text(
        PARAMS.format(
            name=galaxy_name(v_flat),
            snapshot=snapshot,
            snapdir=work_dir / "snapshots",
            stats_dir=galaxy.stats_file[: -len(f"{snapshot:04d}.hdf5")],
            output=work_dir / "vof_output",
            max_radius=MAX_RADIUS,
            n_pixels=N_PIXELS,
            inclination=INCLINATION,
            position_angles=POSITION_ANGLES,
            res_km_s=RES_KM_S,
            n_channels=N_CHANNELS,
            noise=NOISE,
        )
    )
    subprocess.run([sys.executable, "-m", "vof", str(param_file)], cwd=vof_repo, check=True)
    return work_dir / "vof_output"


def trim(source, destination):
    """Copy the datasets in KEEP and every attribute from VOF's output file, as gzipped float32."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as hf_in, h5py.File(destination, "w") as hf_out:
        for key in KEEP:
            hf_out.create_dataset(key, data=hf_in[key][()].astype(np.float32), compression="gzip", shuffle=True)
        hf_out.attrs.update(hf_in.attrs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vof-repo", type=Path, required=True, help="A clone of the VeryObservableFIRE repository.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Default: %(default)s")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    vof_repo = args.vof_repo.resolve()
    sys.path.insert(0, str(vof_repo))  # So the synthetic galaxy can be imported from VOF's tests/
    from tests import synthetic_galaxy

    if args.output.exists():
        shutil.rmtree(args.output)
    for seed, v_flat in enumerate(sum(GALAXIES.values(), []), start=1):
        logger.info("Observing a galaxy rotating at %d km/s", v_flat)
        with tempfile.TemporaryDirectory() as work_dir:
            vof_output = run_vof(synthetic_galaxy, v_flat, seed, Path(work_dir), vof_repo)
            for path in sorted(vof_output.rglob("*_fullSpectra.hdf5")):
                trim(path, args.output / galaxy_name(v_flat) / path.parent.name / path.name)
    size_mb = sum(p.stat().st_size for p in args.output.rglob("*.hdf5")) / 1e6
    logger.info("Wrote %s (%.1f MB)", args.output, size_mb)


if __name__ == "__main__":
    main()
