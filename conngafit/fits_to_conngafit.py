"""Convert an observed HI datacube in a FITS file into the hdf5 format CoNNGaFit runs on, at a trained network's size."""

import logging

import h5py
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

ARCSEC_PER_DEGREE = 3600
HI_FREQUENCY = 1.420405751768 * u.GHz
# Optically thin HI: N_HI [cm^-2] = 1.823e18 * brightness temperature [K] integrated over velocity [km/s]
COLUMN_DENSITY_PER_K_KM_S = 1.823e18


def center_to_length(cube, length, axis):
    """Crop or zero-pad cube along axis to length, keeping it centered."""
    excess = cube.shape[axis] - length
    if excess >= 0:
        start = excess // 2
        return np.take(cube, range(start, start + length), axis=axis)
    pad = [(0, 0)] * cube.ndim
    pad[axis] = (-excess // 2, -excess - (-excess // 2))
    return np.pad(cube, pad)


def column_density_per_jy_beam(header):
    """The factor converting a FITS cube's Jy/beam to HI column density per channel (cm^-2), the units CoNNGaFit's
    training cubes are in, assuming optically thin gas: N_HI = 1.823e18 * T_B * channel width [km/s], with T_B the
    Rayleigh-Jeans brightness temperature [K] of 1 Jy/beam in the header's Gaussian beam (BMAJ, BMIN)."""
    unit = header.get("BUNIT", "Jy/beam").replace(" ", "").upper()
    if unit not in ("JY/BEAM", "JYBEAM-1"):
        raise ValueError(f"the cube is in {header['BUNIT']!r}; fits_to_conngafit converts from Jy/beam")
    beam_area = np.pi / (4 * np.log(2)) * (header["BMAJ"] * u.deg) * (header["BMIN"] * u.deg)
    kelvin = (1 * u.Jy / u.beam).to_value(u.K, equivalencies=u.brightness_temperature(HI_FREQUENCY, beam_area))
    channel_width = abs(header["CDELT3"]) * u.Unit(header.get("CUNIT3", "m/s"))  # FITS velocities default to m/s
    if channel_width.unit.is_equivalent(u.Hz):  # A frequency axis: a channel spans c * dnu / nu of velocity
        channel_width = constants.c * channel_width / HI_FREQUENCY
    return COLUMN_DENSITY_PER_K_KM_S * kelvin * channel_width.to_value(u.km / u.s)


def save_column_density(image, output):
    """Save a cube summed over velocity as output + ".png" and output + ".hdf5"."""
    plt.imshow(image)
    plt.colorbar()
    plt.savefig(output + ".png")
    plt.close()
    with h5py.File(output + ".hdf5", "w") as hf:
        hf.create_dataset("image", data=image)


def fits_to_conngafit(fits_path, output, target_npix, target_nspec, distance_kpc=None, save_images=False):
    """Convert the HI datacube in fits_path to output + ".hdf5": a (target_npix, target_npix, target_nspec) cube in
    CoNNGaFit's (y, x, velocity) layout, the size of the network it will be run on.

    The cube is converted from Jy/beam to HI column density per channel (cm^-2, see column_density_per_jy_beam), the
    units of the training cubes. The image is averaged in square blocks of n_bin pixels, with n_bin chosen so it is
    about target_npix blocks across, then cropped or zero-padded, centered, to exactly target_npix. The velocity
    channels are not resampled, only cropped or zero-padded, centered, to target_nspec, so the cube's channel width
    should match the data the network was trained on.

    distance_kpc: the galaxy's distance. If given, the binned pixel size is also saved in kpc (pixel_res_kpc), which
        use_model.run_inferences uses to plot radial curves against radius.
    save_images: also save the cube summed over velocity, before and after, as output + "_originalImage" and
        output + "_reducedImage" (.png and .hdf5).
    """
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        pixel_arcsec = abs(header["CDELT1"]) * ARCSEC_PER_DEGREE
        to_column_density = column_density_per_jy_beam(header)
        cube = hdul[0].data * to_column_density  # (velocity, y, x) in cm^-2 per channel, copied out of the file
    if cube.ndim == 4:
        cube = cube[0]  # Drop the Stokes axis
    logger.info(
        "Read a %s cube with %g arcsec pixels; 1 Jy/beam = %.4g cm^-2", cube.shape, pixel_arcsec, to_column_density
    )
    if save_images:
        save_column_density(cube.sum(axis=0), output + "_originalImage")

    # Average the image in n_bin x n_bin blocks (column density is per area, so a block's is its pixels' mean), after
    # cropping each axis (centered) to a whole number of blocks
    n_bin = max(1, round(cube.shape[2] / target_npix))
    for axis in (1, 2):
        cube = center_to_length(cube, cube.shape[axis] // n_bin * n_bin, axis)
    cube = block_reduce(cube, (1, n_bin, n_bin), func=np.mean)

    # Center the image and the spectrum on the network's grid, then put velocity last
    for axis, length in ((0, target_nspec), (1, target_npix), (2, target_npix)):
        cube = center_to_length(cube, length, axis)
    spectra = np.moveaxis(cube, 0, -1)  # (y, x, velocity)
    if save_images:
        save_column_density(spectra.sum(axis=2), output + "_reducedImage")

    pixel_res_arcsec = pixel_arcsec * n_bin
    logger.info("Binned %d x %d pixels together: %g arcsec pixels", n_bin, n_bin, pixel_res_arcsec)
    with h5py.File(output + ".hdf5", "w") as hf:
        hf.create_dataset("spectra", data=spectra)
        hf.create_dataset("pixel_res_arcsec", data=pixel_res_arcsec)
        if distance_kpc is not None:
            pixel_res_kpc = np.deg2rad(pixel_res_arcsec / ARCSEC_PER_DEGREE) * distance_kpc
            hf.create_dataset("pixel_res_kpc", data=pixel_res_kpc)
            logger.info("Binned pixels are %g kpc across at %g kpc", pixel_res_kpc, distance_kpc)
