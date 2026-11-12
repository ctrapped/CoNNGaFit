"""fits_to_conngafit: converts FITS cubes of any size to the network's (target_npix, target_npix, target_nspec) grid."""

import io

import h5py
import numpy as np
import pytest
from astropy.io import fits

from conngafit.fits_to_conngafit import center_to_length, column_density_per_jy_beam, fits_to_conngafit

TARGET_NPIX, TARGET_NSPEC = 40, 77
BEAM_ARCSEC = 10.0
CHANNEL_KM_S = 5.0
# Textbook optically thin HI conversion for this beam and channel: 1.823e18 * 6.06e5 / (bmaj * bmin) * dv
EXPECTED_FACTOR = 1.823e18 * 6.0574e5 / BEAM_ARCSEC**2 * CHANNEL_KM_S


def fits_cube(data, arcsec_per_pixel=1.0, **header):
    """An in-memory FITS file holding data (Stokes, velocity, y, x) in Jy/beam, with arcsec_per_pixel pixels, a
    BEAM_ARCSEC beam and CHANNEL_KM_S channels; header overrides or adds keywords."""
    hdu = fits.PrimaryHDU(data)
    hdu.header["CDELT1"] = -arcsec_per_pixel / 3600  # degrees; negative, as RA runs backwards
    hdu.header["BMAJ"] = hdu.header["BMIN"] = BEAM_ARCSEC / 3600
    hdu.header["CDELT3"], hdu.header["CUNIT3"] = CHANNEL_KM_S * 1000, "m/s"
    hdu.header["BUNIT"] = "JY/BEAM"
    for key, value in header.items():
        hdu.header[key] = value
    buffer = io.BytesIO()
    fits.HDUList([hdu]).writeto(buffer)
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize(
    "length, expected",
    [
        (3, [2, 3, 4]),  # cropped; an odd excess leaves the extra value at the end
        (6, [1, 2, 3, 4, 5, 6]),
        (8, [0, 1, 2, 3, 4, 5, 6, 0]),
        (11, [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]),  # padded; odd padding puts the extra zero at the end
    ],
)
def test_center_to_length_keeps_the_middle(length, expected):
    assert center_to_length(np.arange(1, 7), length, axis=0).tolist() == expected


@pytest.mark.parametrize(
    "n_spec, n_pix",
    [
        (70, 80),  # fewer channels than the network takes; the image bins evenly (2 x 2)
        (77, 80),  # exactly as many channels
        (84, 80),  # more channels, by an odd number
        (70, 130),  # an image that doesn't bin evenly (3 x 3 blocks, 43 across, cropped to 40)
        (70, 30),  # an image smaller than the network's grid (padded)
    ],
)
def test_converted_cube_has_the_networks_shape_and_averages_into_blocks(tmp_path, n_spec, n_pix):
    fits_to_conngafit(fits_cube(np.ones((1, n_spec, n_pix, n_pix))), str(tmp_path / "cube"), TARGET_NPIX, TARGET_NSPEC)
    with h5py.File(tmp_path / "cube.hdf5") as hf:
        spectra = hf["spectra"][()]
    assert spectra.shape == (TARGET_NPIX, TARGET_NPIX, TARGET_NSPEC)  # (y, x, velocity)
    n_bin = max(1, round(n_pix / TARGET_NPIX))
    filled = min(TARGET_NPIX, n_pix // n_bin)  # binned pixels along each axis that hold data, the rest padding
    # Every value is 1 Jy/beam, so every filled pixel of every kept channel is one factor's worth of column density
    assert spectra.sum() == pytest.approx(min(n_spec, TARGET_NSPEC) * filled**2 * EXPECTED_FACTOR, rel=1e-3)


def test_pixel_size_in_kpc_needs_a_distance(tmp_path):
    cube = np.ones((1, 70, 80, 80))
    fits_to_conngafit(fits_cube(cube, arcsec_per_pixel=2.0), str(tmp_path / "far"), TARGET_NPIX, TARGET_NSPEC, 1000)
    fits_to_conngafit(fits_cube(cube, arcsec_per_pixel=2.0), str(tmp_path / "none"), TARGET_NPIX, TARGET_NSPEC)
    with h5py.File(tmp_path / "far.hdf5") as far, h5py.File(tmp_path / "none.hdf5") as none:
        assert far["pixel_res_arcsec"][()] == pytest.approx(4.0)  # 2 x 2 blocks of 2-arcsec pixels
        assert far["pixel_res_kpc"][()] == pytest.approx(np.deg2rad(4.0 / 3600) * 1000)
        assert "pixel_res_kpc" not in none


@pytest.mark.parametrize(
    "spectral_axis",
    [
        dict(CDELT3=CHANNEL_KM_S, CUNIT3="km/s"),
        dict(CDELT3=CHANNEL_KM_S / 299792.458 * 1.420405751768e9, CUNIT3="Hz"),  # the same width as a frequency
    ],
)
def test_jy_beam_to_column_density_matches_the_textbook_conversion(spectral_axis):
    header = fits.getheader(fits_cube(np.ones((1, 2, 2, 2)), **spectral_axis))
    assert column_density_per_jy_beam(header) == pytest.approx(EXPECTED_FACTOR, rel=1e-3)


def test_rejects_cubes_not_in_jy_beam():
    with pytest.raises(ValueError, match="Jy/beam"):
        column_density_per_jy_beam(fits.getheader(fits_cube(np.ones((1, 2, 2, 2)), BUNIT="K")))
