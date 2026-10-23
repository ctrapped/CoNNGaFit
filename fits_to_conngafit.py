import logging

import h5py
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# from torchvision.io import read_image
DEG_TO_ARCSEC = 60 * 60

####Function to convert a FITS file of a given resolution to a compatible hdf5 file to run inferences on with CoNNGaFit
####Currently, this will reduce the resolution of the input image
####Input parameter are:
####    image_dir: path to the fits file you wish to convert
####    output: path to where you want to write the converted files and images
####    target_npix: Target spatial dimensionality for inferences (currently 40)
####    target_nspec: Target spectral dimenionality
####    distance: (Optional) Distance to observed galaxy for bookkeeping purposes
####    save_images: (Optional) Save images of column densities for original and reduced images
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023


def fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=-1, save_images=False):
    hdul = fits.open(image_dir)
    original_pixel_res = np.abs(hdul[0].header["CDELT1"] * DEG_TO_ARCSEC)
    logger.info("pixel res = %g arcsec", hdul[0].header["CDELT1"] * DEG_TO_ARCSEC)
    rawdata = hdul[0].data

    logger.debug("Shape of raw data: %s", np.shape(rawdata))

    if save_images:
        plt.imshow(np.sum(rawdata, axis=1)[0, :, :])
        plt.colorbar()
        plt.savefig(output + "_originalImage.png")
        hf_original = h5py.File(output + "_originalImage.hdf5", "w")
        hf_original.create_dataset("image", data=np.sum(rawdata, axis=1)[0, :, :])
        hf_original.close()
        plt.close()

    new_image = np.zeros((target_npix, target_npix, target_nspec))

    n_bin = int(np.round(np.shape(rawdata)[3] / target_npix))

    newdata = rawdata[0, :, :, :]
    logger.debug("Shape of newdata: %s", np.shape(newdata))
    all_but_first = tuple([1, 2])

    n_spec, n_pix, tmp = np.shape(newdata)

    dx = int(np.round((target_npix - n_pix / n_bin) / 2))
    ds = int(np.round((target_nspec - n_spec) / 2))
    logger.debug("ds = %d, target_nspec = %d", ds, target_nspec)

    pix_offset0 = 0
    logger.debug("dx = %d", dx)

    if dx < 0:
        # dx*=-1
        pix_offset0 = int(np.round(-dx / 2))

    n_trunc = target_nspec - 2 * ds
    logger.debug("nspec = %d, ntrunc = %d", n_spec, n_trunc)
    offset = n_spec - n_trunc
    logger.debug("offset = %d", offset)

    pix_offset = 0
    if int(n_pix / n_bin) + dx > target_npix:
        pix_offset = int(np.round((target_npix - int(n_pix / n_bin) - np.abs(dx)) / 2))

    logger.debug("pix_offset = %d, pix_offset0 = %d", pix_offset, pix_offset0)

    for i in range(pix_offset0, int(n_pix / n_bin) + pix_offset):
        for j in range(pix_offset0, int(n_pix / n_bin) + pix_offset):
            if ds > 0:
                new_image[i + dx, j + dx, ds : (target_nspec - ds + offset)] = newdata[
                    :,
                    i * n_bin + pix_offset0 : (i + 1) * n_bin + pix_offset,
                    j * n_bin + pix_offset0 : (j + 1) * n_bin + pix_offset,
                ].sum(axis=all_but_first)
            elif ds < 0:
                new_image[i + dx, j + dx, :] = newdata[
                    -ds : n_spec + ds,
                    i * n_bin + pix_offset0 : (i + 1) * n_bin + pix_offset,
                    j * n_bin + pix_offset0 : (j + 1) * n_bin + pix_offset,
                ].sum(axis=all_but_first)

    if save_images:
        plt.imshow(np.sum(new_image, axis=2))
        plt.colorbar()
        plt.savefig(output + "_reducedImage.png")
        hf_reduced = h5py.File(output + "_reducedImage.hdf5", "w")
        hf_reduced.create_dataset("image", data=np.sum(new_image, axis=2))
        hf_reduced.close()
        plt.close()

    hf = h5py.File(output + ".hdf5", "w")
    hf.create_dataset("spectra", data=new_image)
    hf.create_dataset("pixelRes_arcsec", data=original_pixel_res * n_bin)
    hf.create_dataset("pixelRes_kpc", data=original_pixel_res * n_bin / 60 / 60 * np.pi / 180 * distance)
    hf.close()

    logger.info("n_bin = %d", n_bin)
    logger.info("newPixelRes = %g arcsec", original_pixel_res * n_bin)
    logger.info("newPixelRes = %g kpc", original_pixel_res * n_bin / 60 / 60 * np.pi / 180 * distance)
