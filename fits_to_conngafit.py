import logging

import h5py
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# from torchvision.io import read_image
deg2arcsec = 60 * 60

####Function to convert a FITS file of a given resolution to a compatible hdf5 file to run inferences on with CoNNGaFit
####Currently, this will reduce the resolution of the input image
####Input parameter are:
####    imageDir: path to the fits file you wish to convert
####    output: path to where you want to write the converted files and images
####    targetNpix: Target spatial dimensionality for inferences (currently 40)
####    targetNspec: Target spectral dimenionality
####    distance: (Optional) Distance to observed galaxy for bookkeeping purposes
####    saveImages: (Optional) Save images of column densities for original and reduced images
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023


def FITS_to_CoNNGaFit(imageDir, output, targetNpix, targetNspec, distance=-1, saveImages=False):
    hdul = fits.open(imageDir)
    originalPixelRes = np.abs(hdul[0].header["CDELT1"] * deg2arcsec)
    logger.info("pixel res = %g arcsec", hdul[0].header["CDELT1"] * deg2arcsec)
    rawdata = hdul[0].data

    logger.debug("Shape of raw data: %s", np.shape(rawdata))

    if saveImages:
        plt.imshow(np.sum(rawdata, axis=1)[0, :, :])
        plt.colorbar()
        plt.savefig(output + "_originalImage.png")
        hf_original = h5py.File(output + "_originalImage.hdf5", "w")
        hf_original.create_dataset("image", data=np.sum(rawdata, axis=1)[0, :, :])
        hf_original.close()
        plt.close()

    newImage = np.zeros((targetNpix, targetNpix, targetNspec))

    nBin = int(np.round(np.shape(rawdata)[3] / targetNpix))

    newdata = rawdata[0, :, :, :]
    logger.debug("Shape of newdata: %s", np.shape(newdata))
    all_but_first = tuple([1, 2])

    nSpec, nPix, tmp = np.shape(newdata)

    dx = int(np.round((targetNpix - nPix / nBin) / 2))
    ds = int(np.round((targetNspec - nSpec) / 2))
    logger.debug("ds = %d, targetNspec = %d", ds, targetNspec)

    pixOffset0 = 0
    logger.debug("dx = %d", dx)

    if dx < 0:
        # dx*=-1
        pixOffset0 = int(np.round(-dx / 2))

    nTrunc = targetNspec - 2 * ds
    logger.debug("nspec = %d, ntrunc = %d", nSpec, nTrunc)
    offset = nSpec - nTrunc
    logger.debug("offset = %d", offset)

    pixOffset = 0
    if int(nPix / nBin) + dx > targetNpix:
        pixOffset = int(np.round((targetNpix - int(nPix / nBin) - np.abs(dx)) / 2))

    logger.debug("pixOffset = %d, pixOffset0 = %d", pixOffset, pixOffset0)

    for i in range(pixOffset0, int(nPix / nBin) + pixOffset):
        for j in range(pixOffset0, int(nPix / nBin) + pixOffset):
            if ds > 0:
                newImage[i + dx, j + dx, ds : (targetNspec - ds + offset)] = newdata[
                    :,
                    i * nBin + pixOffset0 : (i + 1) * nBin + pixOffset,
                    j * nBin + pixOffset0 : (j + 1) * nBin + pixOffset,
                ].sum(axis=all_but_first)
            elif ds < 0:
                newImage[i + dx, j + dx, :] = newdata[
                    -ds : nSpec + ds,
                    i * nBin + pixOffset0 : (i + 1) * nBin + pixOffset,
                    j * nBin + pixOffset0 : (j + 1) * nBin + pixOffset,
                ].sum(axis=all_but_first)

    if saveImages:
        plt.imshow(np.sum(newImage, axis=2))
        plt.colorbar()
        plt.savefig(output + "_reducedImage.png")
        hf_reduced = h5py.File(output + "_reducedImage.hdf5", "w")
        hf_reduced.create_dataset("image", data=np.sum(newImage, axis=2))
        hf_reduced.close()
        plt.close()

    hf = h5py.File(output + ".hdf5", "w")
    hf.create_dataset("spectra", data=newImage)
    hf.create_dataset("pixelRes_arcsec", data=originalPixelRes * nBin)
    hf.create_dataset("pixelRes_kpc", data=originalPixelRes * nBin / 60 / 60 * np.pi / 180 * distance)
    hf.close()

    logger.info("nBin = %d", nBin)
    logger.info("newPixelRes = %g arcsec", originalPixelRes * nBin)
    logger.info("newPixelRes = %g kpc", originalPixelRes * nBin / 60 / 60 * np.pi / 180 * distance)
