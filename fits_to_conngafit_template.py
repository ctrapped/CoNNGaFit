import logging

import numpy as np

from fits_to_conngafit import fits_to_conngafit
from logging_setup import configure_logging

####Template for how to convert from fits to CoNNGaFit compatible hdf5 files using fits_to_conngafit.py
####image_dir should point to the FITS file
####output should point to where you want to write the converted datacubes
####distance is not necesarry
####target_npix and target_nspec need to match the current dimensionality of the trained network (currently 40 and 77 respectively)
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023

logger = logging.getLogger(__name__)
configure_logging()

logger.info("Running NGC 2403")
image_dir = "CoNNGaFitData/observations/NGC_2403_NA_CUBE_THINGS.FITS"
output = "CoNNGaFitData/observations/NGC_2403"
distance = 3200  # kpc
target_npix = 40
target_nspec = int(np.round(400 / 5.2))
fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=distance, save_images=True)

logger.info("Running NGC 3351")
image_dir = "CoNNGaFitData/observations/NGC_3351_NA_CUBE_THINGS.FITS"
output = "CoNNGaFitData/observations/NGC_3351"
distance = 10500  # kpc
target_npix = 40
target_nspec = int(np.round(400 / 5.2))
fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=distance, save_images=True)

logger.info("Running NGC 5055")
image_dir = "CoNNGaFitData/observations/NGC_5055_NA_CUBE_THINGS.FITS"
output = "CoNNGaFitData/observations/NGC_5055"
distance = 8900  # kpc
target_npix = 40
target_nspec = int(np.round(400 / 5.2))
fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=distance, save_images=True)

logger.info("Running NGC 7793")
image_dir = "CoNNGaFitData/observations/NGC_7793_NA_CUBE_THINGS.FITS"
output = "CoNNGaFitData/observations/NGC_7793"
distance = 3600  # kpc
target_npix = 40
target_nspec = int(np.round(400 / 5.2))
fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=distance, save_images=True)
