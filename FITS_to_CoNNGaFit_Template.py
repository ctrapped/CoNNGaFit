from FITS_to_CoNNGaFit import FITS_to_CoNNGaFit
####Template for how to convert from fits to CoNNGaFit compatible hdf5 files using FITS_to_CoNNGaFit.py
####imageDir should point to the FITS file
####output should point to where you want to write the converted datacubes
####distance is not necesarry
####targetNpix and targetNspec need to match the current dimensionality of the trained network (currently 40 and 77 respectively)
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023

print("Running NGC 2403")
imageDir = 'CoNNGaFitData/observations/NGC_2403_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_2403'
distance = 3200 #kpc
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
FITS_to_CoNNGaFit(imageDir,output,targetNpix,targetNspec,distance=distance,saveImages=True)

print("Running NGC 3351")
imageDir = 'CoNNGaFitData/observations/NGC_3351_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_3351'
distance = 10500 #kpc
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
FITS_to_CoNNGaFit(imageDir,output,targetNpix,targetNspec,distance=distance,saveImages=True)

print("Running NGC 5055")
imageDir = 'CoNNGaFitData/observations/NGC_5055_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_5055'
distance = 8900 #kpc
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
FITS_to_CoNNGaFit(imageDir,output,targetNpix,targetNspec,distance=distance,saveImages=True)

print("Running NGC 7793")
imageDir = 'CoNNGaFitData/observations/NGC_7793_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_7793'
distance = 3600 #kpc
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
FITS_to_CoNNGaFit(imageDir,output,targetNpix,targetNspec,distance=distance,saveImages=True)
