import os
import pandas as pd
import numpy as np
import h5py
from matplotlib import pyplot as plt

from astropy.io import fits
#from torchvision.io import read_image
deg2arcsec = 60*60

def Convert_THINGS_to_CoNNGaFit(imageDir,output,distance,pixelRes,targetRes_kpc,targetNpix,targetNspec,saveImages=False):
    hdul = fits.open(imageDir)
    originalPixelRes = np.abs(hdul[0].header['CDELT1']*deg2arcsec)
    print('pixel res = ',hdul[0].header['CDELT1']*deg2arcsec)
    rawdata = hdul[0].data 
    
    print("Shape of raw data?",np.shape(rawdata))
    print("max of raw data",np.max(rawdata))
    if saveImages:
        plt.imshow(np.sum(rawdata,axis=1)[0,:,:])
        plt.colorbar()
        plt.savefig(output+"_originalImage.png")
        hf_original = h5py.File(output+"_originalImage.hdf5",'w')
        hf_original.create_dataset('image',data=np.sum(rawdata,axis=1)[0,:,:])
        hf_original.close()
        plt.close()
        
    
    newImage = np.zeros((targetNpix,targetNpix,targetNspec))
    
    imagePixelRes_kpc = pixelRes*distance
    targetPixelRes = 2./5000.
    #nBin = int(np.round(targetPixelRes / pixelRes))
    #nBin = int(np.round(targetRes_kpc / imagePixelRes_kpc))
    nBin = int(np.round(np.shape(rawdata)[3] / 40))

    newdata = rawdata[0,:,:,:]
    print("shape of newdata=",np.shape(newdata))
    all_but_first = tuple([1,2])
    
    nSpec,nPix,tmp = np.shape(newdata)
    
    dx = int(np.round((targetNpix - nPix/nBin)/2))
    ds = int(np.round((targetNspec-nSpec)/2))
    print("ds=",ds)
    print("targetNspec=",targetNspec)
   # if ds<0:
    #    ds*=-1
    pixOffset0=0
    print("dx=",dx)

    if dx<0:
        #dx*=-1
        pixOffset0=int(np.round(-dx/2))

    nTrunc = targetNspec-2*ds
    print("nspec=",nSpec)
    print("ntrunc=",nTrunc)
    offset = nSpec - nTrunc
    print("offset=",offset)
    
    pixOffset=0
    if int(nPix/nBin)+dx > targetNpix:
        pixOffset = int(np.round((targetNpix - int(nPix/nBin) - np.abs(dx))/2))
    
    print('pixOffset=',pixOffset)
    print('pixOffset0=',pixOffset0)

    for i in range(pixOffset0,int(nPix/nBin)+pixOffset):
        for j in range(pixOffset0,int(nPix/nBin)+pixOffset):
            if ds>0:
                newImage[i+dx,j+dx,ds:(targetNspec-ds+offset)] = newdata[:,i*nBin+pixOffset0:(i+1)*nBin+pixOffset,j*nBin+pixOffset0:(j+1)*nBin+pixOffset].sum(axis=all_but_first)
            elif ds<0:
                newImage[i+dx,j+dx,:] = newdata[-ds:nSpec+ds,i*nBin+pixOffset0:(i+1)*nBin+pixOffset,j*nBin+pixOffset0:(j+1)*nBin+pixOffset].sum(axis=all_but_first)


    print("max of new data",np.max(newImage))
    if saveImages:
        plt.imshow(np.sum(newImage,axis=2))
        plt.colorbar()
        plt.savefig(output+"_reducedImage.png")
        hf_reduced = h5py.File(output+"_reducedImage.hdf5",'w')
        hf_reduced.create_dataset('image',data=np.sum(newImage,axis=2))
        hf_reduced.close()
        plt.close()

    hf = h5py.File(output+".hdf5",'w')
    hf.create_dataset('spectra',data=newImage)
    hf.create_dataset('pixelRes_arcsec',data = originalPixelRes*nBin)
    hf.create_dataset('pixelRes_kpc',data = originalPixelRes*nBin /60/60 * np.pi/180 * distance)
    hf.close()
    
    print("nBin=",nBin)
    print("newPixelRes=",originalPixelRes*nBin," arcsec")
    print("newPixelRes=",originalPixelRes*nBin /60/60 * np.pi/180 * distance," kpc")

targetRes_kpc = 1 ###Performance was okay with this set to 1, aside from spatial scale being off while plotting

print("Running NGC 2403")
imageDir = 'CoNNGaFitData/observations/NGC_2403_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_2403'
distance = 3200 #kpc
pixelRes = 1 / 60 / 60 *np.pi/180 #arcSec
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
saveImages = True
Convert_THINGS_to_CoNNGaFit(imageDir,output,distance,pixelRes,targetRes_kpc,targetNpix,targetNspec,saveImages)

print("Running NGC 3351")
imageDir = 'CoNNGaFitData/observations/NGC_3351_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_3351'
distance = 10500 #kpc
pixelRes = 1 / 60 / 60 *np.pi/180 #arcSec
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
saveImages = True
Convert_THINGS_to_CoNNGaFit(imageDir,output,distance,pixelRes,targetRes_kpc,targetNpix,targetNspec,saveImages)

print("Running NGC 5055")
imageDir = 'CoNNGaFitData/observations/NGC_5055_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_5055'
distance = 8900 #kpc
pixelRes = 1 / 60 / 60 *np.pi/180 #arcSec
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
saveImages = True
Convert_THINGS_to_CoNNGaFit(imageDir,output,distance,pixelRes,targetRes_kpc,targetNpix,targetNspec,saveImages)

print("Running NGC 7793")
imageDir = 'CoNNGaFitData/observations/NGC_7793_NA_CUBE_THINGS.FITS'
output = 'CoNNGaFitData/observations/NGC_7793'
distance = 3600 #kpc
pixelRes = 1 / 60 / 60 *np.pi/180 #arcSec
targetNpix = 40
targetNspec = int(np.round(400 / 5.2))
saveImages = True
Convert_THINGS_to_CoNNGaFit(imageDir,output,distance,pixelRes,targetRes_kpc,targetNpix,targetNspec,saveImages)