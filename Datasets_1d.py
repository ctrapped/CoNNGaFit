import os
import pandas as pd
import h5py
import numpy as np
from astropy.io import fits
from torchvision.io import read_image
import scipy.stats as stats

Nlabels = 40*40*1
npix=40
#### Dataset classes for training and inference (1-d radial profile output)

####Dataset class for training. Reads in annotation file. Get loads the image in the directory of the first value of each line, then loads the rest of the entries into 'label'.
####Unlike Datasets.py, the 2-d (npix x npix) label map is azimuthally binned down
####to a 1-d radial profile here, so the network is trained against radial curves rather than
####full 2-d maps.
class CoNNGaFitImageDataset():
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        print(self.img_labels)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx,0])
        
        tmp = img_path.split(".")
        filetype=tmp[len(tmp)-1]
        
        
        if  filetype=="hdf5":
            hf = h5py.File(img_path,'r')
            image = np.array(hf['spectra']).astype('float')
        elif filetype=="fits":
            hdul = fits.open(img_path)
            image = np.array(hdul[0].data)
        else:
            image = read_image(img_path)

        label = np.array(self.img_labels.iloc[idx,1:Nlabels+1]).astype('float')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        #Bin the flattened npix*npix label map by galactocentric radius (in pixels) to get a
        #1-d radial profile. Note: centerIndex uses (npix/2 - 1) here rather than the
        #(npix-1)/2 convention used in PlottingFunctions.py - for npix=40 that's
        #pixel (19,19) vs (19.5,19.5), a half-pixel difference in the assumed galaxy center
        #between how labels are binned here and how predictions are re-binned for plotting.
        label_image = np.reshape(label,[npix,npix])
        indices = np.indices(np.shape(label_image))
        centerIndex = [npix/2-1,npix/2-1]

        rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
        nBins = npix
    
        binRange=[0,np.max(rmag)]
        binned_label,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),label_image.flatten(),"sum",nBins,range=[binRange])
        return image, binned_label



####Dataset class for inferences. Reads in annotation file. Get loads the image in the directory of the first value of each line and ignores everything else
class CoNNGaFitImageInferenceDataset():
    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        print(self.img_labels)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx,0])
        
        tmp = img_path.split(".")
        filetype=tmp[len(tmp)-1]
        
        if  filetype=="hdf5":
            hf = h5py.File(img_path,'r')
            image = np.array(hf['spectra']).astype('float')
        elif filetype=="fits":
            hdul = fits.open(img_path)
            image = np.array(hdul[0].data)
        else:
            image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        return image

