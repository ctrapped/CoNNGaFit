import os
import pandas as pd
import h5py
import numpy as np
from torchvision.io import read_image

Nlabels = 40*40*1
#### Dataset classes for training and inference

####Dataset class for training. Reads in annotation file. Get loads the image in the directory of the first value of each line, then loads the rest of the entries into 'label'
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
            hdul = fits.open(imageDir)
            image = np.array(hdul[0].data) 
        else:
            image = read_image(img_path)

        label = np.array(self.img_labels.iloc[idx,1:Nlabels+1]).astype('float')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



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
            hdul = fits.open(imageDir)
            image = np.array(hdul[0].data) 
        else:
            image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image

