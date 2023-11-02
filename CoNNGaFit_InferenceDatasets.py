import os
import pandas as pd
import h5py
import numpy as np
from torchvision.io import read_image

Nlabels = 20*20*1

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
            i=0
            #load fits file here
        else:
            image = read_image(img_path)

        #label = np.array(self.img_labels.iloc[idx,1:Nlabels+1]).astype('float')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image#, label

