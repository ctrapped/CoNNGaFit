import os
import pandas as pd
import h5py
import numpy as np
from astropy.io import fits
from torchvision.io import read_image

Nlabels = 40*40*1
#### Dataset classes for training and inference (40x40 spatial output, e.g. 2-d radial mass flux/velocity maps)

####Dataset class for training. Reads in annotation file. Get loads the image in the directory of the first value of each line, then loads the rest of the entries into 'label'
class CoNNGaFitImageDataset():
    """PyTorch-style dataset for (datacube, label-map) pairs used to train CoNNGaFit networks.

    annotations_file: CSV where column 0 is the path (relative to root_dir) to the input
        datacube (.hdf5 with a 'spectra' or 'moments' dataset, .fits, or a plain image), and
        columns 1..Nlabels are the flattened target label map (e.g. a 40x40 mass-flux map).
    root_dir: directory prepended to each path in annotations_file.
    transform / target_transform: callables applied to the loaded image / label respectively
        (e.g. torchvision ToTensor/Normalize).
    use_moment_maps: if True, read the 'moments' dataset instead of 'spectra' from hdf5 inputs.
    map_labels_to_tanh: label-normalization mode, one of:
        "False" (default) - no normalization, labels used as-is.
        "symmetric" - divide by max(|label|) so labels fall in [-1, 1] (e.g. mass flux).
        "one-sided" - rescale to [-1, 1] around the midpoint of [min(label), max(label)],
            clipping min to 0 (e.g. rotational velocity, inclination).
        "symmetric_load" / "one-sided_load" - apply the *same* transform as above but using
            normalization stats previously written to output_transform_filedir (for
            validation/test sets, which must reuse the training set's normalization).
    output_transform_filedir: path to an hdf5 file used to persist the running max/min label
        values seen across the dataset, so a "*_load" mode can later reproduce the exact
        same normalization on new data.
    """

    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None, use_moment_maps=False, map_labels_to_tanh="False", output_transform_filedir=None):
        self.img_labels = pd.read_csv(annotations_file)
        print(self.img_labels)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_moment_maps=use_moment_maps
        self.map_labels_to_tanh=map_labels_to_tanh
        self.output_transform_filedir=output_transform_filedir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx,0])

        tmp = img_path.split(".")
        filetype=tmp[len(tmp)-1]


        if  filetype=="hdf5":
            hf = h5py.File(img_path,'r')
            if self.use_moment_maps:
                image = np.array(hf['moments']).astype('float')
            else:
                image = np.array(hf['spectra']).astype('float')
        elif filetype=="fits":
            hdul = fits.open(img_path)
            image = np.array(hdul[0].data)
        else:
            image = read_image(img_path)

        label = np.array(self.img_labels.iloc[idx,1:Nlabels+1]).astype('float')
        if self.map_labels_to_tanh=="symmetric": # For mass flux, -1 to 1
            max_label = np.max(np.abs(label))
            label = np.divide(label,max_label)
            if self.output_transform_filedir is not None:
                try: hf_ot = h5py.File(self.output_transform_filedir,'r+')
                except: hf_ot = h5py.File(self.output_transform_filedir,'w')

                try:
                    current_max_label = hf_ot['max_label'][...]
                    if max_label>current_max_label: current_max_label=max_label
                except:
                    current_max_label=max_label
                try:hf_ot.create_dataset("max_label",data=current_max_label)
                except:hf_ot['max_label'][...]=current_max_label
                hf_ot.close()
        if self.map_labels_to_tanh=="one-sided": # For RC and inc, positive?
            max_label = np.max(label)
            min_label = np.min(label)
            if (min_label<0): min_label=0
            halfRange = (max_label-min_label)/2.
            
            label = (label-min_label-halfRange) / halfRange
            if self.output_transform_filedir is not None:
                try: hf_ot = h5py.File(self.output_transform_filedir,'r+')
                except: hf_ot = h5py.File(self.output_transform_filedir,'w')

                try:
                    current_max_label = hf_ot['max_label'][...]
                    current_min_label = hf_ot['min_label'][...]
                    if max_label>current_max_label: current_max_label=max_label
                    if min_label<current_min_label: current_min_label=min_label
                except:
                    current_max_label=max_label
                    current_min_label=min_label
                try: hf_ot.create_dataset("max_label",data=current_max_label)
                except:hf_ot['max_label'][...]=current_max_label
                try: hf_ot.create_dataset("min_label",data=current_min_label)
                except: hf_ot['min_label'][...]=current_min_label
             
                hf_ot.close()

                
        if self.map_labels_to_tanh=="symmetric_load": # For mass flux, -1 to 1
            hf_ot = h5py.File(self.output_transform_filedir,'r')
            max_label = np.array(hf_ot["max_label"])/1.
            hf_ot.close()

            label = np.divide(label,max_label)

        if self.map_labels_to_tanh=="one-sided_load": # For RC and inc, positive?
            hf_ot = h5py.File(self.output_transform_filedir,'r')
            max_label = hf_ot["max_label"][...]
            min_label = hf_ot["min_label"][...]
            hf_ot.close()

            halfRange = (max_label-min_label)/2.
            label = (label-min_label-halfRange) / halfRange

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



####Dataset class for inferences. Reads in annotation file. Get loads the image in the directory of the first value of each line and ignores everything else
class CoNNGaFitImageInferenceDataset():
    """Dataset for running inference only: yields datacubes with no labels.

    Same annotations_file/root_dir/transform/use_moment_maps semantics as
    CoNNGaFitImageDataset (see above), except only column 0 (the image path) is read -
    any label columns present in the CSV are ignored.
    """

    def __init__(self, annotations_file, root_dir, transform=None, target_transform=None,use_moment_maps=False):
        self.img_labels = pd.read_csv(annotations_file)
        print(self.img_labels)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_moment_maps=use_moment_maps

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx,0])

        tmp = img_path.split(".")
        filetype=tmp[len(tmp)-1]

        if  filetype=="hdf5":
            hf = h5py.File(img_path,'r')
            if self.use_moment_maps:
                image = np.array(hf['moments']).astype('float')
            else:
                image = np.array(hf['spectra']).astype('float')
        elif filetype=="fits":
            hdul = fits.open(img_path)
            image = np.array(hdul[0].data)
        else:
            image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        return image

