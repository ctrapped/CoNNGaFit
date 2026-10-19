import glob
import os
import re
import pandas as pd
import h5py
import numpy as np
from astropy.io import fits
from torchvision.io import read_image

#### Dataset classes for training and inference

# Where VOF's optically thin projection writes each datacube (with its annotations) under data_dir:
#   <data_dir>/<galaxy>/vof_outputs/i<inclination>/<galaxy>_i<inclination>_pa<position angle>_<snapshot>_image_fullSpectra.hdf5
VOF_IMAGE_GLOB = os.path.join("{galaxy}", "vof_outputs", "i*", "{galaxy}_i*_image_fullSpectra.hdf5")
# The part of the filename after the galaxy name
VOF_IMAGE_NAME = re.compile(r"_i(?P<inclination>[^_]+)_pa(?P<position_angle>[^_]+)_(?P<snapshot>\d+)_image_fullSpectra\.hdf5")

KM_PER_KPC = 3.086e16
S_PER_YR = 3.154e7

# Training targets, and the VOF annotation maps each is made from (multiplied together)
TARGETS = {
    "mass": ["mass_annotation"],
    "radial_velocity": ["radial_velocity_annotation"],
    "cylindrical_radial_velocity": ["cylindrical_radial_velocity_annotation"],
    "rotational_velocity": ["rotational_velocity_annotation"],
    "radial_mass_flux": ["mass_annotation", "radial_velocity_annotation"],
    "cylindrical_radial_mass_flux": ["mass_annotation", "cylindrical_radial_velocity_annotation"],
}
# Targets converted from Msun km/s to a mass flux in Msun/yr (see __getitem__)
MASS_FLUX_TARGETS = {"radial_mass_flux", "cylindrical_radial_mass_flux"}


def find_vof_images(data_dir, galaxies):
    """Return the paths of every VOF datacube for each galaxy's snapshot range, sorted.

    galaxies maps each galaxy name to a snapshot number or an inclusive (minSnap, maxSnap) range,
    e.g. {"m12i": (580, 600), "m12f": 600}. Every inclination and position angle found is included.
    """
    paths = []
    for galaxy, snapshots in galaxies.items():
        min_snap, max_snap = (snapshots, snapshots) if np.isscalar(snapshots) else snapshots
        pattern = os.path.join(data_dir, VOF_IMAGE_GLOB.format(galaxy=glob.escape(galaxy)))
        found = []
        for path in glob.glob(pattern):
            match = VOF_IMAGE_NAME.fullmatch(os.path.basename(path)[len(galaxy):])
            if match and min_snap <= int(match["snapshot"]) <= max_snap:
                key = (int(match["snapshot"]), float(match["inclination"]), float(match["position_angle"]))
                found.append((key, path))
        if not found:
            raise FileNotFoundError(f"No VOF images for {galaxy} snapshots {min_snap}-{max_snap} matching {pattern}")
        paths += [path for _, path in sorted(found)]
    return paths


####Dataset class for training. Finds each galaxy's VOF datacubes; get loads one datacube and its target annotation map ('label')
class CoNNGaFitImageDataset():
    """PyTorch-style dataset of (datacube, label-map) pairs from VOF's optically thin outputs, for training CoNNGaFit networks.

    data_dir: root directory holding each galaxy's VOF outputs (see VOF_IMAGE_GLOB above).
    galaxies: which galaxies and snapshots to include, e.g. {"m12i": (580, 600), "m12f": 600};
        snapshot ranges are inclusive, and every inclination/position angle found is used.
    target: which annotation to learn, a key of TARGETS (e.g. "radial_mass_flux",
        "rotational_velocity"). Labels are returned flattened, one value per image pixel, in
        Msun (mass), km/s (velocities), or Msun/yr (mass fluxes).
    transform / target_transform: callables applied to the loaded image / label respectively
        (e.g. torchvision ToTensor/Normalize).
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

    The names attribute holds a short name for each datacube (its filename without extension), in dataset order.
    """

    def __init__(self, data_dir, galaxies, target, transform=None, target_transform=None, map_labels_to_tanh="False", output_transform_filedir=None):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; choose from {sorted(TARGETS)}")
        self.img_paths = find_vof_images(data_dir, galaxies)
        self.names = [os.path.basename(path).split(".")[0] for path in self.img_paths]
        print(f"Found {len(self.img_paths)} images for {galaxies}")
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.map_labels_to_tanh=map_labels_to_tanh
        self.output_transform_filedir=output_transform_filedir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with h5py.File(self.img_paths[idx],'r') as hf:
            image = np.array(hf['spectra']).astype('float')
            # e.g. radial mass flux = mass * radial velocity, per pixel
            label_map = np.prod([hf[name][()] for name in TARGETS[self.target]], axis=0)
            if self.target in MASS_FLUX_TARGETS:
                # Mass flux through a shell one pixel wide: mass * velocity / pixel width [Msun/yr]
                pixel_kpc = hf.attrs["fov_kpc"] / label_map.shape[1]
                label_map = label_map / (pixel_kpc * KM_PER_KPC) * S_PER_YR

        # The annotations are projected onto the same pixels as the datacube's two sky axes
        if label_map.shape != image.shape[:2]:
            raise ValueError(f"{self.img_paths[idx]}: {self.target} map is {label_map.shape}, but the image is {image.shape}")
        label = label_map.flatten().astype('float')
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
        self.img_labels = pd.read_csv(annotations_file, header=None) # No header row; the first line is an image
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

