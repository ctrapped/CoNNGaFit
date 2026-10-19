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

# Augmentations, as (number of 90-degree rotations, left-right flip). VOF puts the galaxy's major
# axis along x; "major_axis" keeps it there, while "all" (every rotation/flip of a square image)
# also includes 90/270-degree rotations that put it along y.
AUGMENTATIONS = {
    None: [(0, False)],
    "major_axis": [(0, False), (2, False), (0, True), (2, True)],
    "all": [(k, flip) for k in range(4) for flip in (False, True)],
}


def augment_array(a, rotations, flip):
    """Rotate a's first two (y, x) axes by rotations*90 degrees, then optionally flip left-right."""
    a = np.rot90(a, rotations, axes=(0, 1))
    if flip:
        a = np.flip(a, axis=1)
    return np.ascontiguousarray(a)  # torch can't wrap the negative strides rot90/flip leave


def augmentation_suffix(rotations, flip):
    """Name suffix for an augmentation, e.g. "_r180_lr" (empty for the original)."""
    return (f"_r{90*rotations}" if rotations else "") + ("_lr" if flip else "")


def radial_profile(hf, target, n_bins):
    """Reduce an open VOF file's annotation maps to a radial curve of target, in n_bins equal-width rings.

    Radius is measured in the image plane from the image center, out to the image edge. Mass is
    the total in each ring [Msun], mass flux the flux through each ring [Msun/yr], and velocities
    the mass-weighted mean [km/s], zero in rings with no mass.
    """
    mass = hf["mass_annotation"][()]
    fov_kpc = hf.attrs["fov_kpc"]
    y, x = np.indices(mass.shape)
    radius = np.hypot(y - (mass.shape[0] - 1) / 2, x - (mass.shape[1] - 1) / 2) * fov_kpc / mass.shape[1]
    edges = np.linspace(0, fov_kpc / 2, n_bins + 1)
    ring = np.digitize(radius, edges) - 1
    inside = ring < n_bins  # pixels in the corners, beyond the image edge, are left out

    def ring_sum(values):
        return np.bincount(ring[inside], weights=values[inside], minlength=n_bins)

    ring_mass = ring_sum(mass)
    if target == "mass":
        return ring_mass
    momentum = ring_sum(mass * hf[TARGETS[target][-1]][()])  # Msun km/s
    if target in MASS_FLUX_TARGETS:
        # Flux through a shell one ring wide: mass * velocity / ring width [Msun/yr]
        ring_kpc = edges[1] - edges[0]
        return momentum / (ring_kpc * KM_PER_KPC) * S_PER_YR
    return np.divide(momentum, ring_mass, out=np.zeros(n_bins), where=ring_mass > 0)


def face_on_path(path):
    """Return the path of the face-on (inclination 0) VOF image for the same galaxy and snapshot as path.

    Uses the same position angle if that file exists; otherwise any face-on position angle (the
    position angle only spins a face-on disk, so its radial curves are the same).
    """
    name = os.path.basename(path)
    match = re.search(VOF_IMAGE_NAME.pattern + "$", name)
    galaxy = name[:match.start()]
    face_on_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "i0")
    same_pa = os.path.join(face_on_dir, f"{galaxy}_i0_pa{match['position_angle']}_{match['snapshot']}_image_fullSpectra.hdf5")
    if os.path.isfile(same_pa):
        return same_pa
    others = sorted(glob.glob(os.path.join(glob.escape(face_on_dir), f"{glob.escape(galaxy)}_i0_pa*_{match['snapshot']}_image_fullSpectra.hdf5")))
    if not others:
        raise FileNotFoundError(f"No face-on (i0) image for {name}; run VOF with inclination 0, or use radial_plane='sky'")
    return others[0]


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


####Dataset class for training. Finds each galaxy's VOF datacubes; get loads one (optionally rotated/flipped) datacube and its target annotation map or radial curve ('label')
class CoNNGaFitImageDataset():
    """PyTorch-style dataset of (datacube, label-map) pairs from VOF's optically thin outputs, for training CoNNGaFit networks.

    data_dir: root directory holding each galaxy's VOF outputs (see VOF_IMAGE_GLOB above).
    galaxies: which galaxies and snapshots to include, e.g. {"m12i": (580, 600), "m12f": 600};
        snapshot ranges are inclusive, and every inclination/position angle found is used.
    target: which annotation to learn, a key of TARGETS (e.g. "radial_mass_flux",
        "rotational_velocity"). Labels are returned flattened, one value per image pixel, in
        Msun (mass), km/s (velocities), or Msun/yr (mass fluxes).
    augment: None (default), "major_axis", or "all" (see AUGMENTATIONS): include each image's
        rotations/flips as extra samples, with the label map transformed to match. "major_axis"
        adds 3 (180-degree rotation, left-right flip, and both), keeping VOF's major axis along x;
        "all" adds 7 (every 90-degree rotation, with and without the flip).
    radial_bins: if given, each label is a radial curve with this many rings (see radial_profile)
        instead of a map; rotations and flips then leave it unchanged.
    radial_plane: where radial curves are measured. "face_on" (default) uses the same galaxy and
        snapshot's face-on (inclination 0) VOF image, so radii are in the disk plane; "sky" uses
        each image's own annotations, with radii in the sky plane.
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

    The names attribute holds a short name for each sample (its filename without extension, plus
    an augmentation suffix such as "_r180_lr"), in dataset order.
    """

    def __init__(self, data_dir, galaxies, target, transform=None, target_transform=None, map_labels_to_tanh="False", output_transform_filedir=None, augment=None, radial_bins=None, radial_plane="face_on"):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; choose from {sorted(TARGETS)}")
        if augment not in AUGMENTATIONS:
            raise ValueError(f"Unknown augment {augment!r}; choose from {list(AUGMENTATIONS)}")
        if radial_plane not in ("face_on", "sky"):
            raise ValueError(f"radial_plane must be 'face_on' or 'sky', not {radial_plane!r}")
        self.img_paths = find_vof_images(data_dir, galaxies)
        # Sample idx is augmentation (idx % n) of file (idx // n)
        self.augmentations = AUGMENTATIONS[augment]
        self.names = [os.path.basename(path).split(".")[0] + augmentation_suffix(*aug)
                      for path in self.img_paths for aug in self.augmentations]
        print(f"Found {len(self.img_paths)} images for {galaxies}, {len(self.names)} samples with augmentation {augment}")
        self.radial_bins = radial_bins
        if radial_bins is not None:
            # Where each image's radial curve comes from; check every face-on file exists up front
            self.curve_paths = [face_on_path(p) for p in self.img_paths] if radial_plane == "face_on" else self.img_paths
            self.curves = {}  # curve per file, computed on first use (rotations/flips don't change it)
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.map_labels_to_tanh=map_labels_to_tanh
        self.output_transform_filedir=output_transform_filedir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        file_idx, aug_idx = divmod(idx, len(self.augmentations))
        rotations, flip = self.augmentations[aug_idx]
        path = self.img_paths[file_idx]
        with h5py.File(path,'r') as hf:
            image = np.array(hf['spectra']).astype('float')
            if self.radial_bins is None:
                # e.g. radial mass flux = mass * radial velocity, per pixel
                label_map = np.prod([hf[name][()] for name in TARGETS[self.target]], axis=0)
                if self.target in MASS_FLUX_TARGETS:
                    # Mass flux through a shell one pixel wide: mass * velocity / pixel width [Msun/yr]
                    pixel_kpc = hf.attrs["fov_kpc"] / label_map.shape[1]
                    label_map = label_map / (pixel_kpc * KM_PER_KPC) * S_PER_YR
        if rotations % 2 and image.shape[0] != image.shape[1]:
            raise ValueError(f"{path}: 90-degree rotations need a square image, not {image.shape[:2]}; use augment='major_axis'")
        image = augment_array(image, rotations, flip)

        if self.radial_bins is None:
            # The annotations are projected onto the same pixels as the datacube's two sky axes
            if label_map.shape != image.shape[:2]:
                raise ValueError(f"{path}: {self.target} map is {label_map.shape}, but the image is {image.shape}")
            label = augment_array(label_map, rotations, flip).flatten().astype('float')
        else:
            if file_idx not in self.curves:
                with h5py.File(self.curve_paths[file_idx],'r') as hf:
                    self.curves[file_idx] = radial_profile(hf, self.target, self.radial_bins)
            label = self.curves[file_idx].astype('float')
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

