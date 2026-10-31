import glob
import logging
import os
import re

import h5py
import numpy as np

logger = logging.getLogger(__name__)

#### Dataset classes for training (VOF images with annotations) and inference (datacubes only)


#### Finding VOF images ####
# Where VOF's optically thin projection writes each datacube (with its annotations), under the
# output directory given to VOF:
#   <output>/i<inclination>/<galaxy>_i<inclination>_pa<position angle>_<snapshot>_image<outputSuffix>_fullSpectra.hdf5
VOF_IMAGE_NAME = "*_image{suffix}_fullSpectra.hdf5"
# The viewing angles and snapshot in a VOF filename, just before "_image"
VOF_VIEW = re.compile(r"_i(?P<inclination>[^_]+)_pa(?P<position_angle>[^_]+)_(?P<snapshot>\d+)_image")


def snapshot_range(snapshots, where):
    """(minSnap, maxSnap) from a snapshot number or an inclusive [minSnap, maxSnap] pair."""
    if isinstance(snapshots, int):
        # One snapshot tha is an interger
        return snapshots, snapshots
    if isinstance(snapshots, (list, tuple)) and len(snapshots) == 2 and all(isinstance(v, int) for v in snapshots):
        # A list of two snapshots, which are integers
        return snapshots[0], snapshots[1]
    raise ValueError(f"{where}: snapshots = {snapshots!r} should be a snapshot or [minSnap, maxSnap]")


def inclination_folders(inclinations, where):
    """The i<inclination>/ folders to search: all of them for "all" (or None), otherwise those listed (in degrees)."""
    if inclinations is None or inclinations == "all":
        return ["i*"]
    if (
        isinstance(inclinations, (list, tuple))
        and inclinations
        and all(isinstance(v, (int, float)) for v in inclinations)
    ):
        return [f"i{inclination:g}" for inclination in inclinations]  # :g writes 45 and 45.0 as "45", like VOF
    raise ValueError(f'{where}: inclinations = {inclinations!r} should be "all" or a list such as [30, 45]')


def find_vof_images(sources, inclinations="all", suffix=""):
    """Return the paths of every VOF datacube in each source directory's snapshot range, in source order.

    sources maps each VOF output directory (the folder holding its i<inclination>/ folders) to a
    snapshot or an inclusive [minSnap, maxSnap] range. inclinations ("all", or a list in degrees)
    and suffix (VOF's outputSuffix, default "": files without one) apply to every source; every
    position angle found is included.
    """
    folders = inclination_folders(inclinations, "find_vof_images")
    filename = VOF_IMAGE_NAME.format(suffix=glob.escape(suffix))
    paths = []
    for data_dir, snapshots in sources.items():
        min_snap, max_snap = snapshot_range(snapshots, data_dir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"No directory {data_dir}")
        found = []
        for folder in folders:
            # escape() so characters like [ ] in the directory name aren't read as wildcards
            for path in glob.glob(os.path.join(glob.escape(data_dir), folder, filename)):
                view = VOF_VIEW.search(os.path.basename(path))
                if view and min_snap <= int(view["snapshot"]) <= max_snap:
                    found.append(path)
        if not found:
            looked_for = os.path.join("{" + ",".join(folders) + "}", filename)
            raise FileNotFoundError(
                f"No VOF images for snapshots {min_snap}-{max_snap} in {data_dir} matching {looked_for}"
            )
        paths += sorted(found)
    return paths


def face_on_path(path):
    """Return the path of the face-on (inclination 0) VOF image for the same galaxy, snapshot, and suffix as path.

    Uses the same position angle if that file exists; otherwise any face-on position angle (the
    position angle only spins a face-on disk, so its radial curves are the same).
    """
    name = os.path.basename(path)
    view = VOF_VIEW.search(name)
    galaxy, rest = name[: view.start()], name[view.end() :]  # rest: VOF's outputSuffix and "_fullSpectra.hdf5"
    face_on_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "i0")
    same_pa = os.path.join(face_on_dir, f"{galaxy}_i0_pa{view['position_angle']}_{view['snapshot']}_image{rest}")
    if os.path.isfile(same_pa):
        return same_pa
    any_pa = f"{glob.escape(galaxy)}_i0_pa*_{view['snapshot']}_image{glob.escape(rest)}"
    others = sorted(glob.glob(os.path.join(glob.escape(face_on_dir), any_pa)))
    if not others:
        raise FileNotFoundError(
            f"No face-on (i0) image for {name}; run VOF with inclination 0, or use radial_projection='sky'"
        )
    return others[0]


#### Labels ####
# Training targets, and the VOF annotation maps each is made from (multiplied together)
TARGETS = {
    "mass": ["mass_annotation"],
    "radial_velocity": ["radial_velocity_annotation"],
    "cylindrical_radial_velocity": ["cylindrical_radial_velocity_annotation"],
    "rotational_velocity": ["rotational_velocity_annotation"],
    "radial_momentum": ["mass_annotation", "radial_velocity_annotation"],
    "cylindrical_radial_momentum": ["mass_annotation", "cylindrical_radial_velocity_annotation"],
    "rotational_momentum": ["mass_annotation", "rotational_velocity_annotation"],
}
# target -> (plot label, plot units, how pixels are combined when binning plots)
TARGET_SETTINGS = {
    "mass": ("Mass", r"[M$_{\odot}$]", "sum"),
    "radial_velocity": ("Rad. Vel.", r"[km s$^{-1}$]", "mean"),
    "cylindrical_radial_velocity": ("Cyl. Rad. Vel.", r"[km s$^{-1}$]", "mean"),
    "rotational_velocity": ("Rot. Vel.", r"[km s$^{-1}$]", "mean"),
    "radial_momentum": ("Radial Momentum", r"[M$_{\odot}$ km s$^{-1}$]", "sum"),
    "cylindrical_radial_momentum": ("Cyl. Radial Momentum", r"[M$_{\odot}$ km s$^{-1}$]", "sum"),
    "rotational_momentum": ("Rot. Momentum", r"[M$_{\odot}$ km s$^{-1}$]", "sum"),
}
# Momentum targets, mass * velocity [Msun km/s]: summed in radial curves, not mass-weighted like velocities
MOMENTUM_TARGETS = {"radial_momentum", "cylindrical_radial_momentum", "rotational_momentum"}


def label_map(hf, target):
    """target's map from an open VOF file: Msun (mass), km/s (velocities), or Msun km/s (momenta) per pixel."""
    # e.g. radial momentum = mass * radial velocity, per pixel
    target_maps = TARGETS[target]
    if len(target_maps) > 1:
        # Multiply Mass and Velocities for Momenta
        label = np.multiply(hf[target_maps[0]][...], hf[target_maps[1]][...])
    else:
        label = hf[target_maps[0]][...]
    return label


def radial_profile(hf, target, n_bins):
    """Reduce an open VOF file's annotation maps to a radial curve of target, in n_bins equal-width rings.

    Radius is measured in the image plane from the image center, out to the image edge. Mass and
    momentum are the totals in each ring [Msun, Msun km/s], and velocities the mass-weighted mean
    [km/s], zero in rings with no mass.
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
    if target in MOMENTUM_TARGETS:
        return momentum
    return np.divide(momentum, ring_mass, out=np.zeros(n_bins), where=ring_mass > 0)


def ring_radii(fov_kpc, n_bins):
    """Centers (kpc) of radial_profile's n_bins rings, for an image fov_kpc across."""
    edges = np.linspace(0, fov_kpc / 2, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


#### Augmentation ####
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
    return (f"_r{90 * rotations}" if rotations else "") + ("_lr" if flip else "")


####Dataset class for training. Finds each galaxy's VOF datacubes; get loads one (optionally rotated/flipped) datacube and its target annotation map or radial curve ('label')
class CoNNGaFitImageDataset:
    """PyTorch-style dataset of (datacube, label) pairs from VeryObservableFIRE's optically thin projection

    sources: which VOF outputs to include, mapping each VOF output directory to its snapshots, e.g.
        {"/scratch/m12i/vof_outputs": (580, 600), "/scratch/m12f/vof_outputs": 600} (see
        find_vof_images); snapshot ranges are inclusive, and every position angle found is used.
    target: which annotation to learn, a key of TARGETS (e.g. "radial_momentum",
        "rotational_velocity"). Labels are returned flattened, one value per image pixel, in
        Msun (mass), km/s (velocities), or Msun km/s (momenta).
    transform / target_transform: callables applied to the loaded image / label respectively
        (e.g. torchvision ToTensor/Normalize for the image, label scaling for the label).
    augment: None (default), "major_axis", or "all" (see AUGMENTATIONS): include each image's
        rotations/flips as extra samples, with the label map transformed to match. "major_axis"
        adds 3 (180-degree rotation, left-right flip, and both), keeping VOF's major axis along x;
        "all" adds 7 (every 90-degree rotation, with and without the flip).
    radial_bins: if given, each label is a radial curve with this many rings (see radial_profile)
        instead of a map; rotations and flips then leave it unchanged.
    radial_projection: where radial curves are measured. "face_on" (default) uses the same galaxy and
        snapshot's face-on (inclination 0) VOF image, so radii are in the disk plane; "sky" uses
        each image's own annotations, with radii in the sky plane.
    inclinations: which inclinations to include, "all" (default) or a list in degrees such as
        [30, 45], for every source.
    suffix: VOF's outputSuffix in the filenames to load (default "": files without one), for every source.

    The names attribute holds a short name for each sample (its filename without extension, plus
    an augmentation suffix such as "_r180_lr"), in dataset order.
    """

    def __init__(
        self,
        sources,
        target,
        transform=None,
        target_transform=None,
        augment=None,
        radial_bins=None,
        radial_projection="face_on",
        inclinations="all",
        suffix="",
    ):
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; choose from {sorted(TARGETS)}")
        if augment not in AUGMENTATIONS:
            raise ValueError(f"Unknown augment {augment!r}; choose from {list(AUGMENTATIONS)}")
        if radial_projection not in ("face_on", "sky"):
            raise ValueError(f"radial_projection must be 'face_on' or 'sky', not {radial_projection!r}")
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        self.radial_bins = radial_bins

        self.img_paths = find_vof_images(sources, inclinations, suffix)
        # Sample idx is augmentation (idx % n) of file (idx // n)
        self.augmentations = AUGMENTATIONS[augment]
        self.names = []
        for path in self.img_paths:
            filename = os.path.splitext(os.path.basename(path))[0]
            for rotations, flip in self.augmentations:
                self.names.append(filename + augmentation_suffix(rotations, flip))
        logger.info(
            "Found %d images in %d folders, %d samples with augmentation %s",
            len(self.img_paths),
            len(sources),
            len(self.names),
            augment,
        )

        self.masses = {}  # mass map or ring masses per file, for mass_weights

        if radial_bins is not None:
            # Where each image's radial curve comes from; check every face-on file exists up front
            if radial_projection == "face_on":
                self.curve_paths = [face_on_path(path) for path in self.img_paths]
            else:
                self.curve_paths = self.img_paths
            self.curves = {}  # curve per file, computed on first use (rotations/flips don't change it)

    def __len__(self):
        return len(self.names)

    def _label(self, hf, file_idx):
        """file_idx's label before augmentation: a 2-d map read from the open file hf, or its radial curve."""
        if self.radial_bins is None:
            return label_map(hf, self.target)
        if file_idx not in self.curves:  # Has the radial curve already been computed?
            with h5py.File(self.curve_paths[file_idx], "r") as hf_curve:
                # Calculate Radial Curve if not already done
                self.curves[file_idx] = radial_profile(hf_curve, self.target, self.radial_bins)
        return self.curves[file_idx]

    def image(self, file_idx):
        """The datacube of image file_idx before augmentation and transform (e.g. to compute image normalization)."""
        with h5py.File(self.img_paths[file_idx], "r") as hf:
            return np.array(hf["spectra"]).astype("float")

    def label(self, file_idx):
        """The flattened label of image file_idx before augmentation and target_transform (e.g. to compute label scaling)."""
        with h5py.File(self.img_paths[file_idx], "r") as hf:
            return self._label(hf, file_idx).flatten().astype("float")

    def mass_weights(self, idx, power):
        """Sample idx's loss weights, mass ** power per pixel (or ring), flattened like its label, with mean 1."""
        file_idx, aug_idx = divmod(idx, len(self.augmentations))
        if file_idx not in self.masses:
            if self.radial_bins is None:
                with h5py.File(self.img_paths[file_idx], "r") as hf:
                    self.masses[file_idx] = hf["mass_annotation"][()]
            else:  # The mass in each ring, from the file the curve is measured on
                with h5py.File(self.curve_paths[file_idx], "r") as hf:
                    self.masses[file_idx] = radial_profile(hf, "mass", self.radial_bins)
        weights = self.masses[file_idx] ** power
        if self.radial_bins is None:
            weights = augment_array(weights, *self.augmentations[aug_idx])  # Rotated/flipped like the label
        weights = weights.flatten().astype("float")
        # Mean 1, so every image counts equally however massive; an image with no mass is weighted uniformly
        return weights / weights.mean() if weights.mean() > 0 else np.ones_like(weights)

    def radii(self, idx):
        """Ring centers (kpc) of sample idx's radial curve, from the field of view of the image it was measured on."""
        with h5py.File(self.curve_paths[idx // len(self.augmentations)], "r") as hf:
            return ring_radii(hf.attrs["fov_kpc"], self.radial_bins)

    def __getitem__(self, idx):
        file_idx, aug_idx = divmod(idx, len(self.augmentations))
        rotations, flip = self.augmentations[aug_idx]
        path = self.img_paths[file_idx]
        with h5py.File(path, "r") as hf:
            image = np.array(hf["spectra"]).astype("float")
            label = self._label(hf, file_idx)
        if rotations % 2 and image.shape[0] != image.shape[1]:
            raise ValueError(
                f"{path}: 90-degree rotations need a square image, not {image.shape[:2]}; use augment='major_axis'"
            )
        image = augment_array(image, rotations, flip)
        if self.radial_bins is None:
            # The annotations are projected onto the same pixels as the datacube's two sky axes
            if label.shape != image.shape[:2]:
                raise ValueError(f"{path}: {self.target} map is {label.shape}, but the image is {image.shape}")
            label = augment_array(label, rotations, flip)
        label = label.flatten().astype("float")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


####Dataset class for inference. Loads datacubes with no labels, e.g. VOF images or real observations converted with fits_to_conngafit.py
class CoNNGaFitImageInferenceDataset:
    """PyTorch-style dataset of datacubes to run a trained network on, with no labels.

    paths: the hdf5 files to load, each holding a 'spectra' datacube (y, x, velocity channel) -
        VOF's *_fullSpectra.hdf5 outputs, or real observations converted with fits_to_conngafit.py.
    transform: callable applied to each loaded datacube (e.g. torchvision ToTensor/Normalize).

    The names attribute holds each datacube's filename without extension, in dataset order.
    """

    def __init__(self, paths, transform=None):
        self.img_paths = list(paths)
        missing = [path for path in self.img_paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"No datacube file(s): {missing}")
        self.transform = transform
        self.names = [os.path.splitext(os.path.basename(path))[0] for path in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with h5py.File(self.img_paths[idx], "r") as hf:
            image = np.array(hf["spectra"]).astype("float")
        if self.transform:
            image = self.transform(image)
        return image
