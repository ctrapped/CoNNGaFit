import logging
import os
import tomllib

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit_datasets import TARGET_SETTINGS, CoNNGaFitImageInferenceDataset, ring_radii
from plotting_functions import make_image
from unet3d_resblock import NeuralNetwork

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

####Functions to run a network trained by train_model_unet.py on HI spectral datacubes: VOF outputs, or
####real observations converted with fits_to_conngafit.py. See inference_template.py for an example.
####
####Written By Cameron Trapp (ctrapped@gmail.com)


def load_model(model_path):
    """Rebuild a network from train_model_unet.py's <model_path>.hdf5 and load <model_path>.pt.
    Returns (model in eval mode, dict of the checkpoint's metadata)."""
    with h5py.File(model_path + ".hdf5", "r") as hf:
        if "hyperparams_text" not in hf:
            raise ValueError(f"{model_path}.hdf5 wasn't written by train_model_unet.py (no hyperparams_text)")
        metadata = {key: hf[key][()] for key in hf}
    for key, value in metadata.items():  # h5py reads strings back as bytes
        if isinstance(value, bytes):
            metadata[key] = value.decode()
    hyperparams = tomllib.loads(metadata["hyperparams_text"])
    k0, k1 = hyperparams["k0"], hyperparams["k1"]
    radial_bins = int(metadata["radial_bins"]) if "radial_bins" in metadata else None
    model = NeuralNetwork(
        tuple(int(n) for n in metadata["input_shape"]),
        hyperparams["n_filt0"],
        (k0, k0, k0),
        (k1, k1, k1),
        hyperparams["n_fc"],
        fc_dropout_rate=hyperparams["dropout_rate"],
        block_dropout_rate=hyperparams["block_dropout_rate"],
        head=metadata["output_head"],
        output_size=radial_bins,
    ).to(device)
    model.load_state_dict(torch.load(model_path + ".pt", map_location=device, weights_only=True))
    return model.eval(), metadata


def predict(model, metadata, image_paths, save_latent_images=False):
    """Yield (name, prediction) for each datacube in image_paths, in physical units: a 2-d map, or a radial curve."""
    image_mean, image_stdv = metadata["image_stats"]  # Normalized like the training images
    label_offset, label_scale = metadata["label_stats"]
    dataset = CoNNGaFitImageInferenceDataset(
        image_paths, transform=Compose([ToTensor(), Normalize(image_mean, image_stdv)])
    )
    n_y, n_x = (int(n) for n in metadata["input_shape"][1:])
    with torch.no_grad():
        for name, image in zip(dataset.names, dataset):
            pred = model(image[None].float().to(device), save_latent_images).cpu().numpy()[0]
            pred = pred * label_scale + label_offset  # Back to physical units
            yield name, (pred if "radial_bins" in metadata else pred.reshape(n_y, n_x))


def ring_radii_kpc(path, n_bins):
    """Ring centers in kpc for path's radial curve (see radial_profile), or None if path doesn't record its size."""
    with h5py.File(path, "r") as hf:
        if "fov_kpc" in hf.attrs:  # VOF outputs
            fov_kpc = hf.attrs["fov_kpc"]
        elif "pixel_res_kpc" in hf:  # fits_to_conngafit.py outputs
            fov_kpc = hf["pixel_res_kpc"][()] * hf["spectra"].shape[1]
        else:
            return None
    return ring_radii(fov_kpc, n_bins)


def save_radial_curve(curve, radii, output, param_label, unit_label):
    """Save a predicted radial curve to output + "radialPlot.png" and output + "radialPlot.hdf5"."""
    x = np.arange(len(curve)) if radii is None else radii
    plt.figure()
    plt.plot(x, curve, "k", lw=2.5)
    plt.axhline(0, color="k", ls="--", lw=1.5)
    plt.xlabel("Radial bin" if radii is None else "Radius [kpc]")
    plt.ylabel(param_label + " " + unit_label)
    plt.savefig(output + "radialPlot.png", bbox_inches="tight")  # "tight" keeps the y-axis label in the figure
    plt.close()

    with h5py.File(output + "radialPlot.hdf5", "w") as hf:
        hf.create_dataset("prediction", data=curve)
        if radii is not None:
            hf.create_dataset("r_plot", data=radii)


def run_inferences(model_path, image_paths, output_prefix, save_latent_images=False):
    """Run the network at model_path (no .pt/.hdf5 extension) on each datacube in image_paths, saving a PNG and an
    hdf5 of each prediction under output_prefix: <name>_projectionMap for maps, <name>_radialPlot for radial curves."""
    logger.info("Using %s device", device)
    model, metadata = load_model(model_path)
    target = metadata["target"]
    param_label, unit_label, _ = TARGET_SETTINGS[target]
    logger.info("Loaded %s (%s, %s head)", model_path, target, metadata["output_head"])

    image_paths = list(image_paths)
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    for path, (name, pred) in zip(image_paths, predict(model, metadata, image_paths, save_latent_images)):
        output = output_prefix + name + "_"
        if pred.ndim == 1:
            radii = ring_radii_kpc(path, len(pred))
            save_radial_curve(pred, radii, output, param_label, unit_label)
        else:
            # make_image takes the flattened map; predictions are already in physical units
            make_image(
                pred.reshape(1, -1),
                output,
                target_shape=pred.shape,
                sim_to_physical_units=1,
                param_label=param_label,
                unit_label=unit_label,
            )
        logger.info("Saved %s", output)

    logger.info("Done!")


def read_image_list(path):
    """Datacube paths from a text/CSV file: the first comma-separated field of each non-blank line."""
    with open(path) as f:
        return [line.split(",")[0].strip() for line in f if line.strip()]
