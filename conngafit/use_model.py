import logging
import os
import tomllib

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit.datasets import TARGET_SETTINGS, CoNNGaFitImageInferenceDataset, ring_radii
from conngafit.networks import build_model
from conngafit.plotting import make_image

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


"""Functions to run a network trained by train_model_unet.py on HI spectral datacubes: VOF outputs, or
real observations converted with fits_to_conngafit.py. See scripts/run_inference.py for an example.

Written By Cameron Trapp (ctrapped@gmail.com)
"""


def load_model(model_path):
    """Rebuild a network from train_model_unet.py's <model_path>.hdf5 and load <model_path>.pt.
    Returns (model in eval mode, dict of the checkpoint's metadata)."""
    with h5py.File(model_path + ".hdf5", "r") as hf:
        if "hyperparams_text" not in hf:
            raise ValueError(f"{model_path}.hdf5 wasn't written by conngafit.train_model (no hyperparams_text)")
        metadata = {key: hf[key][()] for key in hf}
        # The targets in the order of the network's output maps
        targets = hf["target"].asstr()[()]  # One name (older checkpoints) or an array of names
        metadata["target"] = np.atleast_1d(targets).tolist()
    for key, value in metadata.items():  # h5py reads strings back as bytes
        if isinstance(value, bytes):
            metadata[key] = value.decode()
    if "network" not in metadata:
        raise ValueError(f"{model_path}.hdf5 doesn't record its network")
    hyperparams = tomllib.loads(metadata["hyperparams_text"])
    radial_bins = int(metadata["radial_bins"]) if "radial_bins" in metadata else None
    input_shape = tuple(int(n) for n in metadata["input_shape"])
    n_maps = len(metadata["target"])
    model = build_model(metadata["network"], input_shape, hyperparams, output_size=radial_bins, n_maps=n_maps)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path + ".pt", map_location=device, weights_only=True))
    return model.eval(), metadata


def predict(model, metadata, image_paths, save_latent_images=False, return_std=False, return_samples=False):
    """Run the network on each datacube in image_paths.

    Predictions are in physical units, with one map or radial curve per target in metadata["target"]:
    shape (n_targets, nY, nX) for maps, or (n_targets, n_bins) for radial curves.

    Yields:
        (name, prediction) by default.
        (name, prediction, std) with return_std. std is the ddpm network's
            posterior standard deviation, or the predicted 1-sigma of a network trained with predict_variance.
            Otherwise None.
        (name, samples) with return_samples (ddpm only). samples are the n_samples joint posterior samples,
            shape (n_samples, *prediction.shape), for propagating into quantities derived from several targets.

    save_latent_images is not supported by the ddpm network.
    """
    if save_latent_images and metadata["network"] == "ddpm":
        raise ValueError("save_latent_images needs the resblock network")
    if return_samples and metadata["network"] != "ddpm":
        raise ValueError("return_samples needs the ddpm network")
    image_mean, image_stdv = metadata["image_stats"]  # Normalized like the training images
    # One offset and scale per target, shaped to broadcast over each target's map (or radial curve)
    stats_shape = (-1, 1) if "radial_bins" in metadata else (-1, 1, 1)
    label_offset, label_scale = (np.reshape(stats, stats_shape) for stats in metadata["label_stats"])
    dataset = CoNNGaFitImageInferenceDataset(
        image_paths, transform=Compose([ToTensor(), Normalize(image_mean, image_stdv)])
    )
    for name, image in zip(dataset.names, dataset):
        cube = image[None].float().to(device)
        with torch.no_grad():
            if return_samples:
                samples = model.sample(cube, int(metadata["n_samples"]))[:, 0]  # (n_samples, n_maps, nY, nX)
            elif metadata["network"] == "ddpm":
                mean, std = model.predict_with_uncertainty(cube, int(metadata["n_samples"]))
            else:
                output = model(cube, save_latent_images)
                mean, logvar = output if model.predict_variance else (output, None)
                std = None if logvar is None else torch.exp(0.5 * logvar)
        if return_samples:
            yield name, samples.cpu().numpy() * label_scale + label_offset  # Physical units
            continue
        pred = mean.cpu().numpy()[0] * label_scale + label_offset  # Back to physical units
        if not return_std:
            yield name, pred
            continue
        # 1-sigma in physical units: a spread, so scaled by label_scale but not offset
        if std is not None:
            std = std.cpu().numpy()[0] * label_scale
        yield name, pred, std


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


def save_radial_curve(curve, radii, output, param_label, unit_label, uncertainty=None):
    """Save a predicted radial curve to output + "radialPlot.png" and output + "radialPlot.hdf5". With uncertainty
    (the predicted 1-sigma of each ring), its +/-1 sigma is shaded around the curve and saved too."""
    x = np.arange(len(curve)) if radii is None else radii
    plt.figure()
    plt.plot(x, curve, "k", lw=2.5, label="Prediction")
    if uncertainty is not None:
        plt.fill_between(x, curve - uncertainty, curve + uncertainty, color="k", alpha=0.2, lw=0, label=r"$\pm1\sigma$")
    plt.axhline(0, color="k", ls="--", lw=1.5, label="Zero line")
    plt.xlabel("Radial bin" if radii is None else "Radius [kpc]")
    plt.ylabel(param_label + " " + unit_label)
    plt.legend()
    plt.savefig(output + "radialPlot.png", bbox_inches="tight")  # "tight" keeps the y-axis label in the figure
    plt.close()

    with h5py.File(output + "radialPlot.hdf5", "w") as hf:
        hf.create_dataset("prediction", data=curve)
        if uncertainty is not None:
            hf.create_dataset("uncertainty", data=uncertainty)
        if radii is not None:
            hf.create_dataset("r_plot", data=radii)


def run_inferences(model_path, image_paths, output_prefix, save_latent_images=False):
    """Run the network at model_path (no .pt/.hdf5 extension) on each datacube in image_paths, saving a PNG and an
    hdf5 of each prediction under output_prefix: <name>_projectionMap for maps, <name>_radialPlot for radial curves.
    Each target's outputs are named <name>_<target>_. Networks trained with predict_variance also save and plot each
    prediction's 1-sigma uncertainty."""
    logger.info("Using %s device", device)
    model, metadata = load_model(model_path)
    targets = metadata["target"]
    logger.info("Loaded %s (%s, %s network)", model_path, ", ".join(targets), metadata["network"])

    image_paths = list(image_paths)
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    predictions = predict(model, metadata, image_paths, save_latent_images, return_std=True)
    for path, (name, preds, stds) in zip(image_paths, predictions):
        for m, target in enumerate(targets):
            param_label, unit_label, _ = TARGET_SETTINGS[target]
            pred = preds[m]
            std = None if stds is None else stds[m]
            output = f"{output_prefix}{name}_{target}_"
            if pred.ndim == 1:
                radii = ring_radii_kpc(path, len(pred))
                save_radial_curve(pred, radii, output, param_label, unit_label, uncertainty=std)
            else:
                # Predictions are already in physical units
                make_image(
                    pred,
                    output,
                    sim_to_physical_units=1,
                    param_label=param_label,
                    unit_label=unit_label,
                    uncertainty=std,
                )
            logger.info("Saved %s", output)

    logger.info("Done!")


def read_image_list(path):
    """Datacube paths from a text/CSV file: the first comma-separated field of each non-blank line."""
    with open(path) as f:
        return [line.split(",")[0].strip() for line in f if line.strip()]
