import logging
import os

import h5py
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit_datasets import CoNNGaFitImageInferenceDataset
from plotting_functions import create_basic_plots, load_names, make_image

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

SIM_TO_PHYSICAL_UNITS_MASS_FLUX = (
    (2 / np.pi) * 1 / (3.086 * np.power(10.0, 16.0)) * (3.154 * np.power(10.0, 7.0)) * np.power(10, 10)
)  # 1/pixel_res * kpc2km * s2yr * unit mass to solar masses

####Functions to calculate inferences on HI spectral datacubes from a previously trained CoNNGaFit network. Generates a projection map and a radial plot for the fitted parameter of interest.
####For use, you need the trained network and the images to run the inferences on with a corresponding .csv file listing their location.
####Call run_inferences, providing the network type you wish to use, the output directory, and the .csv file containing the input images. See inference_template.py for an example
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023


def run_inferences(
    network_type, image_output_prefix, image_list, model_path=None, params=None, save_latent_images=False
):
    """Load a previously trained CoNNGaFit network and run it on a list of input datacubes,
    writing a projection-map PNG/hdf5 and a radial-profile PNG/hdf5 per input image.

    network_type: which trained network configuration to use. Currently only 'unet18' is
        implemented; any other value raises ValueError.
    image_output_prefix: directory/filename prefix prepended to every output file written.
    image_list: path to a CSV listing the input datacubes to run inference on (see
        CoNNGaFitImageInferenceDataset in conngafit_datasets.py for the expected format).
    model_path: path (without the trailing .pt/.hdf5 extension) to the trained model checkpoint
        to load, overriding the network_type's hardcoded default checkpoint below. Pass None to
        use the default checkpoint for the chosen network_type.
    params: [n_filt0, k0, k1, n_fc] architecture params to build the network with, overriding the
        network_type's hardcoded defaults below (e.g. to load a checkpoint that was trained
        with different hyperparameters than the current default). Pass None to use the
        default params for the chosen network_type.
    save_latent_images: if True, dump intermediate activation images during the forward pass
        (see NeuralNetwork.forward's save_latent_images argument) - useful for debugging only.
    """
    logger.info("Using %s device", device)

    #####Set variables based on desired network type

    if network_type == "unet18":  ####U-Net
        target_shape = [40, 40]  # output dimensions
        from NetworkModels.AlternativeNetworks.Unet3d_HiResTest import NeuralNetwork

        model_output_path = model_path or os.path.join(
            "TrainedNetworks", "hiResTests", "MassFlux_Unet18_FullSpec_finalSnapNoM12m.pt"
        )  # Address of desired model

        if params is None:
            n_filt0 = 6  # Number of filters
            k0 = 9  # Initial kernel size
            k1 = 3  # Kernel Size
            n_fc = 1500  # Nodes in FC layer
            params = [n_filt0, k0, k1, n_fc]

        model = NeuralNetwork(
            params[0], (params[1], params[1], params[1]), (params[2], params[2], params[2]), params[3]
        ).to(device)
    else:
        raise ValueError(f"Unrecognized network_type '{network_type}' - only 'unet18' is currently implemented.")

    model.load_state_dict(torch.load(model_output_path))
    model.eval()

    image_output_suffix = "_massFlux_FacePlot_" + network_type + "_TR_Comp_"
    plot_output_suffix = "_massFlux_RadialPlot_" + network_type + "_TR_Comp_"

    try:  # Get normalization (saved under the 'imageStats' dataset by the training scripts)
        hf_model = h5py.File(model_output_path + ".hdf5", "r")
        image_stats = np.array(hf_model["imageStats"])
        image_mean = image_stats[0]
        image_stdv = image_stats[1]
    except (OSError, KeyError):  # Default to per image normalization if training values not available for some reason
        dataset0 = CoNNGaFitImageInferenceDataset(annotations_file=image_list, root_dir=".", transform=ToTensor())

        loader0 = torch.utils.data.DataLoader(dataset0, batch_size=1, shuffle=False)
        data0 = next(iter(loader0))
        image_mean = data0[0].mean()
        image_stdv = data0[0].std()
        del loader0
        del data0
        del dataset0

    # Load spectral datacubes to run data on
    dataset = CoNNGaFitImageInferenceDataset(
        annotations_file=image_list, root_dir=".", transform=Compose([ToTensor(), Normalize(image_mean, image_stdv)])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Get names of images
    inference_names = load_names(image_list)

    # Make inference for each image loaded
    for i, inputs in enumerate(dataloader):
        x = inputs.to(device)
        pred = model(x.float(), save_latent_images)

        image_output = image_output_prefix + inference_names[i] + image_output_suffix
        plot_output = image_output_prefix + inference_names[i] + plot_output_suffix

        make_image(pred.cpu().detach().numpy(), image_output, target_shape=target_shape)  # Generate Projection Map
        create_basic_plots(pred.cpu().detach().numpy(), plot_output, target_shape)  # Generate Radial Plots

    logger.info("Done!")
