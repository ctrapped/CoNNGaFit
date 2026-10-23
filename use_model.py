import logging
import os

import h5py
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit_datasets import CoNNGaFitImageInferenceDataset
from plotting_functions import CreateBasicPlots, LoadNames, MakeImage

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

Sim2PhysicalUnits_MassFlux = (
    (2 / np.pi) * 1 / (3.086 * np.power(10.0, 16.0)) * (3.154 * np.power(10.0, 7.0)) * np.power(10, 10)
)  # 1/pixel_res * kpc2km * s2yr * unit mass to solar masses

####Functions to calculate inferences on HI spectral datacubes from a previously trained CoNNGaFit network. Generates a projection map and a radial plot for the fitted parameter of interest.
####For use, you need the trained network and the images to run the inferences on with a corresponding .csv file listing their location.
####Call RunInferences, providing the network type you wish to use, the output directory, and the .csv file containing the input images. See inference_template.py for an example
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023


def RunInferences(networkType, imageOutput_prefix, imageList, modelPath=None, params=None, saveLatentImages=False):
    """Load a previously trained CoNNGaFit network and run it on a list of input datacubes,
    writing a projection-map PNG/hdf5 and a radial-profile PNG/hdf5 per input image.

    networkType: which trained network configuration to use. Currently only 'unet18' is
        implemented; any other value raises ValueError.
    imageOutput_prefix: directory/filename prefix prepended to every output file written.
    imageList: path to a CSV listing the input datacubes to run inference on (see
        CoNNGaFitImageInferenceDataset in conngafit_datasets.py for the expected format).
    modelPath: path (without the trailing .pt/.hdf5 extension) to the trained model checkpoint
        to load, overriding the networkType's hardcoded default checkpoint below. Pass None to
        use the default checkpoint for the chosen networkType.
    params: [nFilt0,k0,k1,nFC] architecture params to build the network with, overriding the
        networkType's hardcoded defaults below (e.g. to load a checkpoint that was trained
        with different hyperparameters than the current default). Pass None to use the
        default params for the chosen networkType.
    saveLatentImages: if True, dump intermediate activation images during the forward pass
        (see NeuralNetwork.forward's save_latent_images argument) - useful for debugging only.
    """
    logger.info("Using %s device", device)

    #####Set variables based on desired network type

    if networkType == "unet18":  ####U-Net
        targetShape = [40, 40]  # output dimensions
        from NetworkModels.AlternativeNetworks.Unet3d_HiResTest import NeuralNetwork

        modelOutputPath = modelPath or os.path.join(
            "TrainedNetworks", "hiResTests", "MassFlux_Unet18_FullSpec_finalSnapNoM12m.pt"
        )  # Address of desired model

        if params is None:
            nFilt0 = 6  # Number of filters
            k0 = 9  # Initial kernel size
            k1 = 3  # Kernel Size
            nFC = 1500  # Nodes in FC layer
            params = [nFilt0, k0, k1, nFC]

        model = NeuralNetwork(
            params[0], (params[1], params[1], params[1]), (params[2], params[2], params[2]), params[3]
        ).to(device)
    else:
        raise ValueError(f"Unrecognized networkType '{networkType}' - only 'unet18' is currently implemented.")

    model.load_state_dict(torch.load(modelOutputPath))
    model.eval()

    imageOutput_suffix = "_massFlux_FacePlot_" + networkType + "_TR_Comp_"
    plotOutput_suffix = "_massFlux_RadialPlot_" + networkType + "_TR_Comp_"

    try:  # Get normalization (saved under the 'imageStats' dataset by the training scripts)
        hf_model = h5py.File(modelOutputPath + ".hdf5", "r")
        imageStats = np.array(hf_model["imageStats"])
        imageMean = imageStats[0]
        imageStdv = imageStats[1]
    except (OSError, KeyError):  # Default to per image normalization if training values not available for some reason
        dataset0 = CoNNGaFitImageInferenceDataset(annotations_file=imageList, root_dir=".", transform=ToTensor())

        loader0 = torch.utils.data.DataLoader(dataset0, batch_size=1, shuffle=False)
        data0 = next(iter(loader0))
        imageMean = data0[0].mean()
        imageStdv = data0[0].std()
        del loader0
        del data0
        del dataset0

    # Load spectral datacubes to run data on
    dataset = CoNNGaFitImageInferenceDataset(
        annotations_file=imageList, root_dir=".", transform=Compose([ToTensor(), Normalize(imageMean, imageStdv)])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Get names of images
    inferenceNames = LoadNames(imageList)

    # Make inference for each image loaded
    for i, inputs in enumerate(dataloader):
        X = inputs.to(device)
        pred = model(X.float(), saveLatentImages)

        imageOutput = imageOutput_prefix + inferenceNames[i] + imageOutput_suffix
        plotOutput = imageOutput_prefix + inferenceNames[i] + plotOutput_suffix

        MakeImage(pred.cpu().detach().numpy(), imageOutput, targetShape=targetShape)  # Generate Projection Map
        CreateBasicPlots(pred.cpu().detach().numpy(), plotOutput, targetShape)  # Generate Radial Plots

    logger.info("Done!")
