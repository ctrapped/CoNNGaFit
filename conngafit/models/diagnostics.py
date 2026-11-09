import logging

import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# Diagnostics for the CoNNGaFit 3-d networks:
#   - latent images: PNG dumps of intermediate activations, useful for visually checking what each
#     encoder/decoder stage has learned. Set LATENT_IMAGE_OUTPUT to change where they are written.
#   - parameter counts: helpers each network's summary() uses to report parameters per block/layer.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.

LATENT_IMAGE_OUTPUT = "latentImages/latent_"


def save_all_spectral_channels(x, step):
    """Save one PNG per (filter, spectral channel) of activation tensor x (nBatch,C,nSpec,nX,nY).

    Writes LATENT_IMAGE_OUTPUT + "step<step>_filter<i>_nSpec<j>.png" for the first batch
    element. Diagnostic only.
    """
    np_x = x.cpu().float().detach().numpy()
    logger.debug("Saving latent images for activations of shape %s", tuple(x.shape))
    for i in range(0, np.shape(np_x)[1]):
        for j in range(0, np.shape(np_x)[2]):
            plt.figure()
            plt.imshow(np_x[0, i, j, :, :], cmap="inferno")
            plt.savefig(LATENT_IMAGE_OUTPUT + "step" + str(step) + "_filter" + str(i) + "_nSpec" + str(j) + ".png")
            plt.close()


def save_summed_spectral_channels(x, step, save_itr):
    """Save one PNG per filter of activation tensor x, summed over the spectral axis.

    x has shape (nBatch, C, nSpec, nX, nY); writes LATENT_IMAGE_OUTPUT +
    "sum_step<step>_filter<i>_gal<save_itr>.png" for the first batch element. save_itr is
    the caller's forward() call counter (NeuralNetwork.save_itr). Diagnostic only.
    """
    logger.debug("Saving latent images for activations of shape %s", tuple(x.shape))
    np_x = x.cpu().float().detach().numpy()
    to_plot = np.sum(np_x, axis=2)
    for i in range(0, np.shape(np_x)[1]):
        plt.figure()
        plt.imshow(to_plot[0, i, :, :], cmap="inferno")
        plt.savefig(LATENT_IMAGE_OUTPUT + "sum_step" + str(step) + "_filter" + str(i) + "_gal" + str(save_itr) + ".png")
        plt.close()


def count_params(modules):
    """Number of parameters in a list of modules, counting each parameter once
    (a parameter can appear more than once if the modules overlap or share it)."""
    counted = set()
    total = 0
    for module in modules:
        for param in module.parameters():
            if id(param) not in counted:
                counted.add(id(param))
                total += param.numel()
    return total


def layer_param_lines(modules, indent="      "):
    """Text lines breaking the modules' parameters down by layer, for summary().

    One line per layer that has parameters, in aligned columns, e.g.
        conv1     Conv2d     params=296
        norm1     GroupNorm  params=16
    A module that is a single layer (e.g. a bare Conv2d) gives no lines.
    """
    layers = []
    for module in modules:
        for name, layer in module.named_modules():
            if name == "":  # named_modules() lists the module itself first, with an empty name
                continue
            n_params = sum(p.numel() for p in layer.parameters(recurse=False))  # this layer's own parameters
            if n_params > 0:
                layers.append((name, type(layer).__name__, n_params))
    if not layers:
        return []

    name_width = max(len(name) for name, _, _ in layers)
    type_width = max(len(layer_type) for _, layer_type, _ in layers)
    lines = []
    for name, layer_type, n_params in layers:
        lines.append(f"{indent}{name.ljust(name_width)}  {layer_type.ljust(type_width)}  params={n_params:,}")
    return lines
