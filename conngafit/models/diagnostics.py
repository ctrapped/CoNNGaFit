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
    """Number of parameters in a list of modules, counting each parameter once."""
    return sum({id(p): p.numel() for m in modules for p in m.parameters()}.values())


def layer_param_lines(modules, indent="      "):
    """One line per layer inside `modules` that owns parameters - dotted name, layer type and
    parameter count - so summary() can break a block's total down layer by layer. A module that is
    itself a single layer has nothing to break down and yields no lines."""
    entries = []
    for m in modules:
        for name, layer in m.named_modules():
            n = sum(p.numel() for p in layer.parameters(recurse=False))
            if n and name:
                entries.append((name, type(layer).__name__, n))
    if not entries:
        return []
    nw = max(len(e[0]) for e in entries)
    tw = max(len(e[1]) for e in entries)
    return [f"{indent}{name:<{nw}}  {kind:<{tw}}  params={n:,}" for name, kind, n in entries]
