import numpy as np
from matplotlib import pyplot as plt

# Diagnostics for the CoNNGaFit 3-d networks:
#   - latent images: PNG dumps of intermediate activations, useful for visually checking what each
#     encoder/decoder stage has learned. Set latentImageOutput to change where they are written.
#   - parameter counts: helpers each network's summary() uses to report parameters per block/layer.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.

latentImageOutput="latentImages/latent_"
def SaveAllSpectralChannels(x,nStep):
    """Save one PNG per (filter, spectral channel) of activation tensor x (nBatch,C,nSpec,nX,nY).

    Writes latentImageOutput + "step<nStep>_filter<i>_nSpec<j>.png" for the first batch
    element. Diagnostic only.
    """
    npX = x.cpu().float().detach().numpy();
    print(np.shape(x))
    for i in range(0,np.shape(npX)[1]):
        for j in range(0,np.shape(npX)[2]):
            plt.figure()
            plt.imshow(npX[0,i,j,:,:],cmap='inferno')
            plt.savefig(latentImageOutput+"step"+str(nStep)+"_filter"+str(i)+"_nSpec"+str(j)+".png")
            plt.close()

def SaveSummedSpectralChannels(x,nStep,saveItr):
    """Save one PNG per filter of activation tensor x, summed over the spectral axis.

    x has shape (nBatch, C, nSpec, nX, nY); writes latentImageOutput +
    "sum_step<nStep>_filter<i>_gal<saveItr>.png" for the first batch element. saveItr is
    the caller's forward() call counter (NeuralNetwork.saveItr). Diagnostic only.
    """
    print(np.shape(x))
    npX = x.cpu().float().detach().numpy();
    toPlot = np.sum(npX,axis=2)
    for i in range(0,np.shape(npX)[1]):
        plt.figure()
        plt.imshow(toPlot[0,i,:,:],cmap='inferno')
        plt.savefig(latentImageOutput+"sum_step"+str(nStep)+"_filter"+str(i)+"_gal"+str(saveItr)+".png")
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
