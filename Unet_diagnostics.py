import numpy as np
from matplotlib import pyplot as plt

# Latent-image diagnostics for the CoNNGaFit 3-d networks: PNG dumps of intermediate
# activations, useful for visually checking what each encoder/decoder stage has learned.
# Set latentImageOutput to change where the images are written.
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
