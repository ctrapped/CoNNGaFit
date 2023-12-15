import torch
#from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose
from CoNNGaFit_Datasets import CoNNGaFitImageInferenceDataset


import numpy as np
#from sklearn.preprocessing import StandardScaler    
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

import h5py

from CoNNGaFit_PlottingFunctions import LoadNames,CreateBasicPlots,MakeImage


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')

CUDA_LAUNCH_BLOCKING=1

Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses

####Functions to calculate inferences on HI spectral datacubes from a previously trained CoNNGaFit network. Generates a projection map and a radial plot for the fitted parameter of interest. 
####For use, you need the trained network and the images to run the inferences on with a corresponding .csv file listing their location. 
####Call RunInferences, providing the network type you wish to use, the output directory, and the .csv file containing the input images. See InferenceTemplate.py for an example
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023


    
def RunInferences(networkType,imageOutput_prefix,imageList,params=None,saveLatentImages=False):
 
    #####Set variables based on desired network type
    
    if params is None and networkType=='unet18': ####U-Net
        nFilt0 = 6 #Number of filters
        k0 = 9 #Initial kernel size
        k1 = 3 #Kernel Size
        nFC = 1500 #Nodes in FC layer
        learning_rate= 0.0001 #lr used in training
        weight_decay= 0.001 #wd used in training
        epochs= 100 #epochs used in training
        params=[nFilt0,k0,k1,nFC]
        
        targetShape=[40,40] #output dimensions
        from CoNNGaFit_NeuralNetwork_Unet3d_HiResTest import NeuralNetwork  
        
        modelOutputPath = 'TrainedNetworks\\hiResTests\\MassFlux_Unet18_FullSpec_finalSnapNoM12m.pt' #Address of desired model
        
        model =NeuralNetwork(params[0],(params[1],params[1],params[1]),(params[2],params[2],params[2]),params[3]).to(device)


    model.load_state_dict(torch.load(modelOutputPath))
    model.eval()

    imageOutput_suffix = '_massFlux_FacePlot_'+networkType+"_TR_Comp_"
    plotOutput_suffix = '_massFlux_RadialPlot_'+networkType+"_TR_Comp_"
    compositeOutput_suffix = '_massFlux_CompositePlot_'+networkType+"_TR_Comp_"


    try: #Get normalization
        hf_model = h5py.File(modelOutputPath+".hdf5",'r')
        imageStats = np.array(hf_model['normalizationStats'])
        imageMean = imageStats[0]
        imageStdv = imageStats[1]
    except: #Default to per image normalization if training values not available for some reason
        dataset0 = CoNNGaFitImageInferenceDataset(annotations_file=imageList,
                                   root_dir = '.',
                                   transform = ToTensor())
                                   
                                   
        loader0 = torch.utils.data.DataLoader(dataset0,batch_size=1,shuffle=False)
        data0 = next(iter(loader0))
        imageMean = data0[0].mean()
        imageStdv = data0[0].std()
        del loader0;del data0;del dataset0

    #Load spectral datacubes to run data on
    dataset = CoNNGaFitImageInferenceDataset(annotations_file=imageList,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(imageMean,imageStdv)]))
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

    #Get names of images
    inferenceNames = LoadNames(imageList)

    i=1
    #Make inference for each image loaded
    for inputs in dataloader:
        X=inputs.to(device)
        pred = model(X.float(),saveLatentImages)

        imageOutput = imageOutput_prefix + inferenceNames[i] + imageOutput_suffix 
        plotOutput = imageOutput_prefix + inferenceNames[i] + plotOutput_suffix 

        MakeImage(pred.cpu().detach().numpy(),imageOutput,targetShape) #Generate Projection Map
        CreateBasicPlots(pred.cpu().detach().numpy(),plotOutput,targetShape) #Generate Radial Plots
        i+=1

    print("Done!")    

