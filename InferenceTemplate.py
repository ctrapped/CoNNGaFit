import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler    
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

import h5py
from CoNNGaFit_UseModel import RunInferences

#### For Direct Use ####  

epochs= 4000

dirName = 'fire2_velocitySpace'



networkType = 'unet18'
imageList = "CoNNGaFitData/observations/inferencesToRun.csv"
imageOutput = 'CoNNGaFitData/fire2_velocitySpace/i50/images/comparisons/output_example_'

if networkType=='unet18':
    nFilt0 = 6
    k0 = 9
    k1 = 3
    nFC = 3000
    learning_rate= 0.0001
    weight_decay= 0.001
    
    
outputSuffix = "lr"+str(learning_rate)+"_nFC"+str(nFC)+"_wd"+str(weight_decay)+"_nF"+str(nFilt0)+"_unetFullSpec"
imageOutput = 'CoNNGaFitData\\'+dirName+'\\i50\\images\\comparisons\\'+subFolder+'\\massFlux'+outputSuffix+'_'


#modelPath = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\trainedNetworks\\MassFlux_Unet18_FullSpec'+outputSuffix+'_epoch'+str(epochs)+'.pt'
if subDir is None:
    modelPath = 'CoNNGaFitData\\'+dirName+'\\i50\\trainedNetworks\\MassFlux_Unet18_FullSpec'+outputSuffix+'_epoch'+str(epochs)+'.pt'
else:
    modelPath = 'CoNNGaFitData\\'+dirName+'\\i50\\trainedNetworks\\'+subDir+'\\MassFlux_Unet18_FullSpec'+outputSuffix+'_epoch'+str(epochs)+'.pt'

RunInferences(networkType , modelPath , dirName, imageOutput, imageList, [nFilt0,k0,k1,nFC])
