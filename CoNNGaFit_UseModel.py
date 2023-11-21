import torch
#from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose
from CoNNGaFit_InferenceDatasets import CoNNGaFitImageInferenceDataset


import numpy as np
#from sklearn.preprocessing import StandardScaler    
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

import h5py

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


def RunInferences(networkType,imageOutput_prefix,imageList,params=None):

def LoadNames(inputDir):
    ####Get a distinct name for each image to run inferences on
    Nlines=0
    fid = open(inputDir,'r')
    names = np.zeros((0)).astype('str')
    for line in fid:
        x=line.split("\\")
        Nsplit = np.size(x)
        y=(x[Nsplit-1]).split('.') #get filename in directory
        names=np.append(names,y[0]) #ignore filetype
        
    fid.close()
    
    return names

    


def MakeImage(data,output,vmin=None,vmax=None,targetShape=None):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    
    if targetShape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        targetShape=[npix,npix]
        
    image = np.reshape(image,targetShape) #make into 2d image

    plt.figure()
    
    if vmax is None or vmin is None:
        vmax = np.max( [-np.min(image) , np.max(image)]) # default to symmetric limits
        vmin=-vmax
    
    #Plot in units of Solar Masses per year
    plt.imshow(image*Sim2PhysicalUnits_MassFlux,cmap='seismic',vmin=vmin*Sim2PhysicalUnits_MassFlux,vmax=vmax*Sim2PhysicalUnits_MassFlux)
    plt.colorbar(label = 'Radial Mass Flux [M$_{\odot}$ yr$^{-1}$]');
    plt.savefig(output+"projectionMap.png")
    plt.close()
    
    #Save hdf5 for generating figures
    hf.File(output+"projectionMap.hdf5")
    hf.create_dataset('image',data=image)
    hf.close()

def CreateBasicPlots(data,output,targetShape=None,nBins=20):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    if targetShape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        targetShape=[npix,npix]
    
    npix=targetShape[0]
    image = np.reshape(image,targetShape) #make into 2d image
    
    ## Calculate galactocentric radius of each pixel
    centerIndex = [npix/2-1,npix/2-1]
    indices = np.indices(np.shape(image))
    rmag = 2*np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    ##Calculate total mass flux as function of radius
    binRange=[0,np.max(rmag)]
    binned_data,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image.flatten(),"sum",nBins,range=[binRange])
    
    #Default to symmetric y-limits
    vmax = np.max( [-np.min(binned_data* Sim2PhysicalUnits_MassFlux) , np.max(binned_data* Sim2PhysicalUnits_MassFlux)])*1.05
    vmin=-vmax
    
    #Plot in units of Solar Masses per year
    plt.figure()
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data* Sim2PhysicalUnits_MassFlux , 'k', lw = 2.5) #plot azimuthal average
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data*0 , 'k--', lw = 1.5) #plot zero line
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Radial Mass Flux [M$_{\odot}$ yr$^{-1}$]')
    plt.ylim([vmin,vmax])
    plt.savefig(output+"radialPlot.png")
    plt.close()
    
    #Save hdf5 for generating figures
    hf.File(output+"radialPlot.hdf5")
    hf.create_dataset('binned_data',data=binned_data)
    hf.create_dataset('rPlot',data=np.linspace(0,np.max(rmag),nBins))

    hf.close()
   
    
    
def RunInferences(networkType,imageOutput_prefix,imageList,params=None):
 
    #####Set variables based on desired network type
    
    if params is None and networkType=='unet18': ####U-Net
        nFilt0 = 6 #Number of filters
        k0 = 9 #Initial kernel size
        k1 = 3 #Kernel Size
        nFC = 3000 #Nodes in FC layer
        learning_rate= 0.0001 #lr used in training
        weight_decay= 0.001 #wd used in training
        epochs= 4000 #epochs used in training
        params=[nFilt0,k0,k1,nFC]
        
        targetShape=[40,40] #output dimensions
        from CoNNGaFit_NeuralNetwork_Unet3d_18_parameterized import NeuralNetwork  
        
        paramSuffix = "lr"+str(learning_rate)+"_nFC"+str(nFC)+"_wd"+str(weight_decay)+"_nF"+str(nFilt0)+"_unetFullSpec"
        modelOutputPath = 'TrainedNetworks\\MassFlux_Unet18_FullSpec'+paramSuffix+'_epoch'+str(epochs)+'.pt' #Address of desired model
        
        model =NeuralNetwork(params[0],(params[1],params[1],params[1]),(params[2],params[2],params[2]),params[3]).to(device)


    model.load_state_dict(torch.load(modelOutputPath+".pt"))
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
        pred = model(X.float())

        imageOutput = imageOutput_prefix + inferenceNames[i] + imageOutput_suffix 
        plotOutput = imageOutput_prefix + inferenceNames[i] + plotOutput_suffix 

        MakeImage(pred.cpu().detach().numpy(),imageOutput,targetShape) #Generate Projection Map
        CreateBasicPlots(pred.cpu().detach().numpy(),plotOutput,targetShape) #Generate Radial Plots
        i+=1

    print("Done!")    

