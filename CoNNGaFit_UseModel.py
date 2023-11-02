import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose
import sys
from CoNNGaFit_InferenceDatasets import CoNNGaFitImageInferenceDataset

#try:
#    networkType = sys.argv[1]
#except:
#    networkType = "shallow"



import numpy as np
from sklearn.preprocessing import StandardScaler    
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

import h5py

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')

CUDA_LAUNCH_BLOCKING=1

#training_data = CoNNGaFitDatasets.Fire2_cr_datasets( #Probably need to update completely with your own data loading
#    root="data",
#    train=True,
#    download=True,
#    transform=ToTensor()
#)


Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses
saveItr=0

def LoadNames(inputDir):
    Nlines=0
    fid = open(inputDir,'r')
    for line in fid:
        Nlines+=1
    fid.close()
    fid = open(inputDir,'r')
    names = np.zeros((Nlines)).astype('str')
    i=0
    for line in fid:
        x=line.split("\\")
        Nsplit = np.size(x)
        y=(x[Nsplit-1]).split('.')
        names[i] = y[0]
        i+=1
        #print("y=",y)
        #print("y0=",y[0])
        
    fid.close()
        
    return names

    
def MakeInitialDiagnosticImage(dataloader):
    size = len(dataloader.dataset)
    num_batches=len(dataloader)

    actuals = np.array([])
    
    print("Making initial diagnostic image...")
    
    with torch.no_grad():
        for X, y in dataloader:
            y=y.to(device)
            actuals = np.append(actuals,y.cpu().detach().numpy())
            vmax = np.max(np.abs(actuals))
            vmin = -vmax
            MakeImage(actuals,imageOutput_input+str(saveItr)+"_input_"+str(epoch)+".png",vmin,vmax,tt)


def CompileImages(data,output,tag):
    i=NpixelNeighbors-1
    j = NpixelNeighbors-1
    image = np.zeros((Npixels,Npixels))
    Nsnap=0
    k=0
    
    print("Trying to compile image...")
    
    for idx in range(0,np.size(data)):
        flag = data[idx]
        if k<Nspec:
            image[j,i]+=flag
        else:
            k=-1
            j+=1
            if (j>Npixels-NpixelNeighbors):
                j=NpixelNeighbors-1
                i+=1
                if i>Npixels-NpixelNeighbors:
                    fig = plt.figure()
                    vmin = np.min(image)
                    vmax = np.max(image)
                    if -vmin>vmax:
                        vmax = -vmin
                    else:
                        vmin = -vmax
                    plt.imshow(image)
                    plt.title("Snapshot "+str(Nsnap)+": "+tag)
                    plt.colorbar()
                    plt.savefig(output+"_"+str(Nsnap)+"_"+tag+".png")
                    Nsnap+=1
                    image = 0*image
                    i = NpixelNeighbors-1
        
        k+=1
        
    if Nsnap==0:
                    fig = plt.figure()

                    plt.imshow(image)
                    plt.colorbar()
                    plt.title("Snapshot "+str(Nsnap)+": "+tag)
                    plt.savefig(output+"_"+str(Nsnap)+"_"+tag+".png")



def MakeImage(data,output,vmin,vmax,targetShape=[20,20]):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    npix = int(np.round(np.sqrt(np.size(image))))
    image = np.reshape(image,targetShape)
    #image -= np.min(image)
    #image=np.sum(image,2)
    plt.figure()
    vmax = np.max( [-np.min(image) , np.max(image)])
    vmin=-vmax
    saturationFactor=1
    plt.imshow(image*Sim2PhysicalUnits_MassFlux,cmap='seismic',vmin=vmin*Sim2PhysicalUnits_MassFlux*saturationFactor,vmax=vmax*Sim2PhysicalUnits_MassFlux*saturationFactor)
    plt.colorbar(label = 'Radial Mass Flux [M$_{\odot}$ yr$^{-1}$]');
    #plt.title("Radial Mass Flux")
    plt.savefig(output+"projectionMap.png")
    plt.close()
    
    hf.File(output+"projectionMap.hdf5")
    hf.create_dataset('image',data=image)
    hf.close()

def CreateBasicPlots(data,output):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    npix = int(np.round(np.sqrt(np.size(image))))
    image = np.reshape(image,[20,20])
    centerIndex = [9,9]
    indices = np.indices(np.shape(image))
    rmag = 2*np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    nBins = 20
    binRange=[0,np.max(rmag)]
    print("nBins=",nBins)
    binned_data,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image.flatten(),"sum",nBins,range=[binRange])
    
    plt.figure()
    vmax = np.max( [-np.min(binned_data* Sim2PhysicalUnits_MassFlux) , np.max(binned_data* Sim2PhysicalUnits_MassFlux)])*1.05
    vmin=-vmax
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data* Sim2PhysicalUnits_MassFlux , 'k', lw = 2.5)
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data*0 , 'k--', lw = 1.5)
    plt.xlabel('Radius [kpc]')
    plt.ylabel('Radial Mass Flux [M$_{\odot}$ yr$^{-1}$]')
    plt.ylim([vmin,vmax])
    plt.savefig(output+"radialPlot.png")
    plt.close()
    hf.File(output+"radialPlot.hdf5")
    hf.create_dataset('binned_data',data=binned_data)
    hf.create_dataset('rPlot',data=np.linspace(0,np.max(rmag),nBins))

    hf.close()
   
    
    
def RunInferences(networkType,modelOutputPath, dirName,imageOutput_prefix,imageList,params=None):
 
    targetShape=[40,40]


    elif networkType=="resnet18":
        from CoNNGaFit_NeuralNetwork_ResNet3d_18_parameterized import NeuralNetwork    
    elif networkType=="unet18":
        from CoNNGaFit_NeuralNetwork_Unet3d_18_parameterized import NeuralNetwork      
    elif networkType=="resnet18_3dModel":
        from CoNNGaFit_NeuralNetwork_ResNet3d_18_3dModel import NeuralNetwork
        targetShape=[20,20,16]
    elif networkType=="unet18_3dModel":
        from CoNNGaFit_NeuralNetwork_Unet3d_18_3dModel import NeuralNetwork
        targetShape=[20,20,16]

        
    if networkType=="resnet18" and params is not None:
        model =NeuralNetwork(params[0],(params[1],params[1],params[1]),(params[2],params[2],params[2]),params[3]).to(device)
    elif networkType=="unet18" and params is not None:
        model =NeuralNetwork(params[0],(params[1],params[1],params[1]),(params[2],params[2],params[2]),params[3]).to(device)
    else:
        model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(modelOutputPath+".pt"))
    model.eval()

#    imageList = "CoNNGaFitData/observations/inferencesToRun.csv"



    #dirName = "fire2_md_withNoise"
    #imageOutput_prefix = 'CoNNGaFitData/observations/'
    imageOutput_suffix = '_massFlux_FacePlot_'+networkType+"_TR_Comp_"
    plotOutput_suffix = '_massFlux_RadialPlot_'+networkType+"_TR_Comp_"
    compositeOutput_suffix = '_massFlux_CompositePlot_'+networkType+"_TR_Comp_"


    try:
        hf_model = h5py.File(modelOutputPath+".hdf5",'r')
        imageStats = np.array(hf_model['normalizationStats'])
        imageMean = imageStats[0]
        imageStdv = imageStats[1]
    except:
        dataset0 = CoNNGaFitImageInferenceDataset(annotations_file=imageList,
                                   root_dir = '.',
                                   transform = ToTensor())
                                   
                                   
        loader0 = torch.utils.data.DataLoader(dataset0,batch_size=1,shuffle=False)
        data0 = next(iter(loader0))
        imageMean = data0[0].mean()
        imageStdv = data0[0].std()
        del loader0;del data0;del dataset0


#dataset = datasets.ImageFolder("./CoNNGaFitData/observations/",transform = Compose([ToTensor() , Normalize(imageMean,imageStdv)]))
    dataset = CoNNGaFitImageInferenceDataset(annotations_file=imageList,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(imageMean,imageStdv)]))
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

    inferenceNames = LoadNames(imageList)

    i=1


    for inputs in dataloader:
        X=inputs.to(device)
        pred = model(X.float())

        comparisonDir = comparisonRoot + inferenceNames[i]+ comparisonSuffix

        vmax = np.max(np.abs(pred.cpu().detach().numpy()))
        vmin = -vmax

        imageOutput = imageOutput_prefix + inferenceNames[i] + imageOutput_suffix 
        plotOutput = imageOutput_prefix + inferenceNames[i] + plotOutput_suffix 

        MakeImage(pred.cpu().detach().numpy(),imageOutput,vmin,vmax,targetShape)
        CreateBasicPlots(pred.cpu().detach().numpy(),plotOutput)
        i+=1

    print("Done!")    

