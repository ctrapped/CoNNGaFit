import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose

from CoNNGaFit_Datasets import CoNNGaFitImageDataset

from CoNNGaFit_NeuralNetwork_Unet3d_18_parameterized import NeuralNetwork
import numpy as np
from sklearn.preprocessing import StandardScaler    
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
import h5py
from scipy.stats import gaussian_kde

from CoNNGaFit_UseModel_callable import RunInferences

import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')
eps = 0.000000000000000000001

CUDA_LAUNCH_BLOCKING=1

#training_data = CoNNGaFitDatasets.Fire2_cr_datasets( #Probably need to update completely with your own data loading
#    root="data",
#    train=True,
#    download=True,
#    transform=ToTensor()
#)

Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses

standardError=0
standardError_v2=0
#inclination=50
index2kpc=1

###   HYPERPARAMETERS   ###
nFilt0=6
k0=9
k1=3
nFC=1000
learning_rate = 0.0001
weight_decay = 0.00005

epochs = 10000


learning_rate = float(sys.argv[1])
weight_decay = float(sys.argv[2])

epochs = int(sys.argv[3])

nFC = int(sys.argv[4])
nFilt0 = int(sys.argv[5])

inclination = int(sys.argv[6])



learning_rate_postCutoff = .00001
epochCutoff = epochs
###########################

networkType="unet18"

#dirName = 'fire2_cr_hiRes'
dirName = 'fire2_velocitySpace'
inputDirName = 'fire2_velocitySpace'

##sampleSuffix = "_i50_FinalSnap_fullSpectra"
sampleSuffix = sys.argv[7]

try:
   validationSuffix = sys.argv[8]   
except:
   validationSuffix = sampleSuffix
   
try:
   testingSuffix = sys.argv[9]   
except:
   testingSuffix = sampleSuffix

try:
    outputSuffix = sys.argv[10]
except:
    outputSuffix=sampleSuffix
    
try:
   input_subdir = sys.argv[11]
except:
   input_subdir = ''

trainingDir = 'CoNNGaFitData\\'+inputDirName+'\\i'+str(inclination)+'\\training\\finalTrainingSets\\'+input_subdir+'\\training_annotations_MassFlux_i'+str(inclination)+sampleSuffix+'.csv'
validationDir = 'CoNNGaFitData\\'+inputDirName+'\\i'+str(inclination)+'\\training\\finalTrainingSets\\'+input_subdir+'\\validation_annotations_MassFlux_i'+str(inclination)+validationSuffix+'.csv'
testingDir = 'CoNNGaFitData\\'+inputDirName+'\\i'+str(inclination)+'\\training\\finalTrainingSets\\'+input_subdir+'\\validation_annotations_MassFlux_i'+str(inclination)+testingSuffix+'.csv'


training_data = CoNNGaFitImageDataset(annotations_file=trainingDir,
                                   root_dir = '.',
                                   transform = ToTensor())

test_data = CoNNGaFitImageDataset(annotations_file=validationDir,
                                   root_dir = '.',
                                   transform = ToTensor())
                                   
                                   
batchSizeDefault = None
batchesPerTrainingSet = 1
if batchSizeDefault is None:
    batchSize = len(training_data)
else:
    batchSize = batchSizeDefault
    batchesPerTrainingSet = np.ceil(len(training_data) / batchSize)
    print("Batches per training set = ",batchesPerTrainingSet)
    
loader0 = DataLoader(training_data, batch_size = batchSize)
data0 = next(iter(loader0))
trainingMean = data0[0].mean()
trainingStdv = data0[0].std()
del loader0;del data0

if batchSizeDefault is None:
    batchSize = len(test_data)
else:
    batchSize = batchSizeDefault
    
#testBatchSize=batchSize
#loader1 = DataLoader(test_data, batch_size = batchSize)
#data1 = next(iter(loader1))
#testMean = data1[0].mean()
#testStdv = data1[0].std()
#print("Test Mean=",testMean)
#del loader1;del data1;

#finalTestBatchSize=batchSize
#loader2 = DataLoader(final_test_data, batch_size = batchSize)
#data2 = next(iter(loader2))
#finalTestMean = data2[0].mean()
#finalTestStdv = data2[0].std()
#del loader2;del data2;


training_data = CoNNGaFitImageDataset(annotations_file=trainingDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

test_data = CoNNGaFitImageDataset(annotations_file=validationDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))







#batchSize = 1
if batchSizeDefault is None:
    batchSize = len(training_data)
else:
    batchSize = batchSizeDefault
train_dataloader = DataLoader(training_data, batch_size=batchSize)

if batchSizeDefault is None:
    batchSize = len(test_data)
else:
    batchSize = batchSizeDefault
test_dataloader = DataLoader(test_data, batch_size=batchSize)



print("Data loaded...")

subFolder = 'Comps_09102023_4000'

#subFolder = 'optimalComps/unetFullSpecOptimal/tests'
#outputSuffix = sampleSuffix+"_lr"+str(learning_rate)+"_nFC"+str(nFC)+"_wd"+str(weight_decay)+"_nF"+str(nFilt0)+"_unetFullSpec"


imageOutput_input = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\input\\massFlux'+sampleSuffix
imageOutput=imageOutput_input
imageOutput_prediction = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\prediction\\massFlux'+outputSuffix+'_'
imageOutput_training = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\training\\massFlux'+outputSuffix+'_'
imageOutput_training_input = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\training\\input\\massFlux'+outputSuffix+'_'
imageOutput_comparison = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\comparisons\\'+subFolder+'\\massFlux'+outputSuffix+'_'
imageOutput_comparison_finalTest = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\comparisons\\'+subFolder+'\\testingSet\\massFlux'+outputSuffix+'_'

imageOutput_prediction_finalTest = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\testing\\massFlux'+sampleSuffix
imageOutput_input_finalTest = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\testing\\input\\massFlux'+sampleSuffix

diagnosticOutput = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\images\\comparisons\\'+subFolder+'\\massFlux'+outputSuffix

modelOutputPath = 'CoNNGaFitData\\'+dirName+'\\i'+str(inclination)+'\\trainedNetworks\\MassFlux_Unet18_FullSpec'+outputSuffix+'_epoch'+str(epochs)#+'.pt'

saveItr=0

NsnapsTest=0
NpixTest=0
NsnapsTrain=0
NpixTrain=0

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        #scaler=StandardScaler()
        #X = scaler.fit_transform(X)
        #Compute prediction and loss
        #print("mean,std=",X.mean(),X.std())
        X=X.to(device)
        y=y.to(device)
        
        
        global NsnapsTrain;global NpixTrain;
        tmp,NpixTrain = np.shape(y.cpu().float().detach().numpy());
        
        pred = model(X.float())
        
        #print(pred)
        print(np.shape(pred))

        loss = loss_fn(pred , y.float())
     
        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100==0:
            loss, current = loss.item(), batch*len(X)
            #print("prediction=",np.shape(pred.cpu().detach().numpy()))
           # print(pred)
           # print("y=",np.shape(y.cpu().float().detach().numpy()))
           # print(y.float())
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
    
    if batchSizeDefault is not None:
        if epoch >= (epochs - 1 - batchesPerTrainingSet):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            
            global toPlot_pred;global toPlot_input
            toPlot_pred = np.append(toPlot_pred,pred.cpu().detach().numpy().flatten())
            toPlot_input = np.append(toPlot_input,y.cpu().float().detach().numpy().flatten())
            
           
            
            #MakeCompScatterPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_Epoch"+str(epoch),defineStandardError=True)
            #MakeCompScatterPlot_v2(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_v2_Epoch"+str(epoch),defineStandardError=True)
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1
                NsnapsTrain+=1
                #MakeImage(pred.cpu().detach().numpy(),imageOutput_training+"_"+trainingNames[nSnap]+"_prediction_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                #MakeImage(y.cpu().float().detach().numpy(),imageOutput_training_input+"_"+trainingNames[nSnap]+"_input.png",vmin,vmax,tt)
                
                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_training+"_"+trainingNames[nSnap]+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_training_input+"_"+trainingNames[nSnap]+"_input.hdf5",tt)
                
    elif epoch>=(epochs-1):# or epoch%(int(epochs/10))==0:
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            #MakeCompScatterPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_Epoch"+str(epoch),defineStandardError=True)
            #MakeCompScatterPlot_v2(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_v2_Epoch"+str(epoch),defineStandardError=True)
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1
                #MakeImage(pred.cpu().detach().numpy(),imageOutput_training+"_"+trainingNames[nSnap]+"_prediction_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                #MakeImage(y.cpu().float().detach().numpy(),imageOutput_training_input+"_"+trainingNames[nSnap]+"_input.png",vmin,vmax,tt)
                
                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_training+"_"+trainingNames[nSnap]+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_training_input+"_"+trainingNames[nSnap]+"_input.hdf5",tt)
    
    return loss

    
def test_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct = 0, 0
    total = 0
    
    predictions = np.array([])
    actuals = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(device)
            y=y.to(device)
            pred=model(X.float())
            test_loss += loss_fn(pred, y.float()).item()
            
            #print("shape of pred=")
            #print(np.shape(pred))
            
            #pred = torch.round(torch.sigmoid(pred))
            npPred = pred.cpu().float().detach().numpy();
            npY = y.cpu().float().detach().numpy();
            
            deviation = np.std(npY)
            print("Deviation=",deviation)
            correct += np.size( np.where( ((npPred>npY-deviation) & (npPred<npY+deviation)) )[0] )

            #correct += (pred<y*1.05 and pred>y*.95).sum().float()
            total += np.size(npY)

            predictions = np.append(predictions,pred.cpu().detach().numpy())
            actuals = np.append(actuals,y.cpu().detach().numpy())
            
            global NsnapsTest;global NpixTest
            Nsnaps,Npix = np.shape(npY)
            NpixTest=Npix
            specific_accuracy = np.zeros((Nsnaps))
            for i in range(0,Nsnaps):
                correctForSnap = np.size( np.where( ((npPred[i,:]>npY[i,:]-deviation) & (npPred[i,:]<npY[i,:]+deviation)) )[0] ) 
                specific_accuracy[i] = correctForSnap / np.size(npY[i,:]) * 100
            
            #actuals = np.round(y.cpu().float().detach().numpy())
           # predictions=np.round(pred.cpu().detach().numpy())
            #accuracy = acc.cpu().detach().numpy()
            #print("Predictions range from:",np.min(predictions)," : ",np.max(predictions))
            #print("Actuals range from:",np.min(y.cpu().detach().numpy())," : ",np.max(y.cpu().detach().numpy()))
            



    accuracy = correct/total * 100
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    if batchSizeDefault is not None:
        if epoch >= (epochs - 1 - batchesPerTrainingSet):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            
            
            global toPlot_pred_test;global toPlot_input_test
            toPlot_pred_test = np.append(toPlot_pred_test , pred.cpu().detach().numpy().flatten())
            toPlot_input_test = np.append(toPlot_input_test , y.cpu().float().detach().numpy().flatten())
            
            #MakeCompScatterPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_Epoch"+str(epoch),defineStandardError=True)
            #MakeCompScatterPlot_v2(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison+"_TrainingScatter_v2_Epoch"+str(epoch),defineStandardError=True)
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1
                NsnapsTest+=1;
                #MakeImage(pred.cpu().detach().numpy(),imageOutput_training+"_"+trainingNames[nSnap]+"_prediction_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                #MakeImage(y.cpu().float().detach().numpy(),imageOutput_training_input+"_"+trainingNames[nSnap]+"_input.png",vmin,vmax,tt)
                
                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_prediction+"_"+validationNames[nSnap]+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_input+"_"+validationNames[nSnap]+"_input.hdf5",tt)
                
                
    elif epoch>=(epochs-1):# or epoch%1000==0:
        #CompileImages(actuals,imageOutput,'Input')
        #CompileImages(predictions,imageOutput,'Prediction')
        #if epoch> (epochs-1 - size/batch_size):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            MakeCompScatterPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison+"_i"+str(inclination)+"_ValidationScatter_Epoch"+str(epoch))
            MakeCompScatterPlot_v2(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison+"_i"+str(inclination)+"_ValidationScatter_v2_Epoch"+str(epoch))
            MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,imageOutput_comparison+"_CorrelationPlot_Epoch"+str(epoch))

            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1

                #MakeImage(pred.cpu().detach().numpy(),imageOutput_prediction+"_"+validationNames[nSnap]+"_prediction_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                #MakeImage(y.cpu().float().detach().numpy(),imageOutput_input+"_"+validationNames[nSnap]+"_input.png",vmin,vmax,tt)
                
                MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison+"_"+validationNames[nSnap]+"_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                
                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_prediction+"_"+validationNames[nSnap]+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_input+"_"+validationNames[nSnap]+"_input.hdf5",tt)

                #CreateBasicPlots(pred.cpu().detach().numpy()     ,imageOutput_prediction+"_"+validationNames[nSnap]+"_prediction_Epoch"+str(epoch)+"_plot.png",tt)
                #CreateBasicPlots(y.cpu().float().detach().numpy(),imageOutput_input+"_"+validationNames[nSnap]+"_input_plot.png",tt)
                
                #CreateBasicCompPlots(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison+"_"+validationNames[nSnap]+"_plot.png",tt)

    return test_loss,accuracy,specific_accuracy
    
    
   
def final_test_loop(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct = 0, 0
    total = 0
    
    predictions = np.array([])
    actuals = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(device)
            y=y.to(device)
            pred=model(X.float())
            test_loss += loss_fn(pred, y.float()).item()
            
            #print("shape of pred=")
            #print(np.shape(pred))
            
            #pred = torch.round(torch.sigmoid(pred))
            npPred = pred.cpu().float().detach().numpy();
            npY = y.cpu().float().detach().numpy();
            
            deviation = np.std(npY)
            print("Deviation=",deviation)
            correct += np.size( np.where( ((npPred>npY-deviation) & (npPred<npY+deviation)) )[0] )

            #correct += (pred<y*1.05 and pred>y*.95).sum().float()
            total += np.size(npY)

            predictions = np.append(predictions,pred.cpu().detach().numpy())
            actuals = np.append(actuals,y.cpu().detach().numpy())
            
            global NsnapsTest;global NpixTest
            Nsnaps,Npix = np.shape(npY)
            NpixTest=Npix
            specific_accuracy = np.zeros((Nsnaps))
            for i in range(0,Nsnaps):
                correctForSnap = np.size( np.where( ((npPred[i,:]>npY[i,:]-deviation) & (npPred[i,:]<npY[i,:]+deviation)) )[0] ) 
                specific_accuracy[i] = correctForSnap / np.size(npY[i,:]) * 100
            



    accuracy = correct/total * 100
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    if batchSizeDefault is not None:
        if epoch >= (epochs - 1 - batchesPerTrainingSet):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            
            
            global toPlot_pred_test;global toPlot_input_test
            toPlot_pred_test = np.append(toPlot_pred_test , pred.cpu().detach().numpy().flatten())
            toPlot_input_test = np.append(toPlot_input_test , y.cpu().float().detach().numpy().flatten())
            
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1
                NsnapsTest+=1;

                try:
                    snapName = testingNames[nSnap]
                except:
                    snapName = str(nSnap)+"NameNotFound"


                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_prediction_finalTest+"_"+snapName+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_input_finalTest+"_"+snapName+"_input.hdf5",tt)
                
                
    elif epoch>=(epochs-1):# or epoch%1000==0:
        #CompileImages(actuals,imageOutput,'Input')
        #CompileImages(predictions,imageOutput,'Prediction')
        #if epoch> (epochs-1 - size/batch_size):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax
            nSnap=0;
            MakeCompScatterPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison_finalTest+"_i"+str(inclination)+"_ValidationScatter_Epoch"+str(epoch))
            MakeCompScatterPlot_v2(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),imageOutput_comparison_finalTest+"_i"+str(inclination)+"_ValidationScatter_v2_Epoch"+str(epoch))
            MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,imageOutput_comparison_finalTest+"_CorrelationPlot_Epoch"+str(epoch))

            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                nSnap+=1
                
                try:
                   name=testingNames[nSnap]
                except:
                   print("Warning, could not find name in testing set...")
                   name = str(nSnap)+"NameNotFound"
                
                MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_comparison_finalTest+"_"+name+"_Epoch"+str(epoch)+".png",vmin,vmax,tt)
                
                SaveHDF5(pred.cpu().detach().numpy()     ,imageOutput_prediction_finalTest+"_"+name+"_prediction_Epoch"+str(epoch)+".hdf5",tt)
                SaveHDF5(y.cpu().float().detach().numpy(),imageOutput_input_finalTest+"_"+name+"_input.hdf5",tt)


    return test_loss,accuracy,specific_accuracy
    
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
    print(np.shape(data))
    print("data=",data)
    
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



def MakeImage(data,output,vmin,vmax,Nsnap):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(image))))
    image = np.reshape(image,[npix,npix])
    #image -= np.min(image)
    #image=np.sum(image,2)
    plt.figure()
    vmax = np.max( [-np.min(image) , np.max(image)])
    vmin=-vmax
    plt.imshow(image * Sim2PhysicalUnits_MassFlux,cmap='seismic',vmin=vmin* Sim2PhysicalUnits_MassFlux,vmax=vmax* Sim2PhysicalUnits_MassFlux)
    plt.colorbar();
    plt.title("Radial Mass Flux (Training Units)")
    plt.savefig(output)
    plt.close()
    
def MakeCompImage(data_pred,data_input,data_image,output,vmin,vmax,Nsnap):
    print("Making Comp Image...")
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[Nsnap,:]
    
    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[Nsnap,:]
    
    npix = int(np.round(np.sqrt(np.size(image_pred))))
    image_pred = np.reshape(image_pred,[npix,npix])
    image_input = np.reshape(image_input,[npix,npix])
    #image -= np.min(image)
    #image=np.sum(image,2)
    plt.figure()
    print('data_image shape=',np.shape(data_image))
    plt.subplot(2,2,4)
    imageToPlot = np.sum(data_image[Nsnap,:,:,:],axis=0)
    imageToPlot+=np.abs(np.min(imageToPlot))+1
    plt.imshow(imageToPlot,norm=LogNorm(),cmap='inferno')
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    
    plt.subplot(2,2,3)
    centerIndex = [npix/2-1,npix/2-1]
    indices = np.indices(np.shape(image_pred))
    rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    nBins = 20
    binRange=[0,np.max(rmag)]
    print("nBins=",nBins)
    binned_data_pred,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_pred.flatten(),"sum",nBins,range=[binRange])
    binned_data_input,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_input.flatten(),"sum",nBins,range=[binRange])
    maxPlot = np.max([-np.min(binned_data_pred),-np.min(binned_data_input),np.max(binned_data_pred),np.max(binned_data_input)])
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_input* Sim2PhysicalUnits_MassFlux , 'k',lw = 2)
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_pred * Sim2PhysicalUnits_MassFlux , 'r--' ,lw = 2)
    plt.legend(['Input','Prediction'])
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_pred * 0 , 'k--' ,lw = 1)
    plt.xlabel('Radius')
    plt.ylabel('Radial Mass Flux [M$_{\odot}$ yr$^{-1}$]')
    plt.ylim([-maxPlot*1.05* Sim2PhysicalUnits_MassFlux,maxPlot*1.05* Sim2PhysicalUnits_MassFlux])
    
    
    vmax = np.max( [-np.min(image_pred) , np.max(image_pred), -np.min(image_input) , np.max(image_input)])
    vmin=-vmax
    plt.subplot(2,2,1)
    plt.imshow(image_input * Sim2PhysicalUnits_MassFlux,cmap='seismic',vmin=vmin* Sim2PhysicalUnits_MassFlux,vmax=vmax* Sim2PhysicalUnits_MassFlux)
    plt.title("Radial Mass Flux Input")
    #plt.colorbar();
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.subplot(2,2,2)
    plt.imshow(image_pred * Sim2PhysicalUnits_MassFlux,cmap='seismic',vmin=vmin* Sim2PhysicalUnits_MassFlux,vmax=vmax* Sim2PhysicalUnits_MassFlux)
    plt.colorbar(label = '[M$_{\odot}$ yr$^{-1}$]');
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.title("Radial Mass Flux Prediction")
    plt.savefig(output)
    plt.close()
    
def MakeCompScatterPlot(data_pred,data_input,output,defineStandardError=False):

    points_pred = data_pred.flatten()* Sim2PhysicalUnits_MassFlux
    points_input = data_input.flatten() * Sim2PhysicalUnits_MassFlux
    
    xmin = min(np.min(points_pred),np.min(points_input))
    xmax = max(np.max(points_pred),np.max(points_input))
    
    if np.abs(xmin)>np.abs(xmax):
        xmax = -xmin
    else:
        xmin = -xmax
    

    xmin = -2
    xmax=2


    xy = np.vstack([points_input,points_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    points_input=points_input[idx]
    points_pred=points_pred[idx]
    z=z[idx]

    err = points_pred - points_input
    global standardError
    if defineStandardError:
        standardError = np.sqrt(np.mean(np.power(err,2)))

    plt.scatter(points_input,points_pred,0.5,c=z,alpha=0.5)
    plt.plot([xmin,xmax],[xmin,xmax],'k--',lw=1.5)
    plt.plot([xmin,xmax],[xmin+standardError,xmax+standardError],'k--',lw=0.5)
    plt.plot([xmin,xmax],[xmin-standardError,xmax-standardError],'k--',lw=0.5)

    plt.text(.3,1,"Std. Err: " + str(np.sqrt(np.mean(np.power(err,2)))), horizontalalignment='right',verticalalignment='bottom', transform=plt.gca().transAxes)
    
    
    
    
    plt.xlabel("True Radial MF [M$_{\odot}$ yr$^{-1}$]")
    plt.ylabel("Predicted Radial MF [M$_{\odot}$ yr$^{-1}$]")
    
    plt.xlim([xmin,xmax])
    plt.ylim([xmin,xmax])
    
    plt.savefig(output+".png")
    plt.close()    
    
    
    #plt.figure()
    #data_input[data_input==0]=eps
    #diff = np.divide( np.subtract(data_pred,data_input) , data_input)*100
    #plt.hist(diff,bins=200,range=(-100,100))
    #plt.xlabel("Percent Difference")
    #plt.ylabel("N$_{pixels}$")
    #plt.xlim(-100,100)
    #plt.savefig(output+"_hist.png")
    #plt.close()
    
    return standardError


def RV2coeff(dataList):
    # First compute the scalar product matrices for each data set X
    scalArrList = []

    for arr in dataList:
        scalArr = np.dot(arr, np.transpose(arr))
        diego = np.diag(np.diag(scalArr))
        scalArrMod = scalArr - diego
        scalArrList.append(scalArrMod)

    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(dataList), len(dataList)), float)

    for index, element in np.ndenumerate(C):
        nom = np.trace(
            np.dot(np.transpose(scalArrList[index[0]]),
                      scalArrList[index[1]]))
        denom1 = np.trace(
            np.dot(np.transpose(scalArrList[index[0]]),
                      scalArrList[index[0]]))
        denom2 = np.trace(
            np.dot(np.transpose(scalArrList[index[1]]),
                      scalArrList[index[1]]))
        Rv = nom / np.sqrt(denom1 * denom2)
        C[index[0], index[1]] = Rv

    return C
    
def MakeCorrelationPlot(pred_image,input_image,Nsnaps,output):


    tickSize=14
    labelSize=16
    titleSize=18
    cbarSize=14
    nameSize=14
    legendSize = 14
    
    npix=40

    corr1d = np.zeros((Nsnaps))
    corr2d = np.zeros((Nsnaps))
    for i in range(0,Nsnaps):
        pred_snap = pred_image[i,:]
        input_snap = input_image[i,:]


    
        npix = int(np.round(np.sqrt(np.size(pred_snap))))
        pred_snap = np.reshape(pred_snap,[npix,npix])
        input_snap = np.reshape(input_snap,[npix,npix])
        
        rv2_mat = RV2coeff([input_snap,pred_snap])
        corr2d[i] = rv2_mat[0,1]

        centerIndex = [npix/2-1,npix/2-1]
        indices = np.indices(np.shape(pred_snap))
        rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
        nBins = npix
        binRange=[0,np.max(rmag)]
        binned_pred,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),pred_snap.flatten(),"sum",nBins,range=[binRange])
        binned_input,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),input_snap.flatten(),"sum",nBins,range=[binRange])
        
        correlationMatrix1d = np.corrcoef(binned_pred,binned_input)
        corr1d[i] = correlationMatrix1d[1][0]
        
    plt.figure()
    size = 60
    alpha=0.7
    predColor = 'r'
    inputColor = 'k'
    
    xplot = np.linspace(0,np.size(corr1d),np.size(corr1d))
    
   # plt.scatter(xplot, corr1d, size, c=inputColor,marker='o',alpha=alpha)
   # plt.scatter(xplot, corr2d , size, c=predColor,marker='*',alpha=alpha)
   # plt.plot(xplot,xplot*0,'k--',lw=2)
#    plt.ylim([-1.1,1.1])

    nBins = 20
    alpha = 0.7
    plt.hist(corr1d, bins=nBins,range=[-1,1],histtype='step',color='k',label='1d Corr.',alpha=alpha,linewidth=2)
    plt.hist(corr2d, bins=nBins,range=[-1,1],histtype='step',color='r',label='2D Corr.',alpha=alpha,linewidth=2)
    
    plt.xlim([-1,1])    

    #plt.legend(['1D Corr.','2D Corr.'],fontsize=legendSize)
    plt.legend(fontsize=legendSize,loc='upper left')
    plt.ylabel('Snapshots',fontsize=labelSize)
    plt.xlabel('Correlation',fontsize=labelSize)
    plt.xticks(fontsize=tickSize)
    plt.yticks(fontsize=tickSize)
    plt.tight_layout()
    plt.savefig(output+".png",bbox_inches='tight')
    plt.savefig(output+".pdf",bbox_inches='tight')
    
    plt.close()        
    
    hf = h5py.File(output+".hdf5",'w')
    hf.create_dataset("corr1d",data=corr1d)
    hf.create_dataset("corr2d",data=corr2d)
    hf.close()
    
def MakeCompScatterPlot_v2(data_pred,data_input,output,defineStandardError=False):

    nPix = int(np.round(np.sqrt(np.shape(data_pred)[1])))
    nSnaps = np.shape(data_pred)[0]

    
    ##images_pred = np.reshape(data_pred , [nSnaps,nPix,nPix])
    #images_input = np.reshape(data_input , [nSnaps,nPix,nPix])
    
    
    idx = np.indices((nPix,nPix))
    rmag = np.sqrt( np.power(idx[0,:,:]-nPix/2,2) , np.power(idx[1,:,:]-nPix/2,2) ) * index2kpc

    step=2
    nRings = np.size(range(0,nPix-step))
    
    netMF_pred = np.zeros((nRings,nSnaps))
    netMF_input = np.copy(netMF_pred)
    
    ii = -1
    for rMin in range(0,nPix-step):
      ii+=1
      rMax = rMin+step
      mask = ( (rmag > rMin*index2kpc) & (rmag < rMax*index2kpc))
      

      
      for tt in range(0,nSnaps):
      
          image_pred = np.zeros((np.shape(data_pred)[1]))
          image_pred[:] = data_pred[tt,:]
    
          image_input = np.zeros((np.shape(data_input)[1]))
          image_input[:] = data_input[tt,:]
    
          image_pred = np.reshape(image_pred,[nPix,nPix])
          image_input = np.reshape(image_input,[nPix,nPix])
      
          masked_pred = image_pred[mask]
          netMF_pred[ii,tt] = np.sum(masked_pred)
          
          masked_input = image_input[mask]
          netMF_input[ii,tt] = np.sum(masked_input)

    
      print("Shape of netMF points = ",np.shape(netMF_pred))

      points_pred = netMF_pred[ii,:].flatten()* Sim2PhysicalUnits_MassFlux
      points_input = netMF_input[ii,:].flatten() * Sim2PhysicalUnits_MassFlux
    
      xmin = min(np.min(points_pred),np.min(points_input))
      xmax = max(np.max(points_pred),np.max(points_input))
    
      if np.abs(xmin)>np.abs(xmax):
        xmax = -xmin
      else:
        xmin = -xmax
    



      #xy = np.vstack([points_input,points_pred])
      #z = gaussian_kde(xy)(xy)
      #idx = z.argsort()
      #points_input=points_input[idx]
      #points_pred=points_pred[idx]
      #z=z[idx]

      err = points_pred - points_input
      #global standardError_v2
      #if defineStandardError:
         # standardError_v2 = np.sqrt(np.mean(np.power(err,2)))
          #standardError as stdv between snapshots at a given ring
          #standardError_v2 = np.std(points_input)
          #standardError as stdv between rings at a given snapshots
          #standardError as stdv between both
      #plt.figure()
      #plt.scatter(points_input,points_pred,1,c='k',alpha=1)
      #plt.plot([xmin,xmax],[xmin,xmax],'k--',lw=1.5)
     # plt.plot([xmin,xmax],[xmin,xmax]+standardError_v2,'k--',lw=0.5)
     # plt.plot([xmin,xmax],[xmin,xmax]-standardError_v2,'k--',lw=0.5)

      #plt.text(.3,1,"Std. Err: " + str(np.sqrt(np.mean(np.power(err,2)))), horizontalalignment='right',verticalalignment='bottom', transform=plt.gca().transAxes)
    
    
    
    
      #plt.xlabel("True Net Radial MF [M$_{\odot}$ yr$^{-1}$]")
      #plt.ylabel("Predicted Net Radial MF [M$_{\odot}$ yr$^{-1}$]")
    
      #plt.xlim([xmin,xmax])
     # plt.ylim([xmin,xmax])
    
      #plt.savefig(output+"_"+str(rMin)+"-"+str(rMax)+"kpc.png")
      #plt.cla()
      #plt.close()    
    
    points_pred = netMF_pred.flatten()* Sim2PhysicalUnits_MassFlux #all rings now
    points_input = netMF_input.flatten() * Sim2PhysicalUnits_MassFlux    
    xmin = min(np.min(points_pred),np.min(points_input))
    xmax = max(np.max(points_pred),np.max(points_input))
    
    if np.abs(xmin)>np.abs(xmax):
        xmax = -xmin
    else:
        xmin = -xmax

    #xy = np.vstack([points_input,points_pred])
    #z = gaussian_kde(xy)(xy)
    #idx = z.argsort()
    #points_input=points_input[idx]
    #points_pred=points_pred[idx]
    #z=z[idx]

    err = points_pred - points_input
    global standardError_v2
    if defineStandardError:
        standardError_v2 = np.sqrt(np.mean(np.power(err,2)))
        #standardError as stdv between both
        standardError_v2 = np.std(points_input)

    plt.figure()
    plt.scatter(points_input,points_pred,1,c='k',alpha=1)
    plt.plot([xmin,xmax],[xmin,xmax],'k--',lw=1.5)
    plt.plot([xmin,xmax],[xmin+standardError_v2,xmax+standardError_v2],'k--',lw=0.5)
    plt.plot([xmin,xmax],[xmin-standardError_v2,xmax-standardError_v2],'k--',lw=0.5)

    plt.text(.3,1,"Std. Err: " + str(np.sqrt(np.mean(np.power(err,2)))), horizontalalignment='right',verticalalignment='bottom', transform=plt.gca().transAxes)
    
    plt.xlabel("True Net Radial MF [M$_{\odot}$ yr$^{-1}$]")
    plt.ylabel("Predicted Net Radial MF [M$_{\odot}$ yr$^{-1}$]")
    
    #plt.xlim([xmin,xmax])
    
    plt.ylim([xmin,xmax])
    hf = h5py.File(output+"_allRings.hdf5",'w')
    hf.create_dataset('points_input',data=points_input)
    hf.create_dataset('points_pred',data=points_pred)
    hf.create_dataset('netMF_pred',data=netMF_pred)
    hf.create_dataset('netMF_input',data=netMF_input)
    hf.close()

    plt.savefig(output+"_allRings.png")
    plt.close()   
    
    
    return standardError
    
def SaveHDF5(data,output,Nsnap):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(image))))
    image = np.reshape(image,[npix,npix])
    hf = h5py.File(output,'w')
    hf.create_dataset('data',data=image)
    hf.close()
    
def CreateBasicPlots(data,output,Nsnap):
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[Nsnap,:]
    npix = int(np.round(np.sqrt(np.size(image))))
    image = np.reshape(image,[npix,npix])
    centerIndex = [npix/2-1,npix/2-1]
    indices = np.indices(np.shape(image))
    rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    nBins = 20
    binRange=[0,np.max(rmag)]
    print("nBins=",nBins)
    binned_data,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image.flatten(),"sum",nBins,range=[binRange])
    
    plt.figure()
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data* Sim2PhysicalUnits_MassFlux , lw = 1.5)
    plt.xlabel('Radius')
    plt.savefig(output)
    plt.close()
    
def CreateBasicCompPlots(data_pred,data_input,output,Nsnap):
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[Nsnap,:]
    
    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[Nsnap,:]
    
    npix = int(np.round(np.sqrt(np.size(image_pred))))
    image_pred = np.reshape(image_pred,[npix,npix])
    image_input = np.reshape(image_input,[npix,npix])

    centerIndex = [npix/2-1,npix/2-1]
    indices = np.indices(np.shape(image_pred))
    rmag = np.sqrt( np.power(indices[0,:,:]-centerIndex[0] , 2) + np.power(indices[1,:,:] -centerIndex[1], 2) )
    
    nBins = 20
    binRange=[0,np.max(rmag)]
    print("nBins=",nBins)
    binned_data_pred,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_pred.flatten(),"sum",nBins,range=[binRange])
    binned_data_input,binedge,binnum = stats.binned_statistic_dd(rmag.flatten(),image_input.flatten(),"sum",nBins,range=[binRange])

    plt.figure()
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_input* Sim2PhysicalUnits_MassFlux , 'r--',lw = 1.5)
    plt.plot(np.linspace(0,np.max(rmag),nBins) , binned_data_pred * Sim2PhysicalUnits_MassFlux , 'k' ,lw = 1.5)
    plt.legend(['Input','Prediction'])
    plt.xlabel('Radius')
    plt.savefig(output)
    plt.close()
    

    
model = NeuralNetwork(nFilt0,(k0,k0,k0),(k1,k1,k1),nFC).to(device)

def CustomLossFunction(output, target):
    err = torch.abs(torch.sub(output,target))
    
    
    #loss = 10000*torch.mean(torch.multiply(err,err))
    #loss = torch.mean(torch.multiply(target,torch.multiply(err,err)))
    #L1_weight=.1;
   # loss =  torch.mean(torch.multiply(err,err) + L1_weight*(err))
    
    loss = torch.mean( torch.multiply( torch.multiply(err,err) , torch.abs(target) ) ) #try to bias towards high MF regions


    return loss
    
def LoadNames(inputDir):
    print("IN NAMES")
    Nlines=0
    fid = open(inputDir,'r')
    for line in fid:
        Nlines+=1
    fid.close()
    
    fid = open(inputDir,'r')

      
    #names = np.zeros((Nlines)).astype('str')
    i=0
    for line in fid:
        x=line.split("\\")
        Nsplit = np.size(x)
        y=(x[Nsplit-1]).split('.')
        if i==0:
            names = [y[0]]
        else:
            names.append(y[0])
        if "_md_" in line or "_md\\" in line:
            names[i] = names[i]+"_md"
        i+=1
        #print("y=",y)
        #print("y0=",y[0])
        
    fid.close()
        
    return names

#for layer in model.children():
 #  if hasattr(layer, 'reset_parameters'):
 #      layer.reset_parameters()
print("LOADING NAMES")
validationNames = LoadNames(validationDir);
trainingNames = LoadNames(trainingDir);
testingNames = LoadNames(testingDir);
print(validationNames)
print(model)
#loss_fn = nn.CrossEntropyLoss() #Change to custom loss function most likely
#loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
#loss_fn = CustomLossFunction
#loss_fn = nn.BCELoss()
#loss_fn = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #Steepest gradient descent, can use different optimizers            
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.1) #Steepest gradient descent, can use different optimizers            
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay) #Steepest gradient descent, can use different optimizers            
trainingLoss = np.zeros((epochs))
validationLoss = np.zeros((epochs))
validationAccuracy=np.zeros((epochs))

if batchSizeDefault is not None:
    specificAccuracy=np.zeros((testBatchSize-5,epochs))
else:
    specificAccuracy=np.zeros((len(test_data),epochs))


Nspec = 120
NpixelNeighbors=4
Npixels=30


#MakeInitialDiagnosticImage(test_dataloader)



toPlot_pred = np.array([])
toPlot_input = np.array([])

toPlot_pred_test = np.array([])
toPlot_input_test = np.array([])

for t in range(epochs):
    if t>epochCutoff:
        learning_rate = learning_rate_postCutoff
    print(f"Epoch {t+1}\n-------------------------------")
    trainingLoss[t] = train_loop(train_dataloader, model, loss_fn, optimizer, t)
    if batchSizeDefault is None:
        validationLoss[t] , validationAccuracy[t], specificAccuracy[:,t] = test_loop(test_dataloader, model, loss_fn, t)
    else:
        validationLoss[t] , validationAccuracy[t], tmp = test_loop(test_dataloader, model, loss_fn, t)

    
if batchSizeDefault is not None: #if need to group the last set of batches together for quantitative plot
    toPlot_pred = np.reshape(toPlot_pred, (NsnapsTrain,NpixTrain))
    toPlot_pred_test = np.reshape(toPlot_pred_test, (NsnapsTest,NpixTest))
    toPlot_input = np.reshape(toPlot_input, (NsnapsTrain,NpixTrain))
    toPlot_input_test = np.reshape(toPlot_input_test, (NsnapsTest,NpixTest))

    MakeCompScatterPlot(toPlot_pred,toPlot_input,imageOutput_comparison+"_TrainingScatter_Epoch"+str(epochs),defineStandardError=True)
    MakeCompScatterPlot_v2(toPlot_pred,toPlot_input,imageOutput_comparison+"_TrainingScatter_v2_Epoch"+str(epochs),defineStandardError=True)
    
    MakeCompScatterPlot(toPlot_pred_test,toPlot_input_test,imageOutput_comparison+"_ValidationScatter_Epoch"+str(epochs),defineStandardError=True)
    MakeCompScatterPlot_v2(toPlot_pred_test,toPlot_input_test,imageOutput_comparison+"_ValidationScatter_v2_Epoch"+str(epochs),defineStandardError=True)
    
#Save the Trained Model
torch.save(model.state_dict(),modelOutputPath+".pt")

#Save the normalization information for loading data
hf_norm = h5py.File(modelOutputPath+".hdf5",'w')
hf_norm.create_dataset("imageStats",data=np.array([trainingMean,trainingStdv]))
hf_norm.create_dataset("learningRate",data=learning_rate)
hf_norm.create_dataset("nFC",data=nFC)
hf_norm.create_dataset("weight_decay",data=weight_decay)
hf_norm.create_dataset("nFilt0",data=nFilt0)
hf_norm.create_dataset("sampleSuffix",data=sampleSuffix)
hf_norm.create_dataset("validationSuffix",data=validationSuffix)
hf_norm.create_dataset("testingSuffix",data=testingSuffix)
hf_norm.close()


#Testing Data

final_test_data = CoNNGaFitImageDataset(annotations_file=testingDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

final_test_dataloader = DataLoader(final_test_data, batch_size=batchSize)

final_test_loss , final_test_accuracy, final_test_specific_accuracy = final_test_loop(final_test_dataloader, model, loss_fn, t)
print("Testing Set Loss=",final_test_loss)
print("Testing Set Accuracy=",final_test_accuracy)
print("Testing Set Specific Accuracies=")
for accuracy in final_test_specific_accuracy:
    print(accuracy)

fig1 = plt.figure()    
plt.plot(np.log10(trainingLoss))
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Log(Loss)")
plt.savefig(diagnosticOutput+"_trainingLoss.png")

fig2 = plt.figure()
plt.plot(np.log10(validationLoss))
plt.title("Valdiation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log(Loss)")

plt.savefig(diagnosticOutput+"_validationLoss.png")

fig3 = plt.figure()
plt.plot(validationAccuracy)
plt.title("Valdiation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([60,101])
plt.savefig(diagnosticOutput+"_ValdiationAccuracy.png")


fig4 = plt.figure()
for i in range(0,np.shape(specificAccuracy)[0]):
    plt.plot(specificAccuracy[i,:],lw=0.5)
plt.title("Valdiation Accuracy for each Snapshot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([60,101])
plt.savefig(diagnosticOutput+"_ValdiationAccuracyPerSnapshot.png")

hf_diagnostic = h5py.File(diagnosticOutput+"_LossesAndAccuracies.hdf5",'w')
hf_diagnostic.create_dataset('trainingLoss',data=trainingLoss)
hf_diagnostic.create_dataset('validationLoss',data=validationLoss)
hf_diagnostic.create_dataset('validationAccuracy',data=validationAccuracy)
hf_diagnostic.create_dataset('specificAccuracy',data=specificAccuracy)
hf_diagnostic.create_dataset('trainingNames',data=trainingNames)
hf_diagnostic.create_dataset('validationNames',data=validationNames)
hf_diagnostic.create_dataset('testingNames',data=testingNames)
hf_diagnostic.create_dataset("testingLoss",data=final_test_loss)
hf_diagnostic.create_dataset("testingAccuracy",data=final_test_accuracy)
hf_diagnostic.create_dataset("testingSpecificAccuracy",data=final_test_specific_accuracy)

hf_diagnostic.close()

fig5 = plt.figure()
plt.subplot(211)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Log(Loss)")
plt.plot(np.log10(trainingLoss),lw=3)
plt.plot(np.log10(validationLoss),'k--',lw=3)
plt.legend(['Training','Validation'],fontsize=14)

plt.subplot(212)
for i in range(0,np.shape(specificAccuracy)[0]):
    plt.plot(specificAccuracy[i,:],lw=0.5)
plt.title("Valdiation Accuracy for each Snapshot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")

plt.savefig(diagnosticOutput+"_LossesAndAccuracies.png")

RunInferences(networkType , modelOutputPath , dirName, imageOutput_comparison, [nFilt0,k0,k1,nFC])


print("Done!")    

