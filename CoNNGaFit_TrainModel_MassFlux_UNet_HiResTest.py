import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose

from CoNNGaFit_Datasets import CoNNGaFitImageDataset

from AlternativeNetworks.CoNNGaFit_NeuralNetwork_Unet3d_HiResTest import NeuralNetwork
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
import h5py

from CoNNGaFit_PlottingFunctions import MakeCompImage,RV2coeff,MakeCorrelationPlot,SaveHDF5,LoadNames

import argparse


####Trains the Unet read from CoNNGaFit_NeuralNetwork_Unet3d.py on the given training data. Provides additional diagnostic plots and images on the validation and test datasets provided.
####Training, Validation, and testing datasets must be provided in .csv format as outlined in CoNNGaFit_Datasets.py.
####Hyperparameters were chosen based on paramter space optimization+trial and error.
####Specifically tuned to train for radial mass fluxes and provide appropriate conversions on plots
#
####Run as: python CoNNGaFit_TrainModel_MassFlux_UNet_HiResTest.py [options]
####  Run with --help to see all options (data/network directories, dataset CSV filenames,
####  output filenames). By default, expects
####  CoNNGaFitData/annotation_datasets/{training,validation,test}_annotations_MassFlux_All_Inclinations_<sample-suffix>.csv
####  to already exist (produced by a WriteDatasetsToCsv-style script). Uses
####  AlternativeNetworks/CoNNGaFit_NeuralNetwork_Unet3d_HiResTest.py rather than the standard network.
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023

def parse_args():
    parser = argparse.ArgumentParser(description="Train the HiRes-test CoNNGaFit 3-d U-Net variant to predict radial mass flux maps from HI datacubes.")
    parser.add_argument('--sample-suffix', default='finalSnapNoM12m',
                         help="Suffix used to build default training/validation/test annotation CSV filenames and default output/model names, for any of --training-csv/--validation-csv/--testing-csv/--output-name/--model-name not given explicitly. Default: %(default)s")
    parser.add_argument('--data-dir', default='CoNNGaFitData',
                         help="Root data directory: annotation CSVs are read from <data-dir>/annotation_datasets/, and diagnostic images/plots are written under <data-dir>/outputs/. Default: %(default)s")
    parser.add_argument('--network-dir', default='TrainedNetworks',
                         help="Directory the trained model checkpoint (.pt) and its normalization stats (.hdf5) are saved to. Default: %(default)s")
    parser.add_argument('--training-csv', default=None,
                         help="Training annotations CSV filename, read from <data-dir>/annotation_datasets/. Default: training_annotations_MassFlux_All_Inclinations_<sample-suffix>.csv")
    parser.add_argument('--validation-csv', default=None,
                         help="Validation annotations CSV filename, read from <data-dir>/annotation_datasets/. Default: validation_annotations_MassFlux_All_Inclinations_<sample-suffix>.csv")
    parser.add_argument('--testing-csv', default=None,
                         help="Test annotations CSV filename, read from <data-dir>/annotation_datasets/. Default: test_annotations_MassFlux_All_Inclinations_<sample-suffix>.csv")
    parser.add_argument('--output-name', default=None,
                         help="Base name for diagnostic images/plots written under <data-dir>/outputs/. Default: massFlux_<sample-suffix>")
    parser.add_argument('--model-name', default=None,
                         help="Base filename (no extension) for the saved model checkpoint under <network-dir>. Default: MassFlux_Unet18_FullSpec_<sample-suffix>")
    return parser.parse_args()

args = parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')
eps = 1e-10

Sim2PhysicalUnits_MassFlux = (2/np.pi) * 1/(3.086*np.power(10.,16.)) * (3.154*np.power(10.,7.)) * np.power(10,10) #1/pixel_res * kpc2km * s2yr * unit mass to solar masses

standardError=0
standardError_v2=0
index2kpc=1

####   HYPERPARAMETERS   ####
#Optimized values given in comments
learning_rate = 0.0001
weight_decay = 0.002
epochs = 2500
nFC = 2000
nFilt0 = 6
k0=9
k1=3
dropout_rate = 0.3 #Applied before the output layer, to regularize the large FC bottleneck
block_dropout_rate = 0.0 #Disabled by default. Applies channel-wise dropout (nn.Dropout3d) inside
    #every ResidualBlock/DeconvBlock - a much broader regularizer than dropout_rate above, since
    #it touches every stage of the encoder/decoder and (via featureMap1/2/3) the skip
    #connections too. Only enable this if dropout_rate alone isn't closing the train/validation
    #gap; start low (~0.1-0.15) rather than reusing dropout_rate's 0.3 - the deepest encoder
    #stage here shrinks to a tiny spatial size where dropping whole channels removes a large
    #fraction of that stage's information, so an FC-tuned rate is likely too aggressive here.
#############################



####  Set input/output directories ####
sampleSuffix = args.sample_suffix
trainingSetFilename = args.training_csv or 'training_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
validationSetFilename = args.validation_csv or 'validation_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
testingSetFilename = args.testing_csv or 'test_annotations_MassFlux_All_Inclinations_'+sampleSuffix+'.csv'
outputFilebase = args.output_name or 'massFlux_'+sampleSuffix
modelFilebase = args.model_name or 'MassFlux_Unet18_FullSpec_'+sampleSuffix

trainingDir = os.path.join(args.data_dir,'annotation_datasets',trainingSetFilename)
validationDir = os.path.join(args.data_dir,'annotation_datasets',validationSetFilename)
testingDir = os.path.join(args.data_dir,'annotation_datasets',testingSetFilename)


imageOutput_training = os.path.join(args.data_dir,'outputs','images',outputFilebase+'_training')
imageOutput_validation = os.path.join(args.data_dir,'outputs','images',outputFilebase+'_validation')
imageOutput_finalTest = os.path.join(args.data_dir,'outputs','images',outputFilebase+'_test')

diagnosticOutput = os.path.join(args.data_dir,'outputs','diagnostics',outputFilebase)
modelOutputPath = os.path.join(args.network_dir, modelFilebase)



#### Load and Normalize Data ####
training_data = CoNNGaFitImageDataset(annotations_file=trainingDir,
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


    

training_data = CoNNGaFitImageDataset(annotations_file=trainingDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

validation_data = CoNNGaFitImageDataset(annotations_file=validationDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))




train_dataloader = DataLoader(training_data, batch_size=batchSize)

if batchSizeDefault is None:
    batchSize = len(validation_data)
else:
    batchSize = batchSizeDefault
validation_dataloader = DataLoader(validation_data, batch_size=batchSize)

print("Data loaded...")


#### Define Training Loop ####
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """Run one training epoch: iterate the dataloader, backprop MSE loss each batch, and on
    the final epoch(s) dump per-sample prediction/label hdf5 snapshots for later inspection.
    Returns the loss averaged over all batches in the epoch.
    """
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    num_batches = len(dataloader)
    for batch, (X,y) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)

        pred = model(X.float())
        loss = loss_fn(pred , y.float())
        epoch_loss += loss.item()

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100==0:
            current = batch*len(X)
            print(f"loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")


    if batchSizeDefault is not None:
        if epoch >= (epochs - 1 - batchesPerTrainingSet):
            vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
            vmin = -vmax

            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_training+"_"+trainingNames[tt]+"_Epoch"+str(epoch)+".hdf5",tt)

    elif epoch>=(epochs-1):# or epoch%(int(epochs/10))==0:
        vmax = np.max( [np.max(np.abs(pred.cpu().detach().numpy())) , np.max(np.abs(y.cpu().float().detach().numpy()))] )
        vmin = -vmax
        for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
            SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_training+"_"+trainingNames[tt]+"_Epoch"+str(epoch)+".hdf5",tt)

    return epoch_loss / num_batches

#### Define Validation Loop ####
def validation_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the validation set (no gradient updates) and compute a rough
    "accuracy" (fraction of pixels whose prediction falls within one std-dev of the label -
    a loose sanity metric, not a real accuracy). On the final epoch, also writes correlation
    plots and per-sample comparison images/hdf5 snapshots. Returns
    (validation_loss, accuracy, specific_accuracy) where specific_accuracy is the per-sample
    accuracy for every sample in the validation batch.
    """
    model.eval()
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    validation_loss, correct = 0, 0
    total = 0
    
    predictions = np.array([])
    actuals = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            X=X.to(device)
            y=y.to(device)
            
            pred=model(X.float())
            validation_loss += loss_fn(pred, y.float()).item()
            
            npPred = pred.cpu().float().detach().numpy();
            npY = y.cpu().float().detach().numpy();
            
            #### Roughly estimate accuracy of network. Not a great metric but okay for tracking general peformance ####
            deviation = np.std(npY)
            correct += np.size( np.where( ((npPred>npY-deviation) & (npPred<npY+deviation)) )[0] )
            total += np.size(npY)

            predictions = np.append(predictions,pred.cpu().detach().numpy())
            actuals = np.append(actuals,y.cpu().detach().numpy())

            Nsnaps,Npix = np.shape(npY)
            specific_accuracy = np.zeros((Nsnaps))
            for i in range(0,Nsnaps):
                correctForSnap = np.size( np.where( ((npPred[i,:]>npY[i,:]-deviation) & (npPred[i,:]<npY[i,:]+deviation)) )[0] ) 
                specific_accuracy[i] = correctForSnap / np.size(npY[i,:]) * 100

    accuracy = correct/total * 100
    validation_loss /= num_batches
    print(f"Validation Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {validation_loss:>8f} \n")
    print(f"Validation Error: \n Avg loss: {validation_loss:>8f} \n")
    
    if batchSizeDefault is not None:
        if epoch >= (epochs - 1 - batchesPerTrainingSet):
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_validation+"_"+validationNames[tt]+"_Epoch"+str(epoch)+".hdf5",tt)
    elif epoch>=(epochs-1):# or epoch%1000==0:
        MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,diagnosticOutput+"_ValidationCorrelationPlot_Epoch"+str(epoch))

        for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
            MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_validation+"_"+validationNames[tt]+"_Epoch"+str(epoch)+".png",tt)
            SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_validation+"_"+validationNames[tt]+"_Epoch"+str(epoch)+".hdf5",tt)


    model.train()
    return validation_loss,accuracy,specific_accuracy
    
    
#### Define Test Loop ####
def test_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the held-out test set. Same rough accuracy metric and final-epoch
    diagnostic outputs as validation_loop above, written under imageOutput_finalTest instead.
    Returns (test_loss, accuracy, specific_accuracy).
    """
    model.eval()
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
            
            npPred = pred.cpu().float().detach().numpy();
            npY = y.cpu().float().detach().numpy();
            
            #### Roughly estimate accuracy of network. Not a great metric but okay for tracking general peformance ####
            deviation = np.std(npY)
            correct += np.size( np.where( ((npPred>npY-deviation) & (npPred<npY+deviation)) )[0] )

            total += np.size(npY)

            predictions = np.append(predictions,pred.cpu().detach().numpy())
            actuals = np.append(actuals,y.cpu().detach().numpy())

            Nsnaps,Npix = np.shape(npY)
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
            for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
                try:
                    snapName = testingNames[tt]
                except:
                    snapName = str(tt)+"NameNotFound"

                SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_finalTest+"_"+snapName+"_Epoch"+str(epoch)+".hdf5",tt)

    elif epoch>=(epochs-1):# or epoch%1000==0:
        MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,diagnosticOutput+"_TestCorrelationPlot_Epoch"+str(epoch))
        for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
            try:
                snapName=testingNames[tt]
            except:
                print("Warning, could not find name in testing set...")
                snapName = str(tt)+"NameNotFound"

            MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_finalTest+"_"+snapName+"_Epoch"+str(epoch)+".png",tt)
            SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_finalTest+"_"+snapName+"_Epoch"+str(epoch)+".hdf5",tt)



    model.train()
    return test_loss,accuracy,specific_accuracy


    
#### Define the Model ####
model = NeuralNetwork(nFilt0,(k0,k0,k0),(k1,k1,k1),nFC,dropout_rate=dropout_rate,block_dropout_rate=block_dropout_rate).to(device)
print("Defining model...")
print(model)
print("Loading names...")
validationNames = LoadNames(validationDir);
trainingNames = LoadNames(trainingDir);
testingNames = LoadNames(testingDir);

#### Define Loss Function and Optimizer ####
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay) #Steepest gradient descent, can use different optimizers      

#### Initialize losses and specific accuracies ####      
trainingLoss = np.zeros((epochs))
validationLoss = np.zeros((epochs))
validationAccuracy=np.zeros((epochs))

if batchSizeDefault is not None:
    specificAccuracy=np.zeros((testBatchSize-5,epochs))
else:
    specificAccuracy=np.zeros((len(validation_data),epochs))


#### Perform Training Loop ####
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainingLoss[t] = train_loop(train_dataloader, model, loss_fn, optimizer, t)
    if batchSizeDefault is None:
        validationLoss[t] , validationAccuracy[t], specificAccuracy[:,t] = validation_loop(validation_dataloader, model, loss_fn, t)
    else:
        validationLoss[t] , validationAccuracy[t], tmp = validation_loop(validation_dataloader, model, loss_fn, t)

    
#Save the Trained Model
torch.save(model.state_dict(),modelOutputPath+".pt")

#Save the normalization information for loading data
hf_norm = h5py.File(modelOutputPath+".hdf5",'w')
hf_norm.create_dataset("imageStats",data=np.array([trainingMean,trainingStdv]))
hf_norm.create_dataset("learningRate",data=learning_rate)
hf_norm.create_dataset("nFC",data=nFC)
hf_norm.create_dataset("weight_decay",data=weight_decay)
hf_norm.create_dataset("nFilt0",data=nFilt0)
hf_norm.create_dataset("dropout_rate",data=dropout_rate)
hf_norm.create_dataset("block_dropout_rate",data=block_dropout_rate)
hf_norm.create_dataset("sampleSuffix",data=sampleSuffix)
#hf_norm.create_dataset("trainingData",data=validationSuffix)
#hf_norm.create_dataset("testingSuffix",data=testingSuffix)
hf_norm.close()


#Testing Data

test_data = CoNNGaFitImageDataset(annotations_file=testingDir,
                                   root_dir = '.',
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

test_dataloader = DataLoader(test_data, batch_size=batchSize)

final_test_loss , final_test_accuracy, final_test_specific_accuracy = test_loop(test_dataloader, model, loss_fn, t)
print("Testing Set Loss=",final_test_loss)
print("Testing Set Accuracy=",final_test_accuracy)
print("Testing Set Specific Accuracies=")
for accuracy in final_test_specific_accuracy:
    print(accuracy)


#### Make Basic Diagnostic Outputs ####
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

print("Done!")    