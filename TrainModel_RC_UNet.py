import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose

from Datasets import CoNNGaFitImageDataset

from Unet3d_ResBlock import NeuralNetwork
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats
import h5py

from PlottingFunctions import MakeCompImage,RV2coeff,MakeCorrelationPlot,SaveHDF5

import argparse

####Trains the Unet read from Unet3d_ResBlock.py on the given training data. Provides additional diagnostic plots and images on the validation and test datasets provided.
####Training, validation, and testing datasets are VOF optically thin outputs, chosen by galaxy and snapshot range below (see Datasets.py).
####Hyperparameters were chosen based on paramter space optimization+trial and error.
####Specifically tuned to train for rotational velocities and provide appropriate conversions on plots
#
####Run as: python TrainModel_RC_UNet.py [options]
####  Run with --help to see all options (data/network/output directories,
####  output filenames). With the default directory layout in place, --sample-suffix is normally
####  the only option you need:
####    TrainingData/<galaxy>/vof_outputs/ must hold VOF's outputs for every galaxy in the datasets below
####    NetworkWeights/    receives the trained model checkpoint (.pt/.hdf5)
####    DiagnosticOutputs/ receives per-epoch images/snapshots and loss/accuracy/correlation plots
####  NetworkWeights/ and DiagnosticOutputs/ are created automatically if missing; every directory
####  and filename above can be overridden individually - see --help.
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023

def parse_args():
    parser = argparse.ArgumentParser(description="Train the CoNNGaFit 3-d U-Net to predict rotational velocity maps from HI datacubes.")
    parser.add_argument('--sample-suffix', default='finalSnapNoM12m',
                         help="Suffix identifying this run; every default filename below is built from it. "
                              "With the default directory layout already in place (TrainingData/, NetworkWeights/, "
                              "DiagnosticOutputs/), --sample-suffix is normally the only argument you need to pass. "
                              "Default: %(default)s")

    io = parser.add_argument_group('input/output directories',
        "Override where data is read from and results are written to. Not needed if the default "
        "TrainingData/, NetworkWeights/, and DiagnosticOutputs/ directories exist relative to the "
        "working directory - NetworkWeights/ and DiagnosticOutputs/ are created automatically if missing.")
    io.add_argument('--data-dir', default='TrainingData',
                     help="Root directory holding each galaxy's VOF outputs (<data-dir>/<galaxy>/vof_outputs/). Default: %(default)s")
    io.add_argument('--network-dir', default='NetworkWeights',
                     help="Directory the trained model checkpoint (.pt) and its normalization stats (.hdf5) are written to. Default: %(default)s")
    io.add_argument('--output-dir', default='DiagnosticOutputs',
                     help="Root directory diagnostic outputs are written to: per-epoch comparison images/snapshots "
                          "under <output-dir>/images/, and loss/accuracy curves and correlation plots under "
                          "<output-dir>/diagnostics/. Default: %(default)s")

    filenames = parser.add_argument_group('explicit filenames',
        "Override individual filenames normally derived from --sample-suffix. Not needed for normal runs.")
    filenames.add_argument('--output-name', default=None,
                         help="Base name for diagnostic images/plots under <output-dir>/. Default: rotationalVelocity_<sample-suffix>")
    filenames.add_argument('--model-name', default=None,
                         help="Base filename (no extension) for the saved model checkpoint under <network-dir>. Default: RC_Unet18_FullSpec_<sample-suffix>")

    training = parser.add_argument_group('training options',
        "Override values normally computed automatically from the dataset. Not needed for normal runs.")
    training.add_argument('--batch-size', type=int, default=None,
        help="Mini-batch size for training. Default: a reasonable division of the training set into "
             "~10 batches per epoch (floored at 2 - BatchNorm3d needs more than 1 sample per batch "
             "while training). Pass 0 to instead train on the full dataset as a single batch per epoch.")
    training.add_argument('--test-batch-size', type=int, default=None,
        help="Mini-batch size for validation/testing. Default: a reasonable division of the "
             "validation set into ~10 batches (floored at 1).")
    return parser.parse_args()

args = parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')
eps = 1e-10


standardError=0
standardError_v2=0
index2kpc=1

####   HYPERPARAMETERS   ####
#Optimized values given in comments
learning_rate = 0.0001
weight_decay = 0.001
epochs = 300
nFC = 3000
nFilt0 = 6
k0=9 #Initial kernel size
k1=3 #Kernel size for residual/deconv blocks
dropout_rate = 0.3 #Applied before the output layer, to regularize the large FC bottleneck
block_dropout_rate = 0.0 #Disabled by default. Applies channel-wise dropout (nn.Dropout3d) inside
    #every ResidualBlock/DeconvBlock - a much broader regularizer than dropout_rate above, since
    #it touches every stage of the encoder/decoder and (via featureMap1/2/3) the skip
    #connections too. Only enable this if dropout_rate alone isn't closing the train/validation
    #gap; start low (~0.1-0.15) rather than reusing dropout_rate's 0.3 - the deepest encoder
    #stage here shrinks to a tiny spatial size where dropping whole channels removes a large
    #fraction of that stage's information, so an FC-tuned rate is likely too aggressive here.
outputHead = "conv" #"conv" predicts each pixel from its neighborhood (needed for 128x128/512x512 images);
    #"fc" is the original fully connected output layer (nFC is then the bottleneck size), for 40x40 images
#############################



####   DATASETS   ####
#Each maps galaxy name -> snapshot, or an inclusive (minSnap, maxSnap) range. Every inclination and
#position angle VOF wrote for those snapshots is included.
trainingGalaxies = {"m12i": 600, "m12f": 600, "m12b": 600}
validationGalaxies = {"m12c": 600}
testingGalaxies = {"m12m": 600}
target = "rotational_velocity" #Which annotation to learn; see TARGETS in Datasets.py
#############################



####  Set input/output directories ####
sampleSuffix = args.sample_suffix
outputFilebase = args.output_name or 'rotationalVelocity_'+sampleSuffix
modelFilebase = args.model_name or 'RC_Unet18_FullSpec_'+sampleSuffix

os.makedirs(args.network_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'images'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'diagnostics'), exist_ok=True)

imageOutput_training = os.path.join(args.output_dir,'images',outputFilebase+'_training')
imageOutput_validation = os.path.join(args.output_dir,'images',outputFilebase+'_validation')
imageOutput_finalTest = os.path.join(args.output_dir,'images',outputFilebase+'_test')

diagnosticOutput = os.path.join(args.output_dir,'diagnostics',outputFilebase)
modelOutputPath = os.path.join(args.network_dir, modelFilebase)



#### Load and Normalize Data ####
training_data = CoNNGaFitImageDataset(args.data_dir, trainingGalaxies, target,
                                   transform = ToTensor())
                                   
if args.batch_size is not None:
    batchSizeDefault = None if args.batch_size == 0 else args.batch_size #--batch-size 0 means full-batch
else:
    batchSizeDefault = max(2,len(training_data)//10) #Reasonable division of the training set into ~10
        #batches per epoch; floored at 2 (BatchNorm3d needs more than 1 sample per batch while
        #training). Override with --batch-size (0 for full-batch training).
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

training_data = CoNNGaFitImageDataset(args.data_dir, trainingGalaxies, target,
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

validation_data = CoNNGaFitImageDataset(args.data_dir, validationGalaxies, target,
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))




train_dataloader = DataLoader(training_data, batch_size=batchSize)

if args.test_batch_size is not None:
    testBatchSize = args.test_batch_size
else:
    testBatchSize = max(1,len(validation_data)//10) #Reasonable division of the validation/test
        #sets into ~10 batches; floored at 1 (unlike batchSizeDefault, batch size 1 is safe here
        #since validation/testing run with model.eval(), which uses BatchNorm's running stats
        #rather than computing per-batch statistics). Override with --test-batch-size.
if batchSizeDefault is None:
    batchSize = len(validation_data)
else:
    batchSize = testBatchSize
validation_dataloader = DataLoader(validation_data, batch_size=batchSize)

print("Data loaded...")


#### Define Training Loop ####
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """Run one training epoch (see TrainModel_MassFlux_UNet.py's train_loop for
    the full description; identical structure, just targeting rotational velocity labels).
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
    """Evaluate the model on the validation set (see TrainModel_MassFlux_UNet.py's
    validation_loop for the full description). Returns
    (validation_loss, accuracy, specific_accuracy).
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
        MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,imageOutput_validation+"_CorrelationPlot_Epoch"+str(epoch))

        for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
            MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_validation+"_"+validationNames[tt]+"_Epoch"+str(epoch)+".png",tt,Sim2PhysicalUnits=1,paramLabel='Rot. Vel.',unitLabel='[km s$^{-1}$]',binOp='mean')
            SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_validation+"_"+validationNames[tt]+"_Epoch"+str(epoch)+".hdf5",tt)


    model.train()
    return validation_loss,accuracy,specific_accuracy
    
    
#### Define Test Loop ####
def test_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the held-out test set (see
    TrainModel_MassFlux_UNet.py's test_loop for the full description). Returns
    (test_loss, accuracy, specific_accuracy).
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
        MakeCorrelationPlot(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),Nsnaps,imageOutput_finalTest+"_CorrelationPlot_Epoch"+str(epoch))
        for tt in range(0,np.shape(pred.cpu().detach().numpy())[0]):
            try:
                snapName=testingNames[tt]
            except:
                print("Warning, could not find name in testing set...")
                snapName = str(tt)+"NameNotFound"

            MakeCompImage(pred.cpu().detach().numpy(),y.cpu().float().detach().numpy(),X.cpu().float().detach().numpy(),imageOutput_finalTest+"_"+snapName+"_Epoch"+str(epoch)+".png",tt,Sim2PhysicalUnits=1,paramLabel='Rot. Vel.',unitLabel='[km s$^{-1}$]',binOp='mean')
            SaveHDF5(y.cpu().float().detach().numpy(),pred.cpu().detach().numpy(),imageOutput_finalTest+"_"+snapName+"_Epoch"+str(epoch)+".hdf5",tt)



    model.train()
    return test_loss,accuracy,specific_accuracy


    
#### Define the Model ####
inputShape = tuple(training_data[0][0].shape) #(nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
model = NeuralNetwork(inputShape,nFilt0,(k0,k0,k0),(k1,k1,k1),nFC,fc_dropout_rate=dropout_rate,block_dropout_rate=block_dropout_rate,output_head=outputHead).to(device)
print("Defining model...")
print(model)
print("Loading names...")
validationNames = validation_data.names
trainingNames = training_data.names

#### Define Loss Function and Optimizer ####
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay) #Steepest gradient descent, can use different optimizers      

#### Initialize losses and specific accuracies ####      
trainingLoss = np.zeros((epochs))
validationLoss = np.zeros((epochs))
validationAccuracy=np.zeros((epochs))

if batchSizeDefault is not None:
    #specificAccuracy is only ever filled in per-sample when batchSizeDefault is None (see the
    #main training loop below), so it stays unused in batch mode - sized to 0 rows rather than
    #referencing testBatchSize here.
    specificAccuracy=np.zeros((0,epochs))
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
hf_norm.create_dataset("inputShape",data=inputShape) #Needed to rebuild the network for inference
hf_norm.create_dataset("target",data=target)
hf_norm.create_dataset("outputHead",data=outputHead)
hf_norm.close()


#Testing Data

test_data = CoNNGaFitImageDataset(args.data_dir, testingGalaxies, target,
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]))

test_dataloader = DataLoader(test_data, batch_size=batchSize)
testingNames = test_data.names

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