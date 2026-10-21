import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from Datasets import CoNNGaFitImageDataset, TARGETS, inclination_set

from Unet3d_ResBlock import NeuralNetwork
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import h5py

from PlottingFunctions import MakeCompImage,MakeCorrelationPlot,SaveHDF5,MakeCompImage_1d,SaveHDF5_1d

import argparse
import tomllib


####Trains the Unet read from Unet3d_ResBlock.py on the given training data. Provides additional diagnostic plots and images on the validation and test datasets provided.
####Training, validation, and testing datasets are VOF optically thin outputs, chosen by galaxy and snapshot range (see Datasets.py).
####Hyperparameters were chosen based on paramter space optimization+trial and error.
####Hyperparameters and datasets (including the target annotation) are read from two parameter files,
####configs/hyperparams.toml and configs/datasets.toml by default; plot labels per target are in TARGET_SETTINGS below
#
####Run as: python TrainModel_UNet.py [--hyperparams FILE] [--datasets FILE] [--run-name NAME] [options]
####  Run with --help to see all options. The two parameter files and --run-name are normally the
####  only options you need. The datasets file gives the path to each galaxy's VOF outputs, and by default:
####    NetworkWeights/    receives the trained model checkpoint (<run-name>.pt/.hdf5)
####    DiagnosticOutputs/ receives per-epoch images/snapshots and loss/accuracy/correlation plots
####  NetworkWeights/ and DiagnosticOutputs/ are created automatically if missing.
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 10/21/2026


####   PER-TARGET SETTINGS   ####
#target -> (plot label, plot units, how pixels are combined when binning plots)
TARGET_SETTINGS = {
    "mass": ('Mass', r'[M$_{\odot}$]', 'sum'),
    "radial_velocity": ('Rad. Vel.', r'[km s$^{-1}$]', 'mean'),
    "cylindrical_radial_velocity": ('Cyl. Rad. Vel.', r'[km s$^{-1}$]', 'mean'),
    "rotational_velocity": ('Rot. Vel.', r'[km s$^{-1}$]', 'mean'),
    "radial_mass_flux": ('Radial Mass Flux', r'[M$_{\odot}$ yr$^{-1}$]', 'sum'),
    "cylindrical_radial_mass_flux": ('Cyl. Radial Mass Flux', r'[M$_{\odot}$ yr$^{-1}$]', 'sum'),
}
#############################

####   PARAMETER FILES   ####
#Keys every parameter file must have (see configs/*.toml for what each one does)
HYPERPARAM_KEYS = {"learning_rate", "weight_decay", "epochs", "nFC", "nFilt0", "k0", "k1",
                   "dropout_rate", "block_dropout_rate", "output_head", "label_scaling"}
DATASET_KEYS = {"target", "augment", "inclinations", "radial_bins", "radial_projection", "training", "validation", "testing"}
OPTIONAL_DATASET_KEYS = {"base_dir", "suffix"}
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

def load_parameter_file(path, keys, optional_keys=set()):
    """Read a TOML parameter file, returning (its parameters, its text to save with the model).
    Raises ValueError if any of keys is missing, or any key is unrecognized, so typos aren't silently ignored."""
    with open(path, 'rb') as f:
        text = f.read().decode()
    params = tomllib.loads(text)
    missing, unknown = keys - params.keys(), params.keys() - keys - optional_keys
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unrecognized keys {sorted(unknown)}")
    return params, text

def dataset_sources(section, name, path, base_dir=None):
    """Check a datasets-file section ("VOF output directory" = snapshots lines) and resolve each
    directory relative to base_dir (default: the datasets file's folder). Returns the dict Datasets.py expects."""
    if not isinstance(section, dict):
        raise ValueError(f'{path}: write [{name}] as "/path/to/vof_outputs" = snapshots lines')
    #base_dir itself may be relative, to the datasets file's folder
    base = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.expanduser(base_dir or ""))
    return {os.path.normpath(os.path.join(base, os.path.expanduser(d))): snaps for d, snaps in section.items()}
#############################


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CoNNGaFit 3-d U-Net to predict a gas property map from HI datacubes.")
    parser.add_argument('--hyperparams', default=os.path.join(CONFIG_DIR, 'hyperparams.toml'),
                         help="Hyperparameter file (learning rate, epochs, network size, output head). Default: configs/hyperparams.toml")
    parser.add_argument('--datasets', default=os.path.join(CONFIG_DIR, 'datasets.toml'),
                         help="Datasets file (target, augmentation, training/validation/testing galaxies). Default: configs/datasets.toml")
    parser.add_argument('--run-name', default=None,
                         help="Name for this run; the checkpoint and every diagnostic output are named after it. Default: the target")

    io = parser.add_argument_group('output directories',
        "Override where results are written to (the training data is set in the datasets file). "
        "Both are created automatically if missing.")
    io.add_argument('--network-dir', default='NetworkWeights',
                     help="Directory the trained model checkpoint (.pt) and its normalization stats (.hdf5) are written to. Default: %(default)s")
    io.add_argument('--output-dir', default='DiagnosticOutputs',
                     help="Root directory diagnostic outputs are written to: per-epoch comparison images/snapshots "
                          "under <output-dir>/images/, and loss/accuracy curves and correlation plots under "
                          "<output-dir>/diagnostics/. Default: %(default)s")

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

####   HYPERPARAMETERS   ####
hyperparams, hyperparamsText = load_parameter_file(args.hyperparams, HYPERPARAM_KEYS)
learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
epochs = hyperparams["epochs"]
nFC = hyperparams["nFC"]
nFilt0 = hyperparams["nFilt0"]
k0 = hyperparams["k0"]
k1 = hyperparams["k1"]
dropout_rate = hyperparams["dropout_rate"]
block_dropout_rate = hyperparams["block_dropout_rate"]
outputHead = hyperparams["output_head"]
labelScaling = hyperparams["label_scaling"]
if labelScaling not in ("none", "symmetric", "one-sided"):
    raise ValueError(f'{args.hyperparams}: label_scaling = {labelScaling!r} should be "none", "symmetric" or "one-sided"')
#############################



####   DATASETS   ####
datasets, datasetsText = load_parameter_file(args.datasets, DATASET_KEYS, OPTIONAL_DATASET_KEYS)
target = datasets["target"]
if target not in TARGETS:
    raise ValueError(f"{args.datasets}: unknown target {target!r}; choose from {sorted(TARGETS)}")
trainingSources = dataset_sources(datasets["training"], "training", args.datasets, datasets.get("base_dir"))
validationSources = dataset_sources(datasets["validation"], "validation", args.datasets, datasets.get("base_dir"))
testingSources = dataset_sources(datasets["testing"], "testing", args.datasets, datasets.get("base_dir"))
augmentTraining = None if datasets["augment"] == "none" else datasets["augment"] #TOML has no None
radialBins = datasets["radial_bins"] or None #0 means train on maps
radialProjection = datasets["radial_projection"]
inclinations = datasets["inclinations"] #"all" or a list, for every set (a source line can override it)
inclination_set(inclinations, args.datasets) #Check it here, so an error names this file
suffix = datasets.get("suffix", "") #VOF's outputSuffix in the filenames, for every set (a source line can override it)
if not isinstance(suffix, str):
    raise ValueError(f"{args.datasets}: suffix = {suffix!r} should be a string")
#############################

paramLabel, unitLabel, binOp = TARGET_SETTINGS[target]
runName = args.run_name or target



####  Set input/output directories ####
os.makedirs(args.network_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'images'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir,'diagnostics'), exist_ok=True)

imageOutput_training = os.path.join(args.output_dir,'images',runName+'_training')
imageOutput_validation = os.path.join(args.output_dir,'images',runName+'_validation')
imageOutput_finalTest = os.path.join(args.output_dir,'images',runName+'_test')

diagnosticOutput = os.path.join(args.output_dir,'diagnostics',runName)
modelOutputPath = os.path.join(args.network_dir, runName)



#### Load and Normalize Data ####
training_data = CoNNGaFitImageDataset(trainingSources, target, augment=augmentTraining, radial_bins=radialBins, radial_projection=radialProjection, inclinations=inclinations, suffix=suffix)

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

#### Image normalization, from every training image (augmentation doesn't change their mean/spread) ####
#Each cube's mean and variance are combined exactly (Chan et al.'s parallel algorithm), one cube in memory at a time
nPixels, trainingMean, sumSquaredDeviations = 0, 0.0, 0.0
for i in range(len(training_data.img_paths)):
    cube = training_data.image(i)
    n, mean = cube.size, cube.mean(dtype=np.float64)
    delta = mean - trainingMean
    sumSquaredDeviations += cube.var(dtype=np.float64) * n + delta**2 * nPixels * n / (nPixels + n)
    trainingMean += delta * n / (nPixels + n)
    nPixels += n
trainingStdv = np.sqrt(sumSquaredDeviations / nPixels)
del cube
print(f"Image normalization: mean = {trainingMean:.4g}, stdv = {trainingStdv:.4g}")

#### Label scaling, from every training label (augmentation doesn't change their range) ####
#Labels are scaled to (label - labelOffset) / labelScale for training; predictions and labels are
#scaled back to physical units for every saved output and plot
labelOffset, labelScale = 0.0, 1.0
if labelScaling != "none":
    allLabels = np.concatenate([training_data.label(i) for i in range(len(training_data.img_paths))])
    if labelScaling == "symmetric": #e.g. mass flux: divide by max(|label|), to [-1, 1]
        labelScale = np.max(np.abs(allLabels))
    else: #"one-sided", e.g. rotational velocity: [min(label), max(label)] to [-1, 1]
        labelOffset = (np.max(allLabels) + np.min(allLabels)) / 2
        labelScale = (np.max(allLabels) - np.min(allLabels)) / 2
    if not labelScale > 0:
        raise ValueError(f"label_scaling = {labelScaling!r} needs training labels that aren't all {np.max(allLabels)}")
    del allLabels
    print(f"Label scaling ({labelScaling}): offset = {labelOffset:.4g}, scale = {labelScale:.4g}")

def scale_labels(label):
    return (label - labelOffset) / labelScale

def to_physical(a):
    """Scale labels or predictions back to physical units."""
    return a * labelScale + labelOffset


training_data.transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)])
training_data.target_transform = scale_labels

validation_data = CoNNGaFitImageDataset(validationSources, target, radial_bins=radialBins, radial_projection=radialProjection, inclinations=inclinations, suffix=suffix,
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]), target_transform=scale_labels)




class WithIndices(Dataset):
    """Wraps a dataset so each sample also returns its index, to look up its name after shuffling."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)

train_dataloader = DataLoader(WithIndices(training_data), batch_size=batchSize, shuffle=True) #Reshuffled every epoch

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
validation_dataloader = DataLoader(WithIndices(validation_data), batch_size=batchSize)

print("Data loaded...")


if radialBins is not None: #Labels are radial curves, so plot and save them as curves rather than maps
    MakeCompImage = MakeCompImage_1d
    SaveHDF5 = SaveHDF5_1d

#### Save final-epoch outputs ####
def save_batch_outputs(X, y, pred, idx, names, outputBase, epoch, compImages=True):
    """Save each sample in a batch as an hdf5 snapshot (and a comparison image), named after its
    image via its dataset index (idx), so names stay correct with shuffled or multiple batches."""
    npX = X.cpu().float().detach().numpy()
    npY = to_physical(y.cpu().float().detach().numpy())
    npPred = to_physical(pred.cpu().detach().numpy())
    for tt in range(len(idx)):
        outputName = outputBase+"_"+names[int(idx[tt])]+"_Epoch"+str(epoch)
        if compImages:
            MakeCompImage(npPred,npY,npX,outputName+".png",tt,Sim2PhysicalUnits=1,paramLabel=paramLabel,unitLabel=unitLabel,binOp=binOp) #Labels are already in physical units
        SaveHDF5(npY,npPred,outputName+".hdf5",tt)

#### Define Training Loop ####
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """Run one training epoch: iterate the dataloader, backprop MSE loss each batch, and on
    the final epoch dump every sample's prediction/label hdf5 snapshot for later inspection.
    Returns the loss averaged over all batches in the epoch.
    """
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    num_batches = len(dataloader)
    for batch, (X,y,idx) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)

        pred = model(X.float())
        loss = loss_fn(pred , y.float())
        epoch_loss += loss.item()
        if epoch >= epochs-1:
            save_batch_outputs(X, y, pred, idx, trainingNames, imageOutput_training, epoch, compImages=False)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100==0:
            current = batch*len(X)
            print(f"loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

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
    num_batches=len(dataloader)
    validation_loss, correct = 0, 0
    total = 0
    final = epoch >= epochs-1
    allPred, allY = [], [] #Every sample's prediction/label on the final epoch, for the correlation plot

    with torch.no_grad():
        for X, y, idx in dataloader:
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

            Nsnaps,Npix = np.shape(npY)
            specific_accuracy = np.zeros((Nsnaps))
            for i in range(0,Nsnaps):
                correctForSnap = np.size( np.where( ((npPred[i,:]>npY[i,:]-deviation) & (npPred[i,:]<npY[i,:]+deviation)) )[0] )
                specific_accuracy[i] = correctForSnap / np.size(npY[i,:]) * 100

            if final:
                save_batch_outputs(X, y, pred, idx, validationNames, imageOutput_validation, epoch)
                allPred.append(to_physical(npPred)); allY.append(to_physical(npY))

    accuracy = correct/total * 100
    validation_loss /= num_batches
    print(f"Validation Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {validation_loss:>8f} \n")

    if final and radialBins is None:
        MakeCorrelationPlot(np.concatenate(allPred),np.concatenate(allY),sum(len(a) for a in allPred),diagnosticOutput+"_ValidationCorrelationPlot_Epoch"+str(epoch))

    model.train()
    return validation_loss,accuracy,specific_accuracy


#### Define Test Loop ####
def test_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the held-out test set. Same rough accuracy metric and final-epoch
    diagnostic outputs as validation_loop above, written under imageOutput_finalTest instead.
    Returns (test_loss, accuracy, specific_accuracy).
    """
    model.eval()
    num_batches=len(dataloader)
    test_loss, correct = 0, 0
    total = 0
    final = epoch >= epochs-1
    allPred, allY = [], [] #Every sample's prediction/label on the final epoch, for the correlation plot

    with torch.no_grad():
        for X, y, idx in dataloader:
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

            Nsnaps,Npix = np.shape(npY)
            specific_accuracy = np.zeros((Nsnaps))
            for i in range(0,Nsnaps):
                correctForSnap = np.size( np.where( ((npPred[i,:]>npY[i,:]-deviation) & (npPred[i,:]<npY[i,:]+deviation)) )[0] )
                specific_accuracy[i] = correctForSnap / np.size(npY[i,:]) * 100

            if final:
                save_batch_outputs(X, y, pred, idx, testingNames, imageOutput_finalTest, epoch)
                allPred.append(to_physical(npPred)); allY.append(to_physical(npY))

    accuracy = correct/total * 100
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if final and radialBins is None:
        MakeCorrelationPlot(np.concatenate(allPred),np.concatenate(allY),sum(len(a) for a in allPred),diagnosticOutput+"_TestCorrelationPlot_Epoch"+str(epoch))

    model.train()
    return test_loss,accuracy,specific_accuracy



#### Define the Model ####
inputShape = tuple(training_data[0][0].shape) #(nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
model = NeuralNetwork(inputShape,nFilt0,(k0,k0,k0),(k1,k1,k1),nFC,fc_dropout_rate=dropout_rate,block_dropout_rate=block_dropout_rate,head=outputHead,output_size=radialBins).to(device)
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
hf_norm.create_dataset("labelScaling",data=labelScaling)
hf_norm.create_dataset("labelStats",data=np.array([labelOffset,labelScale])) #physical label = scaled label * labelScale + labelOffset
hf_norm.create_dataset("learningRate",data=learning_rate)
hf_norm.create_dataset("nFC",data=nFC)
hf_norm.create_dataset("weight_decay",data=weight_decay)
hf_norm.create_dataset("nFilt0",data=nFilt0)
hf_norm.create_dataset("dropout_rate",data=dropout_rate)
hf_norm.create_dataset("block_dropout_rate",data=block_dropout_rate)
hf_norm.create_dataset("runName",data=runName)
hf_norm.create_dataset("inputShape",data=inputShape) #Needed to rebuild the network for inference
hf_norm.create_dataset("target",data=target)
hf_norm.create_dataset("outputHead",data=outputHead)
if radialBins is not None:
    hf_norm.create_dataset("radialBins",data=radialBins)
    hf_norm.create_dataset("radialProjection",data=radialProjection)
#The full parameter files, so every trained model records exactly what it was trained with
hf_norm.create_dataset("hyperparamsFile",data=hyperparamsText)
hf_norm.create_dataset("datasetsFile",data=datasetsText)
hf_norm.close()


#Testing Data

test_data = CoNNGaFitImageDataset(testingSources, target, radial_bins=radialBins, radial_projection=radialProjection, inclinations=inclinations, suffix=suffix,
                                   transform = Compose([ToTensor() , Normalize(trainingMean,trainingStdv)]), target_transform=scale_labels)

test_dataloader = DataLoader(WithIndices(test_data), batch_size=batchSize)
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
