import numpy as np
import h5py as h5py
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
print(f'Using {device} device')

modelShape = np.array([40,40,1])
model_size = modelShape[0]*modelShape[1]*modelShape[2]
print("model_size=",model_size);

#Network Class for a 3-d Unet
#Network reads in a 40x40x77 data cube and performs 3d convolutions on each dimension.
#Upsampling branch of the Unet is down with convolutional layers.
#Network is parameterized as follows:
####nFilt0 : number of filters in the inital convolutional layer
####kernel0 : kernel size of the initial convolutional layer. Used to reduce overall data size
####kernel1 : kernel size of the convolutional/deconvolutional blocks
####nFC : number of nodes in final FC layer
####
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 11/21/2023

latentImageOutput="latentImages/latent_"
from matplotlib import pyplot as plt
saveItr=0
def SaveAllSpectralChannels(x,nStep):
#nBatch,1,nSpec,nX,nY
#1,1,77,40,40
    npX = x.cpu().float().detach().numpy();
    print(np.shape(x))
    for i in range(0,np.shape(npX)[1]):
        for j in range(0,np.shape(npX)[2]):
            plt.figure()
            plt.imshow(npX[0,i,j,:,:],cmap='inferno')
            plt.savefig(latentImageOutput+"step"+str(nStep)+"_filter"+str(i)+"_nSpec"+str(j)+".png")
            plt.close()

def SaveSummedSpectralChannels(x,nStep):
    print(np.shape(x))
    npX = x.cpu().float().detach().numpy();
    toPlot = np.sum(npX,axis=2)
    for i in range(0,np.shape(npX)[1]):
        plt.figure()
        plt.imshow(toPlot[0,i,:,:],cmap='inferno')
        plt.savefig(latentImageOutput+"sum_step"+str(nStep)+"_filter"+str(i)+"_gal"+str(saveItr)+".png")
        plt.close()


#### Class for convolutional on the left side of the unet
class ResidualBlock(nn.Module):
    """Encoder-side residual block - see NeuralNetwork_Unet3d.py's ResidualBlock
    for the full description; identical architecture, used here for the HiRes-test network."""

    def __init__(self, in_channels, out_channels, downsample, kernel1, dropout_rate=0.0):
        super().__init__()
        if downsample: #reduce dimensionality
            self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=kernel1,stride=(2,2,2),padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,kernel_size=(1,1,1),stride=(2,2,2)),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=kernel1,stride=(1,1,1),padding=1)
            self.shortcut = nn.Sequential()


        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel1, stride=(1,1,1),padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self,x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x))) #Try replacing with leaky
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        x = nn.ReLU()(x)
        return self.dropout(x)

#### Class for spatially upsampling the decoder path before a skip-connection concatenation ####
class Upsample(nn.Module):
    """See NeuralNetwork_Unet3d.py's Upsample for the full description; identical,
    used here for the HiRes-test network."""

    def __init__(self, in_channels, out_channels, kernel1):
        super().__init__()
        self.upconv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(2,2,2),padding=1)

    def forward(self,x):
        return self.upconv1(x)

#### Class for deconvolutional blocks on the right side of the unet
class DeconvBlock(nn.Module):
    """Decoder-side deconvolution block - see NeuralNetwork_Unet3d.py's DeconvBlock
    for the full description; identical architecture, used here for the HiRes-test network."""

    def __init__(self, in_channels, out_channels, upsample,kernel1, dropout_rate=0.0):
        super().__init__()
        if upsample: #reduce channel count following a concatenation with an upsampled skip connection
            self.conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(1,1,1),padding=1)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels,kernel_size=(1,1,1),stride=(1,1,1)),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(1,1,1),padding=1)
            self.shortcut = nn.Sequential()


        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel1, stride=(1,1,1),padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self,x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x))) #Try replacing with leaky
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        x = nn.ReLU()(x)
        return self.dropout(x)

#### Actual Network ####
class NeuralNetwork(nn.Module):
    """HiRes-test variant of NeuralNetwork_Unet3d.NeuralNetwork: same 3-d residual
    U-Net architecture (including the upsample-then-concatenate skip connections and the
    dropout layer before the output), but with a lighter stem (no MaxPool3d after the initial
    strided conv, and padding=1 instead of 3) and its own hardcoded skip-connection crop
    windows / fc0 input feature count (128*8*nFilt0) tuned for this specific configuration.
    See NeuralNetwork_Unet3d.py's NeuralNetwork docstring for the full description
    and for why these hardcoded values would need to be re-derived for any other input shape.

    Parameters: see NeuralNetwork_Unet3d.NeuralNetwork (nFilt0, kernel0, kernel1,
    nFC, dropout_rate, block_dropout_rate).
    """

    def __init__(self, nFilt0, kernel0, kernel1, nFC, dropout_rate=0.0, block_dropout_rate=0.0):
        print("output size=",modelShape[0]*modelShape[1]*modelShape[2])
        super(NeuralNetwork, self).__init__()
        self.nFC=nFC

        #Initial Convolutional Layer
        self.layer0 = nn.Sequential(
            nn.Conv3d(1,nFilt0,kernel_size=kernel0,stride=(2,2,2),padding=1),
            #nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=1),
            nn.LeakyReLU()
        )

        ##  Residual Layers  ##
        self.layer1 = nn.Sequential(
            ResidualBlock(nFilt0,nFilt0,downsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Downsample
            ResidualBlock(nFilt0,nFilt0,downsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        self.layer2=nn.Sequential(
            ResidualBlock(nFilt0,2*nFilt0,downsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Downsample
            ResidualBlock(2*nFilt0,2*nFilt0,downsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        self.layer3=nn.Sequential(
            ResidualBlock(2*nFilt0,4*nFilt0,downsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Downsample
            ResidualBlock(4*nFilt0,4*nFilt0,downsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        self.layer4=nn.Sequential(
            ResidualBlock(4*nFilt0,8*nFilt0,downsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Downsample
            ResidualBlock(8*nFilt0,8*nFilt0,downsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )
        #######################
        
        ## Upsampling (spatial only, run before each skip-connection concatenation) ##
        self.upsample1 = Upsample(8*nFilt0,4*nFilt0,kernel1=kernel1)
        self.upsample2 = Upsample(4*nFilt0,2*nFilt0,kernel1=kernel1)
        self.upsample3 = Upsample(2*nFilt0,nFilt0,kernel1=kernel1)

        ## Deconvolutional Layers (channel reduction on the concatenated tensors) ##
        self.dc_layer1 = nn.Sequential(
            DeconvBlock(8*nFilt0,4*nFilt0,upsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Reduce channels post-concat
            DeconvBlock(4*nFilt0,4*nFilt0,upsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        self.dc_layer2=nn.Sequential(
            DeconvBlock(4*nFilt0,2*nFilt0,upsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Reduce channels post-concat
            DeconvBlock(2*nFilt0,2*nFilt0,upsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        self.dc_layer3=nn.Sequential(
            DeconvBlock(2*nFilt0,nFilt0,upsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Reduce channels post-concat
            DeconvBlock(nFilt0,nFilt0,upsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
        )

        #######################

        self.featureForwarding = nn.Sequential()


        #Output Layer

        self.avgpool0 = nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.dropout0 = nn.Dropout(p=dropout_rate)
        if self.nFC>0:
            self.fc0 = nn.Linear(in_features=128*8*nFilt0 , out_features=nFC)
            self.relu0 = nn.LeakyReLU()
            self.fc1 = nn.Linear(in_features=nFC,out_features=model_size)
        else:
            self.fc1 = nn.Linear(in_features=128*nFilt0,out_features=model_size)

   
    def forward(self, x, saveLatentImages=False):
        nBatch,nSpec,nX,nY = x.size() 
        x = torch.reshape(x,(nBatch,1,nSpec,nX,nY))


        ##  Initial convolutional layer and pooling  ##
        x = self.layer0(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,0)
        

        ##  Residual Blocks  ##
        x = self.layer1(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,1)

        featureMap1 = self.featureForwarding(x)
        x = self.layer2(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,2)

        featureMap2 = self.featureForwarding(x)
        x = self.layer3(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,3)

        featureMap3 = self.featureForwarding(x) 
        x = self.layer4(x)
        if saveLatentImages:
            SaveSummedSpectralChannels(x,4)
            #SaveAllSpectralChannels(x,4)

        ## Deconvolutional Blocks ##
        x = self.upsample1(x)
        x = torch.cat([x, featureMap3], dim=1) #Feature forwarding from left convolutional -> deconvolutional wing
        x = self.dc_layer1(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,5)

        x = self.upsample2(x)
        x = torch.cat([x, featureMap2[:,:,0:17,:,:]], dim=1) #Cropped Feature forwarding
        x = self.dc_layer2(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,5)

        x = self.upsample3(x)
        x = torch.cat([x, featureMap1[:,:,2:35,:,:]], dim=1) #Cropped Feature forwarding
        x = self.dc_layer3(x)
        if saveLatentImages:
            SaveSummedSpectralChannels(x,7)
            #SaveAllSpectralChannels(x,7)
            
        ##  Output Inference ##
        x = self.avgpool0(x)

        x = torch.flatten(x,1)
        if self.nFC>0:
            x = self.fc0(x)
            x = self.relu0(x)
        x = self.dropout0(x)
        output = self.fc1(x)

        global saveItr
        saveItr+=1
        return output