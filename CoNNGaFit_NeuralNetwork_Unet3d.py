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


##  Hyperparameters  ##
#nFilt0=3

#kernel0 = (9,9,9) #increase first kernel size for spectra??
#kernel1 = (3,3,3)
#nFC = 100

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, kernel1):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=kernel1,stride=(2,2,2),padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,kernel_size=(1,1,1),stride=(2,2,2)),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=kernel1,stride=(1,1,1),padding=1)
            self.shortcut = nn.Sequential() #??Like identity function?
            
            
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel1, stride=(1,1,1),padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x))) #Try replacing with leaky
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample,kernel1):
        super().__init__()
        if downsample:
            self.conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(2,2,2),padding=1)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels,kernel_size=(1,1,1),stride=(2,2,2)),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(1,1,1),padding=1)
            self.shortcut = nn.Sequential() #??Like identity function?
            
            
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel1, stride=(1,1,1),padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
    def forward(self,x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x))) #Try replacing with leaky
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)

class NeuralNetwork(nn.Module):
    def __init__(self, nFilt0, kernel0, kernel1, nFC):
        print("output size=",modelShape[0]*modelShape[1]*modelShape[2])
        super(NeuralNetwork, self).__init__()
        self.nFC=nFC
        
        #Initial Convolutional Layer
        self.layer0 = nn.Sequential(
            nn.Conv3d(1,nFilt0,kernel_size=kernel0,stride=(2,2,2),padding=3),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=1),
            nn.LeakyReLU() ##??Need batchnorm too? Leaky or normal?
        )
        
        ##  Residual Layers  ##
        self.layer1 = nn.Sequential(
            ResidualBlock(nFilt0,nFilt0,downsample=False,kernel1=kernel1),
            ResidualBlock(nFilt0,nFilt0,downsample=False,kernel1=kernel1)
        )
        
        self.layer2=nn.Sequential(
            ResidualBlock(nFilt0,2*nFilt0,downsample=True,kernel1=kernel1),
            ResidualBlock(2*nFilt0,2*nFilt0,downsample=False,kernel1=kernel1)
        )
        
        self.layer3=nn.Sequential(
            ResidualBlock(2*nFilt0,4*nFilt0,downsample=True,kernel1=kernel1),
            ResidualBlock(4*nFilt0,4*nFilt0,downsample=False,kernel1=kernel1)
        )        
        
        self.layer4=nn.Sequential(
            ResidualBlock(4*nFilt0,8*nFilt0,downsample=True,kernel1=kernel1),
            ResidualBlock(8*nFilt0,8*nFilt0,downsample=False,kernel1=kernel1)
        )  
        #######################
        
        ## Deconvolutional Layers ##
        self.dc_layer1 = nn.Sequential(
            DeconvBlock(8*nFilt0,4*nFilt0,downsample=True,kernel1=kernel1), ##Upsample
            DeconvBlock(4*nFilt0,4*nFilt0,downsample=False,kernel1=kernel1)
        )
        
        self.dc_layer2=nn.Sequential(
            DeconvBlock(4*nFilt0,2*nFilt0,downsample=True,kernel1=kernel1),
            DeconvBlock(2*nFilt0,2*nFilt0,downsample=False,kernel1=kernel1)
        )
        
        self.dc_layer3=nn.Sequential(
            DeconvBlock(2*nFilt0,nFilt0,downsample=True,kernel1=kernel1),
            DeconvBlock(nFilt0,nFilt0,downsample=False,kernel1=kernel1)
        )        

        #######################
        
        self.featureForwarding = nn.Sequential()

        
        #Output Layer
        
        self.avgpool0 = nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        if self.nFC>0:
            self.fc0 = nn.Linear(in_features=128*nFilt0 , out_features=nFC)
            self.relu0 = nn.LeakyReLU()
            self.fc1 = nn.Linear(in_features=nFC,out_features=model_size)
        else:
            self.fc1 = nn.Linear(in_features=128*nFilt0,out_features=model_size)

        
        

        
    def forward(self, x):
        nBatch,nSpec,nX,nY = x.size() 
        x = torch.reshape(x,(nBatch,1,nSpec,nX,nY))

        ##  Initial convolutional layer and pooling  ##
        #print("SHAPES OF IMAGE:")
        #print(np.shape(x))
        x = self.layer0(x)
        

        ##  Residual Blocks  ##
        x = self.layer1(x)
        #print(np.shape(x))

        featureMap1 = self.featureForwarding(x)
        x = self.layer2(x)
       # print(np.shape(x))

        featureMap2 = self.featureForwarding(x)
        x = self.layer3(x)
       # print(np.shape(x))

        featureMap3 = self.featureForwarding(x) 
        x = self.layer4(x)
        #print(np.shape(x))


        #print("SIZE OF FEATUREMAP3 = ",featureMap3.size())
        #print("SIZE OF FEATUREMAP2 = ",featureMap2.size())
        #print("SIZE OF FEATUREMAP1 = ",featureMap1.size())

        x = self.dc_layer1(x)
       # print("DECONV LAYERS")
        #print(np.shape(x))

       # print("SIZE of X1 = ",x.size())
        x = x + featureMap3

        x = self.dc_layer2(x)
       # print("SIZE of X2 = ",x.size())
        x = x + featureMap2[:,:,0:9,:,:]
        #print(np.shape(x))

        x = self.dc_layer3(x)
       # print("SIZE of X3 = ",x.size())
        x = x + featureMap1[:,:,1:18,0:9,0:9]
       # print(np.shape(x))

        ##  Output  ##
        x = self.avgpool0(x)
       # print(np.shape(x))

        x = torch.flatten(x,1)
        if self.nFC>0:
            x = self.fc0(x)
            x = self.relu0(x)
        output = self.fc1(x)
 
        #print(np.shape(output))

        return output
        
        


#to predict, use
####logits = model(X)
####pred_probab = nn.Softmax(dim=1)(logits)
####y_pred=pred_probab.argmax(1)
#where X is the image










