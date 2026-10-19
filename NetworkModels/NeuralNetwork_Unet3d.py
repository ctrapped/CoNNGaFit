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

def center_crop_to_match(source, target):
    """Center-crop source's spatial/spectral dims (2,3,4) down to target's shape in those
    dims, trimming any odd leftover pixel from the end rather than the start. No-ops on any
    dim already matching target; raises if source is smaller than target in any dim (that
    would mean the decoder's upsampled features spatially exceed the encoder's skip
    connection - a sign the architecture itself is mismatched, not something to silently
    paper over).

    Used to align a skip-connection feature map with the upsampled decoder tensor it's
    concatenated onto, without hardcoding crop windows for a specific input shape/kernel
    configuration - see forward() below.
    """
    slices = [slice(None), slice(None)]
    for dim in (2, 3, 4):
        src, tgt = source.shape[dim], target.shape[dim]
        if src < tgt:
            raise ValueError(f"skip connection dim {dim} ({src}) smaller than upsampled feature dim ({tgt})")
        start = (src - tgt) // 2
        slices.append(slice(start, start + tgt))
    return source[tuple(slices)]


def center_crop_to_shape(source, shape):
    """Center-crop source's spatial/spectral dims (2,3,4) to shape, trimming any odd pixel from the end."""
    slices = [slice(None), slice(None)]
    for dim, tgt in zip((2, 3, 4), shape):
        start = (source.shape[dim] - tgt) // 2
        slices.append(slice(start, start + tgt))
    return source[tuple(slices)]


#### Class for convolutional on the left side of the unet
class ResidualBlock(nn.Module):
    """Single residual block on the U-Net's encoder (downsampling) side.

    Two 3-d convolutions (conv1 -> BN -> ReLU -> conv2 -> BN) with a skip connection added
    back before the final ReLU. When downsample=True, conv1 uses stride 2 to halve each
    spatial/spectral dimension, and the skip connection is projected through a strided 1x1x1
    conv + BatchNorm so its shape matches the downsampled main path.

    dropout_rate (default 0.0, i.e. off): probability for a channel-wise nn.Dropout3d applied
    to the block's output (after the residual add + final ReLU). Unlike the FC-layer dropout
    in NeuralNetwork, this regularizes every stage of the encoder - and, since featureMap1/2/3
    are captured right after these blocks run, it also regularizes what gets forwarded through
    the skip connections. See NeuralNetwork_Unet3d.NeuralNetwork's block_dropout_rate
    docstring for recommended usage before enabling this.
    """

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
    """Spatially upsamples x via a single strided ConvTranspose3d (roughly doubling each
    spatial/spectral dimension), with no BatchNorm/shortcut/second conv.

    Kept as its own module, separate from DeconvBlock, because the encoder's skip-connection
    feature map must be concatenated onto x *after* it's been brought up to the matching
    spatial resolution but *before* DeconvBlock's own convolutions run - concatenation (unlike
    addition) requires the two tensors to already share the same spatial shape.
    """

    def __init__(self, in_channels, out_channels, kernel1, output_padding=0):
        super().__init__()
        #output_padding=1 makes the output exactly twice the input size (used by the conv head)
        self.upconv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=kernel1,stride=(2,2,2),padding=1,output_padding=output_padding)

    def forward(self,x):
        return self.upconv1(x)

#### Class for deconvolutional blocks on the right side of the unet
class DeconvBlock(nn.Module):
    """Single block on the U-Net's decoder (upsampling) side - the transpose-convolution
    mirror of ResidualBlock. Spatial upsampling itself happens beforehand (see Upsample
    above); when upsample=True here, conv1 instead runs at stride 1 to reduce the channel
    count of the just-concatenated (upsampled decoder features + encoder skip features)
    tensor back down, with the skip connection projected through a matching stride-1 1x1x1
    transpose conv + BatchNorm.

    dropout_rate (default 0.0, i.e. off): probability for a channel-wise nn.Dropout3d applied
    to the block's output (after the residual add + final ReLU) - see ResidualBlock's
    dropout_rate docstring above for the general rationale, and
    NeuralNetwork.block_dropout_rate for recommended usage.
    """

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

#### Per-pixel output head ####
class ConvOutputHead(nn.Module):
    """Predicts one value per image pixel from the decoder's half-resolution features.

    The spectral axis is folded into channels, so each pixel's line-profile features are combined
    by learned weights rather than averaged away. A global summary of the galaxy (the bottleneck,
    averaged over the whole cube) is appended to every pixel. Two 3x3 convolutions then give each
    prediction a 5x5 field of view at half resolution (~10x10 image pixels, about 3 beams across),
    and a 1x1 convolution gives the output, upsampled to the full image.
    """

    def __init__(self, in_channels, global_channels, head_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + global_channels, head_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv3 = nn.Conv2d(head_channels, 1, kernel_size=1)

    def forward(self, x, bottleneck, outputShape):
        nBatch, nFilt, nSpec, nX, nY = x.shape
        x = torch.reshape(x, (nBatch, nFilt*nSpec, nX, nY)) #Fold the spectral axis into channels
        #Global context: average the bottleneck over the whole cube and give every pixel a copy
        context = torch.mean(bottleneck, dim=(2, 3, 4))[:, :, None, None].expand(-1, -1, nX, nY)
        x = torch.cat([x, context], dim=1)
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.conv3(self.dropout(x))
        x = F.interpolate(x, size=outputShape, mode="bilinear", align_corners=False)
        return torch.flatten(x, 1)


#### Actual Network ####
class NeuralNetwork(nn.Module):
    """3-d residual U-Net that maps an HI 21cm spectral datacube (nSpec x nX x nY) to a
    flattened nX*nY output map (e.g. radial mass flux, rotational velocity, or inclination).
    Without inputShape, it is sized for a 40x40x77 input and a 40x40 output (model_size).

    Architecture: a strided conv + maxpool "stem" (layer0) reduces the input, followed by
    4 stages of residual blocks (layer1..layer4) that each roughly halve the spatial/
    spectral dimensions while doubling the filter count. The decoder mirrors this: at each
    stage, an Upsample module spatially upsamples the decoder features, the matching
    pre-downsample encoder feature map is concatenated onto it channel-wise (skip connection),
    and dc_layer1..dc_layer3 then reduce the concatenated channel count back down via
    transpose-conv blocks. The skip-connection tensors are aligned to the upsampled decoder
    tensor via center_crop_to_match (see above), which derives the crop window from the
    tensors' actual runtime shapes rather than hardcoding it for one input/kernel
    configuration - it also always crops symmetrically, rather than trimming any excess
    from one side only. Finally an average pool + optional
    fully-connected bottleneck (fc0) feeds a final linear layer (fc1) that outputs the
    flattened model_size-length prediction. fc0/fc1's in_features is likewise no longer a
    hardcoded literal when inputShape is given (see below).

    Parameters:
        nFilt0: number of filters in the first residual stage (doubles at each subsequent stage).
        kernel0: kernel size (3-tuple) of the initial strided convolution (layer0).
        kernel1: kernel size (3-tuple) used by all ResidualBlock/DeconvBlock convolutions.
        nFC: size of the optional fully-connected bottleneck before the output layer; set to
            0 (or any value <= 0) to skip it and go straight from the flattened features to fc1.
        dropout_rate: dropout probability applied right before the final output layer (fc1) -
            after the optional FC bottleneck's activation if nFC>0, or directly on the
            flattened conv features otherwise. Defaults to 0.0 (no dropout, i.e. identical
            behavior to before this parameter existed) so existing callers are unaffected
            unless they opt in. Only active in training mode - model.eval() disables it.
        block_dropout_rate: dropout probability for a channel-wise nn.Dropout3d (drops whole
            feature-map channels, not individual voxels) applied inside every
            ResidualBlock/DeconvBlock, after that block's residual add + final ReLU. Defaults
            to 0.0 (off, identical behavior to before this parameter existed). This is a much
            broader regularizer than dropout_rate above: it touches every stage of the
            encoder/decoder, and - since featureMap1/2/3 are captured immediately after their
            respective blocks run - it also regularizes what gets forwarded through the skip
            connections. RECOMMENDED USAGE: only enable this if dropout_rate alone isn't
            closing the train/validation gap; start low (~0.1-0.15) rather than reusing
            dropout_rate's typical 0.3, since the deepest encoder stage here shrinks to a tiny
            (~3,2,2) spatial size where dropping whole channels removes a large fraction of
            that stage's total information - a rate tuned for the FC bottleneck is likely too
            aggressive here and can destabilize training. As with dropout_rate, only active in
            training mode.
        inputShape: (nSpec, nX, nY) of the datacubes this instance will actually be run on.
            Defaults to None, which keeps fc0/fc1's in_features hardcoded at 128*nFilt0 (the
            value derived for this file's default 40x40x77 input) - identical behavior to
            before this parameter existed, so existing callers and saved checkpoints are
            unaffected. When given, fc0/fc1's in_features is instead computed by actually
            running a dummy all-zeros batch of this shape through _features() once at
            construction time and reading off its flattened size - so nX/nY/nSpec, kernel0,
            kernel1, and nFilt0 can all be changed freely without hand-deriving a new literal.
            The output layer then has nX*nY outputs, one per image pixel.
        output_head: "fc" (default) for the fully connected output layers above, or "conv" for
            ConvOutputHead, which needs far fewer parameters at large image sizes (e.g. 128x128,
            512x512). "conv" requires inputShape, with nX and nY divisible by 32; it also adds a
            decoder stage at half resolution and pads the stem so every stage exactly halves or
            doubles the image, keeping the decoded features aligned with the image pixels. nFC is
            unused, and dropout_rate applies (channel-wise) before the head's final layer.
        head_channels: number of channels in ConvOutputHead's 3x3 convolutions.
    """

    def __init__(self, nFilt0, kernel0, kernel1, nFC, dropout_rate=0.0, block_dropout_rate=0.0, inputShape=None, output_head="fc", head_channels=64):
        super(NeuralNetwork, self).__init__()
        if output_head not in ("fc", "conv"):
            raise ValueError(f"output_head must be 'fc' or 'conv', not {output_head!r}")
        if output_head == "conv" and (inputShape is None or inputShape[1] % 32 or inputShape[2] % 32):
            raise ValueError(f"output_head='conv' needs inputShape with nX and nY divisible by 32, got {inputShape}")
        self.output_head = output_head
        conv = output_head == "conv"
        #VOF projects the labels onto the image's own pixels, so there is one output per (nX, nY)
        #pixel. Without inputShape, fall back to the 40x40 default (and existing checkpoints).
        self.output_size = model_size if inputShape is None else inputShape[1]*inputShape[2]
        print("output size=",self.output_size)
        self.nFC=nFC
        
        #Initial Convolutional Layer
        self.layer0 = nn.Sequential(
            #With the conv head, padding is chosen so the conv and pool exactly halve the image
            nn.Conv3d(1,nFilt0,kernel_size=kernel0,stride=(2,2,2),padding=kernel0[0]//2 if conv else 3),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2),padding=0 if conv else 1),
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
        self.upsample1 = Upsample(8*nFilt0,4*nFilt0,kernel1=kernel1,output_padding=int(conv))
        self.upsample2 = Upsample(4*nFilt0,2*nFilt0,kernel1=kernel1,output_padding=int(conv))
        self.upsample3 = Upsample(2*nFilt0,nFilt0,kernel1=kernel1,output_padding=int(conv))

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

        if conv:
            #One more stage, up to half resolution, with a skip connection from the stem
            self.upsample4 = Upsample(nFilt0,nFilt0,kernel1=kernel1,output_padding=1)
            self.dc_layer4 = nn.Sequential(
                DeconvBlock(2*nFilt0,nFilt0,upsample=True,kernel1=kernel1,dropout_rate=block_dropout_rate), ##Reduce channels post-concat
                DeconvBlock(nFilt0,nFilt0,upsample=False,kernel1=kernel1,dropout_rate=block_dropout_rate)
            )

        #######################
        
        self.featureForwarding = nn.Sequential() #Identity pass-through; kept as a named module so the encoder->decoder skip connections can be swapped for a learned transform later without changing forward()

        
        #Output Layer

        if conv:
            #Size the head from the decoder's actual output shape, by running a dummy batch through it
            was_training = self.training
            self.eval()
            with torch.no_grad():
                decoded, _ = self._decode(torch.zeros(1,*inputShape))
            self.train(was_training)
            self.head = ConvOutputHead(decoded.shape[1]*decoded.shape[2], 8*nFilt0, head_channels, dropout_rate=dropout_rate)
            print("output size=",self.output_size)
            return

        self.avgpool0 = nn.AvgPool3d(kernel_size=(2,2,2),stride=(2,2,2))

        if inputShape is not None:
            #Compute fc0/fc1's true in_features by actually running a dummy batch through
            #_features below, rather than hardcoding it for one (nX,nY,nSpec,kernel0,kernel1)
            #configuration - see _features' docstring for the eval()/no_grad rationale.
            was_training = self.training
            self.eval()
            with torch.no_grad():
                flat_size = self._features(torch.zeros(1,*inputShape)).shape[1]
            self.train(was_training)
        else:
            #No inputShape given: fall back to the value hardcoded for this network's default
            #40x40x77 input, leaving existing callers' fc0/fc1 sizes (and so their saved
            #checkpoints) completely unaffected.
            flat_size = 128*nFilt0

        self.dropout0 = nn.Dropout(p=dropout_rate)
        if self.nFC>0:
            self.fc0 = nn.Linear(in_features=flat_size , out_features=nFC)
            self.relu0 = nn.LeakyReLU()
            self.fc1 = nn.Linear(in_features=nFC,out_features=self.output_size)
        else:
            self.fc1 = nn.Linear(in_features=flat_size,out_features=self.output_size)

    def _concat_skip(self, x, skip):
        """Concatenate an encoder skip connection onto the upsampled decoder tensor x."""
        if self.output_head == "fc":
            return torch.cat([x, center_crop_to_match(skip, x)], dim=1)
        #Conv head: sizes match exactly except possibly the spectral axis (if nSpec isn't
        #divisible by 32), so crop whichever tensor is larger there
        shape = [min(a, b) for a, b in zip(x.shape[2:], skip.shape[2:])]
        return torch.cat([center_crop_to_shape(x, shape), center_crop_to_shape(skip, shape)], dim=1)

    def _features(self, x, saveLatentImages=False):
        """Run the encoder and decoder (see _decode), then the output average pool and flatten,
        for the fully connected output layers. __init__ calls it once on a dummy batch to size
        fc0/fc1.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; a channel dim of size 1 is inserted before the 3-d convolutions.
        saveLatentImages: if True, dump PNGs of the spectrally-summed activations at each
            encoder/decoder stage to latentImageOutput (see SaveSummedSpectralChannels above) -
            useful for visually debugging what the network is learning, not needed for normal
            training/inference.
        Returns: tensor of shape (nBatch, flat_size) - the flattened pre-fc0 features.
        """
        x, _ = self._decode(x, saveLatentImages=saveLatentImages)

        ##  Output Inference ##
        x = self.avgpool0(x)

        return torch.flatten(x,1)

    def _decode(self, x, saveLatentImages=False):
        """Run the encoder and the upsample -> crop -> concatenate -> decoder stages.

        Returns (decoded, bottleneck): the decoder's output (at 1/4 resolution with the FC
        head, 1/2 with the conv head) and the deepest encoder features.
        """
        nBatch,nSpec,nX,nY = x.size()
        x = torch.reshape(x,(nBatch,1,nSpec,nX,nY))


        ##  Initial convolutional layer and pooling  ##
        if self.output_head == "conv":
            #Run layer0's conv, pool, and activation one at a time to keep the half-resolution
            #features for the last skip connection. LeakyReLU commutes with max pooling, so
            #this is the same as layer0(x).
            stem = self.layer0[0](x)
            stemFeatures = self.layer0[2](stem)
            x = self.layer0[2](self.layer0[1](stem))
        else:
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
        bottleneck = x
            
        ## Deconvolutional Blocks ##
        x = self.upsample1(x)
        x = self._concat_skip(x, featureMap3) #Feature forwarding from left convolutional -> deconvolutional wing
        x = self.dc_layer1(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,5)

        x = self.upsample2(x)
        x = self._concat_skip(x, featureMap2) #Cropped Feature forwarding
        x = self.dc_layer2(x)
        if saveLatentImages: SaveSummedSpectralChannels(x,6)

        x = self.upsample3(x)
        x = self._concat_skip(x, featureMap1) #Cropped Feature forwarding
        x = self.dc_layer3(x)
        if saveLatentImages:
            SaveSummedSpectralChannels(x,7)
            #SaveAllSpectralChannels(x,7)

        if self.output_head == "conv":
            x = self.upsample4(x)
            x = self._concat_skip(x, stemFeatures) #Feature forwarding from the stem, at half resolution
            x = self.dc_layer4(x)
            if saveLatentImages: SaveSummedSpectralChannels(x,8)

        return x, bottleneck

    def forward(self, x, saveLatentImages=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; a channel dim of size 1 is inserted before the 3-d convolutions.
        saveLatentImages: if True, dump PNGs of the spectrally-summed activations at each
            encoder/decoder stage to latentImageOutput (see SaveSummedSpectralChannels above) -
            useful for visually debugging what the network is learning, not needed for normal
            training/inference.
        Returns: tensor of shape (nBatch, self.output_size) - the flattened predicted output map.
        """
        global saveItr
        if self.output_head == "conv":
            decoded, bottleneck = self._decode(x, saveLatentImages=saveLatentImages)
            saveItr+=1
            return self.head(decoded, bottleneck, outputShape=x.shape[2:])

        x = self._features(x, saveLatentImages=saveLatentImages)
        if self.nFC>0:
            x = self.fc0(x)
            x = self.relu0(x)
        x = self.dropout0(x)
        output = self.fc1(x)

        saveItr+=1
        return output
