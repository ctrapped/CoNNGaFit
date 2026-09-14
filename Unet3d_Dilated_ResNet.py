import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 3-d dilated residual network ("Encode-Process-Decode") for HI 21cm spectral datacubes followin
# Stachenfeld et al. 2021, "Learned Simulators for Turbulence" (https://arxiv.org/abs/2112.15275).
# Reads an (nSpec x nX x nY) cube of arbitrary size and predicts a flattened output map,
# matching the input/output conventions of Unet3d_ResBlock.py. There is no spatial
# downsampling/upsampling path or skip connections; the receptive field is grown entirely
# through dilated convolutions. See NeuralNetwork for the full parameter list.
#
# Written by Caleb Choban using Claude LLM, adapting Unet3d_ResBlock.py's conventions.

latentImageOutput="latentImages/latent_"
from matplotlib import pyplot as plt
def SaveAllSpectralChannels(x,nStep):
    """Save one PNG per (filter, spectral channel) of activation tensor x (nBatch,C,nSpec,nX,nY).

    Writes latentImageOutput + "step<nStep>_filter<i>_nSpec<j>.png" for the first batch
    element. Diagnostic only.
    """
    npX = x.cpu().float().detach().numpy();
    print(np.shape(x))
    for i in range(0,np.shape(npX)[1]):
        for j in range(0,np.shape(npX)[2]):
            plt.figure()
            plt.imshow(npX[0,i,j,:,:],cmap='inferno')
            plt.savefig(latentImageOutput+"step"+str(nStep)+"_filter"+str(i)+"_nSpec"+str(j)+".png")
            plt.close()

def SaveSummedSpectralChannels(x,nStep,saveItr):
    """Save one PNG per filter of activation tensor x, summed over the spectral axis.

    x has shape (nBatch, C, nSpec, nX, nY); writes latentImageOutput +
    "sum_step<nStep>_filter<i>_gal<saveItr>.png" for the first batch element. saveItr is
    passed in by the caller (NeuralNetwork.forward passes its self.saveItr) rather than read
    off a module global. Diagnostic only.
    """
    print(np.shape(x))
    npX = x.cpu().float().detach().numpy();
    toPlot = np.sum(npX,axis=2)
    for i in range(0,np.shape(npX)[1]):
        plt.figure()
        plt.imshow(toPlot[0,i,:,:],cmap='inferno')
        plt.savefig(latentImageOutput+"sum_step"+str(nStep)+"_filter"+str(i)+"_gal"+str(saveItr)+".png")
        plt.close()


def _ceil_conv_pad(kernel_size, stride):
    """(before, after) padding per dim so a strided conv gives out = ceil(N/stride).

    A strided conv has out = floor((N + total_pad - k) / stride) + 1, which equals
    ceil(N/stride) for every N iff total_pad == k - stride (clamped to >= 0). For stride=1
    this reduces to the usual k-1 'same' padding. For odd total_pad the extra 1 goes on the
    trailing side. Returned as an F.pad arg for a 5-d tensor: (W_b, W_a, H_b, H_a, D_b, D_a).
    """
    pad = []
    for k, s in zip(kernel_size, stride):  # kernel_size/stride are (D, H, W); F.pad wants W, H, D order
        total = max(k - s, 0)
        before = total // 2
        after = total - before
        pad = [before, after] + pad
    return tuple(pad)


def _strided_out_shape(input_shape, stride):
    """(nSpec, nX, nY) shape after a single conv with the given stride and _ceil_conv_pad
    padding, i.e. ceil(N/stride) per axis."""
    return tuple(-(-n // s) for n, s in zip(input_shape, stride))


#### Encode: project the raw input cube into feature space ####
class Encoder(nn.Module):
    """A single Conv3d projecting the 1-channel input cube to out_channels, at full resolution
    unless `stride` downsamples it. Default stride (1,1,1) keeps the whole network at full
    resolution, since the receptive field is grown via dilation rather than pooling; a larger
    stride is a practical escape hatch for very large input cubes.

    Parameters:
        out_channels: number of feature channels (== nFilt0, constant through the whole net).
        kernel_size: 3-tuple kernel size.
        stride: 3-tuple stride; (1,1,1) (default) keeps every axis exactly N via 'same'
            padding, any other stride brings each axis to ceil(N/stride) via _ceil_conv_pad.
        activation: zero-arg callable returning an nn.Module (see DilatedResidualBlock's
            activation parameter for the accepted forms), or None for no activation.
    """

    def __init__(self, out_channels, kernel_size, stride=(1, 1, 1), activation=None):
        super().__init__()
        self.stride = tuple(stride)
        if all(s == 1 for s in self.stride):
            self.pad = None
            self.conv = nn.Conv3d(1, out_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding='same')
        else:
            self.pad = _ceil_conv_pad(kernel_size, self.stride)
            self.conv = nn.Conv3d(1, out_channels, kernel_size=kernel_size, stride=self.stride, padding=0)
        self.act = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        """x: (nBatch, 1, D, H, W) -> (nBatch, out_channels, D', H', W'), D'/H'/W' = D/H/W
        unmodified if stride=(1,1,1), else ceil(D/stride) etc."""
        if self.pad is not None:
            x = F.pad(x, self.pad)
        return self.act(self.conv(x))


#### Process: a stack of dilated residual blocks, resolution-preserving throughout ####
class DilatedResidualBlock(nn.Module):
    """One dilated residual block: a chain of Conv3d layers whose dilation rates cycle through
    `dilation_rates` (default (1, 2, 4, 8, 4, 2, 1)), each at stride 1 with 'same' padding so
    every layer - and the block as a whole - preserves the spatial/spectral shape exactly,
    regardless of dilation. Every conv is followed by BatchNorm3d and an activation, except the
    last conv in the chain, whose BatchNorm3d output is added to the block's input (a plain
    identity skip, since channels never change through the Process stack) before a final
    activation and channel-wise dropout - the same conv-bn-act...conv-bn + skip -> act ->
    dropout convention as ResidualBlock in Unet3d_ResBlock.py, with dilation added and no
    downsampling.

    Parameters:
        channels: feature channel count (in == out, held constant through the block).
        kernel_size: 3-tuple conv kernel size, used by every layer in the chain.
        dilation_rates: sequence of per-layer dilation factors (applied isotropically, i.e.
            the same rate along D/H/W).
        dropout_rate: probability for the channel-wise nn.Dropout3d on the block output. 0.0
            disables it.
        activation: zero-arg callable returning an nn.Module (the class itself, e.g. nn.ReLU,
            or a lambda such as `lambda: nn.LeakyReLU(0.2)`). Called once per site so every
            activation is an independent module.
    """

    def __init__(self, channels, kernel_size, dilation_rates=(1, 2, 4, 8, 4, 2, 1),
                 dropout_rate=0.0, activation=nn.ReLU):
        super().__init__()
        self.dilation_rates = tuple(dilation_rates)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        for d in self.dilation_rates:
            self.convs.append(nn.Conv3d(channels, channels, kernel_size=kernel_size,
                                         stride=(1, 1, 1), dilation=d, padding='same'))
            self.bns.append(nn.BatchNorm3d(channels))
            self.acts.append(activation())
        self.final_act = activation()
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        """x: (nBatch, channels, D, H, W) -> same shape."""
        skip = x
        out = x
        n = len(self.convs)
        for i, (conv, bn, act) in enumerate(zip(self.convs, self.bns, self.acts)):
            out = bn(conv(out))
            if i < n - 1:
                out = act(out)
        out = out + skip
        out = self.final_act(out)
        return self.dropout(out)


#### Decode: project the Process stack's output back into the head's input features ####
class Decoder(nn.Module):
    """A single Conv3d, at stride 1 with 'same' padding so the spatial/spectral shape is
    unchanged. Feeds the Process stack's output into NeuralNetwork's regression head.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        kernel_size: 3-tuple kernel size.
        activation: zero-arg callable returning an nn.Module, or None for no activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding='same')
        self.act = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D, H, W)."""
        return self.act(self.conv(x))


#### Full network ####
class NeuralNetwork(nn.Module):
    """3-d dilated residual network ("Encode-Process-Decode") mapping an HI 21cm spectral
    datacube (nSpec x nX x nY, any size) to a flattened output map (e.g. radial mass flux,
    rotational velocity, inclination) of length `output_size` (default nX*nY).

    Architecture:
      - Encode (layer_encode): a single Conv3d projecting the 1-channel input to nFilt0
        channels (see Encoder). Stride (1,1,1) by default, so the whole network runs at full
        input resolution - there is no downsampling/upsampling path and therefore no
        encoder/decoder skip connections; the receptive field instead grows via dilation.
      - Process: `n_blocks` DilatedResidualBlock instances applied in sequence, each cycling
        through `dilation_rates` (default (1,2,4,8,4,2,1)) while holding the channel count and
        spatial/spectral shape fixed at nFilt0 and the encoder's output shape.
      - Decode (layer_decode): a single Conv3d (see Decoder) projecting back to nFilt0
        channels, feeding the regression head.
      - Head: identical to Unet3d_ResBlock.py's - nn.AdaptiveAvgPool3d collapses the surviving
        spatial shape to `pool_output`, so the flattened size is fixed regardless of input
        shape or depth; then `n_fc_layers` hidden Linear->activation->dropout blocks feed the
        output Linear and an optional output activation.

    Because every stage preserves shape exactly (or, for the encoder, downsamples via a single
    ceil(N/stride) conv) and nn.AdaptiveAvgPool3d accepts any input size, there is no minimum-
    input-size / stage-count compatibility constraint to check at construction time.

    Parameters:
        input_shape: (nSpec, nX, nY) of one input cube (no batch/channel dim). forward()
            rejects a batch whose shape does not match.
        nFilt0: feature channel count, held constant through the encoder, every Process
            block, and the decoder (no channel doubling - there are no downsampling stages).
        kernel_stem: 3-tuple kernel size for the Encoder and Decoder convolutions.
        kernel_coder: 3-tuple kernel size for every dilated convolution in the Process stack.
        nFC: width of the fully-connected head's hidden layers. An int applies that width to
            every hidden layer; a list/tuple gives explicit per-layer widths (its length
            then overrides n_fc_layers). <= 0 or empty means no hidden layers - the pooled
            features go straight to the output Linear (with a dropout in between).
        n_blocks: number of DilatedResidualBlock instances in the Process stack. With the
            default 7-rate dilation schedule, n_blocks=4 gives 28 dilated conv layers.
        dilation_rates: per-layer dilation schedule cycled by every DilatedResidualBlock
            (default (1, 2, 4, 8, 4, 2, 1)).
        stem_stride: 3-tuple stride for the Encoder conv. (1,1,1) (default) means no
            downsampling; a larger stride (e.g. (2,2,2)) trades resolution for compute on
            large input cubes, bringing every axis to ceil(N/stride).
        n_fc_layers: number of hidden fully-connected layers when nFC is a scalar (ignored
            when nFC is a sequence of widths).
        output_size: length of the flattened prediction. Defaults to nX*nY from input_shape.
        pool_output: 3-tuple target grid for nn.AdaptiveAvgPool3d. The fc input size is
            nFilt0 * prod(pool_output); the default (4, 4, 8) gives 128 * nFilt0.
        fc_dropout_rate: dropout probability after every hidden FC layer's activation (and,
            when there are no hidden layers, once before the output Linear). 0.0 disables.
            Active in training mode only.
        block_dropout_rate: probability for a channel-wise nn.Dropout3d (drops whole feature
            channels) inside every DilatedResidualBlock, after the residual add + final
            activation. 0.0 disables.
        encode_activation: activation after the Encoder conv (default None, i.e. no
            activation).
        block_activation: activation used inside every DilatedResidualBlock, after every conv
            but the chain's last, and once more after the residual add (default nn.ReLU).
        decode_activation: activation after the Decoder conv (default None, i.e. no
            activation).
        fc_activation: activation between hidden fully-connected layers (default nn.LeakyReLU).
        output_activation: activation on the output Linear (default nn.Identity, i.e. none).
        predict_variance: if False (default), the head has a single output Linear and
            forward() returns one tensor of shape (nBatch, output_size), as before. If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar) - train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE. logvar has
            no output_activation applied (it must stay unconstrained for exp() to reach any
            positive variance).

        Each *_activation is a zero-arg callable returning an nn.Module - the activation
        class itself (nn.ReLU, nn.GELU, nn.Tanh, ...) or a lambda for non-default arguments
        (e.g. lambda: nn.LeakyReLU(0.2)). It is called once per site, so every layer gets
        its own module. encode_activation/decode_activation may also be None for no activation.
    """

    def __init__(self, input_shape, nFilt0, kernel_stem, kernel_coder, nFC, n_blocks=4,
                 dilation_rates=(1, 2, 4, 8, 4, 2, 1), stem_stride=(1, 1, 1),
                 n_fc_layers=1, output_size=None, pool_output=(4, 4, 8),
                 fc_dropout_rate=0.0, block_dropout_rate=0.0,
                 encode_activation=None, block_activation=nn.ReLU, decode_activation=None,
                 fc_activation=nn.LeakyReLU, output_activation=nn.Identity,
                 predict_variance=False):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        self.output_size = int(output_size) if output_size is not None else input_shape[1] * input_shape[2]
        print("output size=", self.output_size)
        self.nFC = nFC
        self.saveItr = 0  # bumped once per forward() call; passed by value into the Save*SpectralChannels helpers

        self.nFilt0 = nFilt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.stem_stride = tuple(stem_stride)
        self.n_blocks = n_blocks
        self.dilation_rates = tuple(dilation_rates)
        self.encode_out_shape = _strided_out_shape(input_shape, self.stem_stride)

        # Encode: single conv, 1 -> nFilt0 channels, ceil(N/stem_stride) per axis
        self.layer_encode = Encoder(nFilt0, kernel_stem, stride=self.stem_stride, activation=encode_activation)

        # Process: n_blocks dilated residual blocks, shape and channel count fixed at nFilt0
        self.process = nn.ModuleList([
            DilatedResidualBlock(nFilt0, kernel_coder, dilation_rates=self.dilation_rates,
                                  dropout_rate=block_dropout_rate, activation=block_activation)
            for _ in range(n_blocks)
        ])

        # Decode: single conv, nFilt0 -> nFilt0 channels, shape unchanged
        self.layer_decode = Decoder(nFilt0, nFilt0, kernel_stem, activation=decode_activation)

        # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
        # Identical to Unet3d_ResBlock.py's head.
        self.pool_output = tuple(pool_output)
        self.avgpool0 = nn.AdaptiveAvgPool3d(self.pool_output)
        flat_features = nFilt0 * int(np.prod(self.pool_output))

        if isinstance(nFC, (list, tuple)):
            fc_widths = [int(w) for w in nFC if int(w) > 0]
        elif nFC and nFC > 0:
            fc_widths = [int(nFC)] * int(n_fc_layers)
        else:
            fc_widths = []
        self.fc_widths = fc_widths

        head, in_f = [], flat_features
        for w in fc_widths:
            head += [nn.Linear(in_f, w), fc_activation(), nn.Dropout(p=fc_dropout_rate)]
            in_f = w
        if not fc_widths:
            head.append(nn.Dropout(p=fc_dropout_rate))
        self.fc_hidden = nn.Sequential(*head)
        self.fc_out_mean = nn.Linear(in_f, self.output_size)
        if predict_variance:
            self.fc_out_logvar = nn.Linear(in_f, self.output_size)
        self.out_activation = output_activation()

    def forward(self, x, saveLatentImages=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; (nSpec, nX, nY) must equal the model's input_shape. A channel dim of
            size 1 is inserted before the 3-d convolutions.
        saveLatentImages: if True, dump PNGs of the spectrally-summed activations after the
            encoder and after each Process block to latentImageOutput (see
            SaveSummedSpectralChannels above) - useful for visually debugging what the network
            is learning, not needed for normal training/inference.
        Returns: if predict_variance=False (default), a tensor of shape (nBatch, output_size)
            - the flattened predicted output map. If predict_variance=True, a (mean, logvar)
            tuple of two such tensors - logvar is an unconstrained per-element log-variance
            (exp(logvar) to get the variance), with no output_activation applied.
        """
        nBatch, nSpec, nX, nY = x.size()
        if (nSpec, nX, nY) != self.input_shape:
            raise ValueError(
                f"input spatial/spectral shape {(nSpec, nX, nY)} does not match the "
                f"model's input_shape {self.input_shape}")
        x = torch.reshape(x, (nBatch, 1, nSpec, nX, nY))

        # Encode
        x = self.layer_encode(x)
        step = 0
        if saveLatentImages:
            SaveSummedSpectralChannels(x, step, self.saveItr); step += 1

        # Process: sequence of dilated residual blocks, shape/channels fixed throughout
        for block in self.process:
            x = block(x)
            if saveLatentImages:
                SaveSummedSpectralChannels(x, step, self.saveItr); step += 1

        # Decode
        x = self.layer_decode(x)
        if saveLatentImages:
            SaveSummedSpectralChannels(x, step, self.saveItr); step += 1

        # Head
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        mean = self.out_activation(self.fc_out_mean(x))

        self.saveItr += 1
        if self.predict_variance:
            return mean, self.fc_out_logvar(x)
        return mean

    def stage_shapes(self):
        """Analytically compute the tensor shape at every stage (no forward pass).

        Returns a list of (label, n_features, (nSpec, nX, nY)) tuples in execution order,
        matching what forward() produces: the encoder brings every axis to
        ceil(N/stem_stride) (exactly N when stem_stride is (1,1,1)), every Process block and
        the decoder then preserve that shape exactly (dilation with 'same' padding at stride 1
        never changes the spatial/spectral size).
        """
        rows = [("input datacube", 1, self.input_shape)]
        rows.append(("encode", self.nFilt0, self.encode_out_shape))
        for i in range(self.n_blocks):
            rows.append((f"process block {i}", self.nFilt0, self.encode_out_shape))
        rows.append(("decode", self.nFilt0, self.encode_out_shape))
        rows.append(("adaptive avg pool", self.nFilt0, self.pool_output))
        return rows

    def _enable_mc_dropout(self):
        """Put every Dropout/Dropout3d submodule into train mode while leaving BatchNorm3d
        (and everything else) in eval mode - MC Dropout needs stochastic dropout masks per
        sample, but BatchNorm3d in train mode would normalize against a single-sample batch
        at inference (RunInferences in UseModel.py runs batch_size=1)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    def predict_with_uncertainty(self, x, n_samples=30, saveLatentImages=False):
        """Run n_samples stochastic forward passes (dropout active, BatchNorm fixed).
        Call model.eval() first as usual; this only re-enables dropout for the duration of the
        call, then restores plain eval mode. Only informative if dropout is enabled when the
        model was trained.

        If predict_variance=False: returns (mean, epistemic_std), each of shape
            (nBatch, output_size) - epistemic_std is the MC-Dropout spread across samples.
        If predict_variance=True: returns (mean, aleatoric_std, epistemic_std, total_std).
            aleatoric_std comes from averaging the network's own per-sample variance
            predictions (exp(logvar)); epistemic_std is the spread of the sampled means, as
            above; total_std combines the two via the law of total variance
            (total_var = aleatoric_var + epistemic_var).
        """
        self._enable_mc_dropout()
        with torch.no_grad():
            if self.predict_variance:
                means, logvars = zip(*(self.forward(x, saveLatentImages) for _ in range(n_samples)))
                means = torch.stack(means, dim=0)
                logvars = torch.stack(logvars, dim=0)
            else:
                preds = torch.stack([self.forward(x, saveLatentImages) for _ in range(n_samples)], dim=0)
        self.eval()

        if self.predict_variance:
            mean_pred = means.mean(dim=0)
            aleatoric_std = logvars.exp().mean(dim=0).sqrt()
            epistemic_std = means.std(dim=0)
            total_std = (aleatoric_std**2 + epistemic_std**2).sqrt()
            return mean_pred, aleatoric_std, epistemic_std, total_std
        return preds.mean(dim=0), preds.std(dim=0)

    def summary(self, print_fn=print):
        """Print the network structure, the input dimensions, the data/feature-count
        progression through every layer, which activation runs at each stage, and the
        dilation schedule used by the Process stack. `print_fn` lets output be redirected
        (e.g. to a logger); defaults to the builtin print.
        """
        rows = self.stage_shapes()
        w = max(len(label) for label, _, _ in rows)
        bar = "=" * (w + 40)

        encode_act = type(self.layer_encode.act).__name__
        block_act = type(self.process[0].acts[0]).__name__  # same class at every block/site
        decode_act = type(self.layer_decode.act).__name__
        fc_act = type(self.fc_hidden[1]).__name__ if self.fc_widths else None
        out_act = type(self.out_activation).__name__

        print_fn(bar)
        print_fn(f"{type(self).__name__}  |  {self.n_blocks} process blocks  |  "
                 f"dilation_rates={self.dilation_rates}  |  kernel_stem={self.kernel_stem}  |  "
                 f"kernel_coder={self.kernel_coder}  |  stem_stride={self.stem_stride}  |  "
                 f"predict_variance={self.predict_variance}")
        nS, nX, nY = self.input_shape
        print_fn(f"input:  (nBatch, nSpec={nS}, nX={nX}, nY={nY})  ->  "
                 f"output: (nBatch, {self.output_size})")
        print_fn(f"activations: encode={encode_act}  process blocks={block_act}  "
                 f"decode={decode_act}  fc hidden={fc_act or 'n/a (no hidden FC layers)'}  "
                 f"output={out_act}")
        print_fn("-" * (w + 40))

        def row_note(label):
            if label == "encode":
                return f"activation={encode_act}  stride={self.stem_stride}"
            if label.startswith("process block"):
                return f"activation={block_act}  dilation_rates={self.dilation_rates}"
            if label == "decode":
                return f"activation={decode_act}"
            return None

        for label, feats, shape in rows:
            dims = " x ".join(str(s) for s in shape)
            note = row_note(label)
            suffix = f" {note}" if note else ""
            print_fn(f"  {label:<{w}}   features={feats:<6} dims={dims}{suffix}")

        flat = self.nFilt0 * int(np.prod(self.pool_output))
        print_fn(f"  {'flatten':<{w}}   features={flat}")
        in_f = flat
        for k, out_f in enumerate(self.fc_widths):
            print_fn(f"  {'fc hidden ' + str(k):<{w}}   {in_f} -> {out_f}  activation={fc_act}")
            in_f = out_f
        print_fn(f"  {'fc out (mean)':<{w}}   {in_f} -> {self.output_size}  activation={out_act}")
        if self.predict_variance:
            print_fn(f"  {'fc out (logvar)':<{w}}   {in_f} -> {self.output_size}  activation=None")

        print_fn("-" * (w + 40))
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn(f"parameters: {n_all:,} total ({n_train:,} trainable)")
        print_fn(bar)
        print_fn(str(self))
        print_fn(bar)
