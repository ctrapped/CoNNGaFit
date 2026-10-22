import numpy as np
import torch
from torch import nn

from Unet_blocks import Stem, ResidualBlock, Upsample, DeconvBlock, ConvMapHead, HalfResConvHead, _match_spatial
from Unet_diagnostics import SaveSummedSpectralChannels, count_params, layer_param_lines

# 3-d residual U-Net for HI 21cm spectral datacubes. Reads an (nSpec x nX x nY) cube of
# arbitrary size and predicts a flattened output map. See NeuralNetwork for the full
# parameter list.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.
# Building blocks live in Unet_blocks.py; latent-image diagnostics in Unet_diagnostics.py.


#### Input-shape checks ####
def _stem_out_shape(input_shape):
    """(nSpec, nX, nY) shape after layer_stem: a stride-2 conv then a stride-2 maxpool, each
    exactly halving (ceil(N/2)) via _ceil_conv_pad, so every axis ends at exactly ceil(N/4) -
    true for any kernel size (ceiling division composes exactly: ceil(ceil(n/2)/2) = ceil(n/4))."""
    return tuple(-(-n // 4) for n in input_shape)


def _check_stage_compatibility(input_shape, n_stages):
    """Raise ValueError unless the input survives n_stages-1 halvings after the stem.

    The encoder has n_stages stages: the first keeps its shape, each later one
    downsamples every axis to ceil(N/2). So after the stem there are n_stages-1
    halvings, and an axis that reaches 1 before they are all done cannot be
    downsampled again. The message reports the tightest axis and the largest
    n_stages that axis allows. Returns the post-stem (nSpec, nX, nY) shape on success.
    """
    stem_shape = _stem_out_shape(input_shape)
    n_halvings = n_stages - 1
    tightest = (1 << 30, "", 0, 0)  # (max_halvings, axis_name, input_len, post_stem_len)
    for name, n0, n_stem in zip(("spectral", "x", "y"), input_shape, stem_shape):
        if n_stem < 1:
            raise ValueError(
                f"input {name} axis = {n0} is too small: the stem reduces it to "
                f"{n_stem} (< 1). Enlarge that axis.")
        cap, n = 0, n_stem            # count ceil-halvings this axis can take before hitting 1
        while n >= 2:
            n = -(-n // 2)
            cap += 1
        if cap < tightest[0]:
            tightest = (cap, name, n0, n_stem)
    if tightest[0] < n_halvings:
        cap, name, n0, n_stem = tightest
        raise ValueError(
            f"n_stages={n_stages} needs {n_halvings} downsamplings after the stem, but the "
            f"{name} axis (input {n0}, {n_stem} after the stem) supports at most {cap}. "
            f"Use n_stages <= {cap + 1} or enlarge that axis.")
    return stem_shape

#### Full network ####
class NeuralNetwork(nn.Module):
    """3-d residual U-Net mapping an HI 21cm spectral datacube (nSpec x nX x nY, any size)
    to a flattened output map (e.g. radial mass flux, rotational velocity, inclination) of
    length `output_size` (default nX*nY).

    Architecture:
      - Stem (layer_stem): stride-2 Conv3d + stride-2 MaxPool3d + activation, reducing every
        axis to exactly ceil(N/4) (see Stem).
      - Encoder: `n_stages` stages of two ResidualBlocks each. Stage 0 keeps the stem's
        shape and channel count; every later stage brings each axis to ceil(N/2) and
        doubles the channel count (nFilt0 * 2**i at stage i).
      - Decoder: `n_stages - 1` stages. Each runs Upsample (~2x per axis; see Upsample for
        `upsample_mode`'s two implementations), realigns the result to the matching encoder
        skip map with _match_spatial (center-crop/pad), concatenates channel-wise, then a
        DeconvBlock pair halves the channel count.
      - Head, chosen by `head`:
        'fc' (default): nn.AdaptiveAvgPool3d collapses the surviving spatial shape to
        `pool_output`, so the flattened size is fixed regardless of input shape or depth; then
        `n_fc_layers` hidden Linear->activation->dropout blocks feed the output Linear and an
        optional output activation.
        'conv': ConvMapHead (see Unet_blocks.py) - spectral attention pool, bilinear resize to
        `output_shape`, 2-d residual blocks and a 3x3 output conv; its size does not grow with
        the input or map size.
        'conv_halfres': one more decoder stage up to the stem conv's half resolution (with a
        skip connection from it), then HalfResConvHead (see Unet_blocks.py) - spectral axis
        folded into channels, plus global context from the bottleneck, two 3x3 convs, a 1x1
        output conv, and a bilinear resize to `output_shape`. Its size also does not grow
        with the map size.

    The encoder/decoder are built in loops and the skip alignment + adaptive pool are
    shape-agnostic, so the network runs at any input size and any depth >= 2. __init__
    calls _check_stage_compatibility, which raises a descriptive ValueError (naming the
    tightest axis) if `input_shape` cannot survive the stem plus n_stages-1 ceil-halvings.
    Every activation is configurable via the *_activation args.

    Parameters:
        input_shape: (nSpec, nX, nY) of one input cube (no batch/channel dim). forward()
            rejects a batch whose shape does not match.
        nFilt0: channel count of the first encoder stage; doubles at each later stage.
        kernel_stem: 3-tuple kernel size of the stem's strided convolution.
        kernel_coder: 3-tuple kernel size for every ResidualBlock/DeconvBlock convolution.
        nFC: width of the fully-connected head's hidden layers. An int applies that width to
            every hidden layer; a list/tuple gives explicit per-layer widths (its length
            then overrides n_fc_layers). <= 0 or empty means no hidden layers - the pooled
            features go straight to the output Linear (with a dropout in between).
        n_stages: number of encoder stages (>= 2); the decoder has n_stages - 1. Validated
            against input_shape at construction (see _check_stage_compatibility).
        n_fc_layers: number of hidden fully-connected layers when nFC is a scalar (ignored
            when nFC is a sequence of widths).
        output_size: length of the flattened prediction. Defaults to nX*nY from input_shape
            (or H*W from output_shape); must equal H*W for the conv heads.
        pool_output: 3-tuple target grid for nn.AdaptiveAvgPool3d. The fc input size is
            nFilt0 * prod(pool_output); the default (8, 8, 8) gives 512 * nFilt0.
        fc_dropout_rate: dropout probability after every hidden FC layer's activation (and,
            when there are no hidden layers, once before the output Linear); for the conv heads,
            a channel-wise Dropout2d before the output conv. 0.0 disables. Active in training
            mode only.
        block_dropout_rate: probability for a channel-wise nn.Dropout3d (drops whole feature
            channels) inside every ResidualBlock/DeconvBlock, after the residual add + final
            activation. 0.0 disables. Broader than fc_dropout_rate: it touches every stage
            and, since skip maps are captured straight after their stage, also regularizes
            the skip connections. Enable it only if fc_dropout_rate alone leaves a
            train/validation gap, and start low (~0.1-0.15) - the deepest stage is tiny, so
            dropping whole channels there removes a large fraction of its information.
        stem_activation: activation for the stem (default nn.LeakyReLU).
        block_activation: activation at all three sites in every ResidualBlock/DeconvBlock
            (post-conv1, post-conv2, post residual-add; default nn.ReLU).
        fc_activation: activation between hidden fully-connected layers (default nn.LeakyReLU).
        output_activation: activation on the output Linear (default nn.Identity, i.e. none).
        upsample_mode: how every decoder Upsample resizes spatially - 'conv_transpose'
            (default, a learned stride-2 ConvTranspose3d) or 'trilinear' (nn.Upsample's 3-d
            interpolation mode, followed by a 1x1x1 Conv3d channel projection). See Upsample
            for the tradeoffs.
        predict_variance: if False (default), the head has a single output Linear and
            forward() returns one tensor of shape (nBatch, output_size). If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar) - train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE. logvar has
            no output_activation applied (it must stay unconstrained for exp() to reach any
            positive variance). With a conv head the log-variance is a second output map
            from the head's output conv instead of a second Linear.
        head: 'fc' (default) - adaptive average pool -> flatten -> fully-connected layers, as
            set by pool_output/nFC/n_fc_layers; its output Linear has
            nFilt0 * prod(pool_output) * output_size weights, so it grows with the map size.
            'conv' - ConvMapHead: spectral attention pool -> bilinear resize to output_shape ->
            n_head_blocks 2-d residual blocks -> 3x3 conv, with a parameter count independent of
            the input and map size (pool_output, nFC and n_fc_layers are then unused).
            'conv_halfres' - half-resolution decoder stage -> HalfResConvHead (see the
            Architecture entry above); pool_output, nFC and n_fc_layers are also unused.
        output_shape: (H, W) of the predicted map. Required grid for head='conv' (default
            (nX, nY)) and head='conv_halfres'; for head='fc' it only sets the default output_size = H*W.
        n_head_blocks: number of 2-d residual blocks in head='conv' (>= 0; default 2).
        head_channels: channel count of head='conv_halfres''s two 3x3 convolutions (default 64).

        Each *_activation is a zero-arg callable returning an nn.Module - the activation
        class itself (nn.ReLU, nn.GELU, nn.Tanh, ...) or a lambda for non-default arguments
        (e.g. lambda: nn.LeakyReLU(0.2)). It is called once per site, so every layer gets
        its own module.
    """

    def __init__(self, input_shape, nFilt0, kernel_stem, kernel_coder, nFC, n_stages=4,
                 n_fc_layers=1, output_size=None, pool_output=(8, 8, 8),
                 fc_dropout_rate=0.0, block_dropout_rate=0.0,
                 stem_activation=nn.LeakyReLU, block_activation=nn.ReLU,
                 fc_activation=nn.LeakyReLU, output_activation=nn.Identity,
                 upsample_mode='conv_transpose', predict_variance=False,
                 head='fc', output_shape=None, n_head_blocks=2, head_channels=64):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 (need at least one skip connection)")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        self.n_stages = n_stages
        if head not in ('fc', 'conv', 'conv_halfres'):
            raise ValueError(f"unknown head {head!r}; use 'fc', 'conv' or 'conv_halfres'")
        self.head_type = head
        # Map grid of the prediction: required by the conv head, optional for the fc head.
        self.output_shape = (tuple(int(v) for v in output_shape) if output_shape is not None
                             else (input_shape[1:] if head != 'fc' else None))
        map_size = (int(np.prod(self.output_shape)) if self.output_shape is not None
                    else input_shape[1] * input_shape[2])
        if head != 'fc' and output_size is not None and int(output_size) != map_size:
            raise ValueError(f"output_size={output_size} does not match output_shape "
                             f"{self.output_shape} for head={head!r}")
        self.output_size = int(output_size) if output_size is not None else map_size
        # fail fast if this input can't be halved n_stages-1 times after the stem
        self.stem_out_shape = _check_stage_compatibility(input_shape, n_stages)
        print("output size=", self.output_size)
        self.nFC=nFC
        self.saveItr = 0  # forward() call counter; numbers the latent-image filenames

        # Stem: stride-2 conv + stride-2 maxpool + activation, reducing every axis by exactly ceil(N/4)
        self.layer_stem = Stem(nFilt0, kernel_stem, activation=stem_activation)

        # Encoder: stage 0 keeps the stem's shape/channels; each later stage -> ceil(N/2)
        # per axis and doubles the channel count.
        enc_channels = [nFilt0 * (2 ** i) for i in range(n_stages)]
        self.nFilt0 = nFilt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.enc_channels = list(enc_channels)
        self.encoder = nn.ModuleList()
        prev = nFilt0
        for i, ch in enumerate(enc_channels):
            downsample = i > 0
            self.encoder.append(nn.Sequential(
                ResidualBlock(prev,ch,downsample=downsample,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation),
                ResidualBlock(ch,ch,downsample=False,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation)
            ))
            prev = ch

        # Decoder: stage j upsamples the running tensor, concatenates the skip map from
        # encoder stage enc_idx (same channel count), then a DeconvBlock pair reduces 2C -> C.
        self.upsamples = nn.ModuleList()
        self.dc_layers = nn.ModuleList()
        for j in range(n_stages - 1):
            enc_idx = n_stages - 2 - j
            in_ch = enc_channels[n_stages - 1 - j]
            out_ch = enc_channels[enc_idx]
            self.upsamples.append(Upsample(in_ch,out_ch,kernel1=kernel_coder,mode=upsample_mode))
            self.dc_layers.append(nn.Sequential(
                DeconvBlock(2*out_ch,out_ch,upsample=True,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation),
                DeconvBlock(out_ch,out_ch,upsample=False,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation)
            ))

        if head == 'conv_halfres':
            # One more decoder stage, up to the stem conv's half resolution, with a skip
            # connection from it (see Stem.forward's return_half)
            self.upsample_half = Upsample(nFilt0,nFilt0,kernel1=kernel_coder,mode=upsample_mode)
            self.dc_layer_half = nn.Sequential(
                DeconvBlock(2*nFilt0,nFilt0,upsample=True,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation),
                DeconvBlock(nFilt0,nFilt0,upsample=False,kernel1=kernel_coder,dropout_rate=block_dropout_rate,activation=block_activation)
            )

        # Identity pass-through on the skip connections; a named module so it can later be
        # swapped for a learned transform without touching forward().
        self.featureForwarding = nn.Sequential()

        self.pool_output = tuple(pool_output)
        if head == 'conv':
            # Head: spectral attention pool -> resize to output_shape -> 2-d res blocks -> conv.
            self.fc_widths = []
            self.conv_head = ConvMapHead(nFilt0, out_channels=2 if predict_variance else 1,
                                         n_blocks=n_head_blocks, activation=block_activation,
                                         dropout_rate=fc_dropout_rate)
        elif head == 'conv_halfres':
            # Head: fold the half-resolution spectral axis into channels -> HalfResConvHead
            self.half_shape = tuple(-(-n // 2) for n in input_shape)
            self.fc_widths = []
            self.head_channels = head_channels
            self.halfres_head = HalfResConvHead(nFilt0 * self.half_shape[0], enc_channels[-1], head_channels,
                                                dropout_rate=fc_dropout_rate, activation=fc_activation,
                                                predict_variance=predict_variance)
        else:
            # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
            self.avgpool0 = nn.AdaptiveAvgPool3d(self.pool_output)
            flat_features = nFilt0 * int(np.prod(self.pool_output))

            # Resolve the hidden-layer widths: a scalar nFC repeated n_fc_layers times, or an
            # explicit sequence of widths. Empty -> no hidden layers (a dropout still precedes
            # the output Linear).
            if isinstance(nFC, (list, tuple)):
                fc_widths = [int(w) for w in nFC if int(w) > 0]
            elif nFC and nFC > 0:
                fc_widths = [int(nFC)] * int(n_fc_layers)
            else:
                fc_widths = []
            self.fc_widths = fc_widths

            fc_layers, in_f = [], flat_features
            for w in fc_widths:
                fc_layers += [nn.Linear(in_f, w), fc_activation(), nn.Dropout(p=fc_dropout_rate)]
                in_f = w
            if not fc_widths:
                fc_layers.append(nn.Dropout(p=fc_dropout_rate))
            self.fc_hidden = nn.Sequential(*fc_layers)
            self.fc_out_mean = nn.Linear(in_f, self.output_size)
            if predict_variance:
                self.fc_out_logvar = nn.Linear(in_f, self.output_size)
        self.out_activation = output_activation()


    def forward(self, x, saveLatentImages=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; (nSpec, nX, nY) must equal the model's input_shape. A channel dim of
            size 1 is inserted before the 3-d convolutions.
        saveLatentImages: if True, dump PNGs of the spectrally-summed activations at each
            encoder/decoder stage to latentImageOutput (see SaveSummedSpectralChannels in Unet_diagnostics.py) -
            useful for visually debugging what the network is learning, not needed for normal
            training/inference.
        Returns: if predict_variance=False (default), a tensor of shape (nBatch, output_size)
            - the flattened predicted output map. If predict_variance=True, a (mean, logvar)
            tuple of two such tensors - logvar is an unconstrained per-element log-variance
            (exp(logvar) to get the variance), with no output_activation applied.
        """
        nBatch,nSpec,nX,nY = x.size()
        if (nSpec, nX, nY) != self.input_shape:
            raise ValueError(
                f"input spatial/spectral shape {(nSpec, nX, nY)} does not match the "
                f"model's input_shape {self.input_shape}")
        x = torch.reshape(x,(nBatch,1,nSpec,nX,nY))

        # Stem
        if self.head_type == 'conv_halfres':
            x, stem_half = self.layer_stem(x, return_half=True)
        else:
            x = self.layer_stem(x)
        step = 0
        if saveLatentImages:
            SaveSummedSpectralChannels(x,step,self.saveItr); step += 1

        # Encoder: run every stage, keep all but the last as skip connections
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if saveLatentImages:
                SaveSummedSpectralChannels(x,step,self.saveItr); step += 1
            if i < len(self.encoder) - 1:
                skips.append(self.featureForwarding(x))

        # Decoder: upsample, align to the matching skip map, concatenate, reduce channels
        for up, dc in zip(self.upsamples, self.dc_layers):
            x = up(x)
            skip = skips.pop()
            x = _match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dc(x)
            if saveLatentImages:
                SaveSummedSpectralChannels(x,step,self.saveItr); step += 1

        if self.head_type == 'conv_halfres':
            # Last decoder stage, to half resolution, then the per-pixel head
            x = self.upsample_half(x)
            skip = self.featureForwarding(stem_half)
            x = _match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.dc_layer_half(x)
            if saveLatentImages:
                SaveSummedSpectralChannels(x,step,self.saveItr); step += 1
            outputs = self.halfres_head(x, bottleneck, self.output_shape)
            self.saveItr += 1
            mean = self.out_activation(outputs[0])
            if self.predict_variance:
                return mean, outputs[1]
            return mean

        # Head
        mean, logvar = self._run_head(x)
        self.saveItr += 1
        if self.predict_variance:
            return mean, logvar
        return mean

    def _run_head(self, x):
        """Body features -> (mean, logvar), each flattened to (nBatch, output_size); logvar is None
        unless predict_variance. 'conv': one output map channel each; 'fc': pool -> FC stack."""
        if self.head_type == 'conv':
            out = self.conv_head(x, self.output_shape).flatten(2)   # (nBatch, 1 or 2, H*W)
            mean = self.out_activation(out[:, 0])
            return mean, (out[:, 1] if self.predict_variance else None)
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        mean = self.out_activation(self.fc_out_mean(x))
        return mean, (self.fc_out_logvar(x) if self.predict_variance else None)

    def _enable_mc_dropout(self):
        """Put every Dropout/Dropout2d/Dropout3d submodule into train mode while leaving BatchNorm3d
        (and everything else) in eval mode - MC Dropout needs stochastic dropout masks per
        sample, but BatchNorm3d in train mode would normalize against a single-sample batch
        at inference (RunInferences in UseModel.py runs batch_size=1)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
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

    def stage_shapes(self):
        """Analytically compute the tensor shape at every stage (no forward pass).

        Returns a list of (label, n_features, (nSpec, nX, nY)) tuples in execution order,
        matching what forward() produces: the stem is a stride-2 conv + stride-2 maxpool
        (ceil(N/4) per axis, exactly), each later encoder stage downsamples every axis to
        ceil(N/2), and each decoder stage runs Upsample - N -> 2N-1 for mode='conv_transpose',
        N -> 2N for mode='trilinear' (read off the model's actual Upsample instances, so this
        stays correct for either choice) - then _match_spatial, which realigns the result to
        the corresponding encoder skip map's shape before the concatenation.
        """
        rows = [("input datacube", 1, self.input_shape)]

        stem = _stem_out_shape(self.input_shape)
        rows.append(("stem (conv + maxpool)", self.nFilt0, stem))

        enc_shapes, shape = [], stem
        for i, ch in enumerate(self.enc_channels):
            if i > 0:
                shape = tuple(-(-s // 2) for s in shape)  # ceil(N/2) downsample
            enc_shapes.append(shape)
            tag = f"encoder stage {i}" + (" (no downsample)" if i == 0 else "")
            rows.append((tag, ch, shape))

        shape = enc_shapes[-1]
        for j in range(self.n_stages - 1):
            enc_idx = self.n_stages - 2 - j
            out_ch = self.enc_channels[enc_idx]
            up_mode = self.upsamples[j].mode
            up_shape = tuple((2 * s - 1 if up_mode == 'conv_transpose' else 2 * s) for s in shape)
            rows.append((f"decoder {j}: upsample", out_ch, up_shape))
            skip = enc_shapes[enc_idx]
            rows.append((f"decoder {j}: + skip(enc {enc_idx})", 2 * out_ch, skip))
            rows.append((f"decoder {j}: deconv block", out_ch, skip))
            shape = skip

        if self.head_type == 'conv':
            rows += self.conv_head.stage_shapes(shape[1:], self.output_shape)
        elif self.head_type == 'conv_halfres':
            up_mode = self.upsample_half.mode
            up_shape = tuple((2 * s - 1 if up_mode == 'conv_transpose' else 2 * s) for s in shape)
            rows.append(("half-res decoder: upsample", self.nFilt0, up_shape))
            rows.append(("half-res decoder: + skip(stem conv)", 2 * self.nFilt0, self.half_shape))
            rows.append(("half-res decoder: deconv block", self.nFilt0, self.half_shape))
            folded = self.nFilt0 * self.half_shape[0] + self.enc_channels[-1]
            rows.append(("conv head: fold spectra + global context", folded, self.half_shape[1:]))
            rows.append(("conv head: 2x 3x3 conv", self.head_channels, self.half_shape[1:]))
            rows.append(("conv head: 1x1 conv + resize", 1, self.output_shape))
        else:
            rows.append(("adaptive avg pool", self.nFilt0, self.pool_output))
        return rows

    def _param_groups(self):
        """(label, [modules]) for every summary() row, in order: the stage_shapes() rows, then (fc
        head) the fully-connected layers. Rows with no parameters (input, skip concat, pooling,
        resize) get []."""
        mods = [[], [self.layer_stem]] + [[stage] for stage in self.encoder]
        for up, dc in zip(self.upsamples, self.dc_layers):
            mods += [[up], [], [dc]]   # upsample, + skip (concat), deconv block
        if self.head_type == 'conv':
            head = self.conv_head.param_groups()
            head[-1] = (head[-1][0], head[-1][1] + [self.out_activation])   # output conv + activation
            mods += [m for _, m in head]
            return [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        if self.head_type == 'conv_halfres':
            h = self.halfres_head
            out = [h.dropout, h.out_mean] + ([h.out_logvar] if h.out_logvar is not None else [])
            mods += [[self.upsample_half], [], [self.dc_layer_half],   # half-res decoder stage
                     [], [h.conv1, h.act1, h.conv2, h.act2], out + [self.out_activation]]
            return [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        mods.append([])                # adaptive avg pool
        groups = [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        # fc_hidden is [Linear, activation, Dropout] per hidden layer (see __init__)
        groups += [(f"fc hidden {k}", [self.fc_hidden[3 * k], self.fc_hidden[3 * k + 1]])
                   for k in range(len(self.fc_widths))]
        groups.append(("fc out (mean)", [self.fc_out_mean, self.out_activation]))
        if self.predict_variance:
            groups.append(("fc out (logvar)", [self.fc_out_logvar]))
        return groups

    def param_breakdown(self):
        """(label, n_parameters) for every summary() row that holds parameters, in order. The counts
        add up to the model's total parameter count."""
        return [(label, count_params(mods)) for label, mods in self._param_groups() if mods]

    def summary(self, print_fn=print, layers=True):
        """Print the network structure, the input dimensions, the data/feature-count
        progression through every layer, the number of parameters in every block and layer,
        which activation runs at each stage, and which upsample_mode the decoder uses.
        `print_fn` lets output be redirected (e.g. to a logger); defaults to the builtin print.
        layers: if True (default), list each block's layers and their parameter counts under
        the block's row.
        """
        rows = self.stage_shapes()
        w = max(len(label) for label, _, _ in rows)
        bar = "=" * (w + 58)

        # Pull the actual instantiated activation module for each of the four independently
        # configurable roles (stem_activation, block_activation, fc_activation,
        # output_activation), rather than re-deriving them from the constructor args, so
        # this always reflects what the model is really running. Same idea for the decoder's
        # upsample_mode - read straight off a real Upsample instance.
        stem_act = type(self.layer_stem.act).__name__
        block_act = type(self.encoder[0][0].act1).__name__  # same class at every block/site
        if self.head_type == 'conv_halfres':
            fc_act = type(self.halfres_head.act1).__name__
        else:
            fc_act = type(self.fc_hidden[1]).__name__ if self.fc_widths else None
        out_act = type(self.out_activation).__name__
        if self.head_type == 'conv':
            head_act = ("head blocks=" + type(self.conv_head.blocks[0].act1).__name__
                        if len(self.conv_head.blocks) else "head blocks=n/a (none)")
        elif self.head_type == 'conv_halfres':
            head_act = f"conv head={fc_act}"
        else:
            head_act = f"fc hidden={fc_act or 'n/a (no hidden FC layers)'}"
        upsample_mode = self.upsamples[0].mode  # same mode at every decoder stage

        print_fn(bar)
        print_fn(f"{type(self).__name__}  |  {self.n_stages} encoder stages, "
                 f"{self.n_stages - 1} decoder stages  |  kernel_stem={self.kernel_stem}  |  "
                 f"kernel_coder={self.kernel_coder}  |  upsample_mode={upsample_mode!r}  |  "
                 f"predict_variance={self.predict_variance}  |  head={self.head_type!r}")
        nS, nX, nY = self.input_shape
        print_fn(f"input:  (nBatch, nSpec={nS}, nX={nX}, nY={nY})  ->  "
                 f"output: (nBatch, {self.output_size})")
        print_fn(f"activations: stem={stem_act}  encoder/decoder blocks={block_act}  "
                 f"{head_act}  output={out_act}")
        print_fn("-" * (w + 58))

        def row_note(label):
            # Only rows that actually apply an activation/resize get a note; concatenation/
            # pooling-only rows don't.
            if label.startswith("stem"):
                return f"activation={stem_act}"
            if label.endswith("deconv block") or label.startswith("encoder stage"):
                return f"activation={block_act}"
            if label.endswith(": upsample"):
                return f"mode={upsample_mode!r}"
            if self.head_type == 'conv':
                return self.conv_head.row_note(label)
            if label.startswith("conv head: 2x"):
                return f"activation={fc_act}"
            return None

        groups = dict(self._param_groups())

        def params_col(label):
            mods = groups.get(label, [])
            return f"{count_params(mods):,}" if mods else "-"

        def print_layers(label):
            if layers:
                for line in layer_param_lines(groups.get(label, [])):
                    print_fn(line)

        for label, feats, shape in rows:
            dims = " x ".join(str(s) for s in shape)
            note = row_note(label)
            suffix = f" {note}" if note else ""
            print_fn(f"  {label:<{w}}   features={feats:<6} params={params_col(label):<11} dims={dims}{suffix}")
            print_layers(label)

        if self.head_type == 'fc':
            flat = self.nFilt0 * int(np.prod(self.pool_output))
            print_fn(f"  {'flatten':<{w}}   features={flat}")
            in_f = flat
            for k, out_f in enumerate(self.fc_widths):
                label = f"fc hidden {k}"
                print_fn(f"  {label:<{w}}   {in_f} -> {out_f}  params={params_col(label)}  activation={fc_act}")
                in_f = out_f
            print_fn(f"  {'fc out (mean)':<{w}}   {in_f} -> {self.output_size}  "
                     f"params={params_col('fc out (mean)')}  activation={out_act}")
            if self.predict_variance:
                print_fn(f"  {'fc out (logvar)':<{w}}   {in_f} -> {self.output_size}  "
                         f"params={params_col('fc out (logvar)')}  activation=None")
        else:
            print_fn(f"  {'output (mean)':<{w}}   map -> (nBatch, {self.output_size})  activation={out_act}")
            if self.predict_variance:
                print_fn(f"  {'output (logvar)':<{w}}   map -> (nBatch, {self.output_size})  activation=None")
        print_fn("-" * (w + 58))
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn(f"parameters: {n_all:,} total ({n_train:,} trainable)")
        print_fn(bar)
        print_fn(str(self))
        print_fn(bar)
