import numpy as np
import torch
from torch import nn

from Unet_blocks import Stem, ResidualBlock, Upsample, DeconvBlock, _match_spatial
from Unet_diagnostics import SaveSummedSpectralChannels

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
      - Head: nn.AdaptiveAvgPool3d collapses the surviving spatial shape to `pool_output`,
        so the flattened size is fixed regardless of input shape or depth; then
        `n_fc_layers` hidden Linear->activation->dropout blocks feed the output Linear and
        an optional output activation.

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
        output_size: length of the flattened prediction. Defaults to nX*nY from input_shape.
        pool_output: 3-tuple target grid for nn.AdaptiveAvgPool3d. The fc input size is
            nFilt0 * prod(pool_output); the default (4, 4, 8) gives 128 * nFilt0.
        fc_dropout_rate: dropout probability after every hidden FC layer's activation (and,
            when there are no hidden layers, once before the output Linear). 0.0 disables.
            Active in training mode only.
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
            positive variance).

        Each *_activation is a zero-arg callable returning an nn.Module - the activation
        class itself (nn.ReLU, nn.GELU, nn.Tanh, ...) or a lambda for non-default arguments
        (e.g. lambda: nn.LeakyReLU(0.2)). It is called once per site, so every layer gets
        its own module.
    """

    def __init__(self, input_shape, nFilt0, kernel_stem, kernel_coder, nFC, n_stages=4,
                 n_fc_layers=1, output_size=None, pool_output=(4, 4, 8),
                 fc_dropout_rate=0.0, block_dropout_rate=0.0,
                 stem_activation=nn.LeakyReLU, block_activation=nn.ReLU,
                 fc_activation=nn.LeakyReLU, output_activation=nn.Identity,
                 upsample_mode='conv_transpose', predict_variance=False):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 (need at least one skip connection)")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        self.n_stages = n_stages
        self.output_size = int(output_size) if output_size is not None else input_shape[1] * input_shape[2]
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

        # Identity pass-through on the skip connections; a named module so it can later be
        # swapped for a learned transform without touching forward().
        self.featureForwarding = nn.Sequential()

        # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
        self.pool_output = tuple(pool_output)
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

        # Head
        x = self.avgpool0(x)
        x = torch.flatten(x,1)
        x = self.fc_hidden(x)
        mean = self.out_activation(self.fc_out_mean(x))

        self.saveItr += 1
        if self.predict_variance:
            return mean, self.fc_out_logvar(x)
        return mean

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

        rows.append(("adaptive avg pool", self.nFilt0, self.pool_output))
        return rows

    def summary(self, print_fn=print):
        """Print the network structure, the input dimensions, the data/feature-count
        progression through every layer, which activation runs at each stage, and which
        upsample_mode the decoder uses. `print_fn` lets output be redirected (e.g. to a
        logger); defaults to the builtin print.
        """
        rows = self.stage_shapes()
        w = max(len(label) for label, _, _ in rows)
        bar = "=" * (w + 40)

        # Pull the actual instantiated activation module for each of the four independently
        # configurable roles (stem_activation, block_activation, fc_activation,
        # output_activation), rather than re-deriving them from the constructor args, so
        # this always reflects what the model is really running. Same idea for the decoder's
        # upsample_mode - read straight off a real Upsample instance.
        stem_act = type(self.layer_stem.act).__name__
        block_act = type(self.encoder[0][0].act1).__name__  # same class at every block/site
        fc_act = type(self.fc_hidden[1]).__name__ if self.fc_widths else None
        out_act = type(self.out_activation).__name__
        upsample_mode = self.upsamples[0].mode  # same mode at every decoder stage

        print_fn(bar)
        print_fn(f"{type(self).__name__}  |  {self.n_stages} encoder stages, "
                 f"{self.n_stages - 1} decoder stages  |  kernel_stem={self.kernel_stem}  |  "
                 f"upsample_mode={upsample_mode!r}  |  predict_variance={self.predict_variance}")
        nS, nX, nY = self.input_shape
        print_fn(f"input:  (nBatch, nSpec={nS}, nX={nX}, nY={nY})  ->  "
                 f"output: (nBatch, {self.output_size})")
        print_fn(f"activations: stem={stem_act}  encoder/decoder blocks={block_act}  "
                 f"fc hidden={fc_act or 'n/a (no hidden FC layers)'}  output={out_act}")
        print_fn("-" * (w + 40))

        def row_note(label):
            # Only rows that actually apply an activation/resize get a note; concatenation/
            # pooling-only rows don't.
            if label.startswith("stem"):
                return f"activation={stem_act}"
            if label.endswith("deconv block") or label.startswith("encoder stage"):
                return f"activation={block_act}"
            if label.endswith(": upsample"):
                return f"mode={upsample_mode!r}"
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
