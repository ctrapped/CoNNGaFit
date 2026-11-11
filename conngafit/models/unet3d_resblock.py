import logging
import math

import numpy as np
import torch
from torch import nn

from conngafit.models.blocks import (
    ConvMapHead,
    DeconvBlock,
    HalfResConvHead,
    ResidualBlock,
    Stem,
    Upsample,
    _match_spatial,
)
from conngafit.models.diagnostics import count_params, layer_param_lines, save_summed_spectral_channels

logger = logging.getLogger(__name__)

# 3-d residual U-Net for HI 21cm spectral datacubes. Reads an (nSpec x nX x nY) cube of
# arbitrary size and predicts one or more flattened output maps. See NeuralNetwork for the full
# parameter list.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.
# Building blocks live in blocks.py; latent-image diagnostics in diagnostics.py.


#### Input-shape checks ####
def _stem_out_shape(input_shape):
    """(nSpec, nX, nY) shape after layer_stem: a stride-2 conv then a stride-2 maxpool, each
    exactly halving (ceil(N/2)) via _ceil_conv_pad, so every axis ends at exactly ceil(N/4) -
    true for any kernel size (ceiling division composes exactly: ceil(ceil(n/2)/2) = ceil(n/4))."""
    return tuple(math.ceil(n / 4) for n in input_shape)


def _check_stage_compatibility(input_shape, n_stages):
    """The encoder has n_stages stages: the first keeps its shape, each later one downsamples every axis to
    ceil(N/2). Return the (nSpec, nX, nY) shape after the stem, after checking that every axis can then be halved
    n_stages - 1 times. If not, raise a ValueError naming the smallest axis and the largest n_stages it allows."""
    stem_shape = _stem_out_shape(input_shape)
    n_halvings = n_stages - 1  # Encoder stage 0 keeps its input shape; each later stage halves every axis
    names = ("spectral", "x", "y")
    max_halvings = []
    for n in stem_shape:
        count = 0  # ceil-halvings this axis can take before reaching 1
        while n > 1:
            n = math.ceil(n / 2)
            count += 1
        max_halvings.append(count)
    i = int(np.argmin(max_halvings))  # the axis that allows the fewest halvings
    if max_halvings[i] < n_halvings:
        raise ValueError(
            f"n_stages={n_stages} needs {n_halvings} downsamplings after the stem, but the {names[i]} axis "
            f"(input {input_shape[i]}, {stem_shape[i]} after the stem) supports at most {max_halvings[i]}. "
            f"Use n_stages <= {max_halvings[i] + 1} or enlarge that axis."
        )
    return stem_shape


#### Full network ####
class NeuralNetwork(nn.Module):
    """3-d residual U-Net mapping an HI 21cm spectral datacube (nSpec x nX x nY, any size)
    to `n_maps` flattened output maps (e.g. mass and rotational momentum), each of length
    `output_size` (default nX*nY).

    Architecture:
      - Stem (blocks.Stem): stride-2 Conv3d + stride-2 MaxPool3d + activation, reducing
        every axis to exactly ceil(N/4).
      - Encoder: `n_stages` stages of two ResidualBlocks each. Stage 0 keeps the stem's
        shape and channel count; every later stage brings each axis to ceil(N/2) and
        doubles the channel count (n_filt0 * 2**i at stage i).
      - Decoder: `n_stages - 1` stages. Each upsamples ~2x per axis with blocks.Upsample (set by
        `upsample_mode`), realigns the result to the matching encoder skip map with _match_spatial,
        concatenates channel-wise, then a DeconvBlock pair halves the channel count.
      - Head, chosen by `head`:
        'fc' (default): nn.AdaptiveAvgPool3d collapses the surviving spatial shape to
        `fc_pool_shape`, so the flattened size is fixed regardless of input shape or depth; then
        `n_fc_layers` hidden Linear->activation->dropout blocks feed the output Linear and an
        optional output activation.
        'conv': ConvMapHead (see blocks.py) - spectral attention pool, bilinear resize to
        `output_shape`, 2-d residual blocks and a 3x3 output conv; its size does not grow with
        the input or map size.
        'conv_halfres': one more decoder stage up to the stem conv's half resolution (with a
        skip connection from it), then HalfResConvHead (see blocks.py) - spectral axis
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
        n_filt0: channel count of the first encoder stage; doubles at each later stage.
        kernel_stem: 3-tuple kernel size of the stem's strided convolution.
        kernel_coder: 3-tuple kernel size for every ResidualBlock/DeconvBlock convolution.
        n_fc: width of the fully-connected head's hidden layers. An int applies that width to
            every hidden layer; a list/tuple gives explicit per-layer widths (its length
            then overrides n_fc_layers). <= 0 or empty means no hidden layers - the pooled
            features go straight to the output Linear (with a dropout in between).
        n_stages: number of encoder stages (>= 2); the decoder has n_stages - 1. Validated
            against input_shape at construction (see _check_stage_compatibility).
        n_fc_layers: number of hidden fully-connected layers when n_fc is a scalar (ignored
            when n_fc is a sequence of widths).
        output_size: length of each flattened map. Defaults to nX*nY from input_shape
            (or H*W from output_shape); must equal H*W for the conv heads.
        n_maps: number of maps predicted from the shared body (default 1). Each head's output
            layer predicts every map: n_maps output channels for the conv heads, or
            n_maps * output_size outputs for the fc head.
        fc_pool_shape: (nSpec, nX, nY) grid the fc head averages the decoder's features down to, with
            nn.AdaptiveAvgPool3d. This fixes the fc layers' input size at n_filt0 * prod(fc_pool_shape).
        head_dropout_rate: dropout probability in the output head: after every hidden FC layer's activation
            (or once before the output Linear, with no hidden layers) for the fc head, and a channel-wise
            Dropout2d before the output conv for the conv heads. 0.0 disables. Active in training mode only.
        block_dropout_rate: probability for a channel-wise nn.Dropout3d (drops whole feature
            channels) inside every ResidualBlock/DeconvBlock, after the residual add + final
            activation. 0.0 disables. Enable it only if head_dropout_rate alone leaves a
            train/validation gap, and start low (~0.1-0.15).
        stem_activation: activation for the stem (default nn.LeakyReLU).
        block_activation: activation at all three sites in every ResidualBlock/DeconvBlock
            (post-conv1, post-conv2, post residual-add; default nn.ReLU).
        head_activation: activation in the output head: between the fc head's hidden layers, in the conv
            head's 2-d residual blocks, and after each of conv_halfres's 3x3 convolutions. Default: nn.ReLU for
            head='conv', nn.LeakyReLU otherwise.
        output_activation: activation on the output Linear (default nn.Identity, i.e. none).
        upsample_mode: how each decoder stage upsamples (blocks.Upsample): 'conv_transpose' (default, a
            learned stride-2 ConvTranspose3d) or 'trilinear' (trilinear interpolation, then a 1x1x1 Conv3d).
        predict_variance: if False (default), the head has a single output Linear and
            forward() returns one tensor of shape (nBatch, n_maps, output_size). If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar). Train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE.
            With a conv head the log-variances are n_maps more output
            channels from the head's output conv instead of a second Linear.
        head: the output head that turns the decoder's features into maps (see Architecture above):
            'fc' (default) - pooled to fc_pool_shape, then the fully connected layers set by n_fc and
                n_fc_layers. Its output Linear has (n_fc, or n_filt0 * prod(fc_pool_shape) with no hidden
                layers) * n_maps * output_size weights, so it grows with the map size: best for radial curves
                and small images.
            'conv' - ConvMapHead, with n_head_blocks 2-d residual blocks.
            'conv_halfres' - one more decoder stage to half resolution, then HalfResConvHead with head_channels.
            The conv heads' sizes don't depend on the map size, and they ignore fc_pool_shape, n_fc and
            n_fc_layers.
        output_shape: (H, W) of the predicted map. Required grid for head='conv' (default
            (nX, nY)) and head='conv_halfres'; for head='fc' it only sets the default output_size = H*W.
        n_head_blocks: number of 2-d residual blocks in head='conv' (>= 0; default 2).
        head_channels: channel count of head='conv_halfres''s two 3x3 convolutions (default 64).
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_stem,
        kernel_coder,
        n_fc,
        n_stages=4,
        n_fc_layers=1,
        output_size=None,
        fc_pool_shape=(8, 8, 8),
        head_dropout_rate=0.0,
        block_dropout_rate=0.0,
        stem_activation=nn.LeakyReLU,
        block_activation=nn.ReLU,
        head_activation=None,
        output_activation=nn.Identity,
        upsample_mode="conv_transpose",
        predict_variance=False,
        head="fc",
        output_shape=None,
        n_head_blocks=2,
        head_channels=64,
        n_maps=1,
    ):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        self.n_maps = n_maps
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 (need at least one skip connection)")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        self.n_stages = n_stages
        if head not in ("fc", "conv", "conv_halfres"):
            raise ValueError(f"unknown head {head!r}; use 'fc', 'conv' or 'conv_halfres'")
        self.head_type = head
        if head_activation is None:  # ReLU for the conv head's residual blocks, LeakyReLU otherwise
            head_activation = nn.ReLU if head == "conv" else nn.LeakyReLU
        # Map grid of the prediction: required by the conv head, optional for the fc head.
        self.output_shape = (
            tuple(int(v) for v in output_shape)
            if output_shape is not None
            else (input_shape[1:] if head != "fc" else None)
        )
        map_size = int(np.prod(self.output_shape)) if self.output_shape is not None else input_shape[1] * input_shape[2]
        if head != "fc" and output_size is not None and int(output_size) != map_size:
            raise ValueError(
                f"output_size={output_size} does not match output_shape {self.output_shape} for head={head!r}"
            )
        self.output_size = int(output_size) if output_size is not None else map_size
        # fail fast if this input can't be halved n_stages-1 times after the stem
        self.stem_out_shape = _check_stage_compatibility(input_shape, n_stages)
        logger.debug("output size = %d", self.output_size)
        self.n_fc = n_fc
        self.save_itr = 0  # forward() call counter; numbers the latent-image filenames

        # Stem: stride-2 conv + stride-2 maxpool + activation, reducing every axis by exactly ceil(N/4)
        self.layer_stem = Stem(n_filt0, kernel_stem, activation=stem_activation)

        # Encoder: stage 0 keeps the stem's shape/channels; each later stage -> ceil(N/2)
        # per axis and doubles the channel count.
        enc_channels = [n_filt0 * (2**i) for i in range(n_stages)]
        self.n_filt0 = n_filt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.enc_channels = list(enc_channels)
        self.encoder = nn.ModuleList()
        prev = n_filt0
        for i, ch in enumerate(enc_channels):
            downsample = i > 0
            self.encoder.append(
                nn.Sequential(
                    ResidualBlock(
                        prev,
                        ch,
                        downsample=downsample,
                        kernel1=kernel_coder,
                        dropout_rate=block_dropout_rate,
                        activation=block_activation,
                    ),
                    ResidualBlock(
                        ch,
                        ch,
                        downsample=False,
                        kernel1=kernel_coder,
                        dropout_rate=block_dropout_rate,
                        activation=block_activation,
                    ),
                )
            )
            prev = ch

        # Decoder: stage j upsamples the running tensor, concatenates the skip map from
        # encoder stage enc_idx (same channel count), then a DeconvBlock pair reduces 2C -> C.
        self.upsamples = nn.ModuleList()
        self.dc_layers = nn.ModuleList()
        for j in range(n_stages - 1):
            enc_idx = n_stages - 2 - j
            in_ch = enc_channels[n_stages - 1 - j]
            out_ch = enc_channels[enc_idx]
            self.upsamples.append(Upsample(in_ch, out_ch, kernel1=kernel_coder, mode=upsample_mode))
            self.dc_layers.append(
                nn.Sequential(
                    DeconvBlock(
                        2 * out_ch,
                        out_ch,
                        upsample=True,
                        kernel1=kernel_coder,
                        dropout_rate=block_dropout_rate,
                        activation=block_activation,
                    ),
                    DeconvBlock(
                        out_ch,
                        out_ch,
                        upsample=False,
                        kernel1=kernel_coder,
                        dropout_rate=block_dropout_rate,
                        activation=block_activation,
                    ),
                )
            )

        if head == "conv_halfres":
            # One more decoder stage, up to the stem conv's half resolution, with a skip
            # connection from it (see Stem.forward's return_half)
            self.upsample_half = Upsample(n_filt0, n_filt0, kernel1=kernel_coder, mode=upsample_mode)
            self.dc_layer_half = nn.Sequential(
                DeconvBlock(
                    2 * n_filt0,
                    n_filt0,
                    upsample=True,
                    kernel1=kernel_coder,
                    dropout_rate=block_dropout_rate,
                    activation=block_activation,
                ),
                DeconvBlock(
                    n_filt0,
                    n_filt0,
                    upsample=False,
                    kernel1=kernel_coder,
                    dropout_rate=block_dropout_rate,
                    activation=block_activation,
                ),
            )

        # Identity pass-through on the skip connections; a named module so it can later be
        # swapped for a learned transform without touching forward().
        self.feature_forwarding = nn.Sequential()

        self.fc_pool_shape = tuple(fc_pool_shape)
        if head == "conv":
            # Head: spectral attention pool -> resize to output_shape -> 2-d res blocks -> conv.
            self.fc_widths = []
            self.conv_head = ConvMapHead(
                n_filt0,
                out_channels=2 * n_maps if predict_variance else n_maps,
                n_blocks=n_head_blocks,
                activation=head_activation,
                dropout_rate=head_dropout_rate,
            )
        elif head == "conv_halfres":
            # Head: fold the half-resolution spectral axis into channels -> HalfResConvHead
            self.half_shape = tuple(math.ceil(n / 2) for n in input_shape)
            self.fc_widths = []
            self.head_channels = head_channels
            self.halfres_head = HalfResConvHead(
                n_filt0 * self.half_shape[0],
                enc_channels[-1],
                head_channels,
                dropout_rate=head_dropout_rate,
                activation=head_activation,
                predict_variance=predict_variance,
                n_maps=n_maps,
            )
        else:
            # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
            self.avgpool0 = nn.AdaptiveAvgPool3d(self.fc_pool_shape)
            flat_features = n_filt0 * int(np.prod(self.fc_pool_shape))

            # Resolve the hidden-layer widths: a scalar n_fc repeated n_fc_layers times, or an
            # explicit sequence of widths. Empty -> no hidden layers (a dropout still precedes
            # the output Linear).
            if isinstance(n_fc, (list, tuple)):
                fc_widths = [int(w) for w in n_fc if int(w) > 0]
            elif n_fc and n_fc > 0:
                fc_widths = [int(n_fc)] * int(n_fc_layers)
            else:
                fc_widths = []
            self.fc_widths = fc_widths

            fc_layers, in_f = [], flat_features
            for w in fc_widths:
                fc_layers += [nn.Linear(in_f, w), head_activation(), nn.Dropout(p=head_dropout_rate)]
                in_f = w
            if not fc_widths:
                fc_layers.append(nn.Dropout(p=head_dropout_rate))
            self.fc_hidden = nn.Sequential(*fc_layers)
            self.fc_out_mean = nn.Linear(in_f, n_maps * self.output_size)
            if predict_variance:
                self.fc_out_logvar = nn.Linear(in_f, n_maps * self.output_size)
        self.out_activation = output_activation()

    def forward(self, x, save_latent_images=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; (nSpec, nX, nY) must equal the model's input_shape. A channel dim of
            size 1 is inserted before the 3-d convolutions.
        save_latent_images: if True, dump PNGs of the spectrally-summed activations at each
            encoder/decoder stage to LATENT_IMAGE_OUTPUT (see save_summed_spectral_channels in diagnostics.py) -
            useful for visually debugging what the network is learning, not needed for normal
            training/inference.
        Returns: if predict_variance=False (default), a tensor of shape (nBatch, n_maps, output_size)
            - the flattened predicted output maps. If predict_variance=True, a (mean, logvar)
            tuple of two such tensors - logvar is an unconstrained per-element log-variance
            (exp(logvar) to get the variance), with no output_activation applied.
        """
        n_batch, n_spec, n_x, n_y = x.size()
        if (n_spec, n_x, n_y) != self.input_shape:
            raise ValueError(
                f"input spatial/spectral shape {(n_spec, n_x, n_y)} does not match the "
                f"model's input_shape {self.input_shape}"
            )
        x = torch.reshape(x, (n_batch, 1, n_spec, n_x, n_y))

        # Stem
        if self.head_type == "conv_halfres":
            x, stem_half = self.layer_stem(x, return_half=True)
        else:
            x = self.layer_stem(x)
        step = 0
        if save_latent_images:
            save_summed_spectral_channels(x, step, self.save_itr)
            step += 1

        # Encoder: run every stage, keep all but the last as skip connections
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            if save_latent_images:
                save_summed_spectral_channels(x, step, self.save_itr)
                step += 1
            if i < len(self.encoder) - 1:
                skips.append(self.feature_forwarding(x))
        bottleneck = x

        # Decoder: upsample, align to the matching skip map, concatenate, reduce channels
        for up, dc in zip(self.upsamples, self.dc_layers):
            x = up(x)
            skip = skips.pop()
            x = _match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dc(x)
            if save_latent_images:
                save_summed_spectral_channels(x, step, self.save_itr)
                step += 1

        if self.head_type == "conv_halfres":
            # Last decoder stage, to half resolution, then the per-pixel head
            x = self.upsample_half(x)
            skip = self.feature_forwarding(stem_half)
            x = _match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.dc_layer_half(x)
            if save_latent_images:
                save_summed_spectral_channels(x, step, self.save_itr)
                step += 1
            outputs = self.halfres_head(x, bottleneck, self.output_shape)
            self.save_itr += 1
            mean = self.out_activation(outputs[0])
            if self.predict_variance:
                return mean, outputs[1]
            return mean

        # Head
        mean, logvar = self._run_head(x)
        self.save_itr += 1
        if self.predict_variance:
            return mean, logvar
        return mean

    def _run_head(self, x):
        """Run the 'fc' or 'conv' head on the decoder's output x. Returns (mean, logvar), each of shape
        (nBatch, n_maps, output_size); logvar is None unless predict_variance."""
        if self.head_type == "conv":
            # One output channel per map's mean, followed (with predict_variance) by one per map's log-variance
            out = self.conv_head(x, self.output_shape).flatten(2)  # (nBatch, n_maps or 2 * n_maps, H*W)
            mean = self.out_activation(out[:, : self.n_maps])
            return mean, (out[:, self.n_maps :] if self.predict_variance else None)
        # fc head: pool to fc_pool_shape and flatten, then run the hidden layers, then separate output layers for the
        # means and log-variances
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        map_shape = (self.n_maps, self.output_size)
        mean = self.out_activation(self.fc_out_mean(x).unflatten(1, map_shape))
        logvar = self.fc_out_logvar(x).unflatten(1, map_shape) if self.predict_variance else None
        return mean, logvar

    def _enable_mc_dropout(self):
        """Switch only the dropout layers back to training mode, so they keep dropping random channels. Every other
        layer stays in eval mode: BatchNorm keeps using the statistics stored during training, rather than
        normalizing by the statistics of the batch it is given."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_mc_dropout(self, x, n_samples=30, save_latent_images=False):
        """Estimate how uncertain the trained network is about x with Monte Carlo dropout: run it n_samples times
        with dropout left on, so each pass drops different channels, and take the mean and spread of the results.

        Returns (mean, epistemic_std), each (nBatch, n_maps, output_size). With predict_variance, returns
        (mean, aleatoric_std, epistemic_std, total_std): aleatoric_std is the network's predicted 1-sigma
        averaged over the passes, and total_std = sqrt(aleatoric_std**2 + epistemic_std**2).
        Leaves the network in eval mode.
        """
        self._enable_mc_dropout()
        with torch.no_grad():
            if self.predict_variance:
                means, logvars = zip(*(self.forward(x, save_latent_images) for _ in range(n_samples)))
                means = torch.stack(means, dim=0)
                logvars = torch.stack(logvars, dim=0)
            else:
                preds = torch.stack([self.forward(x, save_latent_images) for _ in range(n_samples)], dim=0)
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
        rows.append(("stem (conv + maxpool)", self.n_filt0, stem))

        enc_shapes, shape = [], stem
        for i, ch in enumerate(self.enc_channels):
            if i > 0:
                shape = tuple(math.ceil(s / 2) for s in shape)  # downsample
            enc_shapes.append(shape)
            tag = f"encoder stage {i}" + (" (no downsample)" if i == 0 else "")
            rows.append((tag, ch, shape))

        shape = enc_shapes[-1]
        for j in range(self.n_stages - 1):
            enc_idx = self.n_stages - 2 - j
            out_ch = self.enc_channels[enc_idx]
            up_mode = self.upsamples[j].mode
            up_shape = tuple((2 * s - 1 if up_mode == "conv_transpose" else 2 * s) for s in shape)
            rows.append((f"decoder {j}: upsample", out_ch, up_shape))
            skip = enc_shapes[enc_idx]
            rows.append((f"decoder {j}: + skip(enc {enc_idx})", 2 * out_ch, skip))
            rows.append((f"decoder {j}: deconv block", out_ch, skip))
            shape = skip

        if self.head_type == "conv":
            rows += self.conv_head.stage_shapes(shape[1:], self.output_shape)
        elif self.head_type == "conv_halfres":
            up_mode = self.upsample_half.mode
            up_shape = tuple((2 * s - 1 if up_mode == "conv_transpose" else 2 * s) for s in shape)
            rows.append(("half-res decoder: upsample", self.n_filt0, up_shape))
            rows.append(("half-res decoder: + skip(stem conv)", 2 * self.n_filt0, self.half_shape))
            rows.append(("half-res decoder: deconv block", self.n_filt0, self.half_shape))
            folded = self.n_filt0 * self.half_shape[0] + self.enc_channels[-1]
            rows.append(("conv head: fold spectra + global context", folded, self.half_shape[1:]))
            rows.append(("conv head: 2x 3x3 conv", self.head_channels, self.half_shape[1:]))
            rows.append(("conv head: 1x1 conv + resize", self.n_maps, self.output_shape))
        else:
            rows.append(("adaptive avg pool", self.n_filt0, self.fc_pool_shape))
        return rows

    def _param_groups(self):
        """(label, [modules]) for every summary() row, in order: the stage_shapes() rows, then (fc
        head) the fully-connected layers. Rows with no parameters (input, skip concat, pooling,
        resize) get []."""
        mods = [[], [self.layer_stem]] + [[stage] for stage in self.encoder]
        for up, dc in zip(self.upsamples, self.dc_layers):
            mods += [[up], [], [dc]]  # upsample, + skip (concat), deconv block
        if self.head_type == "conv":
            head = self.conv_head.param_groups()
            head[-1] = (head[-1][0], head[-1][1] + [self.out_activation])  # output conv + activation
            mods += [m for _, m in head]
            return [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        if self.head_type == "conv_halfres":
            h = self.halfres_head
            out = [h.dropout, h.out_mean] + ([h.out_logvar] if h.out_logvar is not None else [])
            mods += [
                [self.upsample_half],
                [],
                [self.dc_layer_half],  # half-res decoder stage
                [],
                [h.conv1, h.act1, h.conv2, h.act2],
                out + [self.out_activation],
            ]
            return [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        mods.append([])  # adaptive avg pool
        groups = [(label, m) for (label, _, _), m in zip(self.stage_shapes(), mods)]
        # fc_hidden is [Linear, activation, Dropout] per hidden layer (see __init__)
        groups += [
            (f"fc hidden {k}", [self.fc_hidden[3 * k], self.fc_hidden[3 * k + 1]]) for k in range(len(self.fc_widths))
        ]
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
        # configurable roles (stem_activation, block_activation, head_activation,
        # output_activation), rather than re-deriving them from the constructor args, so
        # this always reflects what the model is really running. Same idea for the decoder's
        # upsample_mode - read straight off a real Upsample instance.
        stem_act = type(self.layer_stem.act).__name__
        block_act = type(self.encoder[0][0].act1).__name__  # same class at every block/site
        if self.head_type == "conv_halfres":
            fc_act = type(self.halfres_head.act1).__name__
        else:
            fc_act = type(self.fc_hidden[1]).__name__ if self.fc_widths else None
        out_act = type(self.out_activation).__name__
        if self.head_type == "conv":
            head_act = (
                "head blocks=" + type(self.conv_head.blocks[0].act1).__name__
                if len(self.conv_head.blocks)
                else "head blocks=n/a (none)"
            )
        elif self.head_type == "conv_halfres":
            head_act = f"conv head={fc_act}"
        else:
            head_act = f"fc hidden={fc_act or 'n/a (no hidden FC layers)'}"
        upsample_mode = self.upsamples[0].mode  # same mode at every decoder stage

        print_fn(bar)
        print_fn(
            f"{type(self).__name__}  |  {self.n_stages} encoder stages, "
            f"{self.n_stages - 1} decoder stages  |  kernel_stem={self.kernel_stem}  |  "
            f"kernel_coder={self.kernel_coder}  |  upsample_mode={upsample_mode!r}  |  "
            f"predict_variance={self.predict_variance}  |  head={self.head_type!r}  |  n_maps={self.n_maps}"
        )
        n_spec, n_x, n_y = self.input_shape
        output = f"(nBatch, {self.n_maps}, {self.output_size})"
        print_fn(f"input:  (nBatch, nSpec={n_spec}, nX={n_x}, nY={n_y})  ->  output: {output}")
        print_fn(f"activations: stem={stem_act}  encoder/decoder blocks={block_act}  {head_act}  output={out_act}")
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
            if self.head_type == "conv":
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

        if self.head_type == "fc":
            flat = self.n_filt0 * int(np.prod(self.fc_pool_shape))
            print_fn(f"  {'flatten':<{w}}   features={flat}")
            in_f = flat
            for k, out_f in enumerate(self.fc_widths):
                label = f"fc hidden {k}"
                print_fn(f"  {label:<{w}}   {in_f} -> {out_f}  params={params_col(label)}  activation={fc_act}")
                in_f = out_f
            print_fn(
                f"  {'fc out (mean)':<{w}}   {in_f} -> {self.n_maps} x {self.output_size}  "
                f"params={params_col('fc out (mean)')}  activation={out_act}"
            )
            if self.predict_variance:
                print_fn(
                    f"  {'fc out (logvar)':<{w}}   {in_f} -> {self.n_maps} x {self.output_size}  "
                    f"params={params_col('fc out (logvar)')}  activation=None"
                )
        else:
            print_fn(f"  {'output (mean)':<{w}}   map -> {output}  activation={out_act}")
            if self.predict_variance:
                print_fn(f"  {'output (logvar)':<{w}}   map -> {output}  activation=None")
        print_fn("-" * (w + 58))
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn(f"parameters: {n_all:,} total ({n_train:,} trainable)")
        print_fn(bar)
        print_fn(str(self))
        print_fn(bar)
