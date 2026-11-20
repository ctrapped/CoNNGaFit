"""3-d residual U-Net for HI 21cm spectral datacubes. Reads an (nSpec x nX x nY) cube of arbitrary size and predicts
one or more output maps. See NeuralNetwork for the full parameter list."""

import logging
import math

import numpy as np
import torch
from torch import nn

from conngafit.models.blocks import (
    ConvMapHead,
    DeconvBlock,
    HalfResConvHead,
    Stem,
    Upsample,
    _match_spatial,
    check_stage_compatibility,
)
from conngafit.models.diagnostics import save_summed_spectral_channels
from conngafit.models.unet import UNet3d

logger = logging.getLogger(__name__)


#### Full network ####
class NeuralNetwork(nn.Module):
    """3-d residual U-Net mapping an HI 21cm spectral datacube (nSpec x nX x nY, any size)
    to `n_maps` outputs (e.g. mass and rotational momentum), each of shape `output_shape`: a map, on the cube's
    grid by default, or with head='fc' a 1-d output such as a radial curve.

    Architecture:
      - Stem (blocks.Stem): stride-2 Conv3d + stride-2 MaxPool3d + activation, reducing
        every axis to exactly ceil(N/4).
      - U-Net (unet.UNet3d), with BatchNorm in every block:
        Encoder: `n_stages` stages of two ResidualBlocks each. Stage 0 keeps the stem's
        shape and channel count; every later stage brings each axis to ceil(N/2) and
        doubles the channel count (n_filt0 * 2**i at stage i).
        Decoder: `n_stages - 1` stages. Each upsamples ~2x per axis with blocks.Upsample (set by
        `upsample_mode`), realigns the result to the matching encoder skip map with _match_spatial,
        concatenates channel-wise, then a DeconvBlock pair halves the channel count.
        Attention: SelfAttention3d after the residual blocks of every stage in `attention_stages`, in the
        encoder and in the decoder stage at the same resolution.
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
    calls check_stage_compatibility, which raises a descriptive ValueError (naming the
    tightest axis) if `input_shape` cannot survive the stem plus n_stages-1 ceil-halvings.
    Every activation is configurable via the *_activation args.

    Parameters:
        input_shape: (nSpec, nX, nY) of one input cube (no batch/channel dim). forward()
            rejects a batch whose shape does not match.
        n_filt0: channel count of the first encoder stage; doubles at each later stage.
        kernel_stem: 3-tuple kernel size of the stem's strided convolution.
        kernel_coder: 3-tuple kernel size for every ResidualBlock/DeconvBlock convolution.
        n_stages: number of encoder stages (>= 2); the decoder has n_stages - 1. Validated
            against input_shape at construction (see check_stage_compatibility).
        n_fc: width of the fully-connected head's hidden layers for head='fc'. An int -> width to
            every hidden layer; list/tuple -> explicit per-layer widths. <= 0 or empty means no hidden layers - the pooled
            features go straight to the output Linear (with a dropout in between). Default 64.
        n_fc_layers: number of hidden fully-connected layers when n_fc is a scalar (ignored
            when n_fc is a sequence of widths).
        n_maps: number of maps predicted from the shared body (default 1). Each head's output
            layer predicts every map: n_maps output channels for the conv heads, or
            n_maps * prod(output_shape) outputs for the fc head.
        fc_pool_shape: (nSpec, nX, nY) grid the fc head averages the decoder's features down to, with
            nn.AdaptiveAvgPool3d. This fixes the fc layers' input size at n_filt0 * prod(fc_pool_shape).
        head_dropout_rate: dropout probability in the output head: after every hidden FC layer's activation
            (or once before the output Linear, with no hidden layers) for the fc head, and a channel-wise
            Dropout2d before the output conv for the conv heads. 0.0 disables. Active in training mode only.
        block_dropout_rate: probability for a channel-wise nn.Dropout3d (drops whole feature
            channels) inside every ResidualBlock/DeconvBlock. 0.0 disables.
        stem_activation: activation for the stem (default nn.LeakyReLU).
        block_activation: activation in every ResidualBlock/DeconvBlock (default nn.ReLU).
        head_activation: activation in the output head: between the fc head's hidden layers, in the conv
            head's 2-d residual blocks, and after each of conv_halfres's 3x3 convolutions. Default: nn.ReLU for
            head='conv', nn.LeakyReLU otherwise.
        output_activation: activation on the output Linear (default nn.Identity, i.e. none).
        upsample_mode: how each decoder stage upsamples (blocks.Upsample): 'conv_transpose' (default, a
            learned stride-2 ConvTranspose3d) or 'trilinear' (trilinear interpolation, then a 1x1x1 Conv3d).
        predict_variance: if False (default), the head has a single output Linear and
            forward() returns one tensor of shape (nBatch, n_maps, *output_shape). If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar). Train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE.
            With a conv head the log-variances are n_maps more output
            channels from the head's output conv instead of a second Linear.
        head: the output head that turns the decoder's features into maps (see Architecture above):
            'fc' (default) - pooled to fc_pool_shape, then the fully connected layers set by n_fc and
                n_fc_layers. Its output Linear has (n_fc, or n_filt0 * prod(fc_pool_shape) with no hidden
                layers) * n_maps * prod(output_shape) weights, so it grows with the map size: best for radial curves
                and small images.
            'conv' - ConvMapHead, with n_head_blocks 2-d residual blocks.
            'conv_halfres' - one more decoder stage to half resolution, then HalfResConvHead with head_channels.
            The conv heads' sizes don't depend on the map size, and they ignore fc_pool_shape, n_fc and
            n_fc_layers.
        output_shape: shape of each predicted output: (H, W) for maps (default: (nX, nY) from input),
            or (n_bins,) for 1-d outputs such as radial curves (head='fc' only). Stored as self.output_shape.
        n_head_blocks: number of 2-d residual blocks in head='conv' (>= 0; default 2).
        head_channels: channel count of head='conv_halfres''s two 3x3 convolutions (default 64).
        block_architecture: architecture of every ResidualBlock/DeconvBlock and of head='conv''s 2-d residual
            blocks, 'post_activation' (default) or 'identity_skip' (see blocks.ResidualBlock).
        attention_stages: encoder stage indices (0 to n_stages - 1) that get self-attention (default none).
            Its cost grows as the square of the stage's voxel count, so use coarse stages.
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_stem,
        kernel_coder,
        n_stages=4,
        n_fc=64,
        n_fc_layers=1,
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
        block_architecture="post_activation",
        attention_stages=(),
    ):
        super().__init__()
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

        # Each output's shape: an (H, W) map, on the cube's grid by default, or a 1-d output such as a radial curve
        if output_shape is None:
            output_shape = input_shape[1:]
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.output_shape) == 1 and head != "fc":
            raise ValueError(f"output_shape={self.output_shape}: the conv heads predict maps, not 1-d outputs")
        # fail fast if this input can't be halved n_stages-1 times after the stem
        check_stage_compatibility(input_shape, n_stages)
        logger.debug("output shape = %s", self.output_shape)
        self.n_fc = n_fc
        self.save_itr = 0  # forward() call counter; numbers the latent-image filenames

        # Stem: stride-2 conv + stride-2 maxpool + activation, reducing every axis by exactly ceil(N/4)
        self.layer_stem = Stem(n_filt0, kernel_stem, activation=stem_activation)

        self.n_filt0 = n_filt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.unet = UNet3d(
            n_filt0,
            n_stages,
            kernel_coder,
            dropout_rate=block_dropout_rate,
            activation=block_activation,
            architecture=block_architecture,
            upsample_mode=upsample_mode,
            attention_stages=attention_stages,
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
                    architecture=block_architecture,
                ),
                DeconvBlock(
                    n_filt0,
                    n_filt0,
                    upsample=False,
                    kernel1=kernel_coder,
                    dropout_rate=block_dropout_rate,
                    activation=block_activation,
                    architecture=block_architecture,
                ),
            )

        if head == "conv":
            # Head: spectral attention pool -> resize to output_shape -> 2-d res blocks -> conv.
            self.conv_head = ConvMapHead(
                n_filt0,
                out_channels=2 * n_maps if predict_variance else n_maps,
                n_blocks=n_head_blocks,
                activation=head_activation,
                dropout_rate=head_dropout_rate,
                architecture=block_architecture,
            )
        elif head == "conv_halfres":
            # Head: fold the half-resolution spectral axis into channels -> HalfResConvHead
            half_shape = tuple(math.ceil(n / 2) for n in input_shape)
            self.halfres_head = HalfResConvHead(
                n_filt0 * half_shape[0],
                n_filt0 * 2 ** (n_stages - 1),  # Bottleneck channels
                head_channels,
                dropout_rate=head_dropout_rate,
                activation=head_activation,
                predict_variance=predict_variance,
                n_maps=n_maps,
            )
        else:
            # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
            self.avgpool0 = nn.AdaptiveAvgPool3d(fc_pool_shape)
            flat_features = n_filt0 * int(np.prod(fc_pool_shape))

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
                fc_layers += [
                    nn.Linear(in_f, w),
                    head_activation(),
                    nn.Dropout(p=head_dropout_rate),
                ]
                in_f = w
            if not fc_widths:
                fc_layers.append(nn.Dropout(p=head_dropout_rate))
            self.fc_hidden = nn.Sequential(*fc_layers)
            self.fc_out_mean = nn.Linear(in_f, n_maps * math.prod(self.output_shape))
            if predict_variance:
                self.fc_out_logvar = nn.Linear(in_f, n_maps * math.prod(self.output_shape))
        self.out_activation = output_activation()

    def forward(self, x, save_latent_images=False):
        """Run the network on a batch of datacubes.

        x: Tensor of shape (nBatch, nSpec, nX, nY). (nSpec, nX, nY) must equal the model's input_shape.
        save_latent_images: if True, dump PNGs of the spectrally-summed activations at each
            encoder/decoder stage to LATENT_IMAGE_OUTPUT (see save_summed_spectral_channels in diagnostics.py)

        Returns the predicted maps (nBatch, n_maps, *output_shape) or 1-d outputs.
            If predict_variance=True, a (mean, logvar) tuple (exp(logvar) to get the variance)
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

        # U-Net: every encoder and decoder stage's output, in order
        x, stage_outputs = self.unet(x)
        if save_latent_images:
            for stage_output in stage_outputs:
                save_summed_spectral_channels(stage_output, step, self.save_itr)
                step += 1
        bottleneck = stage_outputs[self.n_stages - 1]

        if self.head_type == "conv_halfres":
            # Last decoder stage, to half resolution, then the per-pixel head
            x = self.upsample_half(x)
            skip = stem_half
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
        (nBatch, n_maps, *output_shape); logvar is None unless predict_variance."""
        if self.head_type == "conv":
            # One output channel per map's mean, followed (with predict_variance) by one per map's log-variance
            out = self.conv_head(x, self.output_shape)  # (nBatch, n_maps or 2 * n_maps, H, W)
            mean = self.out_activation(out[:, : self.n_maps])
            return mean, (out[:, self.n_maps :] if self.predict_variance else None)
        # fc head: pool to fc_pool_shape and flatten, then run the hidden layers, then separate output layers for the
        # means and log-variances
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        maps_shape = (self.n_maps, *self.output_shape)
        mean = self.out_activation(self.fc_out_mean(x).unflatten(1, maps_shape))
        logvar = self.fc_out_logvar(x).unflatten(1, maps_shape) if self.predict_variance else None
        return mean, logvar
