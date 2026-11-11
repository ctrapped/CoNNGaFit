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
    check_stage_compatibility,
)
from conngafit.models.diagnostics import save_summed_spectral_channels

logger = logging.getLogger(__name__)

# 3-d residual U-Net for HI 21cm spectral datacubes. Reads an (nSpec x nX x nY) cube of
# arbitrary size and predicts one or more flattened output maps. See NeuralNetwork for the full
# parameter list.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.
# Building blocks live in blocks.py; latent-image diagnostics in diagnostics.py.


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
    calls check_stage_compatibility, which raises a descriptive ValueError (naming the
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
            against input_shape at construction (see check_stage_compatibility).
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
        check_stage_compatibility(input_shape, n_stages)
        logger.debug("output size = %d", self.output_size)
        self.save_itr = 0  # forward() call counter; numbers the latent-image filenames

        # Stem: stride-2 conv + stride-2 maxpool + activation, reducing every axis by exactly ceil(N/4)
        self.layer_stem = Stem(n_filt0, kernel_stem, activation=stem_activation)

        # Encoder: stage 0 keeps the stem's shape/channels; each later stage -> ceil(N/2)
        # per axis and doubles the channel count.
        enc_channels = [n_filt0 * (2**i) for i in range(n_stages)]
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

        if head == "conv":
            # Head: spectral attention pool -> resize to output_shape -> 2-d res blocks -> conv.
            self.conv_head = ConvMapHead(
                n_filt0,
                out_channels=2 * n_maps if predict_variance else n_maps,
                n_blocks=n_head_blocks,
                activation=head_activation,
                dropout_rate=head_dropout_rate,
            )
        elif head == "conv_halfres":
            # Head: fold the half-resolution spectral axis into channels -> HalfResConvHead
            half_shape = tuple(math.ceil(n / 2) for n in input_shape)
            self.halfres_head = HalfResConvHead(
                n_filt0 * half_shape[0],
                enc_channels[-1],
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
