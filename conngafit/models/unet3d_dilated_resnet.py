import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from conngafit.models.blocks import ConvMapHead, DilatedResidualBlock, _ceil_conv_pad
from conngafit.models.diagnostics import save_summed_spectral_channels

logger = logging.getLogger(__name__)

# 3-d dilated residual network ("Encode-Process-Decode") for HI 21cm spectral datacubes following
# architecture in Stachenfeld et al. 2021, (https://arxiv.org/abs/2112.15275).
# Reads an (nSpec x nX x nY) cube of arbitrary size and predicts one or more output maps,
# matching the input/output conventions of unet3d_resblock.py. There is no spatial
# downsampling/upsampling path or skip connections; the receptive field is grown entirely
# through dilated convolutions. See NeuralNetwork for the full parameter list.


#### Encode: project the raw input cube into feature space ####
class Encoder(nn.Module):
    """A single Conv3d projecting the 1-channel input cube to out_channels, at full resolution
    unless `stride` downsamples it. Default stride (1,1,1) keeps the whole network at full
    resolution, since the receptive field is grown via dilation rather than pooling; a larger
    stride is a practical escape hatch for very large input cubes.

    Parameters:
        out_channels: number of feature channels (== n_filt0, constant through the whole net).
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
            self.conv = nn.Conv3d(1, out_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding="same")
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding="same")
        self.act = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D, H, W)."""
        return self.act(self.conv(x))


#### Full network ####
class NeuralNetwork(nn.Module):
    """3-d dilated residual network ("Encode-Process-Decode") mapping a spectral
    datacube (nSpec x nX x nY, any size) to `n_maps` outputs (e.g. mass and rotational momentum), each of shape
    `output_shape`: a map, on the cube's grid by default, or with head='fc' a 1-d output such as a radial curve.

    Architecture:
      - Encode (layer_encode): a single Conv3d projecting the 1-channel input to n_filt0
        channels (see Encoder). Stride (1,1,1) by default, so the whole network runs at full
        input resolution - there is no downsampling/upsampling path and therefore no
        encoder/decoder skip connections; the receptive field instead grows via dilation.
      - Process: `n_blocks` DilatedResidualBlock instances applied in sequence, each cycling
        through `dilation_rates` (default (1,2,4,8,4,2,1)) while holding the channel count and
        spatial/spectral shape fixed at n_filt0 and the encoder's output shape.
      - Decode (layer_decode): a single Conv3d (see Decoder) projecting back to n_filt0
        channels, feeding the regression head.
      - Head, chosen by `head` (both options identical to unet3d_resblock.py's):
        'fc' (default): nn.AdaptiveAvgPool3d collapses the surviving spatial shape to
        `fc_pool_shape`, so the flattened size is fixed regardless of input shape or depth; then
        `n_fc_layers` hidden Linear->activation->dropout blocks feed the output Linear and an
        optional output activation.
        'conv': ConvMapHead (see blocks.py) - spectral attention pool, bilinear resize to
        `output_shape`, 2-d residual blocks and a 3x3 output conv; its size does not grow with
        the input or map size.

    Because every stage preserves shape exactly (or, for the encoder, downsamples via a single
    ceil(N/stride) conv) and nn.AdaptiveAvgPool3d accepts any input size, there is no minimum-
    input-size / stage-count compatibility constraint to check at construction time.

    Parameters:
        input_shape: (nSpec, nX, nY) of one input cube (no batch/channel dim). forward()
            rejects a batch whose shape does not match.
        n_filt0: feature channel count, held constant through the encoder, every Process
            block, and the decoder (no channel doubling - there are no downsampling stages).
        kernel_stem: 3-tuple kernel size for the Encoder and Decoder convolutions.
        kernel_coder: 3-tuple kernel size for every dilated convolution in the Process stack.
        n_fc: width of the fully-connected head's hidden layers. An int applies that width to
            every hidden layer; a list/tuple gives explicit per-layer widths (its length
            then overrides n_fc_layers). <= 0 or empty means no hidden layers - the pooled
            features go straight to the output Linear (with a dropout in between). Default 64; only the fc
            head uses it.
        n_blocks: number of DilatedResidualBlock instances in the Process stack. With the
            default 7-rate dilation schedule, n_blocks=4 gives 28 dilated conv layers.
        dilation_rates: per-layer dilation schedule cycled by every DilatedResidualBlock
            (default (1, 2, 4, 8, 4, 2, 1)).
        stem_stride: 3-tuple stride for the Encoder conv. (1,1,1) (default) means no
            downsampling; a larger stride (e.g. (2,2,2)) trades resolution for compute on
            large input cubes, bringing every axis to ceil(N/stride).
        n_fc_layers: number of hidden fully-connected layers when n_fc is a scalar (ignored
            when n_fc is a sequence of widths).
        n_maps: number of maps predicted from the shared body (default 1). Each head's output
            layer predicts every map: n_maps output channels for head='conv', or
            n_maps * prod(output_shape) outputs for head='fc'.
        fc_pool_shape: 3-tuple target grid for nn.AdaptiveAvgPool3d. The fc input size is
            n_filt0 * prod(fc_pool_shape); the default (8, 8, 8) gives 512 * n_filt0.
        head_dropout_rate: dropout probability in the output head: after every hidden FC layer's activation
            (or once before the output Linear, with no hidden layers) for head='fc', and a channel-wise
            Dropout2d before the output conv for head='conv'. 0.0 disables. Active in training mode only.
        block_dropout_rate: probability for a channel-wise nn.Dropout3d (drops whole feature
            channels) inside every DilatedResidualBlock, after the residual add + final
            activation. 0.0 disables.
        encode_activation: activation after the Encoder conv (default None, i.e. no
            activation).
        block_activation: activation used inside every DilatedResidualBlock, after every conv
            but the chain's last, and once more after the residual add (default nn.ReLU).
        decode_activation: activation after the Decoder conv (default None, i.e. no
            activation).
        head_activation: activation in the output head: between the fc head's hidden layers, or in the conv
            head's 2-d residual blocks. Default: nn.ReLU for head='conv', nn.LeakyReLU for head='fc'.
        output_activation: activation on the output Linear (default nn.Identity, i.e. none).
        predict_variance: if False (default), the head has a single output Linear and
            forward() returns one tensor of shape (nBatch, n_maps, *output_shape). If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar) - train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE. logvar has
            no output_activation applied (it must stay unconstrained for exp() to reach any
            positive variance). With head='conv' the log-variances are n_maps more output
            channels of the head's final conv instead of a second Linear.
        head: 'fc' (default) - adaptive average pool -> flatten -> fully-connected layers, as
            set by fc_pool_shape/n_fc/n_fc_layers; its output Linear has
            n_filt0 * prod(fc_pool_shape) * n_maps * prod(output_shape) weights, so it grows with the map
            size and number of maps.
            'conv' - ConvMapHead: spectral attention pool -> bilinear resize to output_shape ->
            n_head_blocks 2-d residual blocks -> 3x3 conv, with a parameter count independent of
            the input and map size (fc_pool_shape, n_fc and n_fc_layers are then unused).
        output_shape: shape of each predicted output: (H, W) for maps (default: (nX, nY) from input),
            or (n_bins,) for 1-d outputs such as radial curves (head='fc' only). Stored as self.output_shape.
        n_head_blocks: number of 2-d residual blocks in the conv head (>= 0; default 2).

        Each *_activation is a zero-arg callable returning an nn.Module - the activation
        class itself (nn.ReLU, nn.GELU, nn.Tanh, ...) or a lambda for non-default arguments
        (e.g. lambda: nn.LeakyReLU(0.2)). It is called once per site, so every layer gets
        its own module. encode_activation/decode_activation may also be None for no activation.
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_stem,
        kernel_coder,
        n_fc=64,
        n_blocks=4,
        dilation_rates=(1, 2, 4, 8, 4, 2, 1),
        stem_stride=(1, 1, 1),
        n_fc_layers=1,
        fc_pool_shape=(8, 8, 8),
        head_dropout_rate=0.0,
        block_dropout_rate=0.0,
        encode_activation=None,
        block_activation=nn.ReLU,
        decode_activation=None,
        head_activation=None,
        output_activation=nn.Identity,
        predict_variance=False,
        head="fc",
        output_shape=None,
        n_head_blocks=2,
        n_maps=1,
    ):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        self.n_maps = n_maps
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        if head not in ("fc", "conv"):
            raise ValueError(f"unknown head {head!r}; use 'fc' or 'conv'")
        self.head_type = head
        if head_activation is None:  # ReLU for the conv head's residual blocks, LeakyReLU otherwise
            head_activation = nn.ReLU if head == "conv" else nn.LeakyReLU
        self.head_activation = head_activation
        # Each output's shape: an (H, W) map, on the cube's grid by default, or a 1-d output such as a radial curve
        if output_shape is None:
            output_shape = input_shape[1:]
        self.output_shape = tuple(int(v) for v in output_shape)
        if len(self.output_shape) == 1 and head != "fc":
            raise ValueError(f"output_shape={self.output_shape}: the conv head predicts maps, not 1-d outputs")
        logger.debug("output shape = %s", self.output_shape)
        self.n_fc = n_fc
        self.save_itr = 0  # forward() call counter; numbers the latent-image filenames

        self.n_filt0 = n_filt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.stem_stride = tuple(stem_stride)
        self.n_blocks = n_blocks
        self.dilation_rates = tuple(dilation_rates)

        # Encode: single conv, 1 -> n_filt0 channels, ceil(N/stem_stride) per axis
        self.layer_encode = Encoder(n_filt0, kernel_stem, stride=self.stem_stride, activation=encode_activation)

        # Process: n_blocks dilated residual blocks, shape and channel count fixed at n_filt0
        self.process = nn.ModuleList(
            [
                DilatedResidualBlock(
                    n_filt0,
                    kernel_coder,
                    dilation_rates=self.dilation_rates,
                    dropout_rate=block_dropout_rate,
                    activation=block_activation,
                )
                for _ in range(n_blocks)
            ]
        )

        # Decode: single conv, n_filt0 -> n_filt0 channels, shape unchanged
        self.layer_decode = Decoder(n_filt0, n_filt0, kernel_stem, activation=decode_activation)

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
        else:
            # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
            # Identical to unet3d_resblock.py's head.
            self.avgpool0 = nn.AdaptiveAvgPool3d(self.fc_pool_shape)
            flat_features = n_filt0 * int(np.prod(self.fc_pool_shape))

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
            self.fc_out_mean = nn.Linear(in_f, n_maps * math.prod(self.output_shape))
            if predict_variance:
                self.fc_out_logvar = nn.Linear(in_f, n_maps * math.prod(self.output_shape))
        self.out_activation = output_activation()

    def forward(self, x, save_latent_images=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; (nSpec, nX, nY) must equal the model's input_shape. A channel dim of
            size 1 is inserted before the 3-d convolutions.
        save_latent_images: if True, dump PNGs of the spectrally-summed activations after the
            encoder and after each Process block to LATENT_IMAGE_OUTPUT (see
            save_summed_spectral_channels in diagnostics.py) - useful for visually debugging what the network
            is learning, not needed for normal training/inference.
        Returns: if predict_variance=False (default), a tensor of shape (nBatch, n_maps, *output_shape)
            - the predicted maps (or 1-d outputs). If predict_variance=True, a (mean, logvar)
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

        # Encode
        x = self.layer_encode(x)
        step = 0
        if save_latent_images:
            save_summed_spectral_channels(x, step, self.save_itr)
            step += 1

        # Process: sequence of dilated residual blocks, shape/channels fixed throughout
        for block in self.process:
            x = block(x)
            if save_latent_images:
                save_summed_spectral_channels(x, step, self.save_itr)
                step += 1

        # Decode
        x = self.layer_decode(x)
        if save_latent_images:
            save_summed_spectral_channels(x, step, self.save_itr)
            step += 1

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
        # fc head: pool to fc_pool_shape and flatten, then the hidden layers, then separate output layers for the
        # means and log-variances
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        maps_shape = (self.n_maps, *self.output_shape)
        mean = self.out_activation(self.fc_out_mean(x).unflatten(1, maps_shape))
        logvar = self.fc_out_logvar(x).unflatten(1, maps_shape) if self.predict_variance else None
        return mean, logvar

    def _enable_mc_dropout(self):
        """Switch only the dropout layers back to training mode, so they keep dropping random channels. Every other
        layer stays in eval mode: BatchNorm keeps using the statistics stored during training, rather than
        normalizing by the statistics of the (often single-cube) batch it is given."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_mc_dropout(self, x, n_samples=30, save_latent_images=False):
        """Estimate how uncertain the trained network is about x with Monte Carlo dropout: run it n_samples times
        with dropout left on, so each pass drops different channels, and take the mean and spread of the results.

        This is separate from predict_variance. predict_variance trains the network to predict each value's
        1-sigma, the noise in the data (aleatoric uncertainty). The spread here measures how much the prediction
        depends on which channels dropout removes, the network's own uncertainty (epistemic). It needs a network
        trained with dropout (head_dropout_rate or block_dropout_rate > 0); otherwise every pass is identical and
        the spread is 0.

        Returns (mean, epistemic_std), each (nBatch, n_maps, *output_shape). With predict_variance, returns
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
