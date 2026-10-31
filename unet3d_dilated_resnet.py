import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from unet_blocks import ConvMapHead, DilatedResidualBlock, _ceil_conv_pad
from unet_diagnostics import count_params, layer_param_lines, save_summed_spectral_channels

logger = logging.getLogger(__name__)

# 3-d dilated residual network ("Encode-Process-Decode") for HI 21cm spectral datacubes following
# architecture in Stachenfeld et al. 2021, (https://arxiv.org/abs/2112.15275).
# Reads an (nSpec x nX x nY) cube of arbitrary size and predicts a flattened output map,
# matching the input/output conventions of unet3d_resblock.py. There is no spatial
# downsampling/upsampling path or skip connections; the receptive field is grown entirely
# through dilated convolutions. See NeuralNetwork for the full parameter list.
#
# Written by Caleb Choban using Claude LLM, adapting unet3d_resblock.py's conventions.
# DilatedResidualBlock lives in unet_blocks.py; latent-image diagnostics in unet_diagnostics.py.


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
    """3-d dilated residual network ("Encode-Process-Decode") mapping an HI 21cm spectral
    datacube (nSpec x nX x nY, any size) to a flattened output map (e.g. radial momentum,
    rotational velocity, inclination) of length `output_size` (default nX*nY).

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
        `pool_output`, so the flattened size is fixed regardless of input shape or depth; then
        `n_fc_layers` hidden Linear->activation->dropout blocks feed the output Linear and an
        optional output activation.
        'conv': ConvMapHead (see unet_blocks.py) - spectral attention pool, bilinear resize to
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
            features go straight to the output Linear (with a dropout in between).
        n_blocks: number of DilatedResidualBlock instances in the Process stack. With the
            default 7-rate dilation schedule, n_blocks=4 gives 28 dilated conv layers.
        dilation_rates: per-layer dilation schedule cycled by every DilatedResidualBlock
            (default (1, 2, 4, 8, 4, 2, 1)).
        stem_stride: 3-tuple stride for the Encoder conv. (1,1,1) (default) means no
            downsampling; a larger stride (e.g. (2,2,2)) trades resolution for compute on
            large input cubes, bringing every axis to ceil(N/stride).
        n_fc_layers: number of hidden fully-connected layers when n_fc is a scalar (ignored
            when n_fc is a sequence of widths).
        output_size: length of the flattened prediction. Defaults to nX*nY from input_shape
            (or H*W from output_shape); must equal H*W for head='conv'.
        pool_output: 3-tuple target grid for nn.AdaptiveAvgPool3d. The fc input size is
            n_filt0 * prod(pool_output); the default (8, 8, 8) gives 512 * n_filt0.
        fc_dropout_rate: dropout probability after every hidden FC layer's activation (and,
            when there are no hidden layers, once before the output Linear); for head='conv',
            a channel-wise Dropout2d before the output conv. 0.0 disables. Active in training
            mode only.
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
            forward() returns one tensor of shape (nBatch, output_size). If True,
            a second output Linear is added predicting a per-element log-variance, and
            forward() returns (mean, logvar) - train against a heteroscedastic loss such as
            nn.GaussianNLLLoss()(mean, target, logvar.exp()) rather than plain MSE. logvar has
            no output_activation applied (it must stay unconstrained for exp() to reach any
            positive variance). With head='conv' the log-variance is a second output channel
            of the head's final conv instead of a second Linear.
        head: 'fc' (default) - adaptive average pool -> flatten -> fully-connected layers, as
            set by pool_output/n_fc/n_fc_layers; its output Linear has
            n_filt0 * prod(pool_output) * output_size weights, so it grows with the map size.
            'conv' - ConvMapHead: spectral attention pool -> bilinear resize to output_shape ->
            n_head_blocks 2-d residual blocks -> 3x3 conv, with a parameter count independent of
            the input and map size (pool_output, n_fc and n_fc_layers are then unused).
        output_shape: (H, W) of the predicted map. Required grid for head='conv' (default
            (nX, nY)); for head='fc' it only sets the default output_size = H*W.
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
        n_fc,
        n_blocks=4,
        dilation_rates=(1, 2, 4, 8, 4, 2, 1),
        stem_stride=(1, 1, 1),
        n_fc_layers=1,
        output_size=None,
        pool_output=(8, 8, 8),
        fc_dropout_rate=0.0,
        block_dropout_rate=0.0,
        encode_activation=None,
        block_activation=nn.ReLU,
        decode_activation=None,
        fc_activation=nn.LeakyReLU,
        output_activation=nn.Identity,
        predict_variance=False,
        head="fc",
        output_shape=None,
        n_head_blocks=2,
    ):
        super(NeuralNetwork, self).__init__()
        self.predict_variance = predict_variance
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        if head not in ("fc", "conv"):
            raise ValueError(f"unknown head {head!r}; use 'fc' or 'conv'")
        self.head_type = head
        # Map grid of the prediction: required by the conv head, optional for the fc head.
        self.output_shape = (
            tuple(int(v) for v in output_shape)
            if output_shape is not None
            else (input_shape[1:] if head == "conv" else None)
        )
        map_size = int(np.prod(self.output_shape)) if self.output_shape is not None else input_shape[1] * input_shape[2]
        if head == "conv" and output_size is not None and int(output_size) != map_size:
            raise ValueError(
                f"output_size={output_size} does not match output_shape {self.output_shape} for head='conv'"
            )
        self.output_size = int(output_size) if output_size is not None else map_size
        logger.debug("output size = %d", self.output_size)
        self.n_fc = n_fc
        self.save_itr = 0  # forward() call counter; numbers the latent-image filenames

        self.n_filt0 = n_filt0
        self.kernel_stem = tuple(kernel_stem)
        self.kernel_coder = tuple(kernel_coder)
        self.stem_stride = tuple(stem_stride)
        self.n_blocks = n_blocks
        self.dilation_rates = tuple(dilation_rates)
        self.encode_out_shape = _strided_out_shape(input_shape, self.stem_stride)

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

        self.pool_output = tuple(pool_output)
        if head == "conv":
            # Head: spectral attention pool -> resize to output_shape -> 2-d res blocks -> conv.
            self.fc_widths = []
            self.conv_head = ConvMapHead(
                n_filt0,
                out_channels=2 if predict_variance else 1,
                n_blocks=n_head_blocks,
                activation=block_activation,
                dropout_rate=fc_dropout_rate,
            )
        else:
            # Head: adaptive pool -> flatten -> hidden FC stack -> output Linear -> activation.
            # Identical to unet3d_resblock.py's head.
            self.avgpool0 = nn.AdaptiveAvgPool3d(self.pool_output)
            flat_features = n_filt0 * int(np.prod(self.pool_output))

            if isinstance(n_fc, (list, tuple)):
                fc_widths = [int(w) for w in n_fc if int(w) > 0]
            elif n_fc and n_fc > 0:
                fc_widths = [int(n_fc)] * int(n_fc_layers)
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

    def forward(self, x, save_latent_images=False):
        """Run the network on a batch of datacubes.

        x: tensor of shape (nBatch, nSpec, nX, nY) - a single-channel spectral datacube per
            sample; (nSpec, nX, nY) must equal the model's input_shape. A channel dim of
            size 1 is inserted before the 3-d convolutions.
        save_latent_images: if True, dump PNGs of the spectrally-summed activations after the
            encoder and after each Process block to LATENT_IMAGE_OUTPUT (see
            save_summed_spectral_channels in unet_diagnostics.py) - useful for visually debugging what the network
            is learning, not needed for normal training/inference.
        Returns: if predict_variance=False (default), a tensor of shape (nBatch, output_size)
            - the flattened predicted output map. If predict_variance=True, a (mean, logvar)
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
        """Body features -> (mean, logvar), each flattened to (nBatch, output_size); logvar is None
        unless predict_variance. 'conv': one output map channel each; 'fc': pool -> FC stack."""
        if self.head_type == "conv":
            out = self.conv_head(x, self.output_shape).flatten(2)  # (nBatch, 1 or 2, H*W)
            mean = self.out_activation(out[:, 0])
            return mean, (out[:, 1] if self.predict_variance else None)
        x = self.avgpool0(x)
        x = torch.flatten(x, 1)
        x = self.fc_hidden(x)
        mean = self.out_activation(self.fc_out_mean(x))
        return mean, (self.fc_out_logvar(x) if self.predict_variance else None)

    def stage_shapes(self):
        """Analytically compute the tensor shape at every stage (no forward pass).

        Returns a list of (label, n_features, (nSpec, nX, nY)) tuples in execution order,
        matching what forward() produces: the encoder brings every axis to
        ceil(N/stem_stride) (exactly N when stem_stride is (1,1,1)), every Process block and
        the decoder then preserve that shape exactly (dilation with 'same' padding at stride 1
        never changes the spatial/spectral size).
        """
        rows = [("input datacube", 1, self.input_shape)]
        rows.append(("encode", self.n_filt0, self.encode_out_shape))
        for i in range(self.n_blocks):
            rows.append((f"process block {i}", self.n_filt0, self.encode_out_shape))
        rows.append(("decode", self.n_filt0, self.encode_out_shape))
        if self.head_type == "conv":
            rows += self.conv_head.stage_shapes(self.encode_out_shape[1:], self.output_shape)
        else:
            rows.append(("adaptive avg pool", self.n_filt0, self.pool_output))
        return rows

    def _enable_mc_dropout(self):
        """Put every Dropout/Dropout2d/Dropout3d submodule into train mode while leaving BatchNorm3d
        (and everything else) in eval mode - MC Dropout needs stochastic dropout masks per
        sample, but BatchNorm3d in train mode would normalize against a single-sample batch
        at inference (run_inferences in use_model.py runs batch_size=1)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_uncertainty(self, x, n_samples=30, save_latent_images=False):
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

    def _param_groups(self):
        """(label, [modules]) for every summary() row, in order: the stage_shapes() rows, then (fc
        head) the fully-connected layers. Rows with no parameters (input, pooling, resize) get []."""
        mods = [[], [self.layer_encode]] + [[block] for block in self.process] + [[self.layer_decode]]
        if self.head_type == "conv":
            head = self.conv_head.param_groups()
            head[-1] = (head[-1][0], head[-1][1] + [self.out_activation])  # output conv + activation
            mods += [m for _, m in head]
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
        which activation runs at each stage, and the dilation schedule used by the Process
        stack. `print_fn` lets output be redirected (e.g. to a logger); defaults to the
        builtin print. layers: if True (default), list each block's layers and their parameter
        counts under the block's row.
        """
        rows = self.stage_shapes()
        w = max(len(label) for label, _, _ in rows)
        bar = "=" * (w + 58)

        encode_act = type(self.layer_encode.act).__name__
        block_act = type(self.process[0].acts[0]).__name__  # same class at every block/site
        decode_act = type(self.layer_decode.act).__name__
        fc_act = type(self.fc_hidden[1]).__name__ if self.fc_widths else None
        out_act = type(self.out_activation).__name__
        if self.head_type == "conv":
            head_act = (
                "head blocks=" + type(self.conv_head.blocks[0].act1).__name__
                if len(self.conv_head.blocks)
                else "head blocks=n/a (none)"
            )
        else:
            head_act = f"fc hidden={fc_act or 'n/a (no hidden FC layers)'}"

        print_fn(bar)
        print_fn(
            f"{type(self).__name__}  |  {self.n_blocks} process blocks  |  "
            f"dilation_rates={self.dilation_rates}  |  kernel_stem={self.kernel_stem}  |  "
            f"kernel_coder={self.kernel_coder}  |  stem_stride={self.stem_stride}  |  "
            f"predict_variance={self.predict_variance}  |  head={self.head_type!r}"
        )
        n_spec, n_x, n_y = self.input_shape
        print_fn(f"input:  (nBatch, nSpec={n_spec}, nX={n_x}, nY={n_y})  ->  output: (nBatch, {self.output_size})")
        print_fn(
            f"activations: encode={encode_act}  process blocks={block_act}  "
            f"decode={decode_act}  {head_act}  "
            f"output={out_act}"
        )
        print_fn("-" * (w + 58))

        def row_note(label):
            if label == "encode":
                return f"activation={encode_act}  stride={self.stem_stride}"
            if label.startswith("process block"):
                return f"activation={block_act}  dilation_rates={self.dilation_rates}"
            if label == "decode":
                return f"activation={decode_act}"
            if self.head_type == "conv":
                return self.conv_head.row_note(label)
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
            flat = self.n_filt0 * int(np.prod(self.pool_output))
            print_fn(f"  {'flatten':<{w}}   features={flat}")
            in_f = flat
            for k, out_f in enumerate(self.fc_widths):
                label = f"fc hidden {k}"
                print_fn(f"  {label:<{w}}   {in_f} -> {out_f}  params={params_col(label)}  activation={fc_act}")
                in_f = out_f
            print_fn(
                f"  {'fc out (mean)':<{w}}   {in_f} -> {self.output_size}  "
                f"params={params_col('fc out (mean)')}  activation={out_act}"
            )
            if self.predict_variance:
                print_fn(
                    f"  {'fc out (logvar)':<{w}}   {in_f} -> {self.output_size}  "
                    f"params={params_col('fc out (logvar)')}  activation=None"
                )
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
