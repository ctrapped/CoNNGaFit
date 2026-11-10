import math

import torch
import torch.nn.functional as F
from torch import nn

# Shared building blocks for the CoNNGaFit 3-d networks:
#   Stem, ResidualBlock, Upsample, DeconvBlock - unet3d_resblock.NeuralNetwork and unet3d_ddpm.DenoisingUNet3d
#   DilatedResidualBlock                       - unet3d_dilated_resnet.NeuralNetwork
#   SpectralAttentionPool                      - the conv heads below and unet3d_ddpm's MapHead
#   ResBlock2d                                 - ConvMapHead and unet3d_ddpm's MapHead
#   ConvMapHead                                - head='conv' option of both regression networks
#   HalfResConvHead                            - head='conv_halfres' option of unet3d_resblock.NeuralNetwork
# plus the padding/alignment helpers these blocks need to keep every axis exact for any input size.
#
# Originally written by Cameron Trapp (ctrapped@gmail.com), and refactored into its current form by Caleb Choban using Claude LLM.


#### Padding and alignment helpers used by the blocks ####
def _doubling_deconv_pads(kernel_size):
    """(padding, output_padding) so a stride-2 ConvTranspose3d gives out = 2*N - 1.

    ConvTranspose3d has out = (N-1)*2 - 2*padding + k + output_padding.
    Setting that equal to 2*N - 1 requires 2*padding = k - 1 + output_padding.
    """
    padding, output_padding = [], []
    for k in kernel_size:
        op = 1 - (k % 2)  # 0 for odd kernels, 1 for even kernels
        padding.append((k - 1 + op) // 2)
        output_padding.append(op)
    return tuple(padding), tuple(output_padding)


def _near_same_deconv_pad(kernel_size):
    """Per-axis padding (k-1)//2 for a stride-1 ConvTranspose3d."""
    return tuple((k - 1) // 2 for k in kernel_size)


def _ceil_conv_pad(kernel_size, stride):
    """(before, after) padding per dim so a conv with any stride gives out = ceil(N/stride)."""
    kernel_size, stride = tuple(kernel_size), tuple(stride)
    if len(kernel_size) != len(stride):
        raise ValueError(f"kernel_size {kernel_size} and stride {stride} must have the same length")
    if any(s < 1 for s in stride):
        raise ValueError(f"stride must be >= 1 on every axis; got {stride}")
    pad = []
    for k in kernel_size:  # kernel_size is (D, H, W); F.pad wants W, H, D order
        before = (k - 1) // 2
        after = (k - 1) - before
        pad = [before, after] + pad
    return tuple(pad)


def _film(h, scale_shift):
    """Feature-wise linear modulation (FiLM): h * (1 + scale) + shift.

    h: (nBatch, C, ...) features; scale_shift: (nBatch, 2C), split into per-channel scale and shift
    that are broadcast over all spatial axes. Typically, scale_shift = time_proj(t_emb) for
    diffusion timestep to tell each block how much noise to expect.
    """
    scale, shift = scale_shift.chunk(2, dim=1)
    view = scale.shape + (1,) * (h.dim() - 2)
    return h * (1 + scale.view(view)) + shift.view(view)


def _norm3d(channels, max_groups):
    """BatchNorm3d if max_groups is None, otherwise GroupNorm with gcd(channels, max_groups) groups
    (the largest group count <= max_groups that divides the channels)."""
    if max_groups is None:
        return nn.BatchNorm3d(channels)
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


def _match_spatial(x, ref):
    """Center-crop and/or zero-pad the last three dims of x to match ref's.

    The encoder's downsample (ceil(N/2)) and the decoder's Upsample (2*M-1) are not
    exact inverses. This aligns them for torch.cat.
    """
    # pad any axis where x is shorter than ref (F.pad wants W, H, D order)
    pad = []
    for xs, rs in zip(reversed(x.shape[2:]), reversed(ref.shape[2:])):
        deficit = max(rs - xs, 0)
        pad += [deficit // 2, deficit - deficit // 2]
    if any(pad):
        x = F.pad(x, pad)
    # crop any axis where x is now longer than ref
    slices = [slice(None), slice(None)]
    for xs, rs in zip(x.shape[2:], ref.shape[2:]):
        start = (xs - rs) // 2
        slices.append(slice(start, start + rs))
    return x[tuple(slices)]


#### Input stem ####
class Stem(nn.Module):
    """Reduces the raw input cube by a factor of 4 along every spatial/spectral axis, via a
    stride-2 Conv3d followed by a stride-2 MaxPool3d and an activation.

    Each stage's padding comes from _ceil_conv_pad, so the conv brings every axis to exactly
    ceil(N/2) and the pool then brings that to exactly ceil(N/4).

    Parameters:
        out_channels: number of filters (== n_filt0, the encoder's stage-0 channel count).
        kernel_size: 3-tuple kernel size for the conv (the pool's kernel is fixed at 2).
        activation: zero-arg callable returning an nn.Module (see ResidualBlock's activation
            parameter for the accepted forms).
        in_channels: input channels; 1 for a datacube alone, 1 + n_maps for the DDPM denoiser
            (the datacube plus each noisy map tiled along the spectral axis).
    """

    def __init__(self, out_channels, kernel_size, activation=nn.LeakyReLU, in_channels=1):
        super().__init__()
        stride = (2, 2, 2)  # conv and pool each halve every axis -> ceil(N/4) overall
        self.conv_pad = _ceil_conv_pad(kernel_size, stride)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.pool_pad = _ceil_conv_pad((2, 2, 2), stride)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=stride, padding=0)
        self.act = activation()

    def forward(self, x, return_half=False):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, ceil(D/4), ceil(H/4), ceil(W/4)).

        With return_half=True, also returns the activated conv output, at ceil(N/2) per axis
        (the last skip connection of head='conv_halfres').
        """
        half = self.conv(F.pad(x, self.conv_pad))
        x = self.pool(F.pad(half, self.pool_pad, mode="constant", value=float("-inf")))
        if return_half:
            return self.act(x), self.act(half)
        return self.act(x)


#### Encoder-side residual block ####
class ResidualBlock(nn.Module):
    """One residual block on the encoder (downsampling) side of the U-Net.

    Main path: conv1 -> norm -> [FiLM(t)] -> activation -> conv2 -> norm -> activation
    A skip is added, followed by a final activation and channel-wise dropout.

    When downsample=True, conv1 runs at stride 2 to bring every spatial/spectral axis
    to ceil(N/2) and the skip is projected through a strided 1x1x1 conv + norm to match.

    FiLM time conditioning is applied only when the block is built with t_dim AND forward() is
    given t_emb (as in the DDPM denoiser). With t_dim=None (the default, used by the regression
    networks) the block has no time_proj layer and forward() must be called without t_emb.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        downsample: if True, halve every axis (ceil(N/2)) and use a projected skip.
        kernel1: 3-tuple conv kernel size (used by conv1 and conv2).
        dropout_rate: probability for the channel-wise nn.Dropout3d on the block output.
            0.0 disables it.
        activation: an nn.Module (the class itself, e.g. nn.ReLU, or a lambda such as
           `lambda: nn.LeakyReLU(0.2)`).
        max_groups: None uses BatchNorm3d for bn1, bn2 and the projected skip; an int uses GroupNorm
            with gcd(channels, max_groups) groups instead (e.g. for the DDPM denoiser).
        t_dim: width of the timestep embedding t_emb. If set, a time_proj layer
            (SiLU -> Linear(t_dim, 2*out_channels)) is built to turn t_emb into the FiLM scale/shift
            applied after the first norm. None disables time conditioning.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        downsample,
        kernel1,
        dropout_rate=0.0,
        activation=nn.ReLU,
        max_groups=None,
        t_dim=None,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            # conv1 pads 0; forward() applies F.pad first so any kernel lands on ceil(N/2).
            stride = (2, 2, 2)
            self.down_pad = _ceil_conv_pad(kernel1, stride)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel1, stride=stride, padding=0)
            # a stride-2 1x1x1 conv already yields ceil(N/2), so the skip needs no padding.
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(2, 2, 2)),
                _norm3d(out_channels, max_groups),
            )
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel1, stride=(1, 1, 1), padding="same")
            self.shortcut = nn.Sequential()

        # conv2 keeps the shape fixed
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel1, stride=(1, 1, 1), padding="same")
        self.bn1 = _norm3d(out_channels, max_groups)
        self.bn2 = _norm3d(out_channels, max_groups)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.act1 = activation()
        self.act2 = activation()
        self.act3 = activation()
        if t_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))

    def forward(self, x, t_emb=None):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D', H', W'), where
        D'/H'/W' are ceil(0.5* D/H/W) if downsample=True.

        t_emb: (nBatch, t_dim) timestep embedding. If given, FiLM is applied after bn1 (requires the
        block to be built with t_dim). If None, the block runs without time conditioning.
        """
        shortcut = self.shortcut(x)
        if self.downsample:
            x = F.pad(x, self.down_pad)
        x = self.bn1(self.conv1(x))
        if t_emb is not None:
            x = _film(x, self.time_proj(t_emb))
        x = self.act1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = x + shortcut
        x = self.act3(x)
        return self.dropout(x)


#### Decoder-side spatial upsampler ####
class Upsample(nn.Module):
    """Spatially upsample by ~2x per axis, then project to out_channels. Two
    implementations, selected by `mode`:

      - 'conv_transpose' (default): a single stride-2 ConvTranspose3d does the spatial
        upsample and the channel projection together, in one learned operation.
        Cheap, but can be prone to checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/).

      - 'trilinear': nn.Upsample(scale_factor=2, mode='trilinear') doubles the spatial size
        (exactly 2*N, not 2*N-1) with no learned parameters, followed by a 1x1x1 Conv3d that
        projects in_channels -> out_channels.

    The small 2*N vs 2*N-1 difference between the two modes is accounted for downstream.
    NeuralNetwork.forward realigns the result to the matching encoder skip map's exact shape via
    _match_spatial before concatenating.

    Parameters:
        in_channels, out_channels: channel counts in and out.
        kernel1: 3-tuple kernel size; only used when mode='conv_transpose'.
        mode: 'conv_transpose' (default) or 'trilinear'.
    """

    def __init__(self, in_channels, out_channels, kernel1, mode="conv_transpose"):
        super().__init__()
        if mode not in ("conv_transpose", "trilinear"):
            raise ValueError(f"unknown Upsample mode {mode!r}; use 'conv_transpose' or 'trilinear'")
        self.mode = mode
        if mode == "conv_transpose":
            padding, output_padding = _doubling_deconv_pads(kernel1)
            self.upconv1 = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel1,
                stride=(2, 2, 2),
                padding=padding,
                output_padding=output_padding,
            )
        else:
            self.resize = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.project = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, ~2D, ~2H, ~2W) -
        exactly 2D-1 etc. for mode='conv_transpose', exactly 2D etc. for mode='trilinear'."""
        if self.mode == "conv_transpose":
            return self.upconv1(x)
        return self.project(self.resize(x))


#### Decoder-side deconvolution block ####
class DeconvBlock(nn.Module):
    """One block on the decoder side - the transpose-convolution mirror of ResidualBlock.

    Spatial upsampling is done beforehand by Upsample. Here every conv runs at stride 1, so
    the spatial shape is fixed; the block only changes the channel count. With upsample=True
    it reduces in_channels -> out_channels.
    Main path: convtran1 -> norm -> [FiLM(t)] -> activation -> convtran2 -> norm -> activation.
    A skip is added, followed by a final activation and channel-wise dropout.

    FiLM time conditioning is applied only when the block is built with t_dim AND forward() is
    given t_emb, exactly as in ResidualBlock.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        upsample: if True, project the skip to out_channels; if False, use an identity skip.
        kernel1: 3-tuple conv kernel size.
        dropout_rate: probability for the channel-wise nn.Dropout3d on the block output
            (0.0 disables).
        activation: zero-arg callable returning an nn.Module.
        max_groups, t_dim: as in ResidualBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        upsample,
        kernel1,
        dropout_rate=0.0,
        activation=nn.ReLU,
        max_groups=None,
        t_dim=None,
    ):
        super().__init__()
        pad = _near_same_deconv_pad(kernel1)  # forward() trims any even-kernel overshoot
        self.convtran1 = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel1, stride=(1, 1, 1), padding=pad
        )
        if upsample:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
                _norm3d(out_channels, max_groups),
            )
        else:
            self.shortcut = nn.Sequential()

        self.convtran2 = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size=kernel1, stride=(1, 1, 1), padding=pad
        )
        self.bn1 = _norm3d(out_channels, max_groups)
        self.bn2 = _norm3d(out_channels, max_groups)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.act1 = activation()
        self.act2 = activation()
        self.act3 = activation()
        if t_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))

    def forward(self, x, t_emb=None):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D, H, W).

        t_emb: (nBatch, t_dim) timestep embedding. If given, FiLM is applied after bn1 (requires the
        block to be built with t_dim). If None, the block runs without time conditioning.
        """
        shortcut = self.shortcut(x)  # carries the target spatial shape
        x = self.bn1(_match_spatial(self.convtran1(x), shortcut))
        if t_emb is not None:
            x = _film(x, self.time_proj(t_emb))
        x = self.act1(x)
        x = self.act2(self.bn2(_match_spatial(self.convtran2(x), shortcut)))
        x = x + shortcut
        x = self.act3(x)
        return self.dropout(x)


#### Dilated residual block (unet3d_dilated_resnet's Process stack) ####
class DilatedResidualBlock(nn.Module):
    """A chain of Conv3d layers whose dilation rates cycle through `dilation_rates` (default (1, 2, 4, 8, 4, 2, 1)),
    each at stride 1 with 'same' padding so every layer preserves the spatial/spectral shape exactly.
    Every conv is followed by BatchNorm3d and an activation, except the last conv in the chain, whose BatchNorm3d
    output is added to the block's input (a plain identity skip) before a final
    activation and channel-wise dropout.

    Parameters:
        channels: feature channel count (in == out, held constant through the block).
        kernel_size: 3-tuple conv kernel size, used by every layer in the chain.
        dilation_rates: sequence of per-layer dilation factors (applied isotropically, i.e.
            the same rate along D/H/W).
        dropout_rate: probability for the channel-wise nn.Dropout3d on the block output. 0.0
            disables it.
        activation: zero-arg callable returning an nn.Module
    """

    def __init__(
        self, channels, kernel_size, dilation_rates=(1, 2, 4, 8, 4, 2, 1), dropout_rate=0.0, activation=nn.ReLU
    ):
        super().__init__()
        self.dilation_rates = tuple(dilation_rates)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        for d in self.dilation_rates:
            self.convs.append(
                nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=(1, 1, 1), dilation=d, padding="same")
            )
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


#### 3-d -> 2-d convolutional head ####
class SpectralAttentionPool(nn.Module):
    """Collapse the spectral axis with learned, per-pixel softmax weights.
    Lets each pixel pick the velocity channels that matter (e.g. line wings for outflows),
    and works for any nSpec."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1)  # (B, C, S, X, Y) -> (B, 1, S, X, Y)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=2)  # (B, 1, S, X, Y), sums to 1 over S
        return (x * w).sum(dim=2)  # (B, C, X, Y)


class ResBlock2d(nn.Module):
    """2-d residual block: conv -> GroupNorm -> [FiLM(t)] -> act -> conv -> GroupNorm -> act,
    plus a skip (1x1 conv when the channel count changes). GroupNorm normalizes each sample on its
    own, so the block behaves the same at batch size 1 (inference, MC dropout) as in training.

    FiLM time conditioning is applied only when the block is built with t_dim AND forward() is
    given t_emb.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        activation: zero-arg callable returning an nn.Module.
        max_groups: upper bound on GroupNorm groups (actual = gcd(out_channels, max_groups)).
        kernel_size: kernel size of both 2-d convs.
        t_dim: width of the timestep embedding t_emb. If set, a time_proj layer
            (SiLU -> Linear(t_dim, 2*out_channels)) is built to turn t_emb into the FiLM scale/shift
            applied after norm1. None disables time conditioning.
    """

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, max_groups=8, kernel_size=3, t_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same")
        self.norm1 = nn.GroupNorm(math.gcd(out_channels, max_groups), out_channels)
        self.norm2 = nn.GroupNorm(math.gcd(out_channels, max_groups), out_channels)
        if t_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))
        self.act1, self.act2 = activation(), activation()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb=None):
        """x: (nBatch, in_channels, H, W) -> (nBatch, out_channels, H, W).

        t_emb: (nBatch, t_dim) timestep embedding. If given, FiLM is applied after norm1 (requires the
        block to be built with t_dim). If None, the block runs without time conditioning.
        """
        h = self.norm1(self.conv1(x))
        if t_emb is not None:
            h = _film(h, self.time_proj(t_emb))
        h = self.act1(h)
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class ConvMapHead(nn.Module):
    """Fully convolutional head: 3-d features (nBatch, C, S, X, Y) -> 2-d maps (nBatch, out_channels, H, W).

    Collapses the spectral axis with SpectralAttentionPool, resizes bilinearly to the target map grid
    (H, W), refines with `n_blocks` ResBlock2d and projects with a 3x3 conv. This ensures each output
    pixel is predicted from the features at and around that pixel.

    Parameters:
        channels: feature channels coming out of the network body.
        out_channels: output maps, e.g. 1 (mean) or 2 (mean + log-variance).
        n_blocks: number of ResBlock2d (>= 0).
        activation: zero-arg callable returning an nn.Module.
        dropout_rate: channel-wise nn.Dropout2d before the output conv (0.0 disables).
        max_groups: upper bound on GroupNorm groups (actual = gcd(channels, max_groups)).
    """

    def __init__(self, channels, out_channels=1, n_blocks=2, activation=nn.ReLU, dropout_rate=0.0, max_groups=8):
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        self.channels, self.out_channels = channels, out_channels
        self.pool = SpectralAttentionPool(channels)
        self.blocks = nn.ModuleList([ResBlock2d(channels, channels, activation, max_groups) for _ in range(n_blocks)])
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x, map_shape):
        """x: (nBatch, C, S, X, Y) -> (nBatch, out_channels, *map_shape)."""
        h = self.pool(x)  # (B, C, X, Y)
        h = F.interpolate(h, size=tuple(map_shape), mode="bilinear", align_corners=False)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.dropout(h))

    def stage_shapes(self, in_xy, map_shape):
        """(label, n_features, dims) rows for the head, given the (X, Y) of its input features."""
        c, map_shape = self.channels, tuple(map_shape)
        rows = [("head: spectral attention pool", c, tuple(in_xy)), ("head: resize to map", c, map_shape)]
        rows += [(f"head: 2-d res block {k}", c, map_shape) for k in range(len(self.blocks))]
        rows.append(("head: output conv", self.out_channels, map_shape))
        return rows

    def param_groups(self):
        """(label, [modules]) aligned with stage_shapes() rows."""
        groups = [("head: spectral attention pool", [self.pool]), ("head: resize to map", [])]
        groups += [(f"head: 2-d res block {k}", [b]) for k, b in enumerate(self.blocks)]
        groups.append(("head: output conv", [self.dropout, self.out]))
        return groups

    def row_note(self, label):
        """summary() note for one of this head's rows (None for other labels)."""
        if label == "head: spectral attention pool":
            return "softmax weights over nSpec"
        if label == "head: resize to map":
            return "bilinear"
        if label.startswith("head: 2-d res block"):
            return f"activation={type(self.blocks[0].act1).__name__}"
        if label == "head: output conv":
            return f"{self.out_channels} channels"
        return None


class HalfResConvHead(nn.Module):
    """Predicts one value per map pixel, for each of n_maps maps, from the decoder's half-resolution features.

    The spectral axis is folded into channels, so each pixel's line-profile features are combined
    by learned weights rather than averaged away. A global summary of the galaxy (the deepest
    encoder features, averaged over the whole cube) is appended to every pixel.

    Parameters:
        in_channels: decoder channels * half-resolution spectral length (after folding).
        global_channels: channel count of the deepest encoder stage.
        head_channels: channel count of the two 3x3 convolutions.
        dropout_rate: probability for the channel-wise nn.Dropout2d before the output conv.
        activation: zero-arg callable returning an nn.Module; called once per 3x3 conv.
        predict_variance: if True, a second 1x1 conv predicts a per-pixel log-variance of every map.
        n_maps: number of output maps.
    """

    def __init__(
        self,
        in_channels,
        global_channels,
        head_channels,
        dropout_rate=0.0,
        activation=nn.LeakyReLU,
        predict_variance=False,
        n_maps=1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + global_channels, head_channels, kernel_size=3, padding=1)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1)
        self.act2 = activation()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.out_mean = nn.Conv2d(head_channels, n_maps, kernel_size=1)
        self.out_logvar = nn.Conv2d(head_channels, n_maps, kernel_size=1) if predict_variance else None

    def forward(self, x, bottleneck, map_shape):
        """x: (nBatch, C, nSpec, nX, nY) decoder features; bottleneck: the deepest encoder features.

        Returns a list of (nBatch, n_maps, H*W) maps at map_shape, flattened per map: the mean,
        then the log-variance if predict_variance.
        """
        n_batch, n_filt, n_spec, n_x, n_y = x.shape
        x = torch.reshape(x, (n_batch, n_filt * n_spec, n_x, n_y))  # fold the spectral axis into channels
        # Global context: average the bottleneck over the whole cube and give every pixel a copy
        context = torch.mean(bottleneck, dim=(2, 3, 4))[:, :, None, None].expand(-1, -1, n_x, n_y)
        x = torch.cat([x, context], dim=1)
        x = self.act2(self.conv2(self.act1(self.conv1(x))))
        x = self.dropout(x)
        outputs = [self.out_mean(x)]
        if self.out_logvar is not None:
            outputs.append(self.out_logvar(x))
        return [
            torch.flatten(F.interpolate(o, size=tuple(map_shape), mode="bilinear", align_corners=False), 2)
            for o in outputs
        ]
