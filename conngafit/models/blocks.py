"""Shared building blocks for the CoNNGaFit networks (3-d cubes and 2-d maps):
   Stem                                       - unet3d_resblock.NeuralNetwork and unet3d_ddpm.DenoisingUNet3d
   ResidualBlock, Upsample, DeconvBlock       - unet.UNet (ResidualBlock also in the 2-d map heads)
   SelfAttention                              - self-attention over every position of a coarse feature map
   DilatedResidualBlock                       - unet3d_dilated_resnet.NeuralNetwork
   SpectralAttentionPool                      - the conv heads below and unet3d_ddpm's MapHead
   MapHead2d                                  - 2-d features -> maps, for the 2-d spectral networks
   ConvMapHead                                - head='conv' option of both regression networks
   HalfResConvHead                            - head='conv_halfres' option of unet3d_resblock.NeuralNetwork
   SpectralEncoder                            - datacube -> per-pixel features for spectral_unet2d.py
Plus the padding/alignment helpers these blocks need to keep every axis exact for any input size.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
    """Per-axis padding (k-1)//2 for a stride-1 transposed conv."""
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


def _norm(channels, max_groups, n_dims=3):
    """BatchNorm{n_dims}d if max_groups is None, otherwise GroupNorm with gcd(channels, max_groups) groups
    (the largest group count <= max_groups that divides the channels)."""
    if max_groups is None:
        if n_dims == 3:
            return nn.BatchNorm3d(channels)
        elif n_dims == 2:
            return nn.BatchNorm2d(channels)
        else:
            raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
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


#### Input-shape checks ####
def _check_stage_compatibility(input_shape, n_stages):
    """Raise ValueError unless the input survives n_stages-1 halvings after the stem.

    The encoder has n_stages stages: the first keeps its shape, each later one
    downsamples every axis to ceil(N/2). So after the stem there are n_stages-1
    halvings, and an axis that reaches 1 before they are all done cannot be
    downsampled again. The message reports the tightest axis and the largest
    n_stages that axis allows. Returns the post-stem (nSpec, nX, nY) shape on success.
    """
    # The Stem is a stride-2 conv then a stride-2 maxpool, each padded by _ceil_conv_pad to give exactly
    # ceil(N/2). Ceiling division composes, ceil(ceil(N/2)/2) = ceil(N/4), so every axis ends at ceil(N/4)
    # for any kernel size. -(-n // 4) is integer ceil(n/4): floor division of -n rounds toward -inf.
    stem_shape = tuple(-(-n // 4) for n in input_shape)
    n_halvings = n_stages - 1
    tightest = (1 << 30, "", 0, 0)  # (max_halvings, axis_name, input_len, post_stem_len)
    for name, n0, n_stem in zip(("spectral", "x", "y"), input_shape, stem_shape):
        if n_stem < 1:
            raise ValueError(
                f"input {name} axis = {n0} is too small: the stem reduces it to {n_stem} (< 1). Enlarge that axis."
            )
        cap, n = 0, n_stem  # count ceil-halvings this axis can take before hitting 1
        while n >= 2:
            n = -(-n // 2)  # ceil(n/2), the encoder's downsample
            cap += 1
        if cap < tightest[0]:
            tightest = (cap, name, n0, n_stem)
    if tightest[0] < n_halvings:
        cap, name, n0, n_stem = tightest
        raise ValueError(
            f"n_stages={n_stages} needs {n_halvings} downsamplings after the stem, but the "
            f"{name} axis (input {n0}, {n_stem} after the stem) supports at most {cap}. "
            f"Use n_stages <= {cap + 1} or enlarge that axis."
        )
    return stem_shape


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


#### Input-shape checks ####
def check_stage_compatibility(input_shape, n_stages, stem_reduction=4):
    """The encoder has n_stages stages: the first keeps its shape, each later one downsamples every axis to
    ceil(N/2). Check that every axis can be halved n_stages - 1 times after the stem, which divides every axis by
    stem_reduction (4 for the Stem, 1 for a 2-d map with no stem). input_shape is (nSpec, nX, nY), or (nX, nY) for
    a map. If not, raise a ValueError naming the smallest axis and the largest n_stages it allows."""
    stem_shape = tuple(math.ceil(n / stem_reduction) for n in input_shape)
    n_halvings = n_stages - 1  # Encoder stage 0 keeps its input shape; each later stage halves every axis
    names = ("spectral", "x", "y")[-len(input_shape) :]
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


#### Encoder-side residual block ####
class ResidualBlock(nn.Module):
    """One residual block on the encoder (downsampling) side of the U-Net, in one of two architectures:

      - 'post_activation' (default): conv1 -> norm -> [FiLM(t)] -> activation -> conv2 -> norm -> activation -> + skip -> activation -> dropout.

      - 'identity_skip': conv1 -> norm -> [FiLM(t)] -> activation -> dropout -> conv2 -> norm -> + skip.
        Initialize second norm's weight to zero (i.e. block starts at identity) for easy initial training.

    When downsample=True, conv1 runs at stride 2 to bring every spatial/spectral axis to ceil(N/2). When
    downsampling or changing the channel count, the skip is projected through a 1x1 conv + norm to match
    (strided when downsampling); otherwise it is the identity.

    n_dims sets 3-d (cubes) or 2-d (maps) layers; the input then has n_dims axes after the channels.

    FiLM time conditioning is applied only when the block is built with time_embed_dim AND forward() is
    given t_emb (as in the DDPM denoiser). With time_embed_dim=None (the default, used by the regression
    networks) the block has no time_proj layer and forward() must be called without t_emb.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        downsample: if True, halve every axis (ceil(N/2)).
        kernel1: n_dims-tuple conv kernel size (used by conv1 and conv2).
        n_dims: 3 for Conv3d layers on cubes, 2 for Conv2d layers on maps.
        dropout_rate: probability for the channel-wise dropout (0.0 disables it).
        activation: zero-arg callable returning an nn.Module, e.g. nn.ReLU or lambda: nn.LeakyReLU(0.2).
        time_embed_dim: width of the timestep embedding t_emb. If set, a time_proj layer
            (SiLU -> Linear(time_embed_dim, 2*out_channels)) is built to turn t_emb into the FiLM scale/shift
            applied after the first norm. None disables time conditioning.
        max_groups: upper bound on GroupNorm groups, used only with time_embed_dim; without time conditioning
            the block uses BatchNorm.
        architecture: 'post_activation' or 'identity_skip', as described above.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        downsample,
        kernel1,
        n_dims=3,
        dropout_rate=0.0,
        activation=nn.ReLU,
        time_embed_dim=None,
        max_groups=8,
        architecture="post_activation",
    ):
        super().__init__()
        # Shared by every architecture
        if n_dims == 3:
            conv = nn.Conv3d
            self.dropout = nn.Dropout3d(p=dropout_rate)
        elif n_dims == 2:
            conv = nn.Conv2d
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
        self.downsample = downsample
        if time_embed_dim is None:
            max_groups = None
        if downsample:
            # conv1 pads 0; forward() applies F.pad first so any kernel lands on ceil(N/2).
            stride = (2,) * n_dims
            self.down_pad = _ceil_conv_pad(kernel1, stride)
            self.conv1 = conv(in_channels, out_channels, kernel_size=kernel1, stride=stride, padding=0)
        else:
            stride = 1
            self.conv1 = conv(in_channels, out_channels, kernel_size=kernel1, padding="same")
        if downsample or in_channels != out_channels:
            # a stride-2 1x1 conv already yields ceil(N/2), so the skip needs no padding.
            self.shortcut = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, stride=stride),
                _norm(out_channels, max_groups, n_dims),
            )
        else:
            self.shortcut = nn.Sequential()
        # conv2 keeps the shape fixed
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel1, padding="same")
        self.bn1 = _norm(out_channels, max_groups, n_dims)
        self.bn2 = _norm(out_channels, max_groups, n_dims)
        self.act1 = activation()
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * out_channels))

        self.architecture = architecture
        if architecture == "post_activation":
            self.act2 = activation()
            self.act3 = activation()
        elif architecture == "identity_skip":
            nn.init.zeros_(self.bn2.weight)  # The block starts as its skip
        else:
            raise ValueError(f"unknown architecture {architecture!r}; use 'post_activation' or 'identity_skip'")

    def forward(self, x, t_emb=None):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D', H', W'), where
        D'/H'/W' are ceil(0.5* D/H/W) if downsample=True.

        t_emb: (nBatch, time_embed_dim) timestep embedding. If given, FiLM is applied after bn1 (requires the
        block to be built with time_embed_dim). If None, the block runs without time conditioning.
        """
        if self.architecture == "post_activation":
            shortcut = self.shortcut(x)
            if self.downsample:
                x = F.pad(x, self.down_pad)
            x = self.bn1(self.conv1(x))
            if t_emb is not None:
                x = _film(x, self.time_proj(t_emb))
            x = self.act1(x)
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.act3(x + shortcut)
            return self.dropout(x)

        if self.architecture == "identity_skip":
            shortcut = self.shortcut(x)
            if self.downsample:
                x = F.pad(x, self.down_pad)
            x = self.bn1(self.conv1(x))
            if t_emb is not None:
                x = _film(x, self.time_proj(t_emb))
            x = self.act1(x)
            x = self.bn2(self.conv2(self.dropout(x)))
            return x + shortcut


#### Decoder-side spatial upsampler ####
class Upsample(nn.Module):
    """Spatially upsample by ~2x per axis, then project to out_channels. Two
    implementations, selected by `mode`:

      - 'conv_transpose' (default): a single stride-2 transposed conv does the spatial
        upsample and the channel projection together, in one learned operation.
        Cheap, but can be prone to checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/).

      - 'interpolation': nn.Upsample(scale_factor=2) with linear interpolation (trilinear for cubes, bilinear for
        maps) doubles the spatial size (exactly 2*N, not 2*N-1) with no learned parameters, followed by a 1x1
        conv that projects in_channels -> out_channels.

    The small 2*N vs 2*N-1 difference between the two modes is accounted for downstream.
    NeuralNetwork.forward realigns the result to the matching encoder skip map's exact shape via
    _match_spatial before concatenating.

    Parameters:
        in_channels, out_channels: channel counts in and out.
        kernel1: n_dims-tuple kernel size; only used when mode='conv_transpose'.
        n_dims: 3 for 3-d layers on cubes, 2 for 2-d layers on maps.
        mode: 'conv_transpose' (default) or 'interpolation'.
    """

    def __init__(self, in_channels, out_channels, kernel1, n_dims=3, mode="conv_transpose"):
        super().__init__()
        if mode not in ("conv_transpose", "interpolation"):
            raise ValueError(f"unknown Upsample mode {mode!r}; use 'conv_transpose' or 'interpolation'")
        if n_dims == 3:
            conv, conv_transpose, interpolation = nn.Conv3d, nn.ConvTranspose3d, "trilinear"
        elif n_dims == 2:
            conv, conv_transpose, interpolation = nn.Conv2d, nn.ConvTranspose2d, "bilinear"
        else:
            raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
        self.mode = mode
        if mode == "conv_transpose":
            padding, output_padding = _doubling_deconv_pads(kernel1)
            self.upconv1 = conv_transpose(
                in_channels,
                out_channels,
                kernel_size=kernel1,
                stride=(2,) * n_dims,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            self.resize = nn.Upsample(scale_factor=2, mode=interpolation, align_corners=False)
            self.project = conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """x: (nBatch, in_channels, *spatial) -> (nBatch, out_channels, ~2x every spatial axis) - exactly 2N-1
        for mode='conv_transpose', exactly 2N for mode='interpolation'."""
        if self.mode == "conv_transpose":
            return self.upconv1(x)
        return self.project(self.resize(x))


#### Decoder-side deconvolution block ####
class DeconvBlock(nn.Module):
    """One block on the decoder side - the transpose-convolution mirror of ResidualBlock.

    Spatial upsampling is done beforehand by Upsample. Here every conv runs at stride 1, so
    the spatial shape is fixed; the block only changes the channel count. With upsample=True
    it reduces in_channels -> out_channels. The two architectures are those of ResidualBlock:

      - 'post_activation' (default): convtran1 -> norm -> [FiLM(t)] -> activation -> convtran2 -> norm ->
        activation -> + skip -> activation -> dropout.

      - 'identity_skip': convtran1 -> norm -> [FiLM(t)] -> activation -> dropout -> convtran2 -> norm -> + skip.
      The second norm's weight starts at zero, so the block starts as its skip.

    FiLM time conditioning is applied only when the block is built with time_embed_dim AND forward() is
    given t_emb, exactly as in ResidualBlock.

    Parameters:
        in_channels, out_channels: channel counts of the block's input and output.
        upsample: if True, project the skip to out_channels; if False, use an identity skip.
        kernel1: n_dims-tuple conv kernel size.
        n_dims: 3 for Conv3d layers on cubes, 2 for Conv2d layers on maps.
        dropout_rate: probability for the channel-wise dropout (0.0 disables).
        activation: zero-arg callable returning an nn.Module.
        max_groups, time_embed_dim, architecture: as in ResidualBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        upsample,
        kernel1,
        n_dims=3,
        dropout_rate=0.0,
        activation=nn.ReLU,
        max_groups=8,
        time_embed_dim=None,
        architecture="post_activation",
    ):
        super().__init__()
        # Shared by every architecture
        if n_dims == 3:
            conv_transpose = nn.ConvTranspose3d
            self.dropout = nn.Dropout3d(p=dropout_rate)
        elif n_dims == 2:
            conv_transpose = nn.ConvTranspose2d
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
        if time_embed_dim is None:
            max_groups = None
        pad = _near_same_deconv_pad(kernel1)  # forward() trims any even-kernel overshoot
        self.convtran1 = conv_transpose(in_channels, out_channels, kernel_size=kernel1, padding=pad)
        if upsample:
            self.shortcut = nn.Sequential(
                conv_transpose(in_channels, out_channels, kernel_size=1),
                _norm(out_channels, max_groups, n_dims),
            )
        else:
            self.shortcut = nn.Sequential()
        self.convtran2 = conv_transpose(out_channels, out_channels, kernel_size=kernel1, padding=pad)
        self.bn1 = _norm(out_channels, max_groups, n_dims)
        self.bn2 = _norm(out_channels, max_groups, n_dims)
        self.act1 = activation()
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * out_channels))

        self.architecture = architecture
        if architecture == "post_activation":
            self.act2 = activation()
            self.act3 = activation()
        elif architecture == "identity_skip":
            nn.init.zeros_(self.bn2.weight)  # The block starts as its skip
        else:
            raise ValueError(f"unknown architecture {architecture!r}; use 'post_activation' or 'identity_skip'")

    def forward(self, x, t_emb=None):
        """x: (nBatch, in_channels, D, H, W) -> (nBatch, out_channels, D, H, W).

        t_emb: (nBatch, time_embed_dim) timestep embedding. If given, FiLM is applied after bn1 (requires the
        block to be built with time_embed_dim). If None, the block runs without time conditioning.
        """
        if self.architecture == "post_activation":
            shortcut = self.shortcut(x)  # carries the target spatial shape
            x = self.bn1(_match_spatial(self.convtran1(x), shortcut))
            if t_emb is not None:
                x = _film(x, self.time_proj(t_emb))
            x = self.act1(x)
            x = self.act2(self.bn2(_match_spatial(self.convtran2(x), shortcut)))
            x = self.act3(x + shortcut)
            return self.dropout(x)

        if self.architecture == "identity_skip":
            shortcut = self.shortcut(x)  # carries the target spatial shape
            x = self.bn1(_match_spatial(self.convtran1(x), shortcut))
            if t_emb is not None:
                x = _film(x, self.time_proj(t_emb))
            x = self.act1(x)
            x = self.bn2(_match_spatial(self.convtran2(self.dropout(x)), shortcut))
            return x + shortcut


#### Self-attention over a feature map ####
class SelfAttention(nn.Module):
    """Self-attention over every position (voxel of a cube, or pixel of a map) of a feature map:
    GroupNorm -> q, k, v (1x1 conv) -> scaled dot-product attention -> 1x1 output projection, added to the input.

    Every position attends to every other, so the block links distant parts of the cube or map which cannot reach.
    The output projection starts at zero, so the block starts as the identity. Memory/compute scale as N^2 in the
    number of positions N, so use it only on coarse grids.

    Parameters:
        channels: feature channels of the input and output.
        n_heads: number of attention heads; must divide channels.
        max_groups: upper bound on GroupNorm groups (actual = gcd(channels, max_groups)).
        n_dims: 3 for 3-d layers on cubes, 2 for 2-d layers on maps.
    """

    def __init__(self, channels, n_heads=1, max_groups=8, n_dims=3):
        super().__init__()
        if channels % n_heads:
            raise ValueError(f"n_heads={n_heads} must divide channels={channels}")
        if n_dims == 3:
            conv = nn.Conv3d
        elif n_dims == 2:
            conv = nn.Conv2d
        else:
            raise ValueError(f"n_dims must be 2 or 3; got {n_dims}")
        self.n_heads = n_heads
        # Normalizes each sample's features, so dot products have a consistent scale
        self.norm = nn.GroupNorm(math.gcd(channels, max_groups), channels)
        # Turn each position's features into a query, key, and value
        # (what it looks for in others, what it offers to others,
        # what it passes on to others that attend to it)
        self.qkv = conv(channels, 3 * channels, kernel_size=1)
        # Mixes the heads' outputs back into the feature channels
        self.proj = conv(channels, channels, kernel_size=1)
        # zero projection adds nothing to the input, so the block starts as the identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """x: (nBatch, channels, *spatial) -> (nBatch, channels, *spatial). *spatial is (D, H, W) for cubes or (H, W) for maps."""
        n_batch, channels = x.shape[:2]
        head_dim = channels // self.n_heads

        qkv = self.qkv(self.norm(x))  # (nBatch, 3 * channels, *spatial)
        qkv = qkv.reshape(n_batch, 3, self.n_heads, head_dim, -1)  # (nBatch, 3, n_heads, head_dim, n_positions)
        qkv = qkv.transpose(-1, -2)  # (nBatch, 3, n_heads, n_positions, head_dim): one row per position
        q, k, v = qkv.unbind(dim=1)  # Each (nBatch, n_heads, n_positions, head_dim)

        h = F.scaled_dot_product_attention(q, k, v)  # (nBatch, n_heads, n_positions, head_dim)

        # Back to a feature map, with the heads' outputs side by side along the channel axis
        h = h.transpose(-1, -2)  # (nBatch, n_heads, head_dim, n_positions)
        h = h.reshape(x.shape)  # (nBatch, channels, *spatial)

        # Residual: the projected attention output is added to the unchanged input
        return x + self.proj(h)


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


#### 2-d map head ####
class MapHead2d(nn.Module):
    """2-d features (nBatch, in_channels, H, W) -> maps (nBatch, out_channels, H, W): n_blocks 2-d ResidualBlocks
    (the first maps in_channels -> channels), channel-wise dropout and a 3x3 output conv, so each output pixel is
    predicted from the features at and around that pixel.

    Parameters:
        in_channels: channels of the input features.
        channels: channels of every block.
        out_channels: output maps, e.g. n_maps, or 2 * n_maps for mean + log-variance.
        n_blocks: number of 2-d ResidualBlocks (>= 0).
        activation, max_groups, time_embed_dim, architecture: as in ResidualBlock.
        dropout_rate: channel-wise nn.Dropout2d before the output conv (0.0 disables).
        zero_init_output: start the output conv at zero (the DDPM starts by predicting eps = 0).
    """

    def __init__(
        self,
        in_channels,
        channels,
        out_channels=1,
        n_blocks=2,
        activation=nn.ReLU,
        max_groups=8,
        time_embed_dim=None,
        architecture="post_activation",
        dropout_rate=0.0,
        zero_init_output=False,
    ):
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = ResidualBlock(
                in_channels if i == 0 else channels,
                channels,
                downsample=False,
                kernel1=(3, 3),
                n_dims=2,
                activation=activation,
                max_groups=max_groups,
                time_embed_dim=time_embed_dim,
                architecture=architecture,
            )
            self.blocks.append(block)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.out = nn.Conv2d(channels if n_blocks else in_channels, out_channels, 3, padding=1)
        if zero_init_output:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, h, t_emb=None):
        """h: (nBatch, in_channels, H, W); t_emb: (nBatch, time_embed_dim) or None."""
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(self.dropout(h))


#### 3-d -> 2-d convolutional head ####
class SpectralAttentionPool(nn.Module):
    """Collapse the spectral axis with learned, per-pixel softmax weights.
    Lets each pixel pick the velocity channels that matter (e.g. line wings for outflows),
    and works for any nSpec."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1)  # (B, C, S, X, Y) -> (B, 1, S, X, Y)

    def forward(self, x):
        # Compute per-pixel attention weights over the spectral axis and sum
        w = torch.softmax(self.score(x), dim=2)  # (B, 1, S, X, Y), sums to 1 over S
        return (x * w).sum(dim=2)  # (B, C, X, Y)


class ConvMapHead(nn.Module):
    """Fully convolutional head: 3-d features (nBatch, C, S, X, Y) -> 2-d maps (nBatch, out_channels, H, W).

    Collapses the spectral axis with SpectralAttentionPool, resizes bilinearly to the target map grid
    (H, W), refines with `n_blocks` 2-d ResidualBlocks and projects with a 3x3 conv. This ensures each output
    pixel is predicted from the features at and around that pixel.

    Parameters:
        channels: feature channels coming out of the network body.
        out_channels: output maps, e.g. 1 (mean) or 2 (mean + log-variance).
        n_blocks: number of 2-d ResidualBlocks (>= 0).
        activation: zero-arg callable returning an nn.Module.
        dropout_rate: channel-wise nn.Dropout2d before the output conv (0.0 disables).
        architecture: architecture of every ResidualBlock, 'post_activation' (default) or 'identity_skip'.
    """

    def __init__(
        self,
        channels,
        out_channels=1,
        n_blocks=2,
        activation=nn.ReLU,
        dropout_rate=0.0,
        architecture="post_activation",
    ):
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        self.pool = SpectralAttentionPool(channels)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels,
                    channels,
                    downsample=False,
                    kernel1=(3, 3),
                    n_dims=2,
                    activation=activation,
                    architecture=architecture,
                )
                for _ in range(n_blocks)
            ]
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, x, map_shape):
        """x: (nBatch, C, S, X, Y) -> (nBatch, out_channels, *map_shape)."""
        h = self.pool(x)  # (B, C, X, Y)
        h = F.interpolate(h, size=tuple(map_shape), mode="bilinear", align_corners=False)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.dropout(h))


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

        Returns a list of (nBatch, n_maps, H, W) maps at map_shape: the mean, then the log-variance if
        predict_variance.
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
        return [F.interpolate(o, size=tuple(map_shape), mode="bilinear", align_corners=False) for o in outputs]


#### Spectral encoder: datacube -> per-pixel features ####
class SpectralEncoder(nn.Module):
    """Turn every pixel's spectrum into features at full spatial resolution: n_layers of Conv3d -> GroupNorm ->
    activation, each halving the spectral axis (to ceil(S/2)) and keeping X and Y, then a SpectralAttentionPool that
    collapses what is left of the spectral axis. Each layer doubles the channels.

    Parameters:
        channels: channels of the first layer.
        n_layers: number of conv layers.
        kernel_size: 3-tuple (spectral, x, y) kernel of every conv.
        activation: zero-arg callable returning an nn.Module.
        max_groups: upper bound on GroupNorm groups (actual = gcd(channels, max_groups)).
    """

    def __init__(self, channels, n_layers, kernel_size, activation=nn.SiLU, max_groups=8):
        super().__init__()
        stride = (2, 1, 1)
        self.pad = _ceil_conv_pad(kernel_size, stride)  # ceil(S/2) along the spectrum, X and Y unchanged
        self.layers = nn.ModuleList()
        layer_in = 1  # One intensity per voxel
        for i in range(n_layers):
            layer_out = channels * 2**i
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(layer_in, layer_out, kernel_size=kernel_size, stride=stride),
                    nn.GroupNorm(math.gcd(layer_out, max_groups), layer_out),
                    activation(),
                )
            )
            layer_in = layer_out
        self.out_channels = layer_in
        self.pool = SpectralAttentionPool(self.out_channels)

    def forward(self, x):
        """x: (nBatch, 1, S, X, Y) -> (nBatch, out_channels, X, Y)."""
        for layer in self.layers:
            x = layer(F.pad(x, self.pad))
        return self.pool(x)
