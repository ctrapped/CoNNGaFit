"""Unet_blocks: padding/alignment helpers and the shared 3-d and 2-d building blocks."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from Unet_blocks import (ConvMapHead, DilatedResidualBlock, ResBlock2d, SpectralAttentionPool,
                         _ceil_conv_pad, _doubling_deconv_pads, _match_spatial)

LENGTHS = range(1, 12)
KERNELS = (2, 3, 4, 5)
STRIDES = (1, 2, 3)
BATCH, CHANNELS = 2, 3
KERNEL = (3, 3, 3)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("k", KERNELS)
@pytest.mark.parametrize("n", LENGTHS)
def test_ceil_conv_pad_gives_ceil_n_over_stride(n, k, stride):
    conv = nn.Conv3d(1, 1, kernel_size=k, stride=stride)
    x = F.pad(torch.zeros(1, 1, n, n, n), _ceil_conv_pad((k,) * 3, (stride,) * 3))
    assert conv(x).shape[2:] == (-(-n // stride),) * 3


@pytest.mark.parametrize("kernel, stride", [((3, 3, 3), (2, 2)), ((3, 3, 3), (2, 0, 2))])
def test_ceil_conv_pad_rejects_bad_stride(kernel, stride):
    with pytest.raises(ValueError):
        _ceil_conv_pad(kernel, stride)


@pytest.mark.parametrize("k", KERNELS)
@pytest.mark.parametrize("n", LENGTHS)
def test_doubling_deconv_pads_give_2n_minus_1(n, k):
    padding, output_padding = _doubling_deconv_pads((k,) * 3)
    deconv = nn.ConvTranspose3d(1, 1, kernel_size=k, stride=2, padding=padding, output_padding=output_padding)
    assert deconv(torch.zeros(1, 1, n, n, n)).shape[2:] == (2 * n - 1,) * 3


@pytest.mark.parametrize("x_shape, ref_shape", [
    ((7, 8, 9), (5, 6, 7)),
    ((3, 4, 5), (5, 6, 7)),
    ((3, 8, 5), (6, 4, 5)),
])
def test_match_spatial_aligns_to_reference(x_shape, ref_shape):
    ref = torch.zeros(BATCH, CHANNELS, *ref_shape)
    assert _match_spatial(torch.randn(BATCH, CHANNELS, *x_shape), ref).shape == ref.shape


def test_match_spatial_pad_then_crop_round_trips():
    x = torch.randn(1, 1, 4, 5, 6)
    padded = _match_spatial(x, torch.zeros(1, 1, 7, 8, 9))
    assert torch.equal(_match_spatial(padded, x), x)


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_dilated_block_preserves_shape(dilation):
    block = DilatedResidualBlock(CHANNELS, KERNEL, dilation_rates=(dilation,))
    x = torch.randn(BATCH, CHANNELS, 9, 7, 5)
    assert block(x).shape == x.shape


def test_spectral_attention_pool_weights_sum_to_one():
    pool = SpectralAttentionPool(CHANNELS)
    x = torch.randn(BATCH, CHANNELS, 1, 5, 6).expand(-1, -1, 7, -1, -1)  # identical in every spectral channel
    torch.testing.assert_close(pool(x), x[:, :, 0])


@pytest.mark.parametrize("out_channels", [CHANNELS, 2 * CHANNELS])
def test_resblock2d_shape(out_channels):
    block = ResBlock2d(CHANNELS, out_channels)
    assert block(torch.randn(BATCH, CHANNELS, 5, 6)).shape == (BATCH, out_channels, 5, 6)


def test_resblock2d_is_independent_of_batch():
    block = ResBlock2d(CHANNELS, CHANNELS)
    x = torch.randn(BATCH, CHANNELS, 5, 6)
    torch.testing.assert_close(block(x[:1]), block(x)[:1])


@pytest.mark.parametrize("n_blocks", [0, 2])
@pytest.mark.parametrize("map_shape", [(5, 6), (12, 12)])
@pytest.mark.parametrize("out_channels", [1, 2])
def test_conv_map_head_shape(n_blocks, map_shape, out_channels):
    head = ConvMapHead(CHANNELS, out_channels=out_channels, n_blocks=n_blocks)
    out = head(torch.randn(BATCH, CHANNELS, 7, 4, 4), map_shape)
    assert out.shape == (BATCH, out_channels, *map_shape)


def test_conv_map_head_rejects_negative_blocks():
    with pytest.raises(ValueError, match="n_blocks"):
        ConvMapHead(CHANNELS, n_blocks=-1)
