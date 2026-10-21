"""Unet3d_Dilated_ResNet: strided padding, shape-preserving dilated blocks and analytic stage shapes."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import Unet3d_Dilated_ResNet as dilated
from Unet3d_Dilated_ResNet import DilatedResidualBlock, _ceil_conv_pad

LENGTHS = range(1, 12)
KERNELS = (2, 3, 4, 5)
STRIDES = (1, 2, 3)
CHANNELS = 3
KERNEL = (3, 3, 3)


@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("k", KERNELS)
@pytest.mark.parametrize("n", LENGTHS)
def test_ceil_conv_pad_gives_ceil_n_over_stride(n, k, stride):
    conv = nn.Conv3d(1, 1, kernel_size=k, stride=stride)
    x = F.pad(torch.zeros(1, 1, n, n, n), _ceil_conv_pad((k,) * 3, (stride,) * 3))
    assert conv(x).shape[2:] == (-(-n // stride),) * 3


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_dilated_block_preserves_shape(dilation):
    block = DilatedResidualBlock(CHANNELS, KERNEL, dilation_rates=(dilation,))
    x = torch.randn(2, CHANNELS, 9, 7, 5)
    assert block(x).shape == x.shape


def test_requires_at_least_one_block(make_model):
    with pytest.raises(ValueError, match="n_blocks"):
        make_model(dilated, n_blocks=0)


@pytest.mark.parametrize("stem_stride", [(1, 1, 1), (2, 2, 2), (1, 2, 2)])
@pytest.mark.parametrize("input_shape", [(16, 12, 12), (13, 11, 11)])
def test_stage_shapes_match_forward_pass(make_model, make_cube, stem_stride, input_shape):
    model = make_model(dilated, input_shape=input_shape, stem_stride=stem_stride)
    recorded = []
    for module in [model.layer_encode, *model.process, model.layer_decode, model.avgpool0]:
        module.register_forward_hook(lambda m, inputs, out: recorded.append((out.shape[1], tuple(out.shape[2:]))))
    model(make_cube(input_shape))
    assert recorded == [(feats, tuple(shape)) for _, feats, shape in model.stage_shapes()[1:]]
