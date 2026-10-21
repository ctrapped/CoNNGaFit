"""Unet3d_ResBlock: padding helpers, stage-count validation, output heads and analytic stage shapes."""
from itertools import chain

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import Unet3d_ResBlock as unet
from Unet3d_ResBlock import _ceil_conv_pad, _doubling_deconv_pads, _match_spatial

LENGTHS = range(1, 12)
KERNELS = (2, 3, 4, 5)


@pytest.mark.parametrize("k", KERNELS)
@pytest.mark.parametrize("n", LENGTHS)
def test_ceil_conv_pad_halves_with_ceiling(n, k):
    conv = nn.Conv3d(1, 1, kernel_size=k, stride=2)
    out = conv(F.pad(torch.zeros(1, 1, n, n, n), _ceil_conv_pad((k,) * 3)))
    assert out.shape[2:] == (-(-n // 2),) * 3


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
    ref = torch.zeros(2, 3, *ref_shape)
    assert _match_spatial(torch.randn(2, 3, *x_shape), ref).shape == ref.shape


def test_match_spatial_pad_then_crop_round_trips():
    x = torch.randn(1, 1, 4, 5, 6)
    padded = _match_spatial(x, torch.zeros(1, 1, 7, 8, 9))
    assert torch.equal(_match_spatial(padded, x), x)


def test_too_many_stages_names_tightest_axis(make_model):
    with pytest.raises(ValueError, match="x axis"):
        make_model(unet, input_shape=(64, 12, 64), n_stages=4)


def test_requires_at_least_two_stages(make_model):
    with pytest.raises(ValueError, match="n_stages"):
        make_model(unet, n_stages=1)


@pytest.mark.parametrize("option", [dict(output_head="mlp"), dict(upsample_mode="nearest")])
def test_rejects_unknown_options(make_model, option):
    with pytest.raises(ValueError):
        make_model(unet, **option)


def test_conv_head_requires_one_output_per_pixel(make_model):
    with pytest.raises(ValueError, match="output_size"):
        make_model(unet, output_head="conv", output_size=5)


@pytest.mark.parametrize("output_head", ["fc", "conv"])
@pytest.mark.parametrize("upsample_mode", ["conv_transpose", "trilinear"])
@pytest.mark.parametrize("input_shape, kernel", [
    ((16, 12, 12), (3, 3, 3)),
    ((13, 10, 11), (3, 3, 3)),
    ((16, 12, 12), (4, 4, 4)),
])
def test_forward_shape_for_every_configuration(make_model, make_cube, output_head, upsample_mode, input_shape, kernel):
    model = make_model(unet, input_shape=input_shape, kernel_stem=kernel, kernel_coder=kernel,
                       output_head=output_head, upsample_mode=upsample_mode)
    cube = make_cube(input_shape)
    assert model(cube).shape == (len(cube), input_shape[1] * input_shape[2])


def test_conv_head_predicts_variance(make_model, make_cube):
    model = make_model(unet, output_head="conv", predict_variance=True)
    cube = make_cube()
    mean, logvar = model(cube)
    assert mean.shape == logvar.shape == (len(cube), model.output_size)


@pytest.mark.parametrize("upsample_mode", ["conv_transpose", "trilinear"])
def test_stage_shapes_match_forward_pass(make_model, make_cube, upsample_mode):
    model = make_model(unet, upsample_mode=upsample_mode)
    recorded = []
    decoder = chain.from_iterable(zip(model.upsamples, model.dc_layers))
    for module in [model.layer_stem, *model.encoder, *decoder, model.avgpool0]:
        module.register_forward_hook(lambda m, inputs, out: recorded.append((out.shape[1], tuple(out.shape[2:]))))
    model(make_cube())
    expected = [(feats, tuple(shape)) for label, feats, shape in model.stage_shapes()[1:] if "+ skip" not in label]
    assert recorded == expected
