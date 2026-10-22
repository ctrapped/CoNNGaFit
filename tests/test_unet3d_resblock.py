"""Unet3d_ResBlock: stage-count validation, head/upsample configurations and analytic stage shapes."""
from itertools import chain

import pytest

import Unet3d_ResBlock as unet

HEADS = ["fc", "conv", "conv_halfres"]
CONV_HEADS = ["conv", "conv_halfres"]
# stage_shapes() rows with no module of their own (concatenation, folding, resize) are not hooked;
# the final map shape is checked by the forward-shape tests instead
UNHOOKED_ROWS = ("+ skip", "resize to map", "fold spectra", "1x1 conv + resize")


def head_stages(model):
    """The head's hookable modules, in execution order, for each head type."""
    if model.head_type == "conv":
        return [model.conv_head.pool, *model.conv_head.blocks, model.conv_head.out]
    if model.head_type == "conv_halfres":
        return [model.upsample_half, model.dc_layer_half, model.halfres_head.act2]
    return [model.avgpool0]


def record_stage_shapes(model):
    """Hook every stage of model in execution order; returns the list the hooks append (channels, dims) to."""
    def record(module, inputs, out):
        out = out[0] if isinstance(out, tuple) else out  # the stem also returns its half-resolution map
        recorded.append((out.shape[1], tuple(out.shape[2:])))

    decoder = chain.from_iterable(zip(model.upsamples, model.dc_layers))
    recorded = []
    for module in [model.layer_stem, *model.encoder, *decoder, *head_stages(model)]:
        module.register_forward_hook(record)
    return recorded


def test_too_many_stages_names_tightest_axis(make_model):
    with pytest.raises(ValueError, match="x axis"):
        make_model(unet, input_shape=(64, 12, 64), n_stages=4)


def test_requires_at_least_two_stages(make_model):
    with pytest.raises(ValueError, match="n_stages"):
        make_model(unet, n_stages=1)


def test_rejects_unknown_upsample_mode(make_model):
    with pytest.raises(ValueError, match="Upsample mode"):
        make_model(unet, upsample_mode="nearest")


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("upsample_mode", ["conv_transpose", "trilinear"])
@pytest.mark.parametrize("input_shape, kernel", [
    ((16, 16, 16), (3, 3, 3)),
    ((13, 10, 11), (3, 3, 3)),
    ((16, 16, 16), (4, 4, 4)),
])
def test_forward_shape_for_every_configuration(make_model, make_cube, head, upsample_mode, input_shape, kernel):
    model = make_model(unet, input_shape=input_shape, kernel_stem=kernel, kernel_coder=kernel,
                       head=head, upsample_mode=upsample_mode)
    cube = make_cube(input_shape)
    assert model(cube).shape == (len(cube), input_shape[1] * input_shape[2])


@pytest.mark.parametrize("head", CONV_HEADS)
@pytest.mark.parametrize("output_shape", [(8, 8), (20, 24)])
def test_conv_heads_predict_on_any_map_grid(make_model, make_cube, head, output_shape):
    model = make_model(unet, head=head, output_shape=output_shape)
    cube = make_cube()
    assert model(cube).shape == (len(cube), output_shape[0] * output_shape[1])


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("upsample_mode", ["conv_transpose", "trilinear"])
def test_stage_shapes_match_forward_pass(make_model, make_cube, head, upsample_mode):
    model = make_model(unet, head=head, upsample_mode=upsample_mode)
    recorded = record_stage_shapes(model)
    model(make_cube())
    expected = [(feats, tuple(shape)) for label, feats, shape in model.stage_shapes()[1:]
                if not any(tag in label for tag in UNHOOKED_ROWS)]
    assert recorded == expected
