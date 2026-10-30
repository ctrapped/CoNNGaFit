"""unet3d_dilated_resnet: block-count validation and analytic stage shapes."""

import pytest

import unet3d_dilated_resnet as dilated


def stage_modules(model):
    """Every stage of model with a module of its own, in execution order."""
    if model.head_type == "conv":
        head = [model.conv_head.pool, *model.conv_head.blocks, model.conv_head.out]
    else:
        head = [model.avgpool0]
    return [model.layer_encode, *model.process, model.layer_decode, *head]


def test_requires_at_least_one_block(make_model):
    with pytest.raises(ValueError, match="n_blocks"):
        make_model(dilated, n_blocks=0)


@pytest.mark.parametrize("head", ["fc", "conv"])
@pytest.mark.parametrize("stem_stride", [(1, 1, 1), (2, 2, 2), (1, 2, 2)])
@pytest.mark.parametrize("input_shape", [(16, 16, 16), (13, 11, 11)])
def test_stage_shapes_match_forward_pass(make_model, make_cube, record_shapes, head, stem_stride, input_shape):
    model = make_model(dilated, input_shape=input_shape, stem_stride=stem_stride, head=head)
    recorded = record_shapes(stage_modules(model))
    model(make_cube(input_shape))
    expected = [
        (feats, tuple(shape)) for label, feats, shape in model.stage_shapes()[1:] if "resize to map" not in label
    ]
    assert recorded == expected
