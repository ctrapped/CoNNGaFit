"""Unet3d_Dilated_ResNet: block-count validation and analytic stage shapes."""

import pytest

import Unet3d_Dilated_ResNet as dilated


def record_stage_shapes(model):
    """Hook every stage of model in execution order; returns the list the hooks append (channels, dims) to."""
    head = (
        [model.conv_head.pool, *model.conv_head.blocks, model.conv_head.out]
        if model.head_type == "conv"
        else [model.avgpool0]
    )
    recorded = []
    for module in [model.layer_encode, *model.process, model.layer_decode, *head]:
        module.register_forward_hook(lambda m, inputs, out: recorded.append((out.shape[1], tuple(out.shape[2:]))))
    return recorded


def test_requires_at_least_one_block(make_model):
    with pytest.raises(ValueError, match="n_blocks"):
        make_model(dilated, n_blocks=0)


@pytest.mark.parametrize("head", ["fc", "conv"])
@pytest.mark.parametrize("stem_stride", [(1, 1, 1), (2, 2, 2), (1, 2, 2)])
@pytest.mark.parametrize("input_shape", [(16, 16, 16), (13, 11, 11)])
def test_stage_shapes_match_forward_pass(make_model, make_cube, head, stem_stride, input_shape):
    model = make_model(dilated, input_shape=input_shape, stem_stride=stem_stride, head=head)
    recorded = record_stage_shapes(model)
    model(make_cube(input_shape))
    expected = [
        (feats, tuple(shape)) for label, feats, shape in model.stage_shapes()[1:] if "resize to map" not in label
    ]
    assert recorded == expected
