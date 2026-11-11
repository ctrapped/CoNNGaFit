"""unet3d_resblock: stage-count validation, head/upsample configurations and analytic stage shapes."""

import pytest

from conngafit.models import unet3d_resblock as unet

HEADS = ["fc", "conv", "conv_halfres"]
CONV_HEADS = ["conv", "conv_halfres"]


def test_too_many_stages_names_tightest_axis(make_model):
    with pytest.raises(ValueError, match="x axis"):
        make_model(unet, input_shape=(64, 12, 64), n_stages=4)


def test_requires_at_least_two_stages(make_model):
    with pytest.raises(ValueError, match="n_stages"):
        make_model(unet, n_stages=1)


def test_rejects_unknown_upsample_mode(make_model):
    with pytest.raises(ValueError, match="Upsample mode"):
        make_model(unet, upsample_mode="nearest")


@pytest.mark.parametrize("head", CONV_HEADS)
@pytest.mark.parametrize("output_shape", [(8, 8), (20, 24)])
def test_conv_heads_predict_on_any_map_grid(make_model, make_cube, head, output_shape):
    model = make_model(unet, head=head, output_shape=output_shape)
    cube = make_cube()
    assert model(cube).shape == (len(cube), 1, output_shape[0] * output_shape[1])
