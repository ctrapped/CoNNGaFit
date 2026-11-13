"""unet3d_dilated_resnet: block-count validation and analytic stage shapes."""

import pytest

from conngafit.models import unet3d_dilated_resnet as dilated


def test_requires_at_least_one_block(make_model):
    with pytest.raises(ValueError, match="n_blocks"):
        make_model(dilated, n_blocks=0)
