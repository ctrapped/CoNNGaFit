"""Shared fixtures for the network tests: small, CPU-sized models and input cubes."""

import pytest
import torch

import unet3d_dilated_resnet
import unet3d_resblock

INPUT_SHAPE = (16, 16, 16)  # (nSpec, nX, nY)
BATCH_SIZE = 2
SEED = 0

# Constructor arguments shared by every network, plus per-network arguments that keep it small
COMMON_KWARGS = dict(input_shape=INPUT_SHAPE, nFilt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), nFC=8)
SMALL_KWARGS = {
    unet3d_resblock: dict(n_stages=3),
    unet3d_dilated_resnet: dict(n_blocks=1, dilation_rates=(1, 2, 1)),
}


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


@pytest.fixture(params=list(SMALL_KWARGS), ids=lambda module: module.__name__)
def network(request):
    """Each network module in turn."""
    return request.param


@pytest.fixture
def make_model():
    """Factory for a small module.NeuralNetwork; keyword arguments override the defaults."""

    def make(module, **overrides):
        return module.NeuralNetwork(**{**COMMON_KWARGS, **SMALL_KWARGS[module], **overrides})

    return make


@pytest.fixture
def make_cube():
    """Factory for a random batch of datacubes of shape (BATCH_SIZE, *input_shape)."""

    def make(input_shape=INPUT_SHAPE):
        return torch.randn(BATCH_SIZE, *input_shape)

    return make
