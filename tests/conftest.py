"""Shared fixtures: small, CPU-sized models and input cubes for the network tests, and a script runner for the
end-to-end tests."""

import subprocess
import sys

import pytest
import torch

import unet3d_dilated_resnet
import unet3d_resblock

INPUT_SHAPE = (16, 16, 16)  # (nSpec, nX, nY)
BATCH_SIZE = 2
SEED = 0
SCRIPT_TIMEOUT_S = 600

# Constructor arguments shared by every network, plus per-network arguments that keep it small
COMMON_KWARGS = dict(input_shape=INPUT_SHAPE, n_filt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), n_fc=8)
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


@pytest.fixture
def record_shapes():
    """Factory that hooks each of a list of modules; returns the list every forward pass then appends each module's
    output (channels, spatial shape) to, in execution order."""

    def record(modules):
        recorded = []

        def hook(module, inputs, out):
            out = out[0] if isinstance(out, tuple) else out  # e.g. a stem that also returns its half-resolution map
            recorded.append((out.shape[1], tuple(out.shape[2:])))

        for module in modules:
            module.register_forward_hook(hook)
        return recorded

    return record


@pytest.fixture(scope="session")
def run_script():
    """Run a Python script in a subprocess, as from the command line; fails the test with the end of its output
    if it exits with an error."""

    def run(script, *args, cwd=None):
        result = subprocess.run(
            [sys.executable, str(script), *(str(a) for a in args)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_S,
        )
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        return result

    return run
