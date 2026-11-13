"""models/ddpm_diagnostics: diagnostics of the DDPM recorded while it trains."""

import h5py
import numpy as np
import pytest
import torch
from conftest import BATCH_SIZE, INPUT_SHAPE
from torch import nn

from conngafit.models import unet3d_ddpm as ddpm
from conngafit.models.ddpm_diagnostics import loss_by_timestep
from conngafit.train_model import weighted_mse_loss

N_TIMESTEPS = 10
N_MAPS = 2
MAPS_SHAPE = (N_MAPS, *INPUT_SHAPE[1:])
TARGETS = ["mass", "rotational_velocity"]


@pytest.fixture
def diffusion():
    """A small GaussianDiffusion of N_MAPS maps, in eval mode, whose output depends on its input."""
    denoiser = ddpm.DenoisingUNet3d(
        INPUT_SHAPE, n_filt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), n_stages=3, n_maps=N_MAPS
    )
    nn.init.normal_(denoiser.head.out.weight)  # The output conv starts at zero
    return ddpm.GaussianDiffusion(denoiser, n_timesteps=N_TIMESTEPS).eval()


#### Loss by timestep ####
def test_per_sample_loss_averages_to_the_training_loss():
    pred_noise, noise, w = (torch.rand(BATCH_SIZE, *MAPS_SHAPE) for _ in range(3))
    losses = loss_by_timestep.per_sample_loss(pred_noise, noise, w)
    assert losses.shape == (BATCH_SIZE, N_MAPS)
    torch.testing.assert_close(losses.mean(), weighted_mse_loss(pred_noise, noise, w))


def test_validation_losses_are_each_samples_loss_at_its_fixed_timestep(diffusion):
    n_batches = 2
    batches = [
        (
            torch.randn(BATCH_SIZE, *INPUT_SHAPE),
            torch.randn(BATCH_SIZE, *MAPS_SHAPE),
            torch.ones(BATCH_SIZE, *MAPS_SHAPE),
            None,
        )
        for _ in range(n_batches)
    ]
    draws = [
        (torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,)), torch.randn(BATCH_SIZE, *MAPS_SHAPE)) for _ in range(n_batches)
    ]
    t, losses = loss_by_timestep.validation_losses_by_timestep(diffusion, batches, draws, "cpu")

    assert t.shape == (n_batches * BATCH_SIZE,) and losses.shape == (n_batches * BATCH_SIZE, N_MAPS)
    np.testing.assert_array_equal(t, torch.cat([draw_t for draw_t, _ in draws]).numpy())  # In dataset order
    with torch.no_grad():
        x, y, w, _ = batches[0]
        draw_t, draw_noise = draws[0]
        expected = loss_by_timestep.per_sample_loss(*diffusion(x, y, t=draw_t, noise=draw_noise), w)
    np.testing.assert_allclose(losses[:BATCH_SIZE], expected.numpy(), rtol=1e-5)


def test_loss_by_timestep_is_plotted_and_saved(tmp_path):
    n_samples = 4
    t = np.array([7, 1, 5, 3])
    losses_by_epoch = {epoch: np.random.rand(n_samples, N_MAPS) for epoch in (1, 3)}

    loss_by_timestep.plot_loss_by_timestep(t, losses_by_epoch, TARGETS, N_TIMESTEPS, tmp_path / "loss.png")
    loss_by_timestep.save_loss_by_timestep(t, losses_by_epoch, TARGETS, tmp_path / "loss.hdf5")

    assert (tmp_path / "loss.png").stat().st_size > 0
    with h5py.File(tmp_path / "loss.hdf5", "r") as hf:
        np.testing.assert_array_equal(hf["timesteps"][()], t)
        np.testing.assert_array_equal(hf["epochs"][()], [1, 3])
        np.testing.assert_array_equal(hf["losses"][()], np.stack([losses_by_epoch[1], losses_by_epoch[3]]))
        assert [target.decode() for target in hf["targets"][()]] == TARGETS
