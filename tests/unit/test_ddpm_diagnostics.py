"""models/ddpm_diagnostics: diagnostics of the DDPM recorded while it trains."""

import h5py
import numpy as np
import pytest
import torch
from conftest import BATCH_SIZE, INPUT_SHAPE
from torch import nn

from conngafit.models import unet3d_ddpm as ddpm
from conngafit.models.ddpm_diagnostics import calibration, loss_by_timestep, reverse_chain, spectral_attention
from conngafit.models.ddpm_diagnostics.activations import captured_outputs
from conngafit.models.ddpm_diagnostics.recorder import TrainingDiagnostics
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


#### Calibration ####
N_DRAWS = 19  # Posterior samples per pixel in the calibration tests: ranks 0 to 19
N_PIXELS = 20000


def test_calibrated_samples_give_uniform_ranks_and_nominal_coverage():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=N_PIXELS)
    samples = rng.normal(size=(N_DRAWS, N_PIXELS))  # Drawn from the same distribution as the truth
    ranks = calibration.posterior_ranks(samples, truth, rng)
    counts = np.bincount(ranks, minlength=N_DRAWS + 1)
    expected = N_PIXELS / (N_DRAWS + 1)
    assert np.all(np.abs(counts - expected) < 4 * np.sqrt(expected))  # Uniform within counting noise
    coverage = calibration.coverage_from_ranks(ranks, N_DRAWS, calibration.COVERAGE_LEVELS, rng)
    np.testing.assert_allclose(coverage, calibration.COVERAGE_LEVELS, atol=0.02)


def test_too_narrow_posteriors_cover_too_little():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=N_PIXELS)
    samples = rng.normal(scale=0.5, size=(N_DRAWS, N_PIXELS))  # Half the true spread
    ranks = calibration.posterior_ranks(samples, truth, rng)
    coverage = calibration.coverage_from_ranks(ranks, N_DRAWS, calibration.COVERAGE_LEVELS, rng)
    assert np.all(coverage < calibration.COVERAGE_LEVELS)


def test_tied_samples_spread_the_rank_over_the_tied_positions():
    rng = np.random.default_rng(0)
    truth = np.full(N_PIXELS, -1.0)
    samples = np.full((N_DRAWS, N_PIXELS), -1.0)  # E.g. empty sky: the clipped samples equal the label
    ranks = calibration.posterior_ranks(samples, truth, rng)
    assert ranks.min() == 0 and ranks.max() == N_DRAWS  # Every tied position, not all rank 0
    assert abs(ranks.mean() - N_DRAWS / 2) < 0.1


def test_validation_posterior_samples_have_one_set_per_sample(diffusion):
    n_batches, n_samples = 2, 3
    batches = [
        (torch.randn(BATCH_SIZE, *INPUT_SHAPE), torch.randn(BATCH_SIZE, *MAPS_SHAPE), None, None)
        for _ in range(n_batches)
    ]
    samples, labels = calibration.validation_posterior_samples(diffusion, batches, n_samples, "cpu")
    assert samples.shape == (n_samples, n_batches * BATCH_SIZE, *MAPS_SHAPE)
    np.testing.assert_array_equal(labels, torch.cat([y for _, y, _, _ in batches]).numpy())  # In dataset order


def test_calibration_summary_counts_every_pixel_of_each_target():
    rng = np.random.default_rng(0)
    n_samples, n_val = 5, 3
    samples = rng.normal(size=(n_samples, n_val, *MAPS_SHAPE))
    labels = rng.normal(size=(n_val, *MAPS_SHAPE))
    rank_counts, coverage = calibration.calibration_summary(samples, labels, rng)
    assert rank_counts.shape == (N_MAPS, n_samples + 1)
    assert (rank_counts.sum(axis=1) == n_val * np.prod(MAPS_SHAPE[1:])).all()
    assert coverage.shape == (len(calibration.COVERAGE_LEVELS), N_MAPS)


def test_calibration_is_plotted_and_saved(tmp_path):
    rng = np.random.default_rng(0)
    n_levels = len(calibration.COVERAGE_LEVELS)
    rank_counts = {epoch: rng.integers(0, 100, (N_MAPS, N_DRAWS + 1)) for epoch in (1, 3)}
    coverage = {epoch: rng.random((n_levels, N_MAPS)) for epoch in (1, 3)}

    calibration.plot_calibration(rank_counts, coverage, TARGETS, tmp_path / "calibration.png")
    calibration.save_calibration(rank_counts, coverage, TARGETS, tmp_path / "calibration.hdf5")

    assert (tmp_path / "calibration.png").stat().st_size > 0
    with h5py.File(tmp_path / "calibration.hdf5", "r") as hf:
        np.testing.assert_array_equal(hf["levels"][()], calibration.COVERAGE_LEVELS)
        np.testing.assert_array_equal(hf["epochs"][()], [1, 3])
        np.testing.assert_array_equal(hf["coverage"][()], np.stack([coverage[1], coverage[3]]))
        np.testing.assert_array_equal(hf["rank_counts"][()], np.stack([rank_counts[1], rank_counts[3]]))
        assert [target.decode() for target in hf["targets"][()]] == TARGETS


#### Activation capture ####
def test_captured_outputs_collects_each_call_and_removes_its_hooks():
    layers = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    x = torch.randn(4, 3)
    with captured_outputs({"linear": layers[0]}) as outputs:
        layers(x)
        layers(x)
    assert len(outputs["linear"]) == 2
    torch.testing.assert_close(outputs["linear"][0], layers[0](x))
    assert not layers[0]._forward_hooks  # Hooks are gone after the block


#### Spectral attention ####
def test_attention_weights_sum_to_one_over_the_spectral_channels(diffusion):
    cube, y0, noise = torch.randn(1, *INPUT_SHAPE), torch.randn(1, *MAPS_SHAPE), torch.randn(1, *MAPS_SHAPE)
    timesteps = [0, N_TIMESTEPS - 1]
    weights = spectral_attention.attention_weights(diffusion, cube, y0, noise, timesteps)
    assert weights.shape[0] == len(timesteps) and weights.ndim == 4  # (n_t, S', X', Y')
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=1e-5)


@pytest.mark.parametrize(
    "n_channels, n_pooled, expected",
    [(4, None, [-15, -5, 5, 15]), (5, None, [-20, -10, 0, 10, 20]), (4, 2, [-10, 10])],
)
def test_channel_velocities_are_centred_on_zero(n_channels, n_pooled, expected):
    np.testing.assert_allclose(spectral_attention.channel_velocities(n_channels, 10.0, n_pooled), expected)


def test_attention_on_one_channel_gives_its_velocity_and_no_spread():
    velocities = np.array([-20.0, 0.0, 20.0])
    weights = np.zeros((3, 2, 2))
    weights[2] = 1  # Every pixel attends only to the last channel
    mean, spread = spectral_attention.attention_moments(weights, velocities)
    np.testing.assert_allclose(mean, 20.0)
    np.testing.assert_allclose(spread, 0.0)


def test_cube_mean_velocity_follows_the_emission():
    velocities = np.array([-20.0, 0.0, 20.0])
    cube = np.zeros((3, 2, 2))
    cube[0, 0, 0] = 5.0  # One pixel emits only in the first channel; the others are empty
    velocity = spectral_attention.cube_mean_velocity(cube, velocities)
    assert velocity[0, 0] == -20.0 and np.isnan(velocity[1, 1])


def test_spectral_attention_is_plotted_and_saved(tmp_path):
    rng = np.random.default_rng(0)
    timesteps = [0, 5, 9]
    weights = {epoch: rng.dirichlet(np.ones(4), size=(3, 3, 3)).transpose(0, 3, 1, 2) for epoch in (1, 3)}
    velocities = spectral_attention.channel_velocities(16, 10.0, 4)
    cube_velocity = rng.normal(size=(12, 12))
    mean, spread = spectral_attention.attention_moments(weights[3], velocities)

    spectral_attention.plot_spectral_attention(cube_velocity, mean, spread, timesteps, 3, tmp_path / "attention.png")
    spectral_attention.save_spectral_attention(
        weights, velocities, cube_velocity, timesteps, tmp_path / "attention.hdf5"
    )

    assert (tmp_path / "attention.png").stat().st_size > 0
    with h5py.File(tmp_path / "attention.hdf5", "r") as hf:
        np.testing.assert_array_equal(hf["weights"][()], np.stack([weights[1], weights[3]]))
        np.testing.assert_array_equal(hf["epochs"][()], [1, 3])
        np.testing.assert_array_equal(hf["timesteps"][()], timesteps)
        np.testing.assert_array_equal(hf["velocities"][()], velocities)
        np.testing.assert_array_equal(hf["cube_velocity"][()], cube_velocity)


#### Reverse chain ####
def test_reverse_chain_ends_at_the_returned_sample(diffusion):
    cube, y_true = torch.randn(1, *INPUT_SHAPE), torch.rand(1, *MAPS_SHAPE)
    snapshot_timesteps = [9, 5, 0]

    torch.manual_seed(0)
    snapshots, errors = reverse_chain.reverse_chain(diffusion, cube, y_true, snapshot_timesteps)
    torch.manual_seed(0)
    sample = diffusion.sample(cube, n_samples=1)[0, 0]

    assert snapshots.shape == (len(snapshot_timesteps), 2, *MAPS_SHAPE)
    assert errors.shape == (N_TIMESTEPS, N_MAPS)
    # At t = 0 the sample is the clean-map estimate, so its last snapshot and error are those of the sample
    np.testing.assert_allclose(snapshots[-1, 1], sample.numpy(), rtol=1e-6)
    expected = ((sample - y_true[0]) ** 2).flatten(1).mean(dim=1).numpy()
    np.testing.assert_allclose(errors[0], expected, rtol=1e-5)


def test_reverse_chain_is_plotted_and_saved(tmp_path):
    rng = np.random.default_rng(0)
    snapshot_timesteps = [9, 5, 0]
    snapshots = {epoch: rng.normal(size=(len(snapshot_timesteps), 2, *MAPS_SHAPE)) for epoch in (1, 3)}
    errors = {epoch: rng.random((N_TIMESTEPS, N_MAPS)) for epoch in (1, 3)}
    y_true = rng.normal(size=MAPS_SHAPE)

    reverse_chain.plot_reverse_chain(
        snapshots[3], snapshot_timesteps, y_true, errors, TARGETS, 3, tmp_path / "chain.png"
    )
    reverse_chain.save_reverse_chain(snapshots, errors, snapshot_timesteps, y_true, TARGETS, tmp_path / "chain.hdf5")

    assert (tmp_path / "chain.png").stat().st_size > 0
    with h5py.File(tmp_path / "chain.hdf5", "r") as hf:
        np.testing.assert_array_equal(hf["snapshots"][()], np.stack([snapshots[1], snapshots[3]]))
        np.testing.assert_array_equal(hf["errors"][()], np.stack([errors[1], errors[3]]))
        np.testing.assert_array_equal(hf["epochs"][()], [1, 3])
        np.testing.assert_array_equal(hf["snapshot_timesteps"][()], snapshot_timesteps)
        np.testing.assert_array_equal(hf["truth"][()], y_true)
        assert [name.decode() for name in hf["targets"][()]] == TARGETS


#### Recording during training ####
def test_record_rewrites_every_diagnostic_with_each_recorded_epoch(diffusion, tmp_path):
    n_samples = 3
    batches = [(torch.randn(BATCH_SIZE, *INPUT_SHAPE), torch.rand(BATCH_SIZE, *MAPS_SHAPE), torch.ones(1), None)]
    draws = [(torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,)), torch.randn(BATCH_SIZE, *MAPS_SHAPE))]
    diagnostics = TrainingDiagnostics(
        batches,
        draws,
        n_samples,
        TARGETS,
        str(tmp_path / "run"),
        0,
        "cpu",
        image_mean=0,
        image_stdv=1,
        channel_width=10,
    )

    rng_state = torch.get_rng_state()
    for epoch in (1, 3):
        diagnostics.record(epoch, diffusion)
    assert torch.equal(torch.get_rng_state(), rng_state)  # Training's random draws are left alone

    for name in ("LossByTimestep", "Calibration", "SpectralAttention", "ReverseChain"):
        assert (tmp_path / f"run_{name}.png").stat().st_size > 0
        with h5py.File(tmp_path / f"run_{name}.hdf5", "r") as hf:
            np.testing.assert_array_equal(hf["epochs"][()], [1, 3])  # Both recorded epochs
    with h5py.File(tmp_path / "run_ReverseChain.hdf5", "r") as hf:
        errors = hf["errors"][()]
    np.testing.assert_array_equal(errors[0], errors[1])  # The same model and noise path each epoch
