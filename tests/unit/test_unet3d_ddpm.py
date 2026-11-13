"""unet3d_ddpm: noise schedules, time conditioning, the denoising U-Net and the diffusion process."""

import pytest
import torch
from conftest import BATCH_SIZE, INPUT_SHAPE
from torch import nn

from conngafit.models import unet3d_ddpm as ddpm

N_TIMESTEPS = 10
N_SAMPLES = 3
N_MAPS = 2
# float32 convolutions sum in a different order for different batch sizes
BATCH_RTOL, BATCH_ATOL = 1e-4, 1e-4
OUTPUT_SHAPE = INPUT_SHAPE[1:]
DENOISER_KWARGS = dict(
    input_shape=INPUT_SHAPE, n_filt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), n_stages=3, n_head_blocks=2
)


@pytest.fixture
def make_denoiser():
    """Factory for a small DenoisingUNet3d; keyword arguments override the defaults."""

    def make(**overrides):
        return ddpm.DenoisingUNet3d(**{**DENOISER_KWARGS, **overrides})

    return make


@pytest.fixture
def make_diffusion(make_denoiser):
    """Factory for a GaussianDiffusion around a small denoiser of n_maps maps of output_shape; other keyword arguments
    go to GaussianDiffusion."""

    def make(output_shape=None, n_maps=1, **kwargs):
        denoiser = make_denoiser(output_shape=output_shape, n_maps=n_maps)
        return ddpm.GaussianDiffusion(denoiser, n_timesteps=N_TIMESTEPS, **kwargs)

    return make


def randomize_output(denoiser):
    """Replace the zero-initialised output conv with random weights so the prediction depends on the input."""
    nn.init.normal_(denoiser.head.out.weight)
    nn.init.normal_(denoiser.head.out.bias)
    return denoiser


def denoiser_inputs(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, n_maps=1):
    cube = torch.randn(BATCH_SIZE, *input_shape)
    y_t = torch.randn(BATCH_SIZE, n_maps, *output_shape)
    t = torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,))
    return cube, y_t, t


#### Noise schedule ####
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_beta_schedule_is_a_valid_noising_process(schedule):
    betas = ddpm.make_beta_schedule(schedule, N_TIMESTEPS)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    assert betas.shape == (N_TIMESTEPS,)  # one beta per timestep
    assert ((betas > 0) & (betas < 1)).all()  # every step adds some noise, but never all of it
    assert (alpha_bar[1:] < alpha_bar[:-1]).all()  # remaining signal strictly decreases with t


def test_linear_schedule_matches_ho_et_al():
    betas = ddpm.make_beta_schedule("linear", 1000)
    torch.testing.assert_close(betas[[0, -1]], torch.tensor([1e-4, 2e-2], dtype=torch.float64))


def test_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        ddpm.make_beta_schedule("quadratic", N_TIMESTEPS)


#### Time conditioning ####
@pytest.mark.parametrize("dim", [8, 9])
def test_timestep_embedding_shape_and_dependence_on_t(dim):
    out_dim = 16
    emb = ddpm.TimestepEmbedding(dim, out_dim)(torch.tensor([0, N_TIMESTEPS - 1]))
    assert emb.shape == (2, out_dim)  # one embedding per timestep, odd dims included
    assert not torch.allclose(emb[0], emb[1])  # different timesteps give different embeddings


#### Denoising network ####
@pytest.mark.parametrize("upsample_mode", ["trilinear", "conv_transpose"])
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        (INPUT_SHAPE, OUTPUT_SHAPE),
        (INPUT_SHAPE, (8, 8)),
        ((13, 10, 11), (20, 24)),
    ],
)
@pytest.mark.parametrize("n_maps", [1, N_MAPS])
def test_denoiser_predicts_noise_on_the_map_grid(make_denoiser, upsample_mode, input_shape, output_shape, n_maps):
    model = make_denoiser(
        upsample_mode=upsample_mode, input_shape=input_shape, output_shape=output_shape, n_maps=n_maps
    )
    cube, y_t, t = denoiser_inputs(input_shape, output_shape, n_maps)
    assert model(cube, y_t, t).shape == y_t.shape  # one noise map per input map


def test_untrained_denoiser_predicts_zero_noise(make_denoiser):
    assert not make_denoiser()(*denoiser_inputs()).any()


@pytest.mark.parametrize("option", [dict(n_stages=1), dict(n_head_blocks=0), dict(input_shape=OUTPUT_SHAPE)])
def test_denoiser_rejects_invalid_settings(make_denoiser, option):
    with pytest.raises(ValueError):
        make_denoiser(**option)


def test_denoiser_rejects_cube_too_small_for_its_depth(make_denoiser):
    with pytest.raises(ValueError, match="n_stages"):
        make_denoiser(n_stages=4)


@pytest.mark.parametrize(
    "input_shape, output_shape, n_maps",
    [((13, 10, 11), OUTPUT_SHAPE, 1), (INPUT_SHAPE, (8, 8), 1), (INPUT_SHAPE, OUTPUT_SHAPE, N_MAPS)],
    ids=["cube", "map", "n_maps"],
)
def test_denoiser_rejects_inputs_of_another_shape(make_denoiser, input_shape, output_shape, n_maps):
    with pytest.raises(ValueError, match="built for"):
        make_denoiser()(*denoiser_inputs(input_shape, output_shape, n_maps))


def test_denoiser_uses_no_batchnorm(make_denoiser):
    assert not any(isinstance(m, nn.BatchNorm3d) for m in make_denoiser().modules())


def test_denoiser_samples_are_independent_of_the_batch(make_denoiser):
    model = randomize_output(make_denoiser())
    cube, y_t, t = denoiser_inputs()
    torch.testing.assert_close(
        model(cube[:1], y_t[:1], t[:1]), model(cube, y_t, t)[:1], rtol=BATCH_RTOL, atol=BATCH_ATOL
    )


@pytest.mark.parametrize("n_maps", [1, N_MAPS])
def test_denoiser_backward_reaches_every_parameter(make_denoiser, n_maps):
    model = randomize_output(make_denoiser(n_maps=n_maps))
    model(*denoiser_inputs(n_maps=n_maps)).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name  # parameter is connected to the output
        assert torch.isfinite(param.grad).all(), name  # no NaN/inf gradients


#### Diffusion process ####
def test_alpha_bar_is_a_saved_buffer(make_diffusion):
    diffusion = make_diffusion()
    assert "alpha_bar" in diffusion.state_dict()  # saved with checkpoints and moved by .to(device)
    assert diffusion.alpha_bar.shape == (N_TIMESTEPS,)  # one value per timestep


def test_sample_timesteps_are_uniform_over_all_timesteps(make_diffusion):
    torch.manual_seed(0)
    diffusion = make_diffusion()
    n_draws = 3000  # batches of 3, so the strata don't line up with the timesteps
    t = torch.cat([diffusion.sample_timesteps(3) for _ in range(n_draws)])
    counts = torch.bincount(t, minlength=N_TIMESTEPS).float()
    expected = len(t) / N_TIMESTEPS
    assert ((counts - expected).abs() < 0.1 * expected).all()  # every timestep drawn about equally often


def test_q_sample_mixes_signal_and_noise(make_diffusion):
    diffusion = make_diffusion()
    y0, noise = torch.randn(BATCH_SIZE, 1, *OUTPUT_SHAPE), torch.randn(BATCH_SIZE, 1, *OUTPUT_SHAPE)
    t = torch.tensor([0, N_TIMESTEPS - 1])
    ab = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
    torch.testing.assert_close(diffusion.q_sample(y0, t, noise), ab.sqrt() * y0 + (1 - ab).sqrt() * noise)


@pytest.mark.parametrize("n_maps", [1, N_MAPS])
def test_predicted_noise_matches_the_maps(make_diffusion, n_maps):
    diffusion = make_diffusion(n_maps=n_maps)
    y0 = torch.randn(BATCH_SIZE, n_maps, *OUTPUT_SHAPE)
    pred_noise, noise = diffusion(torch.randn(BATCH_SIZE, *INPUT_SHAPE), y0)
    assert pred_noise.shape == noise.shape == y0.shape  # one predicted noise value per pixel of every map
    assert torch.isfinite(pred_noise).all()  # no NaN/inf from the noising or denoiser


@pytest.mark.parametrize(
    "output_shape, n_maps, expected",
    [
        (None, 1, (N_SAMPLES, BATCH_SIZE, 1, *OUTPUT_SHAPE)),
        ((8, 8), 1, (N_SAMPLES, BATCH_SIZE, 1, 8, 8)),
        ((8, 8), N_MAPS, (N_SAMPLES, BATCH_SIZE, N_MAPS, 8, 8)),
    ],
)
@pytest.mark.parametrize("sample_kw", [dict(), dict(max_batch=2)])
def test_sample_shape(make_diffusion, output_shape, n_maps, expected, sample_kw):
    diffusion = make_diffusion(output_shape=output_shape, n_maps=n_maps)
    assert diffusion.sample(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES, **sample_kw).shape == expected


def test_samples_respect_clip_range(make_diffusion):
    clip_range = (-1.0, 1.0)
    diffusion = make_diffusion(clip_range=clip_range)
    randomize_output(diffusion.denoiser)
    samples = diffusion.sample(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES)
    assert samples.min() >= clip_range[0] and samples.max() <= clip_range[1]


@pytest.mark.parametrize("n_maps", [1, N_MAPS])
def test_predict_with_uncertainty_shapes(make_diffusion, n_maps):
    diffusion = make_diffusion(n_maps=n_maps)
    mean, std = diffusion.predict_with_uncertainty(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES)
    assert mean.shape == std.shape == (BATCH_SIZE, n_maps, *OUTPUT_SHAPE)  # one posterior mean and spread per pixel
    assert (std > 0).any()  # the posterior samples differ


def test_sampling_denoises_every_sample_at_every_timestep(make_diffusion):
    diffusion = make_diffusion()
    timesteps = []
    diffusion.denoiser.register_forward_hook(lambda m, inputs, out: timesteps.append(inputs[2].tolist()))
    diffusion.predict_with_uncertainty(torch.randn(BATCH_SIZE, *INPUT_SHAPE), N_SAMPLES)
    # one call per timestep, from T - 1 down to 0, each denoising every sample of every cube
    assert timesteps == [[t] * N_SAMPLES * BATCH_SIZE for t in reversed(range(N_TIMESTEPS))]


def test_sample_callback_sees_every_step_without_changing_the_samples(make_diffusion):
    diffusion = make_diffusion(n_maps=N_MAPS, clip_range=(-1.0, 1.0))
    randomize_output(diffusion.denoiser)
    cube = torch.randn(BATCH_SIZE, *INPUT_SHAPE)
    steps = []

    def callback(t, y_t, y0_estimate):
        steps.append((t, y_t.shape, y0_estimate.shape, y0_estimate.abs().max().item()))

    torch.manual_seed(0)
    samples = diffusion.sample(cube, n_samples=N_SAMPLES, callback=callback)
    torch.manual_seed(0)
    assert torch.equal(diffusion.sample(cube, n_samples=N_SAMPLES), samples)
    # one call per timestep, from T - 1 down to 0, with the maps of every sample and the clipped estimate
    shape = (N_SAMPLES * BATCH_SIZE, N_MAPS, *OUTPUT_SHAPE)
    assert [step[:3] for step in steps] == [(t, shape, shape) for t in reversed(range(N_TIMESTEPS))]
    assert max(step[3] for step in steps) <= 1.0
