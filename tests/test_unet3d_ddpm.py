"""Unet3d_DDPM: noise schedules, time conditioning, the denoising U-Net and the diffusion process."""

import pytest
import torch
from conftest import BATCH_SIZE, INPUT_SHAPE
from torch import nn

import Unet3d_DDPM as ddpm

N_TIMESTEPS = 10
N_SAMPLES = 3
# float32 convolutions sum in a different order for different batch sizes
BATCH_RTOL, BATCH_ATOL = 1e-4, 1e-4
MAP_SHAPE = INPUT_SHAPE[1:]
DENOISER_KWARGS = dict(nFilt0=4, kernel_stem=(3, 3, 3), kernel_coder=(3, 3, 3), n_stages=3, n_head_blocks=2)
# stage_shapes() rows with no module of their own (inputs, concatenation, resize) are not hooked
UNHOOKED_ROWS = ("input:", "+ skip", "resize to map", "+ noisy map")


@pytest.fixture
def make_denoiser():
    """Factory for a small DenoisingUNet3d; keyword arguments override the defaults."""

    def make(**overrides):
        return ddpm.DenoisingUNet3d(**{**DENOISER_KWARGS, **overrides})

    return make


@pytest.fixture
def make_diffusion(make_denoiser):
    """Factory for a GaussianDiffusion around a small denoiser; keyword arguments go to GaussianDiffusion."""

    def make(**kwargs):
        return ddpm.GaussianDiffusion(make_denoiser(), n_timesteps=N_TIMESTEPS, **kwargs)

    return make


def randomize_output(denoiser):
    """Replace the zero-initialised output conv with random weights so the prediction depends on the input."""
    nn.init.normal_(denoiser.head.out.weight)
    nn.init.normal_(denoiser.head.out.bias)
    return denoiser


def denoiser_inputs(input_shape=INPUT_SHAPE, map_shape=MAP_SHAPE):
    cube = torch.randn(BATCH_SIZE, *input_shape)
    y_t = torch.randn(BATCH_SIZE, *map_shape)
    t = torch.randint(0, N_TIMESTEPS, (BATCH_SIZE,))
    return cube, y_t, t


#### Noise schedule ####
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_beta_schedule_is_a_valid_noising_process(schedule):
    betas = ddpm.make_beta_schedule(schedule, N_TIMESTEPS)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    assert betas.shape == (N_TIMESTEPS,)
    assert ((betas > 0) & (betas < 1)).all()
    assert (alpha_bar[1:] < alpha_bar[:-1]).all()


def test_linear_schedule_matches_ho_et_al():
    betas = ddpm.make_beta_schedule("linear", 1000)
    torch.testing.assert_close(betas[[0, -1]], torch.tensor([1e-4, 2e-2], dtype=torch.float64))


def test_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        ddpm.make_beta_schedule("quadratic", N_TIMESTEPS)


#### Time conditioning ####
@pytest.mark.parametrize("spatial", [(5, 6), (4, 5, 6)])
def test_film_with_zero_scale_and_shift_is_identity(spatial):
    channels = 3
    h = torch.randn(BATCH_SIZE, channels, *spatial)
    torch.testing.assert_close(ddpm._film(h, torch.zeros(BATCH_SIZE, 2 * channels)), h)


def test_batch_to_group_norm_replaces_every_batchnorm():
    module = nn.Sequential(nn.BatchNorm3d(6), nn.Sequential(nn.Conv3d(6, 12, 1), nn.BatchNorm3d(12)))
    ddpm._batch_to_group_norm(module, max_groups=8)
    norms = [m for m in module.modules() if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm))]
    assert [(type(m), m.num_groups) for m in norms] == [(nn.GroupNorm, 2), (nn.GroupNorm, 4)]


@pytest.mark.parametrize("dim", [8, 9])
def test_timestep_embedding_shape_and_dependence_on_t(dim):
    out_dim = 16
    emb = ddpm.TimestepEmbedding(dim, out_dim)(torch.tensor([0, N_TIMESTEPS - 1]))
    assert emb.shape == (2, out_dim)
    assert not torch.allclose(emb[0], emb[1])


#### Denoising network ####
@pytest.mark.parametrize("upsample_mode", ["trilinear", "conv_transpose"])
@pytest.mark.parametrize(
    "input_shape, map_shape",
    [
        (INPUT_SHAPE, MAP_SHAPE),
        (INPUT_SHAPE, (8, 8)),
        ((13, 10, 11), (20, 24)),
    ],
)
def test_denoiser_predicts_noise_on_the_map_grid(make_denoiser, upsample_mode, input_shape, map_shape):
    cube, y_t, t = denoiser_inputs(input_shape, map_shape)
    assert make_denoiser(upsample_mode=upsample_mode)(cube, y_t, t).shape == y_t.shape


def test_untrained_denoiser_predicts_zero_noise(make_denoiser):
    assert not make_denoiser()(*denoiser_inputs()).any()


@pytest.mark.parametrize("option", [dict(n_stages=1), dict(n_head_blocks=0)])
def test_denoiser_rejects_invalid_depths(make_denoiser, option):
    with pytest.raises(ValueError):
        make_denoiser(**option)


def test_denoiser_rejects_cube_too_small_for_its_depth(make_denoiser):
    with pytest.raises(ValueError, match="n_stages"):
        make_denoiser(n_stages=4)(*denoiser_inputs())


def test_denoiser_uses_no_batchnorm(make_denoiser):
    assert not any(isinstance(m, nn.BatchNorm3d) for m in make_denoiser().modules())


def test_denoiser_samples_are_independent_of_the_batch(make_denoiser):
    model = randomize_output(make_denoiser())
    cube, y_t, t = denoiser_inputs()
    torch.testing.assert_close(
        model(cube[:1], y_t[:1], t[:1]), model(cube, y_t, t)[:1], rtol=BATCH_RTOL, atol=BATCH_ATOL
    )


def test_denoiser_backward_reaches_every_parameter(make_denoiser):
    model = randomize_output(make_denoiser())
    model(*denoiser_inputs()).square().mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("upsample_mode", ["trilinear", "conv_transpose"])
@pytest.mark.parametrize("map_shape", [MAP_SHAPE, (8, 8)])
def test_denoiser_stage_shapes_match_forward_pass(make_denoiser, upsample_mode, map_shape):
    model = make_denoiser(upsample_mode=upsample_mode)
    stages = [model.stem, *(blocks[-1] for blocks in model.encoder)]
    for up, blocks in zip(model.upsamples, model.decoder):
        stages += [up, blocks[-1]]
    stages += [model.head.pool, *model.head.blocks, model.head.out]
    recorded = []
    for module in stages:
        module.register_forward_hook(lambda m, inputs, out: recorded.append((out.shape[1], tuple(out.shape[2:]))))
    model(*denoiser_inputs(map_shape=map_shape))
    expected = [
        (feats, tuple(shape))
        for label, feats, shape in model.stage_shapes(INPUT_SHAPE, map_shape)
        if not any(tag in label for tag in UNHOOKED_ROWS)
    ]
    assert recorded == expected


def test_denoiser_param_breakdown_accounts_for_every_parameter(make_denoiser):
    model = make_denoiser()
    assert sum(n for _, n in model.param_breakdown(INPUT_SHAPE)) == sum(p.numel() for p in model.parameters())


#### Diffusion process ####
def test_alpha_bar_is_a_saved_decreasing_buffer(make_diffusion):
    diffusion = make_diffusion()
    assert "alpha_bar" in diffusion.state_dict()
    assert diffusion.alpha_bar.shape == (N_TIMESTEPS,)
    assert (diffusion.alpha_bar[1:] < diffusion.alpha_bar[:-1]).all()


def test_q_sample_mixes_signal_and_noise(make_diffusion):
    diffusion = make_diffusion()
    y0, noise = torch.randn(BATCH_SIZE, *MAP_SHAPE), torch.randn(BATCH_SIZE, *MAP_SHAPE)
    t = torch.tensor([0, N_TIMESTEPS - 1])
    ab = diffusion.alpha_bar[t].view(-1, 1, 1)
    torch.testing.assert_close(diffusion.q_sample(y0, t, noise), ab.sqrt() * y0 + (1 - ab).sqrt() * noise)


@pytest.mark.parametrize("flatten", [False, True])
def test_training_loss_is_a_finite_scalar(make_diffusion, flatten):
    diffusion = make_diffusion(target_shape=MAP_SHAPE if flatten else None)
    y0 = torch.randn(BATCH_SIZE, *MAP_SHAPE)
    loss = diffusion(torch.randn(BATCH_SIZE, *INPUT_SHAPE), y0.flatten(1) if flatten else y0)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_flattened_labels_need_target_shape(make_diffusion):
    with pytest.raises(ValueError, match="target_shape"):
        make_diffusion()(torch.randn(BATCH_SIZE, *INPUT_SHAPE), torch.randn(BATCH_SIZE, MAP_SHAPE[0] * MAP_SHAPE[1]))


@pytest.mark.parametrize(
    "target_shape, expected",
    [
        (None, (N_SAMPLES, BATCH_SIZE, *MAP_SHAPE)),
        ((8, 8), (N_SAMPLES, BATCH_SIZE, 64)),
    ],
)
@pytest.mark.parametrize("sample_kw", [dict(), dict(n_steps=4, eta=0.0), dict(max_batch=2)])
def test_sample_shape(make_diffusion, target_shape, expected, sample_kw):
    diffusion = make_diffusion(target_shape=target_shape)
    assert diffusion.sample(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES, **sample_kw).shape == expected


def test_samples_respect_clip_range(make_diffusion):
    clip_range = (-1.0, 1.0)
    diffusion = make_diffusion(clip_range=clip_range)
    randomize_output(diffusion.denoiser)
    samples = diffusion.sample(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES)
    assert samples.min() >= clip_range[0] and samples.max() <= clip_range[1]


def test_predict_with_uncertainty_shapes(make_diffusion):
    mean, std = make_diffusion().predict_with_uncertainty(torch.randn(BATCH_SIZE, *INPUT_SHAPE), n_samples=N_SAMPLES)
    assert mean.shape == std.shape == (BATCH_SIZE, *MAP_SHAPE)
    assert (std > 0).any()


def test_summary_reports_parameter_count(make_diffusion):
    diffusion = make_diffusion()
    lines = []
    diffusion.summary(INPUT_SHAPE, print_fn=lines.append)
    n_params = sum(p.numel() for p in diffusion.denoiser.parameters())
    assert any(f"{n_params:,} total" in line for line in lines)
