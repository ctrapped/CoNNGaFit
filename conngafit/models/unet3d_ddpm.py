import math

import torch
import torch.nn.functional as F
from torch import nn

from conngafit.models.blocks import (
    DeconvBlock,
    ResBlock2d,
    ResidualBlock,
    SpectralAttentionPool,
    Stem,
    Upsample,
    _match_spatial,
    check_stage_compatibility,
)

# Conditional Denoising Diffusion Probabilistic Model (DDPM) for HI 21cm spectral datacubes.
# Architecture inspired from Xu et al. (2023).
# Learns p(maps | cube): one or more 2-d target maps (e.g. mass and rotational momentum) are diffused with
# Gaussian noise and a 3-d residual U-Net, built from the shared blocks in blocks.py, learns to remove it
# while looking at the datacube. Sampling many maps per cube gives their joint posterior (mean + uncertainty).
#
#   DenoisingUNet3d   - the noise-prediction network eps_theta(cube, y_t, t)
#   GaussianDiffusion - noise schedule, training loss, and DDPM sampling
#
# Like unet3d_resblock.NeuralNetwork, the denoiser is built for one (nSpec, nX, nY) cube size, (nX, nY) target map size
# and number of maps.


#### Noise schedule ####
def make_beta_schedule(name, n_timesteps):
    """Per-step noise variances beta_t.
    'linear' - Ho et al. (2020). 'cosine' - Nichol & Dhariwal (2021); destroys signal more
    gradually, which helps for small, low-resolution maps. Both are clamped to beta <= 0.999 so
    alpha_bar stays positive; the linear schedule's 1000/T scaling otherwise reaches beta >= 1
    for T <= 20."""
    if name == "linear":
        scale = 1000.0 / n_timesteps
        betas = torch.linspace(scale * 1e-4, scale * 2e-2, n_timesteps, dtype=torch.float64)
        return betas.clamp(max=0.999)
    if name == "cosine":
        s = 0.008
        steps = torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps
        alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule {name!r}; use 'linear' or 'cosine'")


#### Timestep embedding ####
class TimestepEmbedding(nn.Module):
    """Sinusoidal encoding of t (as in Transformers) followed by an MLP.
    Gives the network a smooth, learnable representation of the noise level.

    dim // 2 frequencies are spaced geometrically from 1 down to 1/10000; each contributes sin(t*f)
    and cos(t*f), so low frequencies separate distant timesteps and high ones separate neighbours.
    An odd dim gets one zero-padded entry. out_dim: width of the embedding passed to every block."""

    def __init__(self, dim, out_dim, activation=nn.SiLU):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), activation(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]  # (B, dim // 2): every timestep times every frequency
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2:  # pad odd dims
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


#### 3-d -> 2-d head ####
class MapHead(nn.Module):
    """Turn 3-d decoder features into 2-d noise predictions, one per map, at the target map resolution.

    The stem works at 1/4 resolution, but eps is pixel-scale white noise, so the noisy maps y_t
    are re-injected here at full resolution. The U-Net supplies the physics (what the cube says
    about each pixel); this skip supplies the pixel-level noise pattern to subtract.
    """

    def __init__(self, channels, t_dim, n_blocks=2, activation=nn.SiLU, max_groups=8, n_maps=1):
        super().__init__()
        self.pool = SpectralAttentionPool(channels)
        self.blocks = nn.ModuleList(
            [
                ResBlock2d(channels + n_maps if i == 0 else channels, channels, activation, max_groups, t_dim=t_dim)
                for i in range(n_blocks)
            ]
        )
        self.out = nn.Conv2d(channels, n_maps, 3, padding=1)
        nn.init.zeros_(self.out.weight)  # start by predicting eps = 0 -> stable early training
        nn.init.zeros_(self.out.bias)

    def forward(self, feats, y_t, t_emb):
        h = self.pool(feats)  # (B, C, x', y')
        h = F.interpolate(
            h, size=y_t.shape[-2:], mode="bilinear", align_corners=False
        )  # interpolate the pooled features to the full map resolution
        h = torch.cat([h, y_t], dim=1)  # full-res skip of y_t (B, C+n_maps, H, W)
        for blk in self.blocks:
            h = blk(h, t_emb)  # first block (B, C+n_maps, H, W) -> (B, C, H, W), every later block keeps (B, C, H, W)
        return self.out(h)  # (B, n_maps, H, W)


#### Noise-prediction network ####
class DenoisingUNet3d(nn.Module):
    """eps_theta(cube, y_t, t): predicts the noise added to n_maps 2-d maps, conditioned on a 3-d cube.

    Layout mirrors unet3d_resblock.NeuralNetwork, minus its fixed-size FC head:
      input  - cube (1 ch) + noisy maps resized to (nX, nY) and tiled along nSpec (n_maps ch)
      stem   - Stem with 1 + n_maps input channels, every axis -> ceil(N/4)
      encoder- n_stages stages of 2 ResidualBlocks; channels n_filt0 * 2**i, axes halve after stage 0
      decoder- n_stages-1 stages: Upsample -> _match_spatial -> concat skip -> 2 DeconvBlocks
      head   - MapHead: spectral attention pool -> resize to map size -> 2-d res blocks -> eps
    Every block uses GroupNorm (BatchNorm would mix samples at different noise levels) and is
    FiLM-conditioned on the timestep embedding. The cube and map sizes are fixed at
    construction: input_shape is checked against n_stages (check_stage_compatibility) and forward()
    rejects any other size. No weight depends on them, so a trained denoiser's state_dict loads into
    one built for another size (with the same n_maps).

    Parameters:
        input_shape: (nSpec, nX, nY) of one cube.
        n_filt0: channels of encoder stage 0 (doubles per stage).
        kernel_stem, kernel_coder: 3-tuple kernel sizes for the stem and all encoder/decoder blocks.
        n_stages: encoder depth (>= 2).
        t_dim: timestep-embedding width (default 4 * n_filt0).
        n_head_blocks: number of full-resolution 2-d blocks in the head.
        block_dropout_rate: channel-wise dropout inside every 3-d block.
        stem_activation, block_activation: zero-arg callables returning nn.Modules.
        upsample_mode: 'trilinear' (default; no checkerboard artifacts, which would look like
            structured noise to the model) or 'conv_transpose'.
        max_groups: upper bound on GroupNorm groups (actual = gcd(channels, max_groups)).
        output_shape: (H, W) of each map (default (nX, nY) from input).
        n_maps: number of maps denoised together (default 1).
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_stem,
        kernel_coder,
        n_stages=4,
        t_dim=None,
        n_head_blocks=2,
        block_dropout_rate=0.0,
        stem_activation=nn.LeakyReLU,
        block_activation=nn.SiLU,
        upsample_mode="trilinear",
        max_groups=8,
        output_shape=None,
        n_maps=1,
    ):
        super().__init__()
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 (need at least one skip connection)")
        if n_head_blocks < 1:
            raise ValueError("n_head_blocks must be >= 1 (the first block maps C+n_maps -> C channels)")
        input_shape = tuple(int(v) for v in input_shape)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
        self.input_shape = input_shape
        if output_shape is None:
            output_shape = input_shape[1:]
        self.output_shape = tuple(int(v) for v in output_shape)
        self.n_maps = n_maps
        check_stage_compatibility(input_shape, n_stages)  # Raises if the cube is too small
        self.n_stages = n_stages
        t_dim = t_dim or 4 * n_filt0
        self.n_filt0, self.t_dim, self.max_groups = n_filt0, t_dim, max_groups
        self.kernel_stem, self.kernel_coder = tuple(kernel_stem), tuple(kernel_coder)
        block_kw = dict(
            kernel1=self.kernel_coder,
            dropout_rate=block_dropout_rate,
            activation=block_activation,
            max_groups=max_groups,
            t_dim=t_dim,
        )

        self.time_embed = TimestepEmbedding(n_filt0, t_dim)
        self.stem = Stem(n_filt0, self.kernel_stem, stem_activation, in_channels=1 + n_maps)

        # Encoder: stage 0 keeps the stem's shape, later stages halve every axis and double channels.
        enc_ch = [n_filt0 * 2**i for i in range(n_stages)]
        self.encoder = nn.ModuleList()
        prev = n_filt0
        for i, ch in enumerate(enc_ch):
            self.encoder.append(
                nn.ModuleList(
                    [
                        ResidualBlock(prev, ch, downsample=i > 0, **block_kw),
                        ResidualBlock(ch, ch, downsample=False, **block_kw),
                    ]
                )
            )
            prev = ch

        # Decoder: upsample, concat the matching encoder skip, reduce 2C -> C.
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for j in range(n_stages - 1):
            in_ch, out_ch = enc_ch[n_stages - 1 - j], enc_ch[n_stages - 2 - j]
            self.upsamples.append(Upsample(in_ch, out_ch, kernel1=self.kernel_coder, mode=upsample_mode))
            self.decoder.append(
                nn.ModuleList(
                    [
                        DeconvBlock(2 * out_ch, out_ch, upsample=True, **block_kw),
                        DeconvBlock(out_ch, out_ch, upsample=False, **block_kw),
                    ]
                )
            )

        self.head = MapHead(n_filt0, t_dim, n_head_blocks, block_activation, max_groups, n_maps=n_maps)

    def forward(self, cube, y_t, t):
        """cube: (B, nSpec, nX, nY), with (nSpec, nX, nY) = input_shape; y_t: (B, n_maps, H, W) noisy maps, with
        (H, W) = output_shape; t: (B,) timesteps. Returns predicted noise, shape (B, n_maps, H, W)."""
        maps_shape = (self.n_maps, *self.output_shape)
        if tuple(cube.shape[1:]) != self.input_shape or tuple(y_t.shape[1:]) != maps_shape:
            raise ValueError(
                f"cube {tuple(cube.shape[1:])} and maps {tuple(y_t.shape[1:])} don't match the shapes this model "
                f"was built for: input_shape {self.input_shape}, (n_maps, *output_shape) {maps_shape}"
            )
        _, n_spec, n_x, n_y = cube.shape
        t_emb = self.time_embed(t)

        # Put the noisy maps on the cube grid (a no-op when map_shape == (nX, nY)) and tile them along
        # the spectral axis, so the stem sees the cube and the current maps at every voxel.
        y_vol = F.interpolate(y_t, size=(n_x, n_y), mode="bilinear", align_corners=False)
        y_vol = y_vol.unsqueeze(2).expand(-1, -1, n_spec, -1, -1)
        h = self.stem(torch.cat([cube.unsqueeze(1), y_vol], dim=1))

        # Keep every encoder output except the deepest; the decoder pops them in reverse order, so each
        # decoder stage pairs with the encoder stage at the same resolution.
        skips = []
        for i, (b1, b2) in enumerate(self.encoder):
            h = b2(b1(h, t_emb), t_emb)
            if i < self.n_stages - 1:
                skips.append(h)

        for up, (b1, b2) in zip(self.upsamples, self.decoder):
            skip = skips.pop()
            h = torch.cat([_match_spatial(up(h), skip), skip], dim=1)
            h = b2(b1(h, t_emb), t_emb)

        return self.head(h, y_t, t_emb)


#### Diffusion process ####
class GaussianDiffusion(nn.Module):
    """Forward noising, training objective, and reverse sampling for a conditional DDPM.

    Training: pick t ~ U{0..T-1}, form y_t = sqrt(abar_t) y_0 + sqrt(1 - abar_t) eps, and
    regress eps with MSE (Ho et al. 2020 'simple' loss).
    Sampling: start from pure noise and step back through all T steps with the learned eps (DDPM
    sampling, Ho et al. 2020).

    Parameters:
        denoiser: an eps-prediction network with input_shape, output_shape and n_maps, e.g. DenoisingUNet3d. Its
            n_maps maps are sampled together at its output_shape (H, W).
        n_timesteps: number of diffusion steps T.
        schedule: 'cosine' or 'linear' (see make_beta_schedule).
        clip_range: bounds for the predicted clean map during sampling (labels normalized with
            'symmetric' lie in [-1, 1]); None disables clipping.

    alpha_bar is a buffer saved in state_dict, so a checkpoint only loads into a model built with the
    same schedule and n_timesteps (a different n_timesteps raises a size mismatch on load).
    """

    def __init__(
        self,
        denoiser,
        n_timesteps=1000,
        schedule="cosine",
        clip_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.denoiser = denoiser
        self.n_timesteps = n_timesteps
        self.schedule = schedule
        self.output_shape = tuple(denoiser.output_shape)
        self.n_maps = denoiser.n_maps
        self.clip_range = clip_range
        alpha_bar = torch.cumprod(1 - make_beta_schedule(schedule, n_timesteps), dim=0)
        # buffers move with .to(device) and are saved in state_dict
        self.register_buffer("alpha_bar", alpha_bar.float())
        # alpha_bar_{t-1} for every t, with alpha_bar_{-1} = 1 (no noise before the first step). Derived from
        # alpha_bar, so it is not saved (persistent=False) and checkpoints only hold alpha_bar.
        self.register_buffer("alpha_bar_prev", F.pad(self.alpha_bar[:-1], (1, 0), value=1.0), persistent=False)

    def _gather(self, t, like):
        """abar_t reshaped to broadcast against (B, n_maps, H, W) maps."""
        return self.alpha_bar[t].view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, y0, t, noise):
        """Forward process: jump straight from y_0 to y_t in closed form,
        y_t = sqrt(abar_t) y_0 + sqrt(1 - abar_t) eps (Ho et al. 2020, Eq. 4)."""
        ab = self._gather(t, y0)
        return ab.sqrt() * y0 + (1 - ab).sqrt() * noise

    def forward(self, cube, y0):
        """Training loss for a batch of cubes and their (B, n_maps, H, W) maps y0: MSE between true and predicted
        noise."""
        y0 = y0.float()
        t = torch.randint(0, self.n_timesteps, (y0.shape[0],), device=y0.device)
        noise = torch.randn_like(y0)
        return F.mse_loss(self.denoiser(cube.float(), self.q_sample(y0, t, noise), t), noise)

    # TODO: add DDIM sampling (Song et al. 2021): its update depends only on alpha_bar, so it can skip timesteps
    # for faster sampling, and its eta sets the noise added at each step (eta = 1 with all T steps is DDPM; eta = 0 is
    # deterministic given the starting noise). Expose its number of steps and eta as hyperparameters, and check the
    # calibration of the posterior standard deviation for each setting.
    @torch.no_grad()
    def sample(self, cube, n_samples=1, max_batch=None):
        """Draw joint posterior samples of the maps for each cube.

        cube: (B, nSpec, nX, nY).
        max_batch: cap on cubes per denoiser call to bound memory.
        Returns (n_samples, B, n_maps, H, W), with (H, W) = output_shape.
        """
        n_batch = cube.shape[0]
        shape = (self.n_maps, *self.output_shape)
        cubes = cube.float().repeat(n_samples, 1, 1, 1)  # one copy per sample
        chunk = max_batch or cubes.shape[0]
        out = torch.cat([self._reverse(c, shape) for c in cubes.split(chunk)])
        return out.view(n_samples, n_batch, *shape)

    def _reverse(self, cube, shape):
        """Reverse diffusion from y_T ~ N(0, I) to y_0 for one batch of cubes, one step per timestep: each step
        draws y_{t-1} from the Gaussian posterior q(y_{t-1} | y_t, y_0) around the denoiser's estimate of y_0."""
        y = torch.randn(cube.shape[0], *shape, device=cube.device)
        for t in reversed(range(self.n_timesteps)):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar_prev[t]
            alpha = ab / ab_prev  # alpha_t = 1 - beta_t

            eps = self.denoiser(cube, y, torch.full((cube.shape[0],), t, device=cube.device))
            # Invert the forward process y_t = sqrt(abar_t) y_0 + sqrt(1 - abar_t) eps (Ho et al. 2020, Eq. 4)
            # for the current guess of the clean map.
            y0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()
            if self.clip_range is not None:
                y0 = y0.clamp(*self.clip_range)

            # Gaussian posterior q(y_{t-1} | y_t, y_0) (Ho et al. 2020, Eqs. 6-7): its mean weights y_0 and y_t,
            # and its variance beta~_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t is 0 at t = 0, so the last
            # step adds no noise.
            mean = ab_prev.sqrt() * (1 - alpha) / (1 - ab) * y0 + alpha.sqrt() * (1 - ab_prev) / (1 - ab) * y
            variance = (1 - ab_prev) / (1 - ab) * (1 - alpha)
            y = mean + variance.sqrt() * torch.randn_like(y)
        return y

    def predict_with_uncertainty(self, cube, n_samples=30, **sample_kw):
        """Posterior mean and standard deviation per pixel of each map from n_samples draws. sample_kw goes to
        sample()."""
        samples = self.sample(cube, n_samples=n_samples, **sample_kw)
        std, mean = torch.std_mean(samples, dim=0)  # std uses the unbiased (n_samples - 1) estimator
        return mean, std
