import math

import torch
import torch.nn.functional as F
from torch import nn

from unet3d_resblock import _check_stage_compatibility
from unet_blocks import DeconvBlock, ResidualBlock, SpectralAttentionPool, Stem, Upsample, _match_spatial
from unet_diagnostics import count_params, layer_param_lines

# Conditional Denoising Diffusion Probabilistic Model (DDPM) for HI 21cm spectral datacubes.
# Architecture inspired from Xu et al. (2023).
# Learns p(map | cube): a 2-d target map (e.g. radial mass flux) is diffused with Gaussian noise
# and a 3-d residual U-Net, built from the shared blocks in unet_blocks.py, learns to remove it while
# looking at the datacube. Sampling many maps per cube gives a posterior (mean + uncertainty).
#
#   DenoisingUNet3d   - the noise-prediction network eps_theta(cube, y_t, t)
#   GaussianDiffusion - noise schedule, training loss, and DDPM/DDIM sampling
#
# Works for any (nSpec, nX, nY) cube and any target map size; nothing is fixed at construction.


#### Small helpers ####
def _film(h, scale_shift):
    """FiLM: h * (1 + scale) + shift, with per-channel scale/shift broadcast over all spatial axes.
    This is how the timestep tells each block how much noise to expect."""
    scale, shift = scale_shift.chunk(2, dim=1)
    view = scale.shape + (1,) * (h.dim() - 2)
    return h * (1 + scale.view(view)) + shift.view(view)


def _batch_to_group_norm(module, max_groups):
    """Swap every BatchNorm3d in `module` for GroupNorm, in place.
    BatchNorm statistics would mix samples at different noise levels within a batch and break at
    batch size 1 during sampling; GroupNorm normalizes each sample independently."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(math.gcd(c, max_groups), c))
        else:
            _batch_to_group_norm(child, max_groups)


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
    Gives the network a smooth, learnable representation of the noise level."""

    def __init__(self, dim, out_dim, activation=nn.SiLU):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), activation(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2:  # pad odd dims
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


#### Time-conditioned versions of the shared blocks ####
class ConditionalStem(Stem):
    """The shared Stem (ceil(N/4) per axis), but accepting several input channels:
    the datacube plus the noisy map broadcast along the spectral axis."""

    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super().__init__(out_channels, kernel_size, activation)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=0)


class TimeResidualBlock(ResidualBlock):
    """Encoder ResidualBlock with GroupNorm and a FiLM time injection after the first conv."""

    def __init__(
        self, in_channels, out_channels, downsample, kernel1, t_dim, dropout_rate=0.0, activation=nn.SiLU, max_groups=8
    ):
        super().__init__(in_channels, out_channels, downsample, kernel1, dropout_rate, activation)
        _batch_to_group_norm(self, max_groups)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))

    def forward(self, x, t_emb):
        shortcut = self.shortcut(x)
        if self.downsample:
            x = F.pad(x, self.down_pad)  # lands every axis on ceil(N/2)
        x = self.act1(_film(self.bn1(self.conv1(x)), self.time_proj(t_emb)))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.dropout(self.act3(x + shortcut))


class TimeDeconvBlock(DeconvBlock):
    """Decoder DeconvBlock with GroupNorm and a FiLM time injection after the first conv."""

    def __init__(
        self, in_channels, out_channels, upsample, kernel1, t_dim, dropout_rate=0.0, activation=nn.SiLU, max_groups=8
    ):
        super().__init__(in_channels, out_channels, upsample, kernel1, dropout_rate, activation)
        _batch_to_group_norm(self, max_groups)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))

    def forward(self, x, t_emb):
        shortcut = self.shortcut(x)  # carries the target spatial shape
        x = self.act1(_film(self.bn1(_match_spatial(self.convtran1(x), shortcut)), self.time_proj(t_emb)))
        x = self.act2(self.bn2(_match_spatial(self.convtran2(x), shortcut)))
        return self.dropout(self.act3(x + shortcut))


#### 3-d -> 2-d head ####
class TimeResBlock2d(nn.Module):
    """2-d residual block with GroupNorm + FiLM time conditioning, used at full map resolution."""

    def __init__(self, in_channels, out_channels, t_dim, activation=nn.SiLU, max_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(math.gcd(out_channels, max_groups), out_channels)
        self.norm2 = nn.GroupNorm(math.gcd(out_channels, max_groups), out_channels)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, 2 * out_channels))
        self.act1, self.act2 = activation(), activation()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act1(_film(self.norm1(self.conv1(x)), self.time_proj(t_emb)))
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class MapHead(nn.Module):
    """Turn 3-d decoder features into a 2-d noise prediction at the target map resolution.

    The stem works at 1/4 resolution, but eps is pixel-scale white noise, so the noisy map y_t
    is re-injected here at full resolution. The U-Net supplies the physics (what the cube says
    about each pixel); this skip supplies the pixel-level noise pattern to subtract.
    """

    def __init__(self, channels, t_dim, n_blocks=2, activation=nn.SiLU, max_groups=8):
        super().__init__()
        self.pool = SpectralAttentionPool(channels)
        self.blocks = nn.ModuleList(
            [
                TimeResBlock2d(channels + 1 if i == 0 else channels, channels, t_dim, activation, max_groups)
                for i in range(n_blocks)
            ]
        )
        self.out = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)  # start by predicting eps = 0 -> stable early training
        nn.init.zeros_(self.out.bias)

    def forward(self, feats, y_t, t_emb):
        h = self.pool(feats)  # (B, C, x', y')
        h = F.interpolate(
            h, size=y_t.shape[-2:], mode="bilinear", align_corners=False
        )  # interpolate the pooled features to the full map resolution
        h = torch.cat([h, y_t], dim=1)  # full-res skip of y_t (B, C+1, H, W)
        for blk in self.blocks:
            h = blk(h, t_emb)  # first block (B, C+1, H, W) -> (B, C, H, W), every subsequent block keeps (B, C, H, W)
        return self.out(h)  # (B, 1, H, W)


#### Noise-prediction network ####
class DenoisingUNet3d(nn.Module):
    """eps_theta(cube, y_t, t): predicts the noise added to a 2-d map, conditioned on a 3-d cube.

    Layout mirrors unet3d_resblock.NeuralNetwork, minus its fixed-size FC head:
      input  - cube (1 ch) + noisy map resized to (nX, nY) and tiled along nSpec (1 ch)
      stem   - ConditionalStem, every axis -> ceil(N/4)
      encoder- n_stages stages of 2 TimeResidualBlocks; channels nFilt0 * 2**i, axes halve after stage 0
      decoder- n_stages-1 stages: Upsample -> _match_spatial -> concat skip -> 2 TimeDeconvBlocks
      head   - MapHead: spectral attention pool -> resize to map size -> 2-d res blocks -> eps
    Every block is FiLM-conditioned on the timestep embedding. Fully convolutional, so the cube
    and map sizes are only checked at run time (_check_stage_compatibility).

    Parameters:
        nFilt0: channels of encoder stage 0 (doubles per stage).
        kernel_stem, kernel_coder: 3-tuple kernel sizes for the stem and all encoder/decoder blocks.
        n_stages: encoder depth (>= 2).
        t_dim: timestep-embedding width (default 4 * nFilt0).
        n_head_blocks: number of full-resolution 2-d blocks in the head.
        block_dropout_rate: channel-wise dropout inside every 3-d block.
        stem_activation, block_activation: zero-arg callables returning nn.Modules.
        upsample_mode: 'trilinear' (default; no checkerboard artifacts, which would look like
            structured noise to the model) or 'conv_transpose'.
        max_groups: upper bound on GroupNorm groups (actual = gcd(channels, max_groups)).
    """

    def __init__(
        self,
        nFilt0,
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
    ):
        super().__init__()
        if n_stages < 2:
            raise ValueError("n_stages must be >= 2 (need at least one skip connection)")
        if n_head_blocks < 1:
            raise ValueError("n_head_blocks must be >= 1 (the first block maps C+1 -> C channels)")
        self.n_stages = n_stages
        t_dim = t_dim or 4 * nFilt0
        # kept for stage_shapes()/summary()
        self.nFilt0, self.t_dim, self.max_groups = nFilt0, t_dim, max_groups
        self.kernel_stem, self.kernel_coder = tuple(kernel_stem), tuple(kernel_coder)
        block_kw = dict(
            kernel1=tuple(kernel_coder),
            t_dim=t_dim,
            dropout_rate=block_dropout_rate,
            activation=block_activation,
            max_groups=max_groups,
        )

        self.time_embed = TimestepEmbedding(nFilt0, t_dim)
        self.stem = ConditionalStem(2, nFilt0, tuple(kernel_stem), stem_activation)

        # Encoder: stage 0 keeps the stem's shape, later stages halve every axis and double channels.
        enc_ch = [nFilt0 * 2**i for i in range(n_stages)]
        self.enc_channels = list(enc_ch)
        self.encoder = nn.ModuleList()
        prev = nFilt0
        for i, ch in enumerate(enc_ch):
            self.encoder.append(
                nn.ModuleList(
                    [
                        TimeResidualBlock(prev, ch, downsample=i > 0, **block_kw),
                        TimeResidualBlock(ch, ch, downsample=False, **block_kw),
                    ]
                )
            )
            prev = ch

        # Decoder: upsample, concat the matching encoder skip, reduce 2C -> C.
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for j in range(n_stages - 1):
            in_ch, out_ch = enc_ch[n_stages - 1 - j], enc_ch[n_stages - 2 - j]
            self.upsamples.append(Upsample(in_ch, out_ch, kernel1=tuple(kernel_coder), mode=upsample_mode))
            self.decoder.append(
                nn.ModuleList(
                    [
                        TimeDeconvBlock(2 * out_ch, out_ch, upsample=True, **block_kw),
                        TimeDeconvBlock(out_ch, out_ch, upsample=False, **block_kw),
                    ]
                )
            )

        self.head = MapHead(nFilt0, t_dim, n_head_blocks, block_activation, max_groups)

    def forward(self, cube, y_t, t):
        """cube: (B, nSpec, nX, nY); y_t: (B, H, W) noisy map; t: (B,) timesteps.
        Returns predicted noise, shape (B, H, W)."""
        _, nSpec, nX, nY = cube.shape
        _check_stage_compatibility((nSpec, nX, nY), self.n_stages)
        t_emb = self.time_embed(t)

        # Put the noisy map on the cube grid and tile it along the spectral axis.
        y_img = y_t.unsqueeze(1)
        # Interpolate noisy map to the cube's spatial resolution if resolutions differ.
        y_vol = F.interpolate(y_img, size=(nX, nY), mode="bilinear", align_corners=False)
        y_vol = y_vol.unsqueeze(2).expand(-1, -1, nSpec, -1, -1)
        h = self.stem(torch.cat([cube.unsqueeze(1), y_vol], dim=1))

        skips = []
        for i, (b1, b2) in enumerate(self.encoder):
            h = b2(b1(h, t_emb), t_emb)
            if i < self.n_stages - 1:
                skips.append(h)

        for up, (b1, b2) in zip(self.upsamples, self.decoder):
            skip = skips.pop()
            h = torch.cat([_match_spatial(up(h), skip), skip], dim=1)
            h = b2(b1(h, t_emb), t_emb)

        return self.head(h, y_img, t_emb).squeeze(1)

    def stage_shapes(self, input_shape, map_shape=None):
        """Analytically compute the tensor shape at every stage (no forward pass).

        The network is fully convolutional, so shapes depend on the input: `input_shape` is
        (nSpec, nX, nY) of one cube and `map_shape` is (H, W) of the noisy map (default (nX, nY)).
        Returns a list of (label, n_features, dims) tuples in execution order, matching forward():
        stem -> ceil(N/4) per axis, each later encoder stage -> ceil(N/2), each decoder stage
        upsamples (2N-1 for 'conv_transpose', 2N for 'trilinear') then _match_spatial realigns it
        to the encoder skip map, and the head collapses the spectral axis and resizes to map_shape.
        Raises ValueError if the cube is too small for n_stages (see _check_stage_compatibility).
        """
        input_shape = tuple(int(v) for v in input_shape)
        map_shape = tuple(int(v) for v in map_shape) if map_shape is not None else input_shape[1:]
        stem = _check_stage_compatibility(input_shape, self.n_stages)
        rows = [("input: cube + tiled noisy map", 2, input_shape), ("stem (conv + maxpool)", self.nFilt0, stem)]

        enc_shapes, shape = [], stem
        for i, ch in enumerate(self.enc_channels):
            if i > 0:
                shape = tuple(-(-s // 2) for s in shape)  # ceil(N/2) downsample
            enc_shapes.append(shape)
            rows.append((f"encoder stage {i}" + (" (no downsample)" if i == 0 else ""), ch, shape))

        shape = enc_shapes[-1]
        for j, up in enumerate(self.upsamples):
            enc_idx = self.n_stages - 2 - j
            out_ch = self.enc_channels[enc_idx]
            up_shape = tuple((2 * s - 1 if up.mode == "conv_transpose" else 2 * s) for s in shape)
            rows.append((f"decoder {j}: upsample", out_ch, up_shape))
            skip = enc_shapes[enc_idx]
            rows.append((f"decoder {j}: + skip(enc {enc_idx})", 2 * out_ch, skip))
            rows.append((f"decoder {j}: deconv block", out_ch, skip))
            shape = skip

        c = self.nFilt0
        rows += [
            ("head: spectral attention pool", c, shape[1:]),  # spectral axis removed
            ("head: resize to map", c, map_shape),
            ("head: + noisy map y_t", c + 1, map_shape),
        ]
        rows += [(f"head: 2-d res block {k}", c, map_shape) for k in range(len(self.head.blocks))]
        rows.append(("head: output conv -> eps", 1, map_shape))
        return rows

    def _param_groups(self, input_shape, map_shape=None):
        """(label, [modules]) for every summary() row, in order: the timestep embedding, then the
        stage_shapes() rows. Rows with no parameters (input, skip concat, resize, y_t concat) get []."""
        mods = [[], [self.stem]] + [[stage] for stage in self.encoder]
        for up, blocks in zip(self.upsamples, self.decoder):
            mods += [[up], [], [blocks]]  # upsample, + skip (concat), deconv blocks
        mods += [[self.head.pool], [], []] + [[b] for b in self.head.blocks] + [[self.head.out]]
        rows = self.stage_shapes(input_shape, map_shape)
        return [("timestep embedding", [self.time_embed])] + [(label, m) for (label, _, _), m in zip(rows, mods)]

    def param_breakdown(self, input_shape, map_shape=None):
        """(label, n_parameters) for every summary() row that holds parameters, in order. The counts
        add up to the model's total parameter count; the shapes only set the row labels."""
        return [(label, count_params(mods)) for label, mods in self._param_groups(input_shape, map_shape) if mods]

    def summary(self, input_shape, map_shape=None, print_fn=print, layers=True):
        """Print the network structure for a cube of `input_shape` (nSpec, nX, nY) and a map of
        `map_shape` (H, W; default (nX, nY)): input/output dimensions, the data/feature-count
        progression through every layer, the number of parameters in every block and layer, the
        activation and time conditioning at each stage, and the decoder's upsample_mode.
        `print_fn` lets output be redirected (e.g. to a logger). layers: if True (default), list
        each block's layers and their parameter counts under the block's row.
        """
        rows = self.stage_shapes(input_shape, map_shape)
        nS, nX, nY = rows[0][2]
        H, W = rows[-1][2]
        w = max(len(label) for label, _, _ in rows + [("timestep embedding", 0, ())])
        bar = "=" * (w + 58)

        # Read activations and modes off the instantiated modules so this reflects what actually runs.
        stem_act = type(self.stem.act).__name__
        block_act = type(self.encoder[0][0].act1).__name__  # same class at every 3-d block/site
        head_act = type(self.head.blocks[0].act1).__name__
        time_act = type(self.time_embed.mlp[1]).__name__
        upsample_mode = self.upsamples[0].mode  # same mode at every decoder stage

        print_fn(bar)
        print_fn(
            f"{type(self).__name__}  |  {self.n_stages} encoder stages, "
            f"{self.n_stages - 1} decoder stages  |  kernel_stem={self.kernel_stem}  |  "
            f"kernel_coder={self.kernel_coder}  |  upsample_mode={upsample_mode!r}  |  "
            f"t_dim={self.t_dim}  |  GroupNorm max_groups={self.max_groups}"
        )
        print_fn(
            f"input:  cube (nBatch, nSpec={nS}, nX={nX}, nY={nY}) + noisy map (nBatch, {H}, {W}) "
            f"+ t (nBatch,)  ->  output: eps (nBatch, {H}, {W})"
        )
        print_fn(
            f"timestep embedding: t -> {self.time_embed.dim} sinusoidal features -> MLP ({time_act}) "
            f"-> {self.t_dim}; FiLM scale/shift into every encoder, decoder and head block"
        )
        print_fn(f"activations: stem={stem_act}  encoder/decoder blocks={block_act}  head blocks={head_act}")
        print_fn("-" * (w + 58))

        def row_note(label):
            if label.startswith("stem"):
                return f"activation={stem_act}"
            if label.startswith("encoder stage") or label.endswith("deconv block"):
                return f"activation={block_act}  FiLM(t)"
            if label.endswith(": upsample"):
                return f"mode={upsample_mode!r}"
            if label == "head: spectral attention pool":
                return "softmax weights over nSpec"
            if label == "head: resize to map":
                return "bilinear"
            if label.startswith("head: 2-d res block"):
                return f"activation={head_act}  FiLM(t)"
            if label.startswith("head: output conv"):
                return "zero-initialised"
            return None

        groups = dict(self._param_groups(input_shape, map_shape))

        def params_col(label):
            mods = groups.get(label, [])
            return f"{count_params(mods):,}" if mods else "-"

        def print_layers(label):
            if layers:
                for line in layer_param_lines(groups.get(label, [])):
                    print_fn(line)

        label = "timestep embedding"
        print_fn(
            f"  {label:<{w}}   features={self.t_dim:<6} params={params_col(label):<11} "
            f"t (nBatch,) -> (nBatch, {self.t_dim})"
        )
        print_layers(label)
        for label, feats, shape in rows:
            dims = " x ".join(str(s) for s in shape)
            note = row_note(label)
            suffix = f" {note}" if note else ""
            print_fn(f"  {label:<{w}}   features={feats:<6} params={params_col(label):<11} dims={dims}{suffix}")
            print_layers(label)

        print_fn("-" * (w + 58))
        n_all = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print_fn(f"parameters: {n_all:,} total ({n_train:,} trainable)")
        print_fn(bar)
        print_fn(str(self))
        print_fn(bar)


#### Diffusion process ####
class GaussianDiffusion(nn.Module):
    """Forward noising, training objective, and reverse sampling for a conditional DDPM.

    Training: pick t ~ U{0..T-1}, form y_t = sqrt(abar_t) y_0 + sqrt(1 - abar_t) eps, and
    regress eps with MSE (Ho et al. 2020 'simple' loss).
    Sampling: start from pure noise and step backwards with the learned eps. sample() uses the
    DDIM update (Song et al. 2021): eta=1 with all T steps is exactly DDPM ancestral sampling;
    fewer steps trade fidelity for speed.

    Parameters:
        denoiser: an eps-prediction network, e.g. DenoisingUNet3d.
        n_timesteps: number of diffusion steps T.
        schedule: 'cosine' or 'linear' (see make_beta_schedule).
        target_shape: (H, W) of the map. Needed only to reshape flattened (B, H*W) labels, as
            produced by conngafit_datasets.py; if None, 3-d (B, H, W) labels are required and sampling
            defaults to the cube's (nX, nY).
        clip_range: bounds for the predicted clean map during sampling (labels normalized with
            'symmetric' lie in [-1, 1]); None disables clipping.
    """

    def __init__(self, denoiser, n_timesteps=1000, schedule="cosine", target_shape=None, clip_range=(-1.0, 1.0)):
        super().__init__()
        self.denoiser = denoiser
        self.n_timesteps = n_timesteps
        self.schedule = schedule
        self.target_shape = tuple(target_shape) if target_shape is not None else None
        self.clip_range = clip_range
        alpha_bar = torch.cumprod(1 - make_beta_schedule(schedule, n_timesteps), dim=0)
        # buffers move with .to(device) and are saved in state_dict
        self.register_buffer("alpha_bar", alpha_bar.float())

    # -- shape handling: accept flattened (B, H*W) labels from conngafit_datasets.py --
    def _to_map(self, y):
        if y.dim() == 3:
            return y
        if self.target_shape is None:
            raise ValueError("flattened labels need target_shape to be set")
        return y.view(y.shape[0], *self.target_shape)

    def _gather(self, t, like):
        """abar_t reshaped to broadcast against a (B, H, W) map."""
        return self.alpha_bar[t].view(-1, *([1] * (like.dim() - 1)))

    def q_sample(self, y0, t, noise):
        """Forward process: jump straight from y_0 to y_t in closed form."""
        ab = self._gather(t, y0)
        return ab.sqrt() * y0 + (1 - ab).sqrt() * noise

    def forward(self, cube, y0):
        """Training loss for a batch: MSE between true and predicted noise."""
        y0 = self._to_map(y0.float())
        t = torch.randint(0, self.n_timesteps, (y0.shape[0],), device=y0.device)
        noise = torch.randn_like(y0)
        return F.mse_loss(self.denoiser(cube.float(), self.q_sample(y0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, cube, n_samples=1, n_steps=None, eta=1.0, max_batch=None):
        """Draw posterior map samples for each cube.

        cube: (B, nSpec, nX, nY). n_steps: reverse steps (default T). eta: 1 = stochastic DDPM-like,
        0 = deterministic DDIM. max_batch: cap on cubes per denoiser call to bound memory.
        Returns (n_samples, B, H, W), or (n_samples, B, H*W) if target_shape is set.
        """
        B = cube.shape[0]
        shape = self.target_shape or tuple(cube.shape[-2:])
        cubes = cube.float().repeat(n_samples, 1, 1, 1)  # one copy per sample
        chunk = max_batch or cubes.shape[0]
        out = torch.cat([self._reverse(c, shape, n_steps, eta) for c in cubes.split(chunk)])
        out = out.view(n_samples, B, *shape)
        return out.flatten(2) if self.target_shape else out

    def _reverse(self, cube, shape, n_steps, eta):
        """Reverse diffusion from y_T ~ N(0, I) to y_0 for one batch of cubes."""
        n_steps = n_steps or self.n_timesteps
        ts = torch.linspace(self.n_timesteps - 1, 0, n_steps, device=cube.device).round().long().unique_consecutive()
        y = torch.randn(cube.shape[0], *shape, device=cube.device)
        for i, t in enumerate(ts):
            t_b = t.expand(cube.shape[0])
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.ones_like(ab)

            eps = self.denoiser(cube, y, t_b)
            y0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()  # current guess of the clean map
            if self.clip_range is not None:
                y0 = y0.clamp(*self.clip_range)
                eps = (y - ab.sqrt() * y0) / (1 - ab).sqrt()  # keep eps consistent with clipped y0

            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            noise = torch.randn_like(y) if i + 1 < len(ts) else torch.zeros_like(y)
            y = ab_prev.sqrt() * y0 + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise
        return y

    def predict_with_uncertainty(self, cube, n_samples=30, **sample_kw):
        """Posterior mean and standard deviation per pixel from n_samples draws."""
        samples = self.sample(cube, n_samples=n_samples, **sample_kw)
        return samples.mean(dim=0), samples.std(dim=0)

    def summary(self, input_shape, print_fn=print, layers=True):
        """Print the diffusion settings (T, noise schedule, map shape, clipping, how much signal
        survives at the first and last step), then the denoiser's own summary for a cube of
        `input_shape` (nSpec, nX, nY). The map shape is target_shape, or (nX, nY) if unset.
        layers is passed on to the denoiser's summary (per-layer parameter counts).
        """
        input_shape = tuple(int(v) for v in input_shape)
        map_shape = self.target_shape or input_shape[1:]
        ab = self.alpha_bar
        bar = "=" * 60
        print_fn(bar)
        print_fn(
            f"{type(self).__name__}  |  T={self.n_timesteps}  |  schedule={self.schedule!r}  |  "
            f"target_shape={self.target_shape}  |  clip_range={self.clip_range}"
        )
        print_fn(
            f"signal fraction sqrt(alpha_bar): {ab[0].sqrt().item():.4f} at t=0 -> "
            f"{ab[-1].sqrt().item():.2e} at t={self.n_timesteps - 1}"
        )
        print_fn(
            "training: MSE(eps_theta(cube, y_t, t), eps) with t ~ U{0..T-1};  "
            "sampling: DDIM update (eta=1 with all T steps = DDPM)"
        )
        print_fn(
            f"map: (nBatch, {map_shape[0]}, {map_shape[1]})"
            + (f", labels flattened to (nBatch, {map_shape[0] * map_shape[1]})" if self.target_shape else "")
        )
        print_fn(bar)
        if hasattr(self.denoiser, "summary"):
            self.denoiser.summary(input_shape, map_shape, print_fn=print_fn, layers=layers)
