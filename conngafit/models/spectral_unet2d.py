"""Networks that compress each pixel's spectrum first and then work in 2-d at full map resolution: a SpectralEncoder
turns the datacube into per-pixel features, a 2-d UNet refines them and a MapHead2d predicts the output maps.

  SpectralResNet2d        - residual U-Net predicting the maps (and their log-variance) directly, like
                            unet3d_resblock.NeuralNetwork
  SpectralDenoisingUNet2d - noise-prediction network for the DDPM (wrapped by unet3d_ddpm.GaussianDiffusion),
                            like unet3d_ddpm.DenoisingUNet3d
"""

import torch
from torch import nn

from conngafit.models.blocks import MapHead2d, SpectralEncoder, check_stage_compatibility
from conngafit.models.unet import UNet
from conngafit.models.unet3d_ddpm import TimestepEmbedding


def _check_input_shape(input_shape, n_stages):
    """input_shape as a tuple of ints, after checking it is (nSpec, nX, nY) and its maps allow n_stages - 1
    halvings."""
    input_shape = tuple(int(v) for v in input_shape)
    if len(input_shape) != 3:
        raise ValueError(f"input_shape must be (nSpec, nX, nY); got {input_shape}")
    check_stage_compatibility(input_shape[1:], n_stages, stem_reduction=1)
    return input_shape


class SpectralResNet2d(nn.Module):
    """Spectral datacube (nSpec, nX, nY) -> n_maps maps on the cube's (nX, nY) grid.

    Layout:
      encoder    - SpectralEncoder: n_spectral_layers strided 3-d convs halving the spectral axis, then a
                   per-pixel attention pool over it -> (C_enc, nX, nY)
      input_conv - 3x3 Conv2d to n_filt0 channels
      unet       - UNet(n_dims=2) with n_stages stages
      head       - MapHead2d: n_head_blocks 2-d residual blocks, dropout and a 3x3 conv

    Parameters:
        input_shape: (nSpec, nX, nY) of the cube; nX and nY must allow n_stages - 1 halvings.
        n_filt0: channels of the encoder's first layer, of the U-Net's first stage and of the head.
        kernel_encoder: 3-tuple (spectral, x, y) kernel of the encoder's convs.
        kernel_size: 2-tuple kernel of the U-Net's convs.
        n_spectral_layers: encoder layers, each halving the spectral axis and doubling the channels.
        n_stages: U-Net encoder stages (the last is the bottleneck).
        n_maps: number of predicted maps.
        n_head_blocks: residual blocks in the head (>= 0).
        predict_variance: if True, forward() also returns a per-pixel log-variance for each map.
        output_activation: zero-arg callable applied to the predicted mean (default nn.Identity).
        head_dropout_rate: channel-wise dropout before the head's output conv.
        block_dropout_rate: channel-wise dropout inside every U-Net block.
        block_architecture: architecture of every residual block, 'post_activation' or 'identity_skip'.
        attention_stages: U-Net stage indices followed by self-attention.
        upsample_mode: Upsample mode of the U-Net's decoder, 'conv_transpose' or 'interpolation'.
        activation: zero-arg callable returning an nn.Module, used throughout.
        max_groups: upper bound on GroupNorm groups in self-attention.
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_encoder,
        kernel_size,
        n_spectral_layers=3,
        n_stages=5,
        n_maps=1,
        n_head_blocks=2,
        predict_variance=False,
        output_activation=nn.Identity,
        head_dropout_rate=0.0,
        block_dropout_rate=0.0,
        block_architecture="post_activation",
        attention_stages=(),
        upsample_mode="conv_transpose",
        activation=nn.ReLU,
        max_groups=8,
    ):
        super().__init__()
        self.input_shape = _check_input_shape(input_shape, n_stages)
        self.output_shape = self.input_shape[1:]
        self.n_maps = n_maps
        self.predict_variance = predict_variance

        # No time conditioning, so BatchNorm in the encoder, as in the U-Net and head
        self.encoder = SpectralEncoder(n_filt0, n_spectral_layers, kernel_encoder, activation, max_groups=None)
        self.input_conv = nn.Conv2d(self.encoder.out_channels, n_filt0, kernel_size, padding="same")
        self.unet = UNet(
            n_filt0,
            n_stages,
            kernel_size,
            n_dims=2,
            dropout_rate=block_dropout_rate,
            activation=activation,
            max_groups=max_groups,
            architecture=block_architecture,
            upsample_mode=upsample_mode,
            attention_stages=attention_stages,
        )
        self.head = MapHead2d(
            n_filt0,
            n_filt0,
            out_channels=2 * n_maps if predict_variance else n_maps,
            n_blocks=n_head_blocks,
            activation=activation,
            architecture=block_architecture,
            dropout_rate=head_dropout_rate,
        )
        self.out_activation = output_activation()

    def forward(self, x, save_latent_images=False):
        """x: (nBatch, nSpec, nX, nY) -> mean (nBatch, n_maps, nX, nY), or (mean, logvar) with predict_variance.
        save_latent_images is not supported: the latent-image diagnostics are written for 3-d features."""
        if save_latent_images:
            raise ValueError("save_latent_images is not supported by SpectralResNet2d")
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"input (nSpec, nX, nY) = {tuple(x.shape[1:])} doesn't match the model's input_shape {self.input_shape}"
            )
        h = self.encoder(x.unsqueeze(1))  # (nBatch, C_enc, nX, nY)
        h = self.input_conv(h)  # (nBatch, n_filt0, nX, nY)
        h, _ = self.unet(h)
        out = self.head(h)  # (nBatch, n_maps or 2 * n_maps, nX, nY)
        mean = self.out_activation(out[:, : self.n_maps])
        if self.predict_variance:
            return mean, out[:, self.n_maps :]
        return mean


class SpectralDenoisingUNet2d(nn.Module):
    """eps_theta(cube, y_t, t): predicts the noise added to n_maps maps on the cube's (nX, nY) grid, conditioned on the
    cube.

    Layout:
      encoder    - SpectralEncoder: n_spectral_layers strided 3-d convs halving the spectral axis, then a
                   per-pixel attention pool over it -> (C_enc, nX, nY)
      input      - the noisy maps y_t concatenated at full resolution -> (C_enc + n_maps, nX, nY)
      input_conv - 3x3 Conv2d to n_filt0 channels
      unet       - UNet(n_dims=2) with n_stages stages
      head       - MapHead2d: n_head_blocks 2-d residual blocks and a 3x3 conv that starts at zero, so the network
                   starts by predicting eps = 0
    Every U-Net and head block uses GroupNorm and is FiLM-conditioned on the timestep embedding, which is as wide as
    the widest U-Net stage (n_filt0 * 2**(n_stages - 1)).
    The encoder sees only the spectral cube: encode() runs its spectral encoder, and denoise() runs the rest.

    Parameters:
        input_shape: (nSpec, nX, nY) of the cube; nX and nY must allow n_stages - 1 halvings.
        n_filt0: channels of the encoder's first layer, of the U-Net's first stage and of the head.
        kernel_encoder: 3-tuple (spectral, x, y) kernel of the encoder's convs.
        kernel_size: 2-tuple kernel of the U-Net's convs.
        n_spectral_layers: encoder layers, each halving the spectral axis and doubling the channels.
        n_stages: U-Net encoder stages (the last is the bottleneck).
        n_maps: number of maps denoised together.
        n_head_blocks: residual blocks in the head (>= 0).
        time_encoding_dim: width of the fixed sinusoidal timestep encoding.
        block_dropout_rate: channel-wise dropout inside every U-Net block.
        block_architecture: architecture of every residual block, 'post_activation' or 'identity_skip'.
        attention_stages: U-Net stage indices followed by self-attention.
        upsample_mode: Upsample mode of the U-Net's decoder, 'interpolation' or 'conv_transpose'.
        activation: zero-arg callable returning an nn.Module, used throughout.
        max_groups: upper bound on GroupNorm groups.
    """

    def __init__(
        self,
        input_shape,
        n_filt0,
        kernel_encoder,
        kernel_size,
        n_spectral_layers=3,
        n_stages=5,
        n_maps=1,
        n_head_blocks=2,
        time_encoding_dim=128,
        block_dropout_rate=0.0,
        block_architecture="post_activation",
        attention_stages=(),
        upsample_mode="interpolation",
        activation=nn.SiLU,
        max_groups=8,
    ):
        super().__init__()
        self.input_shape = _check_input_shape(input_shape, n_stages)
        self.output_shape = self.input_shape[1:]
        self.n_maps = n_maps
        self.time_embed_dim = n_filt0 * 2 ** (n_stages - 1)  # As wide as the widest U-Net stage

        self.time_embed = TimestepEmbedding(time_encoding_dim, self.time_embed_dim)
        self.encoder = SpectralEncoder(n_filt0, n_spectral_layers, kernel_encoder, activation, max_groups)
        self.input_conv = nn.Conv2d(self.encoder.out_channels + n_maps, n_filt0, kernel_size, padding="same")
        self.unet = UNet(
            n_filt0,
            n_stages,
            kernel_size,
            n_dims=2,
            dropout_rate=block_dropout_rate,
            activation=activation,
            max_groups=max_groups,
            time_embed_dim=self.time_embed_dim,
            architecture=block_architecture,
            upsample_mode=upsample_mode,
            attention_stages=attention_stages,
        )
        self.head = MapHead2d(
            n_filt0,
            n_filt0,
            out_channels=n_maps,
            n_blocks=n_head_blocks,
            activation=activation,
            max_groups=max_groups,
            time_embed_dim=self.time_embed_dim,
            architecture=block_architecture,
            zero_init_output=True,
        )

    def encode(self, cube):
        """cube: (B, nSpec, nX, nY), with (nSpec, nX, nY) = input_shape. Returns the encoder's features,
        (B, C_enc, nX, nY)."""
        if tuple(cube.shape[1:]) != self.input_shape:
            raise ValueError(
                f"cube {tuple(cube.shape[1:])} doesn't match the input_shape {self.input_shape} this model was built for"
            )
        return self.encoder(cube.unsqueeze(1))

    def denoise(self, features, y_t, t):
        """features: encode()'s output; y_t: (B, n_maps, nX, nY) noisy maps; t: (B,) timesteps.
        Returns predicted noise, shape (B, n_maps, nX, nY)."""
        maps_shape = (self.n_maps, *self.output_shape)
        if tuple(y_t.shape[1:]) != maps_shape:
            raise ValueError(
                f"maps {tuple(y_t.shape[1:])} don't match the (n_maps, *output_shape) {maps_shape} this model was "
                "built for"
            )
        t_emb = self.time_embed(t)
        h = self.input_conv(torch.cat([features, y_t], dim=1))  # (B, n_filt0, nX, nY)
        h, _ = self.unet(h, t_emb)
        return self.head(h, t_emb)  # (B, n_maps, nX, nY)

    def forward(self, cube, y_t, t):
        """cube: (B, nSpec, nX, nY); y_t: (B, n_maps, nX, nY) noisy maps; t: (B,) timesteps.
        Returns predicted noise, shape (B, n_maps, nX, nY)."""
        return self.denoise(self.encode(cube), y_t, t)
