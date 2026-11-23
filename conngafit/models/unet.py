"""U-Net bodies shared by the CoNNGaFit networks. Each network adds its own input stage (stem) and output head.
UNet3d - unet3d_resblock.NeuralNetwork and unet3d_ddpm.DenoisingUNet3d
"""

import torch
from torch import nn

from conngafit.models.blocks import DeconvBlock, ResidualBlock, SelfAttention3d, Upsample, _match_spatial


class UNet3d(nn.Module):
    """3-d residual U-Net: n_stages encoder stages of two ResidualBlocks (each stage after the first halving every
    axis to ceil(N/2) and doubling the channels), and a mirrored decoder in which each stage upsamples, concatenates
    the encoder output at the same resolution and applies two DeconvBlocks. Stages listed in attention_stages are
    followed by SelfAttention3d, in the encoder and in the decoder stage at the same resolution.

    Without time_embed_dim the blocks use BatchNorm; with it they use GroupNorm and are FiLM-conditioned on t_emb.

    Parameters:
        channels: channels of the input and of stage 0; doubled at each later stage.
        n_stages: number of encoder stages (the last is the bottleneck).
        kernel_size: 3-tuple kernel of every block conv.
        dropout_rate: probability of the channel-wise dropout inside every block.
        activation: zero-arg callable returning an nn.Module.
        max_groups: upper bound on GroupNorm groups, in the blocks (with time_embed_dim) and in SelfAttention3d.
        time_embed_dim: width of the timestep embedding t_emb; None disables time conditioning.
        architecture: architecture of every block, 'post_activation' or 'identity_skip'.
        upsample_mode: Upsample mode of the decoder, 'interpolation' or 'conv_transpose'.
        attention_stages: encoder stage indices (0 to n_stages - 1) followed by self-attention.
    """

    def __init__(
        self,
        channels,
        n_stages,
        kernel_size,
        dropout_rate=0.0,
        activation=nn.ReLU,
        max_groups=8,
        time_embed_dim=None,
        architecture="post_activation",
        upsample_mode="interpolation",
        attention_stages=(),
    ):
        super().__init__()
        self.n_stages = n_stages
        stage_channels = [channels * 2**i for i in range(n_stages)]
        block_kw = dict(
            kernel1=tuple(kernel_size),
            dropout_rate=dropout_rate,
            activation=activation,
            max_groups=max_groups,
            time_embed_dim=time_embed_dim,
            architecture=architecture,
        )

        # Encoder: stage 0 keeps the input shape, later stages halve every axis
        self.encoder = nn.ModuleList()
        prev = channels
        for i, ch in enumerate(stage_channels):
            blocks = [
                ResidualBlock(prev, ch, downsample=i > 0, **block_kw),
                ResidualBlock(ch, ch, downsample=False, **block_kw),
            ]
            self.encoder.append(nn.ModuleList(blocks))
            prev = ch

        # Decoder: decoder stage j is at encoder stage n_stages - 2 - j; upsample, concat that stage's output,
        # reduce 2C -> C
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for j in range(n_stages - 1):
            in_ch, out_ch = stage_channels[n_stages - 1 - j], stage_channels[n_stages - 2 - j]
            self.upsamples.append(Upsample(in_ch, out_ch, kernel1=tuple(kernel_size), mode=upsample_mode))
            blocks = [
                DeconvBlock(2 * out_ch, out_ch, upsample=True, **block_kw),
                DeconvBlock(out_ch, out_ch, upsample=False, **block_kw),
            ]
            self.decoder.append(nn.ModuleList(blocks))

        # Self-attention after each listed stage's blocks; nn.Identity at every other stage
        attention_stages = set(attention_stages)
        if not attention_stages <= set(range(n_stages)):
            raise ValueError(f"attention_stages {sorted(attention_stages)} must be within 0..{n_stages - 1}")
        self.encoder_attention = nn.ModuleList()
        for i, ch in enumerate(stage_channels):
            if i in attention_stages:
                self.encoder_attention.append(SelfAttention3d(ch, max_groups=max_groups))
            else:
                self.encoder_attention.append(nn.Identity())
        self.decoder_attention = nn.ModuleList()
        for j in range(n_stages - 1):
            stage = n_stages - 2 - j
            if stage in attention_stages:
                self.decoder_attention.append(SelfAttention3d(stage_channels[stage], max_groups=max_groups))
            else:
                self.decoder_attention.append(nn.Identity())

    def forward(self, x, t_emb=None):
        """x: (nBatch, channels, S, X, Y); t_emb: (nBatch, time_embed_dim) or None.

        Returns (output, stage_outputs): output is the last decoder stage's (nBatch, channels, S, X, Y), and
        stage_outputs lists the output of every encoder stage, then every decoder stage. The bottleneck is
        stage_outputs[n_stages - 1].
        """
        stage_outputs = []
        skips = []
        for i, (blocks, attention) in enumerate(zip(self.encoder, self.encoder_attention)):
            for block in blocks:
                x = block(x, t_emb)
            x = attention(x)
            stage_outputs.append(x)
            if i < self.n_stages - 1:
                skips.append(x)

        # Pop the skips in reverse, so each decoder stage pairs with the encoder stage at the same resolution
        for up, blocks, attention in zip(self.upsamples, self.decoder, self.decoder_attention):
            skip = skips.pop()
            x = torch.cat([_match_spatial(up(x), skip), skip], dim=1)
            for block in blocks:
                x = block(x, t_emb)
            x = attention(x)
            stage_outputs.append(x)
        return x, stage_outputs
