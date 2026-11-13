"""Where along the 21 cm line profile the DDPM's head looks, from its spectral attention weights.

The head's SpectralAttentionPool collapses the spectral axis with a softmax over velocity channels for every pixel.
Its attention-weighted mean velocity and spread show which part of the line each pixel's prediction uses (e.g. the
line centre or its wings), and comparing them with the cube's intensity-weighted mean velocity shows whether the
attention follows the emission. They are shown at a few noise levels t, as the denoiser's task changes with t.

The head works after the stem, at ~1/4 of the cube's spectral and spatial resolution, so its channel velocities are
those of evenly spread pooled channels and approximate to within the stem's receptive field (a few channels).
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

from conngafit.models.ddpm_diagnostics.activations import captured_outputs

TIMESTEP_FRACTIONS = (0.0, 0.5, 1.0)  # Noise levels the attention is shown at, as fractions of the last timestep


def attention_weights(model, cube, y0, noise, timesteps):
    """The head's spectral attention weights for one sample at each timestep.

    model: a GaussianDiffusion in eval mode. cube: (1, nSpec, nX, nY), y0 and noise: (1, n_maps, H, W) tensors on
    the model's device. timesteps: the integer timesteps t to run the denoiser at, on y0 noised with noise.
    Returns a NumPy array of shape (n_t, S', X', Y'), summing to 1 over S' (the head's spectral channels).
    """
    score = model.denoiser.head.pool.score  # Its softmax over the spectral axis gives the attention weights
    with torch.no_grad(), captured_outputs({"score": score}) as outputs:
        for t in timesteps:
            model(cube, y0, t=torch.full((1,), t, device=cube.device), noise=noise)
    scores = torch.cat(outputs["score"])  # (n_t, 1, S', X', Y')
    return torch.softmax(scores, dim=2)[:, 0].cpu().numpy()


def channel_velocities(n_channels, channel_width, n_pooled=None):
    """Line-of-sight velocity (km/s) of each spectral channel, with 0 km/s at the central channel (or between the
    two central channels). channel_width: km/s per channel. If n_pooled is given, the velocities of n_pooled evenly
    spread channels covering the same range instead (the head's channels after the stem)."""
    centers = np.arange(n_channels) - (n_channels - 1) / 2  # In channels from the cube's centre
    if n_pooled is not None:
        centers = (np.arange(n_pooled) + 0.5) * n_channels / n_pooled - n_channels / 2
    return centers * channel_width


def attention_moments(weights, velocities):
    """Attention-weighted mean velocity and spread (standard deviation) of every pixel.

    weights: (..., S', X', Y') attention weights summing to 1 over S'. velocities: (S',) channel velocities.
    Returns (mean, spread) arrays of shape (..., X', Y').
    """
    v = velocities[:, None, None]
    mean = (weights * v).sum(axis=-3)
    spread = np.sqrt((weights * (v - mean[..., None, :, :]) ** 2).sum(axis=-3))
    return mean, spread


def cube_mean_velocity(cube, velocities):
    """Intensity-weighted mean velocity (first moment) of every pixel of a (nSpec, nX, nY) datacube in intensity
    units, NaN where a pixel has no emission. Negative intensities (noise) are ignored."""
    intensity = np.clip(cube, 0, None)
    total = intensity.sum(axis=0)
    weighted = (intensity * velocities[:, None, None]).sum(axis=0)
    return np.divide(weighted, total, out=np.full(total.shape, np.nan), where=total > 0)


def plot_spectral_attention(cube_velocity, attention_mean, attention_spread, timesteps, epoch, output):
    """Plot one row per timestep: the cube's intensity-weighted mean velocity, the attention-weighted mean velocity
    and the attention's velocity spread.

    cube_velocity: (nX, nY) from cube_mean_velocity. attention_mean, attention_spread: (n_t, X', Y') from
    attention_moments. timesteps: the n_t timesteps. epoch: the epoch shown in the title. output: the image file.
    Every panel spans the same field of view, so the coarser attention maps line up with the cube's.
    """
    extent = (0, cube_velocity.shape[1], 0, cube_velocity.shape[0])  # The full map, in cube pixels
    v_max = np.nanmax(np.abs(np.concatenate([cube_velocity.ravel(), attention_mean.ravel()])))
    velocity_style = dict(cmap="RdBu_r", vmin=-v_max, vmax=v_max, origin="lower", extent=extent)
    spread_style = dict(cmap="viridis", vmin=0, vmax=attention_spread.max(), origin="lower", extent=extent)

    fig, axes = plt.subplots(len(timesteps), 3, figsize=(12, 3.6 * len(timesteps)), squeeze=False)
    for row, t in enumerate(timesteps):
        panels = [
            (cube_velocity, velocity_style, "cube: intensity-weighted velocity", "km/s"),
            (attention_mean[row], velocity_style, f"t = {t}: attention-weighted velocity", "km/s"),
            (attention_spread[row], spread_style, f"t = {t}: attention velocity spread", "km/s"),
        ]
        for ax, (image, style, title, unit) in zip(axes[row], panels):
            fig.colorbar(ax.imshow(image, **style), ax=ax, label=unit)
            ax.set(title=title, xticks=[], yticks=[])
    fig.suptitle(f"Spectral attention of the head, epoch {epoch}")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_spectral_attention(weights_by_epoch, velocities, cube_velocity, timesteps, output):
    """Write the attention to an hdf5 file: weights (E, n_t, S', X', Y') of every recorded epoch, epochs (E,),
    timesteps (n_t,), the head's channel velocities (S',) and the cube's intensity-weighted velocity (nX, nY).

    weights_by_epoch: dict of epoch -> (n_t, S', X', Y') weights from attention_weights; other arguments as for
    channel_velocities, cube_mean_velocity and plot_spectral_attention.
    """
    epochs = sorted(weights_by_epoch)
    with h5py.File(output, "w") as hf:
        hf.create_dataset("weights", data=np.stack([weights_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("epochs", data=np.array(epochs))
        hf.create_dataset("timesteps", data=np.array(timesteps))
        hf.create_dataset("velocities", data=velocities)
        hf.create_dataset("cube_velocity", data=cube_velocity)
