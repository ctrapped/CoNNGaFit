"""Where along the 21 cm line profile the DDPM's head looks, from its spectral attention weights.

The head's SpectralAttentionPool collapses the spectral axis with a softmax over velocity channels for every pixel.
Its attention-weighted mean velocity shows which part of the line each pixel's prediction uses, and comparing it
with the cube's intensity-weighted mean velocity shows whether the attention follows the emission. The normalized
entropy of the weights shows how concentrated the attention is: 1 for uniform weights, which make the pool a plain
mean over the spectral axis, and 0 for a single channel. They are shown at a few noise levels t, as the denoiser's
task changes with t, and the entropy at every recorded epoch shows whether the attention learns to concentrate.

The head works after the stem, at ~1/4 of the cube's spectral and spatial resolution, so its channel velocities are
those of evenly spread pooled channels and approximate to within the stem's receptive field (a few channels).
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import entr

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


def attention_summary(weights, velocities):
    """Attention-weighted mean velocity and normalized entropy of every pixel's attention weights.

    weights: (..., S', X', Y') attention weights summing to 1 over S'. velocities: (S',) channel velocities.
    Returns (mean, entropy) arrays of shape (..., X', Y'): entropy is 1 for uniform weights and 0 for weights all on
    one channel.
    """
    mean = (weights * velocities[:, None, None]).sum(axis=-3)
    entropy = entr(weights).sum(axis=-3) / np.log(weights.shape[-3])
    return mean, entropy


def cube_mean_velocity(cube, velocities):
    """Intensity-weighted mean velocity (first moment) of every pixel of a (nSpec, nX, nY) datacube in intensity
    units, NaN where a pixel has no emission. Negative intensities (noise) are ignored."""
    intensity = np.clip(cube, 0, None)
    total = intensity.sum(axis=0)
    weighted = (intensity * velocities[:, None, None]).sum(axis=0)
    return np.divide(weighted, total, out=np.full(total.shape, np.nan), where=total > 0)


def plot_spectral_attention(cube_velocity, attention_mean, entropy_by_epoch, timesteps, output):
    """Plot the cube's intensity-weighted mean velocity and the attention's median entropy against epoch (top row),
    then one row per timestep of the latest epoch's attention-weighted mean velocity and entropy.

    cube_velocity: (nX, nY) from cube_mean_velocity. attention_mean: (n_t, X', Y') mean velocities of the latest
    epoch, from attention_summary. entropy_by_epoch: dict of epoch -> (n_t, X', Y') entropies from
    attention_summary. timesteps: the n_t timesteps. output: the image file to write.
    Every map spans the same field of view, so the coarser attention maps line up with the cube's, and the
    velocity maps share a colour scale.
    """
    epochs = sorted(entropy_by_epoch)
    extent = (0, cube_velocity.shape[1], 0, cube_velocity.shape[0])  # The full map, in cube pixels
    v_max = np.nanmax(np.abs(np.concatenate([cube_velocity.ravel(), attention_mean.ravel()])))
    velocity_style = dict(cmap="RdBu_r", vmin=-v_max, vmax=v_max, origin="lower", extent=extent)
    entropy_style = dict(cmap="viridis", vmin=0, vmax=1, origin="lower", extent=extent)

    fig, axes = plt.subplots(len(timesteps) + 1, 2, figsize=(8.4, 3.6 * (len(timesteps) + 1)), squeeze=False)
    ax = axes[0, 0]
    fig.colorbar(ax.imshow(cube_velocity, **velocity_style), ax=ax, label="km/s")
    ax.set(title="cube: intensity-weighted velocity", xticks=[], yticks=[])
    ax = axes[0, 1]
    median_entropy = np.array([np.median(entropy_by_epoch[epoch], axis=(1, 2)) for epoch in epochs])  # (E, n_t)
    for row, t in enumerate(timesteps):
        ax.plot(epochs, median_entropy[:, row], "o-", label=f"t = {t}")
    ax.axhline(1, color="k", linestyle="--", label="uniform")
    ax.set(ylim=(0, 1.05), xlabel="epoch", ylabel="median normalized entropy", title="attention concentration")
    ax.legend(fontsize="small")

    for row, t in enumerate(timesteps):
        panels = [
            (attention_mean[row], velocity_style, f"t = {t}: attention-weighted velocity", "km/s"),
            (entropy_by_epoch[epochs[-1]][row], entropy_style, f"t = {t}: attention entropy", "1 = uniform"),
        ]
        for ax, (image, style, title, label) in zip(axes[row + 1], panels):
            fig.colorbar(ax.imshow(image, **style), ax=ax, label=label)
            ax.set(title=title, xticks=[], yticks=[])
    fig.suptitle(f"Spectral attention of the head, epoch {epochs[-1]}")
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
