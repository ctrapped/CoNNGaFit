"""Validation denoising loss as a function of the diffusion timestep t.

The training loss averages over every noise level, so it cannot show which ones the model has learned. Scoring every
validation sample at the same grid of timesteps, with fixed noise, every recorded epoch shows how the loss at each
noise level falls as training goes on: small t sets the fine detail of the maps, large t their overall structure.
The noise-prediction loss is small at large t even when the model is poor there, so the error of the clean-map
estimate it implies is shown as well, the quantity the reverse chain's error curve tracks.
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

N_GRID_TIMESTEPS = 10  # Timesteps, evenly spaced over 0..T-1, at which every validation sample is scored


def per_sample_loss(pred_noise, noise, w):
    """Weighted MSE between predicted and true noise for each sample and map, shape (B, n_maps): the training loss of
    train_model.compute_loss before it is averaged over the batch and maps. pred_noise, noise, w: (B, n_maps, ...)."""
    squared_error = w * nn.functional.mse_loss(pred_noise, noise, reduction="none")
    return squared_error.flatten(2).mean(dim=2)


def validation_losses_by_timestep(model, dataloader, noises, timesteps, device):
    """Every validation sample's denoising loss, and the error of the clean-map estimate it implies, at each timestep.

    model: a GaussianDiffusion in eval mode. dataloader: the (unshuffled) validation dataloader, yielding
    (x, y, w, idx) batches. noises: one fixed noise tensor per batch, shaped like its maps, used at every timestep.
    timesteps: the timesteps to score at.
    The clean-map estimate is y0 - sqrt((1 - abar_t) / abar_t) (eps_pred - eps), what the reverse chain inverts y_t
    to, clipped to model.clip_range as it is there.
    Returns (noise_losses, clean_map_losses) as NumPy arrays of shape (n_t, N, n_maps), in dataset order: the
    weighted MSE of the predicted noise (the training loss) and the MSE of the estimate against the true maps.
    """
    noise_losses, clean_map_losses = [], []
    with torch.no_grad():
        for (x, y, w, _), noise in zip(dataloader, noises):
            x, y, w = x.to(device).float(), y.to(device).float(), w.to(device).float()
            batch_noise_losses, batch_clean_map_losses = [], []
            for t in timesteps:
                pred_noise, _ = model(x, y, t=torch.full((x.shape[0],), t, device=device), noise=noise)
                ab = model.alpha_bar[t]
                y0_estimate = y - ((1 - ab) / ab).sqrt() * (pred_noise - noise)
                if model.clip_range is not None:
                    y0_estimate = y0_estimate.clamp(*model.clip_range)
                batch_noise_losses.append(per_sample_loss(pred_noise, noise, w).cpu().numpy())
                batch_clean_map_losses.append(((y0_estimate - y) ** 2).flatten(2).mean(dim=2).cpu().numpy())
            noise_losses.append(np.stack(batch_noise_losses))
            clean_map_losses.append(np.stack(batch_clean_map_losses))
    return np.concatenate(noise_losses, axis=1), np.concatenate(clean_map_losses, axis=1)


def plot_loss_by_timestep(timesteps, noise_losses_by_epoch, clean_map_losses_by_epoch, targets, n_timesteps, output):
    """Plot the noise-prediction loss (top row) and the clean-map error (bottom row) against the timestep, one column
    per target and one line per recorded epoch (coloured from early to late): the mean over the validation samples,
    with a band spanning their range.

    timesteps: (n_t,) timesteps. noise_losses_by_epoch, clean_map_losses_by_epoch: dicts of epoch -> (n_t, N, n_maps)
    losses from validation_losses_by_timestep. targets: the n_maps target names. n_timesteps: the number of
    diffusion steps T, the extent of the t axis. output: the image file to write.
    """
    epochs = sorted(noise_losses_by_epoch)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    rows = [
        (noise_losses_by_epoch, "MSE of ε̂ vs ε (weighted)"),
        (clean_map_losses_by_epoch, "MSE of ŷ0 vs truth (scaled units)"),
    ]
    n_validation = noise_losses_by_epoch[epochs[0]].shape[1]
    fig, axes = plt.subplots(len(rows), len(targets), figsize=(6.4 * len(targets), 8.4), squeeze=False, sharex=True)
    for row, (losses_by_epoch, ylabel) in enumerate(rows):
        for m, target in enumerate(targets):
            ax = axes[row, m]
            for epoch, color in zip(epochs, colors):
                losses = losses_by_epoch[epoch][:, :, m]  # (n_t, N)
                ax.plot(timesteps, losses.mean(axis=1), "o-", color=color, label=f"epoch {epoch}")
                ax.fill_between(timesteps, losses.min(axis=1), losses.max(axis=1), color=color, alpha=0.2)
            ax.set(xlim=(0, n_timesteps - 1), yscale="log", ylabel=ylabel)
            if row == 0:
                ax.set_title(target)
    for ax in axes[-1]:
        ax.set_xlabel("timestep t")
    axes[0, -1].legend(fontsize="small")
    fig.suptitle(f"Validation loss by timestep\nmean (line) and range (band) of the {n_validation} validation images")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_loss_by_timestep(timesteps, noise_losses_by_epoch, clean_map_losses_by_epoch, targets, output):
    """Write the losses to an hdf5 file: timesteps (n_t,), epochs (E,), noise_losses and clean_map_losses
    (E, n_t, N, n_maps) and the target names. Arguments as for plot_loss_by_timestep."""
    epochs = sorted(noise_losses_by_epoch)
    with h5py.File(output, "w") as hf:
        hf.create_dataset("timesteps", data=timesteps)
        hf.create_dataset("epochs", data=np.array(epochs))
        hf.create_dataset("noise_losses", data=np.stack([noise_losses_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("clean_map_losses", data=np.stack([clean_map_losses_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("targets", data=targets)
