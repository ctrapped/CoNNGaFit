"""Validation denoising loss as a function of the diffusion timestep t.

The training loss averages over every noise level, so it cannot show which ones the model has learned. Scoring each
validation sample at its fixed timestep (train_model's fixed validation draws) every recorded epoch shows how the
loss at each noise level falls as training goes on: small t sets the fine detail of the maps, large t their overall
structure.
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def per_sample_loss(pred_noise, noise, w):
    """Weighted MSE between predicted and true noise for each sample and map, shape (B, n_maps): the training loss of
    train_model.compute_loss before it is averaged over the batch and maps. pred_noise, noise, w: (B, n_maps, ...)."""
    squared_error = w * nn.functional.mse_loss(pred_noise, noise, reduction="none")
    return squared_error.flatten(2).mean(dim=2)


def validation_losses_by_timestep(model, dataloader, fixed_draws, device):
    """Each validation sample's denoising loss at its fixed timestep.

    model: a GaussianDiffusion in eval mode. dataloader: the (unshuffled) validation dataloader, yielding
    (x, y, w, idx) batches. fixed_draws: one (t, noise) pair per batch, as train_model makes for validation.
    Returns (t, losses) as NumPy arrays of shapes (N,) and (N, n_maps), in dataset order.
    """
    timesteps, losses = [], []
    with torch.no_grad():
        for (x, y, w, _), (t, noise) in zip(dataloader, fixed_draws):
            x, y, w = x.to(device).float(), y.to(device).float(), w.to(device).float()
            pred_noise, noise = model(x, y, t=t, noise=noise)
            timesteps.append(t.cpu().numpy())
            losses.append(per_sample_loss(pred_noise, noise, w).cpu().numpy())
    return np.concatenate(timesteps), np.concatenate(losses)


def plot_loss_by_timestep(t, losses_by_epoch, targets, n_timesteps, output):
    """Plot each sample's loss against its timestep, one panel per target and one line per recorded epoch (coloured
    from early to late).

    t: (N,) timesteps. losses_by_epoch: dict of epoch -> (N, n_maps) losses at t. targets: the n_maps target names.
    n_timesteps: the number of diffusion steps T, the extent of the t axis. output: the image file to write.
    """
    order = np.argsort(t)
    epochs = sorted(losses_by_epoch)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    fig, axes = plt.subplots(1, len(targets), figsize=(6.4 * len(targets), 4.8), squeeze=False)
    for m, (ax, target) in enumerate(zip(axes[0], targets)):
        for epoch, color in zip(epochs, colors):
            ax.plot(t[order], losses_by_epoch[epoch][order, m], "o-", color=color, label=f"epoch {epoch}")
        ax.set(xlim=(0, n_timesteps - 1), yscale="log", xlabel="timestep t", ylabel="denoising loss", title=target)
    axes[0, -1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_loss_by_timestep(t, losses_by_epoch, targets, output):
    """Write the losses to an hdf5 file: timesteps (N,), epochs (E,), losses (E, N, n_maps) and the target names.
    Arguments as for plot_loss_by_timestep."""
    epochs = sorted(losses_by_epoch)
    with h5py.File(output, "w") as hf:
        hf.create_dataset("timesteps", data=t)
        hf.create_dataset("epochs", data=np.array(epochs))
        hf.create_dataset("losses", data=np.stack([losses_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("targets", data=targets)
