"""How a posterior sample of the maps emerges along the DDPM's reverse chain, from pure noise at t = T-1 to t = 0.

At every reverse step the denoiser gives an estimate of the clean maps. Snapshots of the noisy maps y_t and of
that estimate at a few steps show when the large-scale structure appears and when the detail does; the error of the
estimate at every step against the true maps shows how late in the chain the sample settles.
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

TIMESTEP_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)  # Snapshot timesteps, as fractions of the last timestep


def reverse_chain(model, cube, y_true, snapshot_timesteps):
    """Run the reverse chain for one posterior sample, recording snapshots and the error at every step.

    model: a GaussianDiffusion in eval mode. cube: (1, nSpec, nX, nY) and y_true: (1, n_maps, H, W) tensors on the
    model's device. snapshot_timesteps: the timesteps to keep snapshots of.
    Returns (snapshots, errors) as NumPy arrays: snapshots (n_snap, 2, n_maps, H, W) holding y_t and the estimate
    of the clean maps at each snapshot timestep, in their order, and errors (T, n_maps), the MSE of that estimate
    against y_true at every timestep t.
    """
    snapshots = {}
    errors = np.zeros((model.n_timesteps, y_true.shape[1]))

    def record_step(t, y_t, y0_estimate):
        errors[t] = ((y0_estimate - y_true) ** 2).flatten(2).mean(dim=2)[0].cpu().numpy()
        if t in snapshot_timesteps:
            snapshots[t] = torch.stack([y_t[0], y0_estimate[0]]).cpu().numpy()

    with torch.no_grad():
        model.sample(cube, n_samples=1, callback=record_step)
    return np.stack([snapshots[t] for t in snapshot_timesteps]), errors


def plot_reverse_chain(snapshots, snapshot_timesteps, y_true, errors_by_epoch, targets, epoch, output):
    """Plot, for each target, y_t (top row), the clean-map estimate next to the true map (middle row) and the
    estimate minus the true map (bottom row) at each snapshot timestep, and below them the estimate's error against
    t for every recorded epoch.

    snapshots: (n_snap, 2, n_maps, H, W) from reverse_chain. snapshot_timesteps: its n_snap timesteps.
    y_true: (n_maps, H, W) true maps. errors_by_epoch: dict of epoch -> (T, n_maps) errors. targets: the n_maps
    target names. epoch: the epoch of the snapshots, shown in the title. output: the image file to write.
    Maps are in scaled label units. y_t, the estimates and the truth share the true map's colour scale, so values
    outside its range are drawn saturated. The differences share a symmetric scale set by the sample's difference
    at the lowest snapshot timestep.
    """
    n_snap, n_maps = len(snapshot_timesteps), len(targets)
    n_cols = n_snap + 1  # The snapshots, then the true map
    n_rows = 3  # Rows of maps per target
    fig = plt.figure(figsize=(2.4 * n_cols + 0.6, 2.4 * (n_rows * n_maps + 1.5)))
    grid = fig.add_gridspec(
        n_rows * n_maps + 1,
        n_cols + 1,  # The last, narrow column holds each row's colour bar
        height_ratios=[1] * (n_rows * n_maps) + [1.5],
        width_ratios=[1] * n_cols + [0.08],
    )
    for m, target in enumerate(targets):
        map_style = dict(cmap="viridis", vmin=y_true[m].min(), vmax=y_true[m].max(), origin="lower")
        differences = snapshots[:, 1, m] - y_true[m]
        # The sample's (lowest timestep's) difference sets the scale; earlier, noisier estimates saturate
        difference_limit = np.abs(differences[np.argmin(snapshot_timesteps)]).max()
        difference_style = dict(cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit, origin="lower")
        rows = [
            (snapshots[:, 0, m], map_style, "noisy maps y_t"),
            (snapshots[:, 1, m], map_style, "clean-map estimate ŷ0"),
            (differences, difference_style, "ŷ0 - truth"),
        ]
        for r, (images, style, label) in enumerate(rows):
            row = n_rows * m + r
            for col, t in enumerate(snapshot_timesteps):
                ax = fig.add_subplot(grid[row, col])
                image = ax.imshow(images[col], **style)
                ax.set(xticks=[], yticks=[])
                if r == 0:
                    ax.set_title(f"t = {t}")
                if col == 0:
                    ax.set_ylabel(f"{target}\n{label}")
            fig.colorbar(image, cax=fig.add_subplot(grid[row, -1]), label="scaled units")
        ax = fig.add_subplot(grid[n_rows * m + 1, n_snap])
        ax.imshow(y_true[m], **map_style)
        ax.set(title="true map", xticks=[], yticks=[])

    epochs = sorted(errors_by_epoch)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    span = n_cols // n_maps  # Columns of the bottom row per target's error plot
    for m, target in enumerate(targets):
        ax = fig.add_subplot(grid[-1, m * span : (m + 1) * span])
        for each_epoch, color in zip(epochs, colors):
            ax.plot(errors_by_epoch[each_epoch][:, m], color=color, label=f"epoch {each_epoch}")
        ax.set(
            yscale="log",
            xlabel="timestep t",
            ylabel="MSE ŷ0 - truth (scaled units)",
            title=target,
        )
        ax.invert_xaxis()
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_reverse_chain(snapshots_by_epoch, errors_by_epoch, snapshot_timesteps, y_true, targets, output):
    """Write the reverse chains to an hdf5 file: snapshots (E, n_snap, 2, n_maps, H, W) and errors (E, T, n_maps)
    of every recorded epoch, epochs (E,), snapshot_timesteps (n_snap,), the true maps (n_maps, H, W) and the target
    names. snapshots_by_epoch: dict of epoch -> snapshots from reverse_chain; other arguments as for
    plot_reverse_chain."""
    epochs = sorted(snapshots_by_epoch)
    with h5py.File(output, "w") as hf:
        hf.create_dataset("snapshots", data=np.stack([snapshots_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("errors", data=np.stack([errors_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("epochs", data=np.array(epochs))
        hf.create_dataset("snapshot_timesteps", data=np.array(snapshot_timesteps))
        hf.create_dataset("truth", data=y_true)
        hf.create_dataset("targets", data=targets)
