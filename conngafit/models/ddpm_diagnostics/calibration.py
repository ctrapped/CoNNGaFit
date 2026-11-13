"""Posterior calibration of the validation maps: whether the spread of the DDPM's posterior samples matches its errors.

For every validation pixel the true value is ranked among its posterior samples (simulation-based calibration, see
Deistler et al. 2025). Calibrated posteriors give uniformly distributed ranks: a U-shaped rank histogram means they
are too narrow, a hump too wide, and a slope biased. The same ranks give the fraction of true values inside each
central credible interval, which equals its level for calibrated posteriors.

These are per-pixel marginal checks: neighbouring pixels are correlated, so the histogram is less certain than its
pixel counts suggest, and they do not test the joint posterior of a map.
"""

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

COVERAGE_LEVELS = np.linspace(0.05, 0.95, 19)  # Central credible interval levels the coverage is measured at


def validation_posterior_samples(model, dataloader, n_samples, device):
    """n_samples posterior samples of every validation sample's maps, and their labels, in scaled label units.

    model: a GaussianDiffusion in eval mode. dataloader: the validation dataloader, yielding (x, y, w, idx) batches.
    Returns (samples, labels) as NumPy arrays of shapes (n_samples, N, n_maps, H, W) and (N, n_maps, H, W).
    """
    samples, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            samples.append(model.sample(x.to(device).float(), n_samples=n_samples).cpu().numpy())
            labels.append(y.float().numpy())
    return np.concatenate(samples, axis=1), np.concatenate(labels)


def posterior_ranks(samples, y, rng):
    """Rank of each true value among its posterior samples: the number of samples below it, 0 to n_samples.

    samples: (n_samples, ...) and y: (...) arrays. A true value equal to some of its samples is given a random rank
    among the tied positions, so ties (e.g. empty sky, where the label and the clipped samples are both -1) do not
    pile up at one rank. rng: a NumPy Generator for the tie-breaking.
    """
    below = (samples < y).sum(axis=0)
    tied = (samples == y).sum(axis=0)
    return below + rng.integers(0, tied + 1)


def coverage_from_ranks(ranks, n_samples, levels, rng):
    """Fraction of true values inside the central credible interval at each level.

    Each rank is turned into a value uniform on [0, 1) for calibrated posteriors, (rank + U) / (n_samples + 1) with U
    uniform on [0, 1); the true value lies inside the central interval at level a when that value is within a / 2 of
    1 / 2. ranks: array of ranks from posterior_ranks. levels: (L,) levels. rng: a NumPy Generator for U.
    """
    pit = (ranks + rng.random(ranks.shape)) / (n_samples + 1)
    distance = np.abs(pit - 0.5).ravel()
    return np.array([(distance <= level / 2).mean() for level in levels])


def calibration_summary(samples, labels, rng, levels=COVERAGE_LEVELS):
    """Rank histogram and coverage of each target's maps.

    samples: (n_samples, N, n_maps, H, W) and labels: (N, n_maps, H, W) arrays, from validation_posterior_samples.
    Returns (rank_counts, coverage) of shapes (n_maps, n_samples + 1) and (len(levels), n_maps).
    """
    n_samples, _, n_maps = samples.shape[:3]
    ranks = posterior_ranks(samples, labels, rng)
    rank_counts = np.stack([np.bincount(ranks[:, m].ravel(), minlength=n_samples + 1) for m in range(n_maps)])
    coverage = np.stack([coverage_from_ranks(ranks[:, m], n_samples, levels, rng) for m in range(n_maps)], axis=1)
    return rank_counts, coverage


def plot_calibration(rank_counts_by_epoch, coverage_by_epoch, targets, output, levels=COVERAGE_LEVELS):
    """Plot, for each target, the rank histogram of the latest epoch (top) and the coverage against the interval
    level for every recorded epoch (bottom, coloured from early to late).

    rank_counts_by_epoch: dict of epoch -> (n_maps, n_samples + 1) counts. coverage_by_epoch: dict of epoch ->
    (len(levels), n_maps) coverage. targets: the n_maps target names. output: the image file to write.
    The shaded band is where 95% of a uniform histogram's bins would fall if every pixel were independent.
    """
    epochs = sorted(coverage_by_epoch)
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    fig, axes = plt.subplots(2, len(targets), figsize=(6.4 * len(targets), 9.6), squeeze=False)
    for m, target in enumerate(targets):
        counts = rank_counts_by_epoch[epochs[-1]][m]
        n_bins, n_pixels = len(counts), counts.sum()
        expected = n_pixels / n_bins
        band = 2 * np.sqrt(expected * (1 - 1 / n_bins))  # Binomial standard deviation of a bin's count
        ax = axes[0, m]
        ax.axhspan(expected - band, expected + band, color="C0", alpha=0.3, label="uniform")
        ax.stairs(counts, np.arange(n_bins + 1) - 0.5, color="k")
        ax.set(
            xlabel="rank of the true value among the samples", ylabel="pixels", title=f"{target}, epoch {epochs[-1]}"
        )

        ax = axes[1, m]
        ax.plot([0, 1], [0, 1], "k--", label="calibrated")
        for epoch, color in zip(epochs, colors):
            ax.plot(levels, coverage_by_epoch[epoch][:, m], "o-", color=color, label=f"epoch {epoch}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="credible interval level", ylabel="fraction of true values inside")
    axes[1, -1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_calibration(rank_counts_by_epoch, coverage_by_epoch, targets, output, levels=COVERAGE_LEVELS):
    """Write the calibration to an hdf5 file: levels (L,), epochs (E,), coverage (E, L, n_maps), rank_counts
    (E, n_maps, n_samples + 1) and the target names. Arguments as for plot_calibration."""
    epochs = sorted(coverage_by_epoch)
    with h5py.File(output, "w") as hf:
        hf.create_dataset("levels", data=levels)
        hf.create_dataset("epochs", data=np.array(epochs))
        hf.create_dataset("coverage", data=np.stack([coverage_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("rank_counts", data=np.stack([rank_counts_by_epoch[epoch] for epoch in epochs]))
        hf.create_dataset("targets", data=targets)
