"""Records every DDPM diagnostic of this package at each recorded epoch of training (train(diagnostic_mode=True))."""

import numpy as np
import torch

from conngafit.models.ddpm_diagnostics import calibration, loss_by_timestep


class TrainingDiagnostics:
    """Records the DDPM diagnostics of models/ddpm_diagnostics while it trains, keeping every recorded epoch's
    results and rewriting their outputs each time, so they can be inspected while training runs.

    dataloader: the (unshuffled) validation dataloader. fixed_draws: its fixed (t, noise) pair per batch.
    n_samples: posterior samples per validation sample, for the calibration. targets: the target names, one per map.
    output: the path prefix of every output (e.g. <output_dir>/diagnostics/<run_name>). seed: seeds the diagnostics'
    own NumPy generator, so they leave training's random draws alone. device: where the validation batches go.
    """

    def __init__(self, dataloader, fixed_draws, n_samples, targets, output, seed, device):
        self.dataloader = dataloader
        self.fixed_draws = fixed_draws
        self.n_samples = n_samples
        self.targets = targets
        self.output = output
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.losses_by_epoch = {}  # Each recorded epoch's validation loss per sample, at its fixed timestep
        self.rank_counts_by_epoch, self.coverage_by_epoch = {}, {}  # Each recorded epoch's posterior calibration

    def record(self, epoch, model):
        """Record the diagnostics of model (a GaussianDiffusion in eval mode) at epoch, and rewrite every output."""
        # fork_rng restores the random number generators afterwards. Diagnostics would change every later
        # random draw of training otherwise
        with torch.random.fork_rng():
            timesteps, self.losses_by_epoch[epoch] = loss_by_timestep.validation_losses_by_timestep(
                model, self.dataloader, self.fixed_draws, self.device
            )
            samples, labels = calibration.validation_posterior_samples(
                model, self.dataloader, self.n_samples, self.device
            )
        self.rank_counts_by_epoch[epoch], self.coverage_by_epoch[epoch] = calibration.calibration_summary(
            samples, labels, self.rng
        )

        loss_by_timestep.plot_loss_by_timestep(
            timesteps, self.losses_by_epoch, self.targets, model.n_timesteps, self.output + "_LossByTimestep.png"
        )
        loss_by_timestep.save_loss_by_timestep(
            timesteps, self.losses_by_epoch, self.targets, self.output + "_LossByTimestep.hdf5"
        )
        calibration.plot_calibration(
            self.rank_counts_by_epoch, self.coverage_by_epoch, self.targets, self.output + "_Calibration.png"
        )
        calibration.save_calibration(
            self.rank_counts_by_epoch, self.coverage_by_epoch, self.targets, self.output + "_Calibration.hdf5"
        )
