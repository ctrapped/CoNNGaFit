"""Plotting functions for the diagnostic figures and hdf5 snapshots that training and inference save: predictions
against labels, as maps or radial curves."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import h5py
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from conngafit.datasets import MOMENTUM_TARGETS, TARGET_SETTINGS, ring_index, ring_radii


def make_comp_image(
    data_pred,
    data_input,
    data_image,
    output,
    index,
    sim_to_physical_units=1,
    param_label="Radial Momentum",
    unit_label=r"[M$_{\odot}$ km s$^{-1}$]",
    bin_op="sum",
    uncertainty=None,
):
    """Save a diagnostic figure (input map, prediction map, radially-binned
    input/prediction curves, and the summed input datacube) for one snapshot in a batch.
    With uncertainty, a fifth panel shows the predicted 1-sigma map, next to the prediction.

    data_pred / data_input: (nBatch, nY, nX) prediction/label maps.
    data_image: (nBatch, nSpec, nX, nY) input datacube batch.
    output: path to save the PNG to.
    index: index of the sample within the batch to plot.
    sim_to_physical_units: scalar multiplied into the data before plotting
    param_label/unit_label: axis/colorbar labels for the parameter being plotted.
    bin_op: statistic passed to scipy.stats.binned_statistic_dd for the radial profile
        ("sum" for momenta, "mean" for velocity/inclination, etc.).
    uncertainty: (nBatch, nY, nX) predicted 1-sigma maps in the same units as data_pred, or None (the default) for
        the 4-panel figure.
    """
    image_pred, image_input = data_pred[index], data_input[index]
    npix = image_pred.shape[0]

    # Maps along the top row (input, prediction, and the 1-sigma map if given), curve and summed cube below
    n_cols = 2 if uncertainty is None else 3
    panel = {"input": 1, "prediction": 2, "sigma": 3, "curve": n_cols + 1, "cube": n_cols + 2}
    plt.figure(figsize=(3.2 * n_cols, 4.8))  # matplotlib's default size for 2 columns

    plt.subplot(2, n_cols, panel["cube"])
    image_to_plot = np.sum(data_image[index, :, :, :], axis=0)
    image_to_plot += np.abs(np.min(image_to_plot)) + 1
    plt.imshow(image_to_plot, norm=LogNorm(), cmap="inferno")
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.subplot(2, n_cols, panel["curve"])
    center_index = [(npix - 1.0) / 2.0, (npix - 1.0) / 2.0]
    indices = np.indices(np.shape(image_pred))
    rmag = np.sqrt(np.power(indices[0, :, :] - center_index[0], 2) + np.power(indices[1, :, :] - center_index[1], 2))

    n_bins = 20

    bin_range = [0, np.max(rmag)]
    binned_data_pred, binedge, binnum = stats.binned_statistic_dd(
        rmag.flatten(), image_pred.flatten(), bin_op, n_bins, range=[bin_range]
    )
    binned_data_input, binedge, binnum = stats.binned_statistic_dd(
        rmag.flatten(), image_input.flatten(), bin_op, n_bins, range=[bin_range]
    )
    max_plot = np.max(
        [-np.min(binned_data_pred), -np.min(binned_data_input), np.max(binned_data_pred), np.max(binned_data_input)]
    )
    plt.plot(np.linspace(0, np.max(rmag), n_bins), binned_data_input * sim_to_physical_units, "k", lw=2)
    plt.plot(np.linspace(0, np.max(rmag), n_bins), binned_data_pred * sim_to_physical_units, "r--", lw=2)
    plt.legend(["Input", "Prediction"])
    plt.plot(np.linspace(0, np.max(rmag), n_bins), binned_data_pred * 0, "k--", lw=1)
    plt.xlabel("Radius")
    plt.ylabel(param_label + " " + unit_label)
    plt.ylim([-max_plot * 1.05 * sim_to_physical_units, max_plot * 1.05 * sim_to_physical_units])

    vmax = np.max([-np.min(image_pred), np.max(image_pred), -np.min(image_input), np.max(image_input)])
    vmin = -vmax
    plt.subplot(2, n_cols, panel["input"])
    plt.imshow(
        image_input * sim_to_physical_units,
        cmap="seismic",
        vmin=vmin * sim_to_physical_units,
        vmax=vmax * sim_to_physical_units,
    )
    plt.title(param_label + " Input")
    # plt.colorbar();
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.subplot(2, n_cols, panel["prediction"])
    plt.imshow(
        image_pred * sim_to_physical_units,
        cmap="seismic",
        vmin=vmin * sim_to_physical_units,
        vmax=vmax * sim_to_physical_units,
    )
    plt.colorbar(label=unit_label)
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    plt.title(param_label + " Prediction")

    if uncertainty is not None:
        plt.subplot(2, n_cols, panel["sigma"])
        sigma_map = uncertainty[index]
        plt.imshow(sigma_map * sim_to_physical_units, cmap="viridis", vmin=0)  # sequential: sigma is >= 0
        plt.colorbar(label=unit_label)
        plt.gca().axes.xaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticks([])
        plt.title(r"Predicted 1$\sigma$")
    plt.savefig(output)
    plt.close()


def make_image(
    data,
    output,
    vmin=None,
    vmax=None,
    sim_to_physical_units=1,
    param_label="Radial Momentum",
    unit_label=r"[M$_{\odot}$ km s$^{-1}$]",
    uncertainty=None,
):
    """Save a predicted map as a PNG (with a physical-units colorbar) and an accompanying hdf5 file. With
    uncertainty, the predicted 1-sigma map is plotted beside the prediction and saved too.

    data: the (nY, nX) predicted map.
    output: path prefix; writes output+"projectionMap.png"/".hdf5".
    vmin/vmax: fixed color limits; Defaults to symmetric limits.
    uncertainty: the (nY, nX) predicted 1-sigma map, or None.
    """
    image = data

    n_cols = 1 if uncertainty is None else 2
    plt.figure(figsize=(6.4 * n_cols, 4.8))  # matplotlib's default size per panel

    if vmax is None or vmin is None:
        vmax = np.max([-np.min(image), np.max(image)])  # default to symmetric limits
        vmin = -vmax

    plt.subplot(1, n_cols, 1)
    plt.imshow(
        image * sim_to_physical_units,
        cmap="seismic",
        vmin=vmin * sim_to_physical_units,
        vmax=vmax * sim_to_physical_units,
    )
    plt.colorbar(label=param_label + " " + unit_label)

    if uncertainty is not None:
        sigma_map = uncertainty
        plt.subplot(1, n_cols, 2)
        plt.imshow(sigma_map * sim_to_physical_units, cmap="viridis", vmin=0)  # sequential: sigma is >= 0
        plt.colorbar(label=r"Predicted 1$\sigma$ " + unit_label)
    plt.savefig(output + "projectionMap.png")
    plt.close()

    # Save hdf5 for generating figures
    hf = h5py.File(output + "projectionMap.hdf5", "w")
    hf.create_dataset("image", data=image)
    if uncertainty is not None:
        hf.create_dataset("uncertainty", data=sigma_map)
    hf.close()


def save_hdf5(input_data, pred_data, output, index, uncertainty=None):
    """Save one snapshot's prediction (and optionally its label and predicted 1-sigma), maps or radial curves, to an
    hdf5 file for later re-plotting.

    input_data: (nBatch, nY, nX) label maps or (nBatch, n_bins) curves, or None to skip saving the label
        (e.g. when running inference with no ground truth available).
    pred_data: predictions, shaped like input_data.
    output: full path to the hdf5 file to write (datasets 'input', 'prediction' and 'uncertainty').
    index: index of the sample within the batch to save.
    uncertainty: predicted 1-sigma, shaped like pred_data, or None (the default) to skip it.
    """
    hf = h5py.File(output, "w")
    if input_data is not None:
        hf.create_dataset("input", data=input_data[index])
    hf.create_dataset("prediction", data=pred_data[index])
    if uncertainty is not None:
        hf.create_dataset("uncertainty", data=uncertainty[index])
    hf.close()


def make_comp_image_1d(
    data_pred,
    data_input,
    data_image,
    output,
    index,
    sim_to_physical_units=1,
    param_label="Radial Momentum",
    unit_label=r"[M$_{\odot}$ km s$^{-1}$]",
    bin_op="sum",
    radii=None,
    uncertainty=None,
):
    """1-d analog of make_comp_image: save a 2-panel figure (summed input datacube, and the
    input/prediction radial curves already stored in data_pred/data_input) for one snapshot.
    With uncertainty, the prediction's +/-1 sigma is shaded around it.

    data_pred / data_input: (nBatch, nRadialBins) radially-binned prediction/label curves
    data_image: (nBatch, nSpec, nX, nY) input datacube batch.
    output/index/sim_to_physical_units/param_label/unit_label: see make_comp_image above.
    radii: the curves' ring centers in kpc (see ring_radii in datasets.py); if None, plot against bin number.
    uncertainty: (nBatch, nRadialBins) predicted 1-sigma of each ring.
    """

    fig, (ax_curve, ax_image) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    image_to_plot = np.sum(data_image[index, :, :, :], axis=0)
    image_to_plot += np.abs(np.min(image_to_plot)) + 1  # offset to positive values, for the log scale
    im = ax_image.imshow(image_to_plot, norm=LogNorm(), cmap="inferno")
    ax_image.xaxis.set_ticks([])
    ax_image.yaxis.set_ticks([])
    fig.colorbar(im, ax=ax_image, label="Moment 0 Map of Normalized Input")

    plt.sca(ax_curve)
    ax_curve.set_box_aspect(1)  # square, the same size as the image

    rmag = np.arange(np.size(data_pred[index, :])) if radii is None else radii

    max_plot = np.max(
        [
            -np.min(np.array(data_pred[index, :])),
            -np.min(np.array(data_input[index, :])),
            np.max(np.array(data_pred[index, :])),
            np.max(np.array(data_input[index, :])),
        ]
    )
    curve_pred = np.array(data_pred[index, :]) * sim_to_physical_units
    plt.plot(rmag, np.array(data_input[index, :]) * sim_to_physical_units, "k", lw=2, label="Input")
    plt.plot(rmag, curve_pred, "r--", lw=2, label="Prediction")
    if uncertainty is not None:  # shade the predicted +/-1 sigma, and widen the y-range to fit it
        sigma = np.array(uncertainty[index, :]) * sim_to_physical_units
        plt.fill_between(
            rmag, curve_pred - sigma, curve_pred + sigma, color="r", alpha=0.25, lw=0, label=r"$\pm1\sigma$"
        )
        max_plot = max(max_plot, np.max(np.abs(curve_pred) + sigma) / sim_to_physical_units)
    plt.plot(rmag, rmag * 0, "k--", lw=1)
    plt.legend()
    plt.xlabel("Radial bin" if radii is None else "Radius [kpc]")
    plt.ylabel(param_label + " " + unit_label)
    plt.ylim([-max_plot * 1.05 * sim_to_physical_units, max_plot * 1.05 * sim_to_physical_units])

    plt.savefig(output)
    plt.close()


def rv2_coeff(data_list):
    """Compute the pairwise RV coefficient matrix (a multivariate generalization of squared
    correlation, in [0, 1]) between a list of 2-d arrays of the same shape - used by
    make_correlation_plot below to score 2-d map similarity between prediction and truth.
    Returns a (len(data_list), len(data_list)) matrix c where c[i,j] is the RV coefficient
    between data_list[i] and data_list[j].
    """
    # First compute the scalar product matrices for each data set X
    scal_arr_list = []

    for arr in data_list:
        scal_arr = np.dot(arr, np.transpose(arr))
        scal_arr_mod = scal_arr - np.diag(np.diag(scal_arr))
        scal_arr_list.append(scal_arr_mod)

    # Now compute the 'between study cosine matrix' c
    c = np.zeros((len(data_list), len(data_list)), float)

    for index in np.ndindex(c.shape):
        nom = np.trace(np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[1]]))
        denom1 = np.trace(np.dot(np.transpose(scal_arr_list[index[0]]), scal_arr_list[index[0]]))
        denom2 = np.trace(np.dot(np.transpose(scal_arr_list[index[1]]), scal_arr_list[index[1]]))
        rv = nom / np.sqrt(denom1 * denom2)
        c[index[0], index[1]] = rv

    return c


def make_correlation_plot(pred_image, input_image, output):
    """Score prediction-vs-truth agreement across a batch of snapshots and save a histogram
    of two correlation metrics, plus the raw per-snapshot values to an hdf5 file.

    pred_image / input_image: (n_snaps, nY, nX) prediction/label maps.
    output: path prefix; writes output+".png"/".pdf"/".hdf5".

    For each snapshot, computes:
      corr2d - the RV coefficient (see rv2_coeff) between the full 2-d prediction and truth maps.
      corr1d - the Pearson correlation between the two maps' radially-binned 1-d profiles.
    """
    tick_size = 14
    label_size = 16
    legend_size = 14

    n_snaps = len(pred_image)
    corr1d = np.zeros((n_snaps))
    corr2d = np.zeros((n_snaps))
    for i in range(0, n_snaps):
        pred_snap, input_snap = pred_image[i], input_image[i]
        npix = pred_snap.shape[0]

        rv2_mat = rv2_coeff([input_snap, pred_snap])
        corr2d[i] = rv2_mat[0, 1]

        center_index = [(npix - 1.0) / 2.0, (npix - 1.0) / 2.0]
        indices = np.indices(np.shape(pred_snap))
        rmag = np.sqrt(
            np.power(indices[0, :, :] - center_index[0], 2) + np.power(indices[1, :, :] - center_index[1], 2)
        )

        bin_range = [0, np.max(rmag)]
        binned_pred, _binedge, _binnum = stats.binned_statistic_dd(
            rmag.flatten(), pred_snap.flatten(), "sum", npix, range=[bin_range]
        )
        binned_input, _binedge, _binnum = stats.binned_statistic_dd(
            rmag.flatten(), input_snap.flatten(), "sum", npix, range=[bin_range]
        )

        correlation_matrix_1d = np.corrcoef(binned_pred, binned_input)
        corr1d[i] = correlation_matrix_1d[1][0]

    plt.figure()
    n_bins = 20
    alpha = 0.7
    plt.hist(corr1d, bins=n_bins, range=[-1, 1], histtype="step", color="k", label="1d Corr.", alpha=alpha, linewidth=2)
    plt.hist(corr2d, bins=n_bins, range=[-1, 1], histtype="step", color="r", label="2D Corr.", alpha=alpha, linewidth=2)

    plt.xlim([-1, 1])

    plt.legend(fontsize=legend_size, loc="upper left")
    plt.ylabel("Snapshots", fontsize=label_size)
    plt.xlabel("Correlation", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(output + ".png", bbox_inches="tight")
    plt.savefig(output + ".pdf", bbox_inches="tight")

    plt.close()

    hf = h5py.File(output + ".hdf5", "w")
    hf.create_dataset("corr1d", data=corr1d)
    hf.create_dataset("corr2d", data=corr2d)
    hf.close()


def save_correlation_plots(saved, targets, output_base, epoch):
    """Write a correlation plot of each target's predicted and true maps, from the (predictions, labels) batches
    save_batch_outputs collected in saved, in physical units."""
    all_pred = np.concatenate([batch_pred for batch_pred, _ in saved])
    all_y = np.concatenate([batch_y for _, batch_y in saved])
    for m, target in enumerate(targets):
        make_correlation_plot(all_pred[:, m], all_y[:, m], f"{output_base}_{target}_Epoch{epoch}")


def plot_loss(losses, title, ylabel, output):
    """Plot a loss curve per epoch, on a log axis unless some value isn't positive (the Gaussian NLL can be negative)."""
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    if np.all(losses > 0):
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(output)
    plt.close()


def plot_history(output, result, validation_names, loss_label, accuracy_label):
    """Plot a run's training and validation losses (loss_label names the training objective), its validation
    accuracy, and each validation sample's accuracy, to output + "_trainingLoss.png" and so on. result is
    train_model.train's TrainingResult; accuracy_label says what counts as accurate (e.g. "within predicted 1-sigma")."""
    plot_loss(result.training_losses, "Training Loss", loss_label, output + "_trainingLoss.png")
    plot_loss(result.validation_losses, "Validation Loss", loss_label, output + "_validationLoss.png")

    # Accuracies are NaN on the epochs between predict_every, so only the predicted epochs are plotted
    epoch_numbers = np.arange(1, len(result.validation_accuracies) + 1)
    predicted = np.isfinite(result.validation_accuracies)

    plt.figure()
    plt.plot(epoch_numbers[predicted], result.validation_accuracies[predicted], marker="o", ms=3)
    plt.title(f"Validation Accuracy ({accuracy_label})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.ylim([0, 101])
    plt.savefig(output + "_ValidationAccuracy.png")
    plt.close()

    plt.figure()
    for name, accuracies in zip(validation_names, result.specific_accuracies):
        plt.plot(epoch_numbers[predicted], accuracies[predicted], lw=0.5, marker="o", ms=2, label=name)
    if len(validation_names) <= 10:  # Name each line, unless there are too many to read
        plt.legend(fontsize=6)
    plt.title("Validation Accuracy for each Snapshot")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.ylim([0, 101])
    plt.savefig(output + "_ValidationAccuracyPerSnapshot.png")
    plt.close()


def save_test_summary(figure_file, images, names, curves, sigmas, result, target):
    """Save a one-page summary of a network's radial curves over a large testing set to figure_file: the
    counterpart of examples/demo_plots.save_demo_figure for hundreds to thousands of images, called the same way.

      Row 1: the median-error image's input column density; every image's residual against radius (median, 16-84%
             and 2.5-97.5% bands); predicted against true for every ring of every image (2-d histogram, 1:1 line).
      Row 2: a histogram of (predicted - true) / predicted 1-sigma against N(0, 1); the MSE and loss histories.
      Row 3: the images at the 0th, 25th, 50th, 75th, 90th and 100th percentile of RMS error, each with its true and
             predicted curve.

    images: as examples/demo_utils.load_images returns, one entry per test image; only "cubes" (of the median-error
        image), "fovs_kpc" and "true_curves" are used.
    names: a label for each image, for the example titles (e.g. its filename).
    curves, sigmas: (n_images, n_rings) predicted curves of target and their 1-sigma; sigmas may be None.
    result: the TrainingResult train() returned.
    target: the target the curves are of, e.g. "rotational_velocity".
    Returns each image's RMS error, in target's units.
    """
    true_curves = images["true_curves"]
    residuals = curves - true_curves
    errors = np.sqrt(np.mean(np.square(residuals), axis=1))  # Each image's RMS error
    param_label, unit_label, _ = TARGET_SETTINGS[target]
    radii = np.array([ring_radii(fov, curves.shape[1]) for fov in images["fovs_kpc"]])
    order = np.argsort(errors)

    def at_percentile(percentile):
        """The image whose RMS error is at percentile (nearest rank)."""
        return int(order[round(percentile / 100 * (len(errors) - 1))])

    fig = plt.figure(figsize=(16, 14), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2, 0.85])
    (ax_input, ax_residual, ax_scatter), (ax_check, ax_mse, ax_loss) = top.subplots(2, 3)
    example_axes = bottom.subplots(1, 6, sharey=True)

    # Input column density of the median-error image (the cube in cm^-2 per channel, summed), on a log scale spanning
    # its 1st to 99.9th percentiles
    median = at_percentile(50)
    column_density = images["cubes"][median].sum(axis=2)
    half_width = images["fovs_kpc"][median] / 2
    low, high = np.percentile(column_density[column_density > 0], [1, 99.9])
    im = ax_input.imshow(
        np.clip(column_density, low, None),
        origin="lower",
        extent=[-half_width, half_width, -half_width, half_width],
        cmap="inferno",
        norm=LogNorm(low, high),
    )
    fig.colorbar(im, ax=ax_input, label="HI column density [cm$^{-2}$]")
    ax_input.set(xlabel="x [kpc]", ylabel="y [kpc]", title=f"Input, median-error image\n{names[median]}")

    # Residual bands. Ring k is at the same radius in every image only if they share a field of view; otherwise rings
    # are compared at the same fraction of each image's half-width
    if np.allclose(radii, radii[0]):
        x, xlabel = radii[0], "Radius [kpc]"
    else:
        x, xlabel = (np.arange(curves.shape[1]) + 0.5) / curves.shape[1], "Radius / image half-width"
    low2, low1, mid, high1, high2 = np.nanpercentile(residuals, [2.5, 16, 50, 84, 97.5], axis=0)
    ax_residual.fill_between(x, low2, high2, color="C0", alpha=0.15, lw=0, label="2.5–97.5%")
    ax_residual.fill_between(x, low1, high1, color="C0", alpha=0.35, lw=0, label="16–84%")
    ax_residual.plot(x, mid, color="C0", marker="o", ms=3, label="Median")
    ax_residual.axhline(0, color="k", lw=1.5)
    ax_residual.set(xlabel=xlabel, ylabel=f"Predicted − true {unit_label}", title=f"Residuals ({len(curves)} images)")
    ax_residual.legend(fontsize=8)

    # Predicted against true, on log axes when every value is positive and they span more than 3 orders of magnitude
    values = np.concatenate([true_curves.ravel(), curves.ravel()])
    scale = "log" if np.all(values > 0) and values.max() / values.min() > 1e3 else "linear"
    hexbins = ax_scatter.hexbin(
        true_curves.ravel(), curves.ravel(), gridsize=40, bins="log", mincnt=1, xscale=scale, yscale=scale
    )
    fig.colorbar(hexbins, ax=ax_scatter, label="Rings")
    ax_scatter.plot([values.min(), values.max()], [values.min(), values.max()], "k--", lw=1, label="1:1")
    ax_scatter.set(
        xlabel=f"True {param_label} {unit_label}",
        ylabel=f"Predicted {param_label} {unit_label}",
        title="Predicted against true, every ring",
    )
    ax_scatter.legend(fontsize=8, loc="upper left")

    # Uncertainty check: residual / sigma should follow N(0, 1). Rings with sigma = 0 (no mass) are left out, and
    # values beyond ±5 are counted in the end bins
    if sigmas is None:
        ax_check.hist(residuals.ravel(), bins=50)
        ax_check.set(xlabel=f"Predicted − true {unit_label}", ylabel="Rings", title="Residuals (no predicted 1σ)")
    else:
        has_sigma = sigmas > 0
        z = residuals[has_sigma] / sigmas[has_sigma]
        ax_check.hist(np.clip(z, -5, 5), bins=np.linspace(-5, 5, 51), density=True, alpha=0.7)
        grid = np.linspace(-5, 5, 200)
        ax_check.plot(grid, stats.norm.pdf(grid), "k", lw=1.5, label="N(0, 1)")
        within1, within3 = np.mean(np.abs(z) < 1), np.mean(np.abs(z) < 3)
        ax_check.text(
            0.03,
            0.97,
            f"within 1σ: {within1:.0%} (expect 68%)\nwithin 3σ: {within3:.0%} (expect 99.7%)",
            transform=ax_check.transAxes,
            va="top",
            fontsize=8,
        )
        ax_check.set(xlabel="(predicted − true) / predicted 1σ", ylabel="Density", title="Uncertainty check")
        ax_check.legend(fontsize=8, loc="upper right")

    # Training histories; the MSE of the mean is NaN on epochs whose predictions weren't scored. Log axes unless a
    # value isn't positive (the Gaussian NLL can be negative)
    for ax, training, validation, title in (
        (ax_mse, result.training_mses, result.validation_mses, "MSE of the mean (scaled labels)"),
        (ax_loss, result.training_losses, result.validation_losses, "Loss (training objective)"),
    ):
        epochs = np.arange(1, len(training) + 1)
        evaluated = np.isfinite(training) & np.isfinite(validation)
        ax.plot(epochs[evaluated], training[evaluated], label="Training")
        ax.plot(epochs[evaluated], validation[evaluated], label="Validation")
        positive = np.all(training[evaluated] > 0) and np.all(validation[evaluated] > 0)
        ax.set(xlabel="Epoch", ylabel=title, title=title, yscale="log" if positive else "linear")
        ax.legend(fontsize=8)

    # Examples by RMS error, best to worst; a set with fewer than 6 images shows each once
    examples = []
    for percentile, label in zip(
        (0, 25, 50, 75, 90, 100), ("Best", "25th pct.", "Median", "75th pct.", "90th pct.", "Worst")
    ):
        i = at_percentile(percentile)
        if i not in [j for j, _ in examples]:
            examples.append((i, label))
    for ax, (i, label) in zip(example_axes, examples):
        ax.plot(radii[i], true_curves[i], "k", lw=2, label="True")
        ax.plot(radii[i], curves[i], "--", color="C1", marker="o", ms=3, label="Predicted")
        if sigmas is not None:
            ax.fill_between(radii[i], curves[i] - sigmas[i], curves[i] + sigmas[i], color="C1", alpha=0.25, lw=0)
        ax.set_title(f"{label}: RMS {errors[i]:.3g}\n{names[i]}", fontsize=9)
        ax.set_xlabel("Radius [kpc]")
    for ax in example_axes[len(examples) :]:
        ax.set_axis_off()
    example_axes[0].set_ylabel(f"{param_label} {unit_label}")
    handles = example_axes[0].get_legend_handles_labels()[0]
    if sigmas is not None:
        handles.append(Patch(color="C1", alpha=0.25, lw=0, label="Predicted ±1σ"))
    example_axes[0].legend(handles=handles, fontsize=8)

    fig.suptitle(f"{param_label} on the testing set")
    fig.savefig(figure_file, dpi=110)
    plt.close(fig)
    return errors


# Map targets whose errors are measured in dex, log10(predicted / true), as they span orders of magnitude; pixels where
# either map isn't positive (e.g. counter-rotating gas) are left out of those errors. Other targets can be negative,
# so their errors are predicted - true
LOG_MAP_TARGETS = {"mass", "rotational_momentum"}


def _ring_curve(ring, n_rings, target, values, mass, sigma=None):
    """A map of target binned into n_rings rings (ring: each pixel's ring, see datasets.ring_index), as
    datasets.radial_profile bins annotations: masses and momenta summed, velocities mass-weighted by mass. With sigma,
    a per-pixel 1-sigma map, also the curve's 1-sigma, assuming independent pixel errors (otherwise None)."""
    inside = ring < n_rings  # Pixels in the corners, beyond the image edge, are left out

    def ring_sum(a):
        return np.bincount(ring[inside], weights=a[inside], minlength=n_rings)

    if target == "mass" or target in MOMENTUM_TARGETS:
        return ring_sum(values), None if sigma is None else np.sqrt(ring_sum(np.square(sigma)))
    ring_mass = ring_sum(mass)

    def per_mass(a):
        return np.divide(a, ring_mass, out=np.zeros(n_rings), where=ring_mass > 0)

    curve = per_mass(ring_sum(mass * values))
    return curve, None if sigma is None else per_mass(np.sqrt(ring_sum(np.square(mass * sigma))))


def save_test_map_summary(
    figure_file, images, names, true_maps, maps, map_sigmas, target, true_mass, detected, n_rings=10
):
    """Save a one-page summary of a network's maps of target over a large testing set to figure_file: the map
    counterpart of save_test_summary.

      Row 1: predicted against true for every detected pixel (2-d histogram, 1:1 line); a histogram of
             (predicted - true) / predicted 1-sigma against N(0, 1); each image's RMS error, marking the examples.
      Rows 2-4: the best, median and worst images by RMS error, each as its input column density, true and predicted
             maps, their error, the prediction's signal-to-noise, and the radial curves binned from both maps.

    Only detected pixels count in the statistics and are shown on the map panels. Errors are in dex for
    LOG_MAP_TARGETS, otherwise predicted - true in target's units.

    images: as examples/demo_utils.load_images returns; "fovs_kpc" is used, and "cubes" only for the three example
        images, so it can be any sequence that loads a cube when indexed instead of every cube in memory.
    names: a label for each image, for the example rows.
    true_maps, maps, map_sigmas: (n_images, nY, nX) true and predicted maps of target and the predicted 1-sigma;
        map_sigmas may be None.
    true_mass: (n_images, nY, nX) true mass maps, to weight the velocity curves as the true curves are weighted.
    detected: (n_images, nY, nX) True where a pixel is detected, e.g. cube.sum(axis=2) > a cutoff in cm^-2 chosen
        for each image's noise.
    n_rings: rings in each radial curve.
    Returns each image's RMS error over its detected pixels.
    """
    param_label, unit_label, _ = TARGET_SETTINGS[target]
    log_target = target in LOG_MAP_TARGETS
    if log_target:
        valid = detected & (true_maps > 0) & (maps > 0)
        errors_map = np.log10(np.where(valid, maps, 1) / np.where(valid, true_maps, 1))
        error_label = "log$_{10}$(predicted / true) [dex]"
    else:
        valid = detected
        errors_map = maps - true_maps
        error_label = f"Predicted − true {unit_label}"
    n_valid = valid.sum(axis=(1, 2))
    squared = np.where(valid, np.square(errors_map), 0).sum(axis=(1, 2))
    errors = np.sqrt(np.divide(squared, n_valid, out=np.full(len(maps), np.nan), where=n_valid > 0))
    order = np.argsort(errors)  # NaN (no detected pixels) sorts last
    n_ranked = int(np.isfinite(errors).sum())
    examples = []
    for rank, label in ((0, "Best"), ((n_ranked - 1) // 2, "Median"), (n_ranked - 1, "Worst")):
        if int(order[rank]) not in [i for i, _ in examples]:
            examples.append((int(order[rank]), label))

    fig = plt.figure(figsize=(26, 19), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 3.2])
    ax_scatter, ax_check, ax_errors = top.subplots(1, 3)
    example_axes = bottom.subplots(3, 6, squeeze=False)

    # Predicted against true, every detected pixel; log axes for LOG_MAP_TARGETS
    scale = "log" if log_target else "linear"
    true_values, predicted_values = true_maps[valid], maps[valid]
    hexbins = ax_scatter.hexbin(
        true_values, predicted_values, gridsize=60, bins="log", mincnt=1, xscale=scale, yscale=scale
    )
    fig.colorbar(hexbins, ax=ax_scatter, label="Pixels")
    limits = [min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())]
    ax_scatter.plot(limits, limits, "k--", lw=1, label="1:1")
    ax_scatter.set(
        xlabel=f"True {param_label} {unit_label}",
        ylabel=f"Predicted {param_label} {unit_label}",
        title=f"Predicted against true, {valid.sum():,} detected pixels",
    )
    ax_scatter.legend(fontsize=8, loc="upper left")

    # Uncertainty check: (predicted - true) / sigma, in target's units, should follow N(0, 1); values beyond ±5 are
    # counted in the end bins
    if map_sigmas is None:
        ax_check.hist(errors_map[valid], bins=50)
        ax_check.set(xlabel=error_label, ylabel="Pixels", title="Errors (no predicted 1σ)")
    else:
        has_sigma = detected & (map_sigmas > 0)
        z = (maps - true_maps)[has_sigma] / map_sigmas[has_sigma]
        ax_check.hist(np.clip(z, -5, 5), bins=np.linspace(-5, 5, 51), density=True, alpha=0.7)
        grid = np.linspace(-5, 5, 200)
        ax_check.plot(grid, stats.norm.pdf(grid), "k", lw=1.5, label="N(0, 1)")
        within1, within3 = np.mean(np.abs(z) < 1), np.mean(np.abs(z) < 3)
        ax_check.text(
            0.03,
            0.97,
            f"within 1σ: {within1:.0%} (expect 68%)\nwithin 3σ: {within3:.0%} (expect 99.7%)",
            transform=ax_check.transAxes,
            va="top",
            fontsize=9,
        )
        ax_check.set(xlabel="(predicted − true) / predicted 1σ", ylabel="Density", title="Uncertainty check")
        ax_check.legend(fontsize=8, loc="upper right")

    # Each image's RMS error, with the examples marked
    rms_label = "RMS error [dex]" if log_target else f"RMS error {unit_label}"
    ax_errors.hist(errors[np.isfinite(errors)], bins=30)
    for (i, label), color in zip(examples, ("C2", "C1", "C3")):
        ax_errors.axvline(errors[i], color=color, lw=2, label=f"{label}: {errors[i]:.3g}")
    ax_errors.set(xlabel=rms_label, ylabel="Images", title=f"Per-image error ({n_ranked} images)")
    ax_errors.legend(fontsize=8)

    # Example rows. Each map column shares one color scale over the examples' detected pixels, so the rows compare
    picked = [i for i, _ in examples]
    cubes = {i: np.asarray(images["cubes"][i]) for i in picked}
    column_densities = {i: cubes[i].sum(axis=2) for i in picked}
    all_density = np.concatenate([column_densities[i][column_densities[i] > 0] for i in picked])
    density_norm = LogNorm(*np.percentile(all_density, [1, 99.9]))
    example_true = np.concatenate([true_maps[i][valid[i]] for i in picked])
    example_predicted = np.concatenate([maps[i][valid[i]] for i in picked])
    value_low, value_high = np.percentile(np.concatenate([example_true, example_predicted]), [1, 99])
    value_style = {"norm": LogNorm(value_low, value_high)} if log_target else {"vmin": value_low, "vmax": value_high}
    error_limit = np.percentile(np.abs(np.concatenate([errors_map[i][valid[i]] for i in picked])), 99)
    if map_sigmas is not None:
        snr = maps / np.where(map_sigmas > 0, map_sigmas, np.nan)
        snr_low, snr_high = np.nanpercentile(np.concatenate([snr[i][valid[i]] for i in picked]), [1, 99])

    for row, (axes, (i, label)) in enumerate(zip(example_axes, examples)):
        half_width = images["fovs_kpc"][i] / 2
        extent = [-half_width, half_width, -half_width, half_width]
        hidden = ~valid[i]
        panels = [
            (column_densities[i], "Input: HI column density [cm$^{-2}$]", {"norm": density_norm, "cmap": "inferno"}),
            (
                np.ma.masked_where(hidden, true_maps[i]),
                f"True {param_label} {unit_label}",
                {**value_style, "cmap": "magma"},
            ),
            (
                np.ma.masked_where(hidden, maps[i]),
                f"Predicted {param_label} {unit_label}",
                {**value_style, "cmap": "magma"},
            ),
            (
                np.ma.masked_where(hidden, errors_map[i]),
                error_label,
                {"vmin": -error_limit, "vmax": error_limit, "cmap": "RdBu_r"},
            ),
        ]
        if map_sigmas is not None:
            panels.append(
                (
                    np.ma.masked_where(hidden, snr[i]),
                    "Predicted / 1σ",
                    {"vmin": snr_low, "vmax": snr_high, "cmap": "viridis"},
                )
            )
        for ax, (image, colorbar_label, style) in zip(axes, panels):
            if colorbar_label.startswith("Input"):
                image = np.clip(image, density_norm.vmin, None)
            im = ax.imshow(image, origin="lower", extent=extent, **style)
            if row == 0:  # One color bar per column, as its header
                fig.colorbar(im, ax=ax, location="top", shrink=0.9, label=colorbar_label)
            ax.set_xlabel("x [kpc]")
        if map_sigmas is None:
            axes[4].set_axis_off()
        axes[0].set_ylabel(f"{label}: RMS {errors[i]:.3g}\n{names[i]}\ny [kpc]")

        # Radial curves binned from the true and predicted maps over every pixel, as the training curves are
        ring = ring_index(true_maps[i].shape, images["fovs_kpc"][i], n_rings)
        radii = ring_radii(images["fovs_kpc"][i], n_rings)
        true_curve, _ = _ring_curve(ring, n_rings, target, true_maps[i], true_mass[i])
        sigma_map = None if map_sigmas is None else map_sigmas[i]
        curve, curve_sigma = _ring_curve(ring, n_rings, target, maps[i], true_mass[i], sigma_map)
        ax_curve = axes[5]
        ax_curve.plot(radii, true_curve, "k", lw=2, label="True")
        ax_curve.plot(radii, curve, "--", color="C1", marker="o", ms=3, label="Predicted")
        if curve_sigma is not None:
            ax_curve.fill_between(radii, curve - curve_sigma, curve + curve_sigma, color="C1", alpha=0.25, lw=0)
        ax_curve.set(xlabel="Radius [kpc]", ylabel=f"{param_label} {unit_label}")
        if row == 0:
            handles = ax_curve.get_legend_handles_labels()[0]
            if curve_sigma is not None:
                handles.append(Patch(color="C1", alpha=0.25, lw=0, label="Predicted ±1σ"))
            ax_curve.legend(handles=handles, fontsize=8)
            ax_curve.set_title("Radial curve")
    for axes in example_axes[len(examples) :]:  # Fewer than 3 images
        for ax in axes:
            ax.set_axis_off()

    fig.suptitle(f"{param_label} maps on the testing set")
    fig.savefig(figure_file, dpi=90)
    plt.close(fig)
    return errors
