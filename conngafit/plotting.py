"""Plotting functions for the diagnostic figures and hdf5 snapshots that training and inference save: predictions
against labels, as maps or radial curves."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import h5py
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


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


def plot_history(output, result, validation_names, loss_label, accuracy_tolerance):
    """Plot a run's training and validation losses (loss_label names the training objective), its validation
    accuracy, and each validation sample's accuracy, to output + "_trainingLoss.png" and so on. result is
    train_model.train's TrainingResult; accuracy_tolerance is the accuracy's tolerance, in label standard deviations."""
    plot_loss(result.training_losses, "Training Loss", loss_label, output + "_trainingLoss.png")
    plot_loss(result.validation_losses, "Validation Loss", loss_label, output + "_validationLoss.png")

    # Accuracies are NaN on the epochs between predict_every, so only the predicted epochs are plotted
    epoch_numbers = np.arange(1, len(result.validation_accuracies) + 1)
    predicted = np.isfinite(result.validation_accuracies)

    plt.figure()
    plt.plot(epoch_numbers[predicted], result.validation_accuracies[predicted], marker="o", ms=3)
    plt.title(f"Validation Accuracy (within {accuracy_tolerance:g} label std. dev.)")
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
