import matplotlib
import numpy as np

matplotlib.use("Agg")
import h5py
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm



#### Plotting Functions ####
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

    data_pred / data_input: (nBatch, model_size) flattened prediction/label maps,
        or (nBatch, 1, npix, npix) maps.
    data_image: (nBatch, nSpec, nX, nY) input datacube batch.
    output: path to save the PNG to.
    index: index of the sample within the batch to plot.
    sim_to_physical_units: scalar multiplied into the data before plotting
    param_label/unit_label: axis/colorbar labels for the parameter being plotted.
    bin_op: statistic passed to scipy.stats.binned_statistic_dd for the radial profile
        ("sum" for mass flux, "mean" for velocity/inclination, etc.).
    uncertainty: the predicted 1-sigma of each value, in the same units and either layout as
        data_pred, or None (the default) for the 4-panel figure.
    """
    try:
        image_pred = np.zeros((np.shape(data_pred)[1]))
        image_pred[:] = data_pred[index, :]

        image_input = np.zeros((np.shape(data_input)[1]))
        image_input[:] = data_input[index, :]

        npix = int(np.round(np.sqrt(np.size(image_pred))))
        image_pred = np.reshape(image_pred, [npix, npix])
        image_input = np.reshape(image_input, [npix, npix])
    except ValueError:  # data_pred/data_input were already (nBatch,1,npix,npix) rather than flattened
        image_pred = data_pred[index, 0, :, :]
        image_input = data_input[index, 0, :, :]
        npix = int(np.round(np.sqrt(np.size(image_pred))))

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
        sigma_map = np.reshape(np.asarray(uncertainty)[index], np.shape(image_pred))
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
    target_shape=None,
    sim_to_physical_units=1,
    param_label="Radial Momentum",
    unit_label=r"[M$_{\odot}$ km s$^{-1}$]",
    uncertainty=None,
):
    """Reshape a single flattened prediction into a 2-d map, save it as a PNG (with a
    physical-units colorbar) and an accompanying hdf5 file. With uncertainty, the predicted
    1-sigma map is plotted beside the prediction and saved too.

    data: 1-d array of length npix*npix (a single sample's flattened prediction)
    output: path prefix; writes output+"projectionMap.png"/".hdf5".
    vmin/vmax: fixed color limits; Defaults to symmetric limits.
    target_shape: (nY, nX) shape to reshape the flat data into
    uncertainty: the predicted 1-sigma of each value
    """
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]

    if target_shape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        target_shape = [npix, npix]

    image = np.reshape(image, target_shape)  # make into 2d image

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
        sigma_map = np.reshape(uncertainty, image.shape)
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
    """Reshape one snapshot's flattened prediction (and optionally its label and predicted
    1-sigma) into 2-d maps and save them to an hdf5 file for later re-plotting.

    input_data: (nBatch, model_size) flattened label maps, or None to skip saving the label
        (e.g. when running inference with no ground truth available).
    pred_data: (nBatch, model_size) flattened prediction maps.
    output: full path to the hdf5 file to write (datasets 'input', 'prediction' and 'uncertainty').
    index: index of the sample within the batch to save.
    uncertainty: (nBatch, model_size) flattened predicted 1-sigma maps, or None (the default) to skip it.
    """
    pred_image = np.zeros((np.shape(pred_data)[1]))
    pred_image[:] = pred_data[index, :]
    npix = int(np.round(np.sqrt(np.size(pred_image))))
    pred_image = np.reshape(pred_image, [npix, npix])

    hf = h5py.File(output, "w")
    if input_data is not None:
        input_image = np.zeros((np.shape(input_data)[1]))
        input_image[:] = input_data[index, :]
        npix = int(np.round(np.sqrt(np.size(input_image))))
        input_image = np.reshape(input_image, [npix, npix])
        hf.create_dataset("input", data=input_image)
    hf.create_dataset("prediction", data=pred_image)
    if uncertainty is not None:
        hf.create_dataset("uncertainty", data=np.reshape(uncertainty[index, :], pred_image.shape))
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
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[index, :]

    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[index, :]

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


def save_hdf5_1d(input_data, pred_data, output, index, uncertainty=None):
    """1-d analog of save_hdf5: save sample index's radial curves (and optionally their
    predicted 1-sigma) from (nBatch, n_bins) arrays to an hdf5 file."""
    hf = h5py.File(output, "w")
    hf.create_dataset("input", data=input_data[index])
    hf.create_dataset("prediction", data=pred_data[index])
    if uncertainty is not None:
        hf.create_dataset("uncertainty", data=uncertainty[index])
    hf.close()



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
        diego = np.diag(np.diag(scal_arr))
        scal_arr_mod = scal_arr - diego
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

    pred_image / input_image: (n_snaps, model_size) flattened prediction/label maps.
    n_snaps: number of snapshots (rows) to score.
    output: path prefix; writes output+".png"/".pdf"/".hdf5".

    For each snapshot, reshapes to a 2-d npix x npix map and computes:
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
        pred_snap = pred_image[i, :]
        input_snap = input_image[i, :]

        npix = int(np.round(np.sqrt(np.size(pred_snap))))
        pred_snap = np.reshape(pred_snap, [npix, npix])
        input_snap = np.reshape(input_snap, [npix, npix])

        rv2_mat = rv2_coeff([input_snap, pred_snap])
        corr2d[i] = rv2_mat[0, 1]

        center_index = [(npix - 1.0) / 2.0, (npix - 1.0) / 2.0]
        indices = np.indices(np.shape(pred_snap))
        rmag = np.sqrt(
            np.power(indices[0, :, :] - center_index[0], 2) + np.power(indices[1, :, :] - center_index[1], 2)
        )

        n_bins = npix
        bin_range = [0, np.max(rmag)]
        binned_pred, binedge, binnum = stats.binned_statistic_dd(
            rmag.flatten(), pred_snap.flatten(), "sum", n_bins, range=[bin_range]
        )
        binned_input, binedge, binnum = stats.binned_statistic_dd(
            rmag.flatten(), input_snap.flatten(), "sum", n_bins, range=[bin_range]
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