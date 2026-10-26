import matplotlib
import numpy as np

matplotlib.use("Agg")
import h5py
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

SIM_TO_PHYSICAL_UNITS_MASS_FLUX = (
    (2 / np.pi) * 1 / (3.086 * np.power(10.0, 16.0)) * (3.154 * np.power(10.0, 7.0)) * np.power(10, 10)
)  # 1/pixel_res * kpc2km * s2yr * unit mass to solar masses


#### Plotting Functions ####
def make_comp_image(
    data_pred,
    data_input,
    data_image,
    output,
    index,
    sim_to_physical_units=SIM_TO_PHYSICAL_UNITS_MASS_FLUX,
    param_label="Radial Mass Flux",
    unit_label=r"[M$_{\odot}$ yr$^{-1}$]",
    bin_op="sum",
):
    """Save a 4-panel diagnostic figure (input map, prediction map, radially-binned
    input/prediction curves, and the summed input datacube) for one snapshot in a batch.

    data_pred / data_input: (nBatch, model_size) flattened prediction/label maps, or
        (nBatch, 1, npix, npix) already-2-d maps - both shapes are handled (see the
        try/except below, which falls back to the already-2-d case if the flat reshape fails).
    data_image: (nBatch, nSpec, nX, nY) input datacube batch, summed over the spectral axis
        for the bottom-right panel.
    output: path to save the PNG to.
    index: index of the sample within the batch to plot.
    sim_to_physical_units: scalar multiplied into the data before plotting, to convert from
        simulation units to physical units (see SIM_TO_PHYSICAL_UNITS_MASS_FLUX above).
    param_label/unit_label: axis/colorbar labels for the parameter being plotted.
    bin_op: statistic passed to scipy.stats.binned_statistic_dd for the radial profile
        ("sum" for mass flux, "mean" for velocity/inclination, etc.).
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

    plt.figure()

    plt.subplot(2, 2, 4)
    image_to_plot = np.sum(data_image[index, :, :, :], axis=0)
    image_to_plot += np.abs(np.min(image_to_plot)) + 1
    plt.imshow(image_to_plot, norm=LogNorm(), cmap="inferno")
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])

    plt.subplot(2, 2, 3)
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
    plt.subplot(2, 2, 1)
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
    plt.subplot(2, 2, 2)
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
    plt.savefig(output)
    plt.close()


def make_image(
    data,
    output,
    vmin=None,
    vmax=None,
    target_shape=None,
    sim_to_physical_units=SIM_TO_PHYSICAL_UNITS_MASS_FLUX,
    param_label="Radial Mass Flux",
    unit_label=r"[M$_{\odot}$ yr$^{-1}$]",
):
    """Reshape a single flattened prediction into a 2-d map, save it as a PNG (with a
    physical-units colorbar) and an accompanying hdf5 file.

    data: 1-d array of length npix*npix (a single sample's flattened prediction) - note the
        (1, N) shape convention (data[1] indexing) mirrors the batched functions above but
        here data is expected to already be for one sample.
    output: path prefix; writes output+"projectionMap.png"/".hdf5".
    vmin/vmax: fixed color limits; if either is None, both default to symmetric limits
        derived from the data's own min/max.
    target_shape: (nY, nX) shape to reshape the flat data into; pass this explicitly as a
        keyword when calling (not positionally - the 3rd positional argument is vmin, not
        target_shape).
    """
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]

    if target_shape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        target_shape = [npix, npix]

    image = np.reshape(image, target_shape)  # make into 2d image

    plt.figure()

    if vmax is None or vmin is None:
        vmax = np.max([-np.min(image), np.max(image)])  # default to symmetric limits
        vmin = -vmax

    # Plot in units of Solar Masses per year
    plt.imshow(
        image * sim_to_physical_units,
        cmap="seismic",
        vmin=vmin * sim_to_physical_units,
        vmax=vmax * sim_to_physical_units,
    )
    plt.colorbar(label=param_label + " " + unit_label)
    plt.savefig(output + "projectionMap.png")
    plt.close()

    # Save hdf5 for generating figures
    hf = h5py.File(output + "projectionMap.hdf5", "w")
    hf.create_dataset("image", data=image)
    hf.close()


def create_basic_plots(
    data,
    output,
    target_shape=None,
    n_bins=20,
    sim_to_physical_units=SIM_TO_PHYSICAL_UNITS_MASS_FLUX,
    param_label="Radial Mass Flux",
    unit_label=r"[M$_{\odot}$ yr$^{-1}$]",
):
    """Reshape a single flattened prediction into a 2-d map, azimuthally bin it into a
    radial profile by galactocentric pixel radius, and save the profile as a PNG + hdf5.

    data: 1-d array of length npix*npix for one sample (see make_image above for the shape
        convention).
    output: path prefix; writes output+"radialPlot.png"/".hdf5".
    target_shape: (nY, nX) shape to reshape data into; defaults to a square inferred from
        len(data).
    n_bins: number of radial bins to sum the map into.
    """
    image = np.zeros((np.shape(data)[1]))
    image[:] = data[:]
    if target_shape is None:
        npix = int(np.round(np.sqrt(np.size(image))))
        target_shape = [npix, npix]

    npix = target_shape[0]
    image = np.reshape(image, target_shape)  # make into 2d image

    ## Calculate galactocentric radius of each pixel
    center_index = [(npix - 1.0) / 2.0, (npix - 1.0) / 2.0]
    indices = np.indices(np.shape(image))
    rmag = 2 * np.sqrt(
        np.power(indices[0, :, :] - center_index[0], 2) + np.power(indices[1, :, :] - center_index[1], 2)
    )

    ##Calculate total mass flux as function of radius
    bin_range = [0, np.max(rmag)]
    binned_data, binedge, binnum = stats.binned_statistic_dd(
        rmag.flatten(), image.flatten(), "sum", n_bins, range=[bin_range]
    )

    # Default to symmetric y-limits
    vmax = np.max([-np.min(binned_data * sim_to_physical_units), np.max(binned_data * sim_to_physical_units)]) * 1.05
    vmin = -vmax

    # Plot in units of Solar Masses per year
    plt.figure()
    plt.plot(
        np.linspace(0, np.max(rmag), n_bins), binned_data * sim_to_physical_units, "k", lw=2.5
    )  # plot azimuthal average
    plt.plot(np.linspace(0, np.max(rmag), n_bins), binned_data * 0, "k--", lw=1.5)  # plot zero line
    plt.xlabel("Radius [kpc]")
    plt.ylabel(param_label + " " + unit_label)
    plt.ylim([vmin, vmax])
    plt.savefig(output + "radialPlot.png")
    plt.close()

    # Save hdf5 for generating figures
    hf = h5py.File(output + "radialPlot.hdf5", "w")
    hf.create_dataset("binned_data", data=binned_data)
    hf.create_dataset("r_plot", data=np.linspace(0, np.max(rmag), n_bins))

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


def make_correlation_plot(pred_image, input_image, n_snaps, output):
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

    npix = 40

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


def save_hdf5(input_data, pred_data, output, index):
    """Reshape one snapshot's flattened prediction (and optionally its label) into a 2-d
    map and save both to an hdf5 file for later re-plotting.

    input_data: (nBatch, model_size) flattened label maps, or None to skip saving the label
        (e.g. when running inference with no ground truth available).
    pred_data: (nBatch, model_size) flattened prediction maps.
    output: full path to the hdf5 file to write (datasets 'input' and/or 'prediction').
    index: index of the sample within the batch to save.
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
    hf.close()


def load_names(input_dir):
    """Extract a short, distinct name for each row of a CoNNGaFit annotations CSV, by taking
    the basename (without extension) of the image path in each line.

    input_dir: path to an annotations CSV (as produced for training/validation/testing, or an
        inference image list) - only the leading image-path portion of each line is used.
    Returns: a 0-indexed list of names, one per line/row, in file order - i.e. names[0]
        corresponds to the CSV's first data row. Callers must index this list starting at 0,
        not 1.
    """
    fid = open(input_dir, "r")
    names = []
    for line in fid:
        x = line.split("/")
        n_split = np.size(x)
        y = (x[n_split - 1]).split(".")  # get filename in directory
        if "_md" in line and "_md" not in y[0]:
            names.append(y[0] + "_md")
        else:
            names.append(y[0])  # ignore filetype

    fid.close()

    return names


def make_comp_image_1d(
    data_pred,
    data_input,
    data_image,
    output,
    index,
    sim_to_physical_units=SIM_TO_PHYSICAL_UNITS_MASS_FLUX,
    param_label="Radial Mass Flux",
    unit_label=r"[M$_{\odot}$ yr$^{-1}$]",
    bin_op="sum",
    radii=None,
):
    """1-d analog of make_comp_image: save a 2-panel figure (summed input datacube, and the
    input/prediction radial curves already stored in data_pred/data_input) for one snapshot.

    data_pred / data_input: (nBatch, nRadialBins) already-radially-binned prediction/label
        curves (unlike make_comp_image, these are not reshaped into a 2-d map here).
    data_image: (nBatch, nSpec, nX, nY) input datacube batch.
    output/index/sim_to_physical_units/param_label/unit_label: see make_comp_image above.
    radii: the curves' ring centers in kpc (see ring_radii in conngafit_datasets.py); if None, plot against bin number.
    """
    image_pred = np.zeros((np.shape(data_pred)[1]))
    image_pred[:] = data_pred[index, :]

    image_input = np.zeros((np.shape(data_input)[1]))
    image_input[:] = data_input[index, :]

    # npix = int(np.round(np.sqrt(np.size(image_pred))))
    # image_pred = np.reshape(image_pred,[npix,npix])
    # image_input = np.reshape(image_input,[npix,npix])

    fig, (ax_curve, ax_image) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    image_to_plot = np.sum(data_image[index, :, :, :], axis=0)
    image_to_plot += np.abs(np.min(image_to_plot)) + 1  # offset to positive values, for the log scale
    im = ax_image.imshow(image_to_plot, norm=LogNorm(), cmap="inferno")
    ax_image.xaxis.set_ticks([])
    ax_image.yaxis.set_ticks([])
    fig.colorbar(im, ax=ax_image, label="Input summed over channels (normalized)")

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
    plt.plot(rmag, np.array(data_input[index, :]) * sim_to_physical_units, "k", lw=2)
    plt.plot(rmag, np.array(data_pred[index, :]) * sim_to_physical_units, "r--", lw=2)
    plt.plot(rmag, rmag * 0, "k--", lw=1)
    plt.legend(["Input", "Prediction"])
    plt.xlabel("Radial bin" if radii is None else "Radius [kpc]")
    plt.ylabel(param_label + " " + unit_label)
    plt.ylim([-max_plot * 1.05 * sim_to_physical_units, max_plot * 1.05 * sim_to_physical_units])

    plt.savefig(output)
    plt.close()


def save_hdf5_1d(input_data, pred_data, output, index):
    """1-d analog of save_hdf5: save one snapshot's already-1-d input/prediction radial
    curves to an hdf5 file. Note index is currently unused - the full input_data/pred_data
    arrays are saved as-is rather than being indexed down to a single snapshot; pass
    already-sliced 1-d arrays for a single sample.
    """
    hf = h5py.File(output, "w")
    hf.create_dataset("input", data=input_data)
    hf.create_dataset("prediction", data=pred_data)
    hf.close()
