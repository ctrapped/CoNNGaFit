"""Plotting functions shared by examples/demo.py and the scripts in experiments/. save_demo_figure draws and saves
examples/demo.py's whole figure. demo_figure lays out a figure, and each plot_ function draws one part of it from
arrays onto the axes it's given: the input maps, the rotation curves and their residuals, a training history, and
the predicted maps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

from conngafit.datasets import TARGET_SETTINGS, ring_radii
from demo_utils import COLUMN_DENSITY_LIMITS, DETECTION, LOS_VELOCITY_LIMITS, MAP_PANELS, SNR_LIMITS, moment_maps

# The training objective's title, axis label and scale, for networks trained with predict_variance: the Gaussian NLL
# can be < 0
NLL_LOSS = ("Gaussian NLL", "NLL (training objective)", "linear")


def save_demo_figure(
    figure_file, images, position_angles, curves, sigmas, result, target, true_maps=None, maps=None, map_sigmas=None
):
    """Save examples/demo.py's figure to figure_file: the first test image's input maps, every test image's true and
    predicted rotation curves and their residuals, with the predicted 1-sigma, and the training history; with maps,
    also the first image's true and predicted maps of target, their difference and signal-to-noise.

    images: the test images, from demo_utils.load_images; position_angles: theirs, in degrees.
    curves, sigmas: (n_images, n_rings) predicted rotation curves and their 1-sigma.
    result: the TrainingResult train() returned.
    true_maps, maps, map_sigmas: (n_images, nY, nX) true and predicted maps of target and the predicted 1-sigma, for
        a network that predicts maps.
    """
    fig, ((ax_density, ax_velocity, ax_curve), (ax_residual, ax_mse, ax_loss)), map_axes = demo_figure(
        n_map_rows=0 if maps is None else 1
    )
    column_density, velocity = moment_maps(images["cubes"][0], images["channel_widths"][0])
    half_width = images["fovs_kpc"][0] / 2
    extent = [-half_width, half_width, -half_width, half_width]
    plot_inputs((ax_density, ax_velocity), column_density, velocity, extent, COLUMN_DENSITY_LIMITS, LOS_VELOCITY_LIMITS)
    radii = ring_radii(images["fovs_kpc"][0], curves.shape[1])
    plot_curves(ax_curve, ax_residual, radii, images["true_curves"], curves, sigmas, position_angles)
    # The MSE of the predicted mean is NaN on epochs whose predictions weren't scored
    plot_history(ax_mse, result.training_mses, result.validation_mses, "MSE of the mean", "MSE (scaled labels)", "log")
    plot_history(ax_loss, result.training_losses, result.validation_losses, *NLL_LOSS)
    if maps is not None:
        label = " ".join(TARGET_SETTINGS[target][:2])  # e.g. "Rot. Vel. [km s$^{-1}$]"
        undetected = column_density < DETECTION  # Hidden in the map panels
        plot_map(
            map_axes[0],
            label,
            true_maps[0],
            maps[0],
            map_sigmas[0],
            undetected,
            extent,
            SNR_LIMITS,
            **MAP_PANELS[target],
        )
    fig.savefig(figure_file, dpi=110, bbox_inches="tight")
    plt.close(fig)


def demo_figure(n_map_rows=0, n_map_columns=4):
    """(figure, curve axes, map axes): a 2 x 3 grid of axes on top for the input maps and rotation curve, then the
    residuals and two training histories; below it, n_map_rows rows of n_map_columns axes for the predicted maps
    (plot_map fills 4 per row), or None if n_map_rows is 0."""
    fig = plt.figure(figsize=(14.25, 4.6 * (2 + n_map_rows)), layout="constrained")
    if n_map_rows == 0:
        return fig, fig.subplots(2, 3), None
    curves_figure, maps_figure = fig.subfigures(2, 1, height_ratios=[2, n_map_rows])
    return fig, curves_figure.subplots(2, 3), maps_figure.subplots(n_map_rows, n_map_columns, squeeze=False)


def plot_image(ax, image, label, extent, **style):
    """Show image on ax with a labeled color bar on top; extent is in kpc and style is passed to imshow."""
    im = ax.imshow(image, origin="lower", extent=extent, **style)
    colorbar = ax.figure.colorbar(im, ax=ax, location="top", shrink=0.9, pad=0.02)
    colorbar.set_label(label)
    # A scale factor such as 1e9 goes beside the color bar, clear of its label
    colorbar.ax.xaxis.get_offset_text().set(x=1.02, horizontalalignment="left", verticalalignment="center")
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]")


def plot_inputs(axes, column_density, velocity, extent, column_density_limits, velocity_limits):
    """The column density (cm^-2, on a log scale) and line-of-sight velocity (km/s) maps on the two axes, within their
    (min, max) color limits; extent is in kpc."""
    plot_image(
        axes[0],
        np.clip(column_density, column_density_limits[0], None),
        "Input: HI column density [cm$^{-2}$]",
        extent,
        norm=LogNorm(*column_density_limits),
        cmap="inferno",
    )
    plot_image(
        axes[1],
        velocity,
        "Input: line-of-sight velocity [km/s]",
        extent,
        vmin=velocity_limits[0],
        vmax=velocity_limits[1],
        cmap="RdBu_r",
    )


def plot_curves(ax_curve, ax_residual, radii, true_curves, curves, sigmas, position_angles):
    """The true and predicted rotation curves at radii (kpc) on ax_curve, and their residuals on ax_residual, each
    with its predicted ±1-sigma band; true_curves, curves and sigmas are (n_images, n_rings), for images seen at
    position_angles in degrees."""
    for i, position_angle in enumerate(position_angles):
        residual = curves[i] - true_curves[i]
        ax_curve.plot(radii, true_curves[i], "k", lw=2, label="True" if i == 0 else None)
        (line,) = ax_curve.plot(radii, curves[i], "--", marker="o", ms=3, label=f"Predicted, PA {position_angle:g}°")
        color = line.get_color()
        ax_curve.fill_between(radii, curves[i] - sigmas[i], curves[i] + sigmas[i], color=color, alpha=0.15, lw=0)
        # The same bands around each residual: a ring is within its 1-sigma where its band crosses zero
        ax_residual.plot(radii, residual, "--", marker="o", ms=3, color=color, label=f"PA {position_angle:g}°")
        ax_residual.fill_between(radii, residual - sigmas[i], residual + sigmas[i], color=color, alpha=0.15, lw=0)
    ax_curve.set(xlabel="Radius [kpc]", ylabel="Rotational velocity [km/s]", title="Test galaxy's rotation curve")
    ax_curve.set_ylim(bottom=0)
    ax_residual.axhline(0, color="k", lw=2)
    ax_residual.set(xlabel="Radius [kpc]", ylabel="Predicted − true [km/s]", title="Residuals")
    # One legend entry for every line's band, in grey since the bands share their line's color
    sigma_band = Patch(color="grey", alpha=0.3, lw=0, label="Predicted ±1σ")
    for ax in (ax_curve, ax_residual):
        ax.legend(handles=[*ax.get_legend_handles_labels()[0], sigma_band], fontsize=8)


def plot_history(ax, training, validation, title, ylabel, yscale):
    """A training and validation history, one value per epoch, on ax; epochs where it is NaN (not evaluated) are
    skipped."""
    epochs = np.arange(1, len(training) + 1)
    evaluated = np.isfinite(training)
    ax.plot(epochs[evaluated], training[evaluated], marker="o", label="Training")
    ax.plot(epochs[evaluated], validation[evaluated], marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel=ylabel, yscale=yscale, title=title)
    ax.legend()


def plot_map(
    axes, label, true_map, predicted_map, sigma_map, mask, extent, snr_limits, log_scale, value_limits, difference_limit
):
    """A row of 4 axes for one map: its true and predicted values within value_limits (min, max), how they differ,
    within ±difference_limit, and the prediction's signal-to-noise (predicted / its 1-sigma) within snr_limits. The
    maps are (nY, nX), label names the map with its units, pixels where mask is True are hidden, and extent is in kpc.

    With log_scale, the values are on a log color scale and differ by log10(predicted / true), in dex; pixels that are
    not positive, such as counter-rotating gas in a momentum map, are hidden on these panels. Otherwise the values are
    on a linear scale and differ by predicted - true, for maps that can be negative, such as velocity."""
    true_map = np.ma.masked_where(mask, true_map)
    predicted_map = np.ma.masked_where(mask, predicted_map)
    if log_scale:
        value_style = {"norm": LogNorm(*value_limits), "cmap": "magma"}
        # Masked division and log10 also hide pixels where the true map is zero or the ratio is not positive
        difference = np.ma.log10(predicted_map / true_map)
        difference_label = "log$_{10}$(predicted / true)"
    else:
        value_style = {"vmin": value_limits[0], "vmax": value_limits[1], "cmap": "magma"}
        difference = predicted_map - true_map
        difference_label = "Predicted − true"
    plot_image(axes[0], true_map, f"True {label}", extent, **value_style)
    plot_image(axes[1], predicted_map, f"Predicted {label}", extent, **value_style)
    plot_image(
        axes[2], difference, difference_label, extent, vmin=-difference_limit, vmax=difference_limit, cmap="RdBu_r"
    )
    plot_image(
        axes[3],
        predicted_map / sigma_map,
        "Predicted / 1σ",
        extent,
        vmin=snr_limits[0],
        vmax=snr_limits[1],
        cmap="viridis",
    )
