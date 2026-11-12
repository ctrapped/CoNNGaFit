"""Train a CoNNGaFit network: train() takes the hyperparameters and datasets as dicts, with the same keys as the
files in configs/ (see configs/*.toml for what each one does; networks.py lists each network's hyperparameter keys).
scripts/train_model_unet.py runs it from the command line on those files.

train() saves the trained network to <network_dir>/<run_name>.pt and .hdf5, and diagnostic images and plots to
<output_dir>/images/ and <output_dir>/diagnostics/.
"""

import json
import logging
import os
import random
import tomllib
from dataclasses import dataclass
from functools import partial

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit import plotting
from conngafit.datasets import TARGET_SETTINGS, CoNNGaFitImageDataset, inclination_folders
from conngafit.models.unet3d_ddpm import GaussianDiffusion
from conngafit.networks import HYPERPARAM_KEYS, OPTIONAL_HYPERPARAM_KEYS, build_model, model_summary

logger = logging.getLogger(__name__)

# Keys every datasets dict must have, and may have
DATASET_KEYS = {
    "target",
    "augment",
    "inclinations",
    "radial_bins",
    "radial_projection",
    "training",
    "validation",
    "testing",
}
OPTIONAL_DATASET_KEYS = {"data_dir", "suffix"}
# Predictions within this many standard deviations of their target's training labels count as accurate
ACCURACY_TOLERANCE = 0.1


def check_keys(params, keys, optional_keys, name):
    """Raise ValueError if params is missing any of keys, or has a key that isn't in keys or optional_keys, so typos
    aren't silently ignored."""
    missing, unknown = keys - params.keys(), params.keys() - keys - optional_keys
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {sorted(missing)}, unrecognized keys {sorted(unknown)}")


def to_toml(params):
    """TOML text of a parameter dict: each value, then each dict value as a [table] (JSON literals are valid TOML for
    strings, numbers, booleans and lists)."""

    def line(key, value):
        key = key if key.isidentifier() else json.dumps(key)
        return f"{key} = {json.dumps(value, default=str)}"  # default=str writes Paths as strings

    values = [line(key, value) for key, value in params.items() if not isinstance(value, dict)]
    tables = []
    for name, table in params.items():
        if isinstance(table, dict):
            tables += [f"[{name}]"] + [line(str(key), value) for key, value in table.items()]
    return "\n".join(values + tables) + "\n"


def load_hyperparams(path):
    """The hyperparameters in a TOML file, as a dict for train()."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_datasets(path):
    """The datasets in a TOML file, as a dict for train(). Its galaxy directories are relative to data_dir, which is
    itself relative to the file's folder (and is the file's folder if not given)."""
    with open(path, "rb") as f:
        datasets = tomllib.load(f)
    data_dir = os.path.expanduser(datasets.get("data_dir", ""))
    datasets["data_dir"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), data_dir))
    return datasets


def dataset_sources(section, name, data_dir=None):
    """Check a datasets section ({VOF output directory: snapshots}) and resolve each directory relative to data_dir
    (default: the current directory). Returns the dict datasets.py expects."""
    if not isinstance(section, dict):
        raise ValueError(f"datasets: {name} should be a dict of VOF output directories and their snapshots")
    base = os.path.expanduser(str(data_dir or ""))
    return {os.path.abspath(os.path.join(base, os.path.expanduser(d))): snaps for d, snaps in section.items()}


def check_hyperparams(hyperparams):
    """Raise ValueError if hyperparams is missing a key its network needs, has an unrecognized key, or has an
    invalid value."""
    network = hyperparams.get("network")  # Sets which keys hyperparams must have
    if network not in HYPERPARAM_KEYS:
        raise ValueError(f"hyperparams: network = {network!r} should be one of {sorted(HYPERPARAM_KEYS)}")
    check_keys(hyperparams, HYPERPARAM_KEYS[network], OPTIONAL_HYPERPARAM_KEYS[network], "hyperparams")
    label_scaling = hyperparams["label_scaling"]
    if label_scaling not in ("none", "symmetric", "one-sided"):
        raise ValueError(f'hyperparams: label_scaling = {label_scaling!r} should be "none", "symmetric" or "one-sided"')
    mass_weight_power = hyperparams.get("mass_weight_power", 0)
    if not mass_weight_power >= 0:
        raise ValueError(f"hyperparams: mass_weight_power = {mass_weight_power!r} should be 0 or more")
    predict_variance = hyperparams.get("predict_variance", False)
    if not isinstance(predict_variance, bool):
        raise ValueError(f"hyperparams: predict_variance = {predict_variance!r} should be true or false")
    if predict_variance and label_scaling == "none":
        raise ValueError(
            'hyperparams: predict_variance = true needs label_scaling = "symmetric" or "one-sided", '
            "since nn.GaussianNLLLoss floors the variance at 1e-6 in the units of the labels it trains on"
        )
    predict_every = hyperparams["predict_every"]
    if not (isinstance(predict_every, int) and predict_every >= 1):
        raise ValueError(f"hyperparams: predict_every = {predict_every!r} should be an integer of 1 or more")


def check_datasets(datasets):
    """Raise ValueError if datasets is missing a key, has an unrecognized key, or has invalid inclinations or suffix.
    The training, validation and testing sources are checked by dataset_sources, and the targets by the datasets."""
    check_keys(datasets, DATASET_KEYS, OPTIONAL_DATASET_KEYS, "datasets")
    inclination_folders(datasets["inclinations"], "datasets")
    suffix = datasets.get("suffix", "")
    if not isinstance(suffix, str):
        raise ValueError(f"datasets: suffix = {suffix!r} should be a string")


@dataclass
class TrainingResult:
    """What train() returns: the trained network's path (without .pt/.hdf5), each epoch's losses (the training
    objective), MSEs of the predicted mean and validation accuracies (NaN on epochs between predict_every), each
    validation sample's accuracy every epoch, (n_validation, epochs), and the test set's final loss, MSE, accuracy and
    accuracy of each sample."""

    model_path: str
    training_losses: np.ndarray
    validation_losses: np.ndarray
    training_mses: np.ndarray
    validation_mses: np.ndarray
    validation_accuracies: np.ndarray
    specific_accuracies: np.ndarray
    final_test_loss: float
    final_test_mse: float
    final_test_accuracy: float
    final_test_specific_accuracy: np.ndarray


def weighted_mse_loss(pred, y, w):
    """Mean squared error, each pixel (or ring) weighted by w; all ones is plain MSE."""
    return torch.mean(w * nn.functional.mse_loss(pred, y, reduction="none"))


def weighted_gaussian_nll_loss(pred, y, logvar, w):
    """Gaussian NLL of (pred, exp(logvar)), each pixel (or ring) weighted by w; all ones is nn.GaussianNLLLoss."""
    return torch.mean(w * nn.functional.gaussian_nll_loss(pred, y, logvar.exp(), reduction="none"))


def compute_loss(model, x, y, w, predict=True, n_samples=None):
    """
    Returns (loss, prediction, 1-sigma) for a batch of cubes x and labels y, with weights by w. The
    prediction and 1-sigma are None unless predict=True.

    resblock: If the network predicts its variance, use Gaussian NLL of (mean, exp(logvar)) with sigma = exp(logvar / 2)
              Otherwise, use MSE loss, with sigma = None.
    ddpm: the denoising loss (unweighted), and the mean and standard deviation of n_samples posterior
    samples, which are only drawn if predict since this has high compute cost.
    """
    if isinstance(model, GaussianDiffusion):
        loss = model(x, y)
        if not predict:
            return loss, None, None
        pred, sigma = model.predict_with_uncertainty(x, n_samples)
        return loss, pred, sigma

    output = model(x)
    if model.predict_variance:
        pred, logvar = output
        loss, sigma = weighted_gaussian_nll_loss(pred, y, logvar, w), torch.exp(0.5 * logvar)
    else:
        pred, sigma = output, None
        loss = weighted_mse_loss(pred, y, w)
    if not predict:
        return loss, None, None
    return loss, pred, sigma


def count_accurate(pred, y, tolerance):
    """Return (each sample's accuracy [%], number of accurate values) for a batch of scaled predictions and labels:
    the values within tolerance of their label."""
    within = np.abs(pred - y) < tolerance
    return within.mean(axis=tuple(range(1, within.ndim))) * 100, within.sum()  # Over each sample's every value


def save_batch_outputs(
    x,
    y,
    pred,
    sigma,
    idx,
    dataset,
    image_output,
    epoch,
    targets,
    to_physical,
    label_scale,
    radial_bins,
    comp_images=True,
    saved=None,
):
    """Save each sample's map (or curve) of each target in a batch as an hdf5 snapshot (and, with comp_images, a
    comparison image), named after its image via its dataset index (idx), so names stay correct with shuffled
    batches, after the target and after the epoch.

    x, y, pred, sigma: the batch's cubes, labels, predictions and their 1-sigma (see compute_loss, or None), in
        scaled label units; to_physical and label_scale (see train) convert them back to physical units.
    saved: if given, a list the batch's (predictions, labels) in physical units are appended to, for
        plotting.save_correlation_plots.
    """
    np_x = x.cpu().float().detach().numpy()
    np_y = to_physical(y.cpu().float().detach().numpy())
    np_pred = to_physical(pred.cpu().detach().numpy())
    # 1-sigma in physical units: a spread, so scaled by label_scale but not offset
    np_sigma = None if sigma is None else sigma.cpu().detach().numpy() * label_scale
    # Radial curves are plotted as curves, not maps
    make_comp_image = plotting.make_comp_image_1d if radial_bins is not None else plotting.make_comp_image
    for m, target in enumerate(targets):
        param_label, unit_label, bin_op = TARGET_SETTINGS[target]
        target_sigma = None if np_sigma is None else np_sigma[:, m]
        for tt in range(len(idx)):
            output_name = f"{image_output}_{dataset.names[int(idx[tt])]}_{target}_Epoch{epoch}"
            if comp_images:
                # Radial curves are plotted against the radius of each ring, from the image's field of view
                radii = {"radii": dataset.radii(int(idx[tt]))} if radial_bins is not None else {}
                make_comp_image(
                    np_pred[:, m],
                    np_y[:, m],
                    np_x,
                    output_name + ".png",
                    tt,
                    sim_to_physical_units=1,
                    param_label=param_label,
                    unit_label=unit_label,
                    bin_op=bin_op,
                    uncertainty=target_sigma,
                    **radii,
                )  # Labels are already in physical units
            plotting.save_hdf5(np_y[:, m], np_pred[:, m], output_name + ".hdf5", tt, uncertainty=target_sigma)
    if saved is not None:
        saved.append((np_pred, np_y))


def train_loop(model, dataloader, optimizer, device, predict, n_samples=None, save_batch=None):
    """Run one training epoch: iterate the dataloader and backprop compute_loss each batch.

    predict: also predict each batch, to score the MSE of the predicted mean (made every predict_every epochs).
    save_batch: if given, called with each batch's (x, y, pred, sigma, idx) to save its outputs (see
        save_batch_outputs).
    Returns (loss, MSE of the predicted mean), each averaged over all batches in the epoch; the MSE is NaN if no
    batch was predicted.
    """
    model.train()
    epoch_loss, epoch_mse = 0.0, 0.0
    n_predicted = 0  # Batches with predictions, made every predict_every epochs
    num_batches = len(dataloader)

    for batch, (x, y, w, idx) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        loss, pred, sigma = compute_loss(model, x.float(), y.float(), w.float(), predict, n_samples)
        epoch_loss += loss.item()
        if pred is not None:
            n_predicted += 1
            epoch_mse += nn.functional.mse_loss(pred, y.float()).item()
        if save_batch is not None:
            save_batch(x, y, pred, sigma, idx)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            logger.info("loss: %7f [%5d/%5d]", loss.item(), batch * len(x), len(dataloader.dataset))

    return epoch_loss / num_batches, (epoch_mse / n_predicted if n_predicted else np.nan)


def evaluate(model, dataloader, device, predict, accuracy_tolerance, name, n_samples=None, save_batch=None):
    """Evaluate the model on the validation or test set (no gradient updates) and compute a rough "accuracy": the
    percentage of values within accuracy_tolerance of their label. name ("Validation" or "Test") labels the log line.

    predict: also predict each batch, to score the MSE of the predicted mean and the accuracy (made every
        predict_every epochs and on the final epoch).
    save_batch: if given, called with each batch's (x, y, pred, sigma, idx) to save its outputs (see
        save_batch_outputs).
    Returns (loss, mse, accuracy, specific_accuracy) where mse is the MSE of the predicted mean and
    specific_accuracy is the accuracy of every sample in the set. The mse and accuracies are NaN without predict.
    """
    model.eval()
    num_batches = len(dataloader)
    summed_loss, summed_mse, correct = 0, 0, 0
    total = 0
    specific_accuracy = np.full(len(dataloader.dataset), np.nan)

    with torch.no_grad():
        for x, y, w, idx in dataloader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            loss, pred, sigma = compute_loss(model, x.float(), y.float(), w.float(), predict, n_samples)
            summed_loss += loss.item()
            if pred is None:  # Predictions are made every predict_every epochs
                continue
            summed_mse += nn.functional.mse_loss(pred, y.float()).item()

            np_pred = pred.cpu().float().detach().numpy()
            np_y = y.cpu().float().detach().numpy()

            specific_accuracy[idx.numpy()], n_correct = count_accurate(np_pred, np_y, accuracy_tolerance)
            correct += n_correct
            total += np.size(np_y)

            if save_batch is not None:
                save_batch(x, y, pred, sigma, idx)

    accuracy = correct / total * 100 if total else np.nan
    mean_loss = summed_loss / num_batches
    mse = summed_mse / num_batches if total else np.nan
    logger.info("%s: accuracy %.1f%%, avg loss %8f, avg MSE %8f", name, accuracy, mean_loss, mse)

    model.train()
    return mean_loss, mse, accuracy, specific_accuracy


def training_statistics(dataset):
    """One pass over the training files, one cube in memory at a time: the image normalization (the mean and
    standard deviation of every pixel of every cube) and each file's label, (n_maps, nY, nX) or (n_maps, n_rings)."""
    # Every cube has the same size, so the training set's mean (and mean of squares) is the mean of each cube's
    cube_means, cube_mean_squares, labels = [], [], []
    for i in range(len(dataset.img_paths)):
        cube = dataset.image(i)
        cube_means.append(cube.mean(dtype=np.float64))
        cube_mean_squares.append(np.square(cube, dtype=np.float64).mean())
        labels.append(dataset.label(i))
    training_mean = np.mean(cube_means)
    training_stdv = np.sqrt(np.mean(cube_mean_squares) - training_mean**2)  # var = E[x²] - E[x]²
    logger.info("Image normalization: mean = %.4g, stdv = %.4g", training_mean, training_stdv)
    return training_mean, training_stdv, labels


def calc_label_scaling(labels, label_scaling, targets):
    """Each target's label offset and scale, from every training label. Labels are scaled to
    (label - label_offset) / label_scale for training and predictions. labels are scaled back to physical units
    for every saved output and plot. Also gives an accuracy tolerance, in scaled units.

    Returns (label_offset, label_scale, accuracy_tolerance), each shaped to broadcast over a target's map or curve:
    (n_maps, 1, 1) for maps, (n_maps, 1) for radial curves.
    """
    n_maps = len(targets)
    all_labels = np.concatenate([label.reshape(n_maps, -1) for label in labels], axis=1)  # (n_maps, every value)
    label_offset, label_scale = np.zeros((n_maps, 1)), np.ones((n_maps, 1))
    if label_scaling != "none":
        if label_scaling == "symmetric":  # e.g. radial momentum: divide by max(|label|), to [-1, 1]
            label_scale = np.max(np.abs(all_labels), axis=1, keepdims=True)
        else:  # "one-sided", e.g. rotational velocity: [min(label), max(label)] to [-1, 1]
            label_max = np.max(all_labels, axis=1, keepdims=True)
            label_min = np.min(all_labels, axis=1, keepdims=True)
            label_offset = (label_max + label_min) / 2
            label_scale = (label_max - label_min) / 2
        for target, offset, scale in zip(targets, label_offset[:, 0], label_scale[:, 0]):
            if not scale > 0:
                raise ValueError(
                    f"label_scaling = {label_scaling!r} needs {target} training labels that aren't constant"
                )
            logger.info("Label scaling of %s (%s): offset = %.4g, scale = %.4g", target, label_scaling, offset, scale)

    accuracy_tolerance = ACCURACY_TOLERANCE * np.std(all_labels, axis=1, keepdims=True) / label_scale
    for target, tolerance in zip(targets, (accuracy_tolerance * label_scale)[:, 0]):
        logger.info("Accuracy tolerance of %s = %.4g (physical units)", target, tolerance)

    # Shape each target's statistics, (n_maps, 1) so far, to broadcast over its map (or curve)
    per_target = (n_maps,) + (1,) * (labels[0].ndim - 1)
    return label_offset.reshape(per_target), label_scale.reshape(per_target), accuracy_tolerance.reshape(per_target)


def make_dataloaders(training_data, validation_data, test_data, batch_size=None, test_batch_size=None):
    """DataLoaders for the training, validation and test sets. The training set is reshuffled every epoch.

    batch_size: training batch size. Default: about a tenth of the training set, at least 2 (BatchNorm needs more
        than 1 sample per batch while training); 0 trains on the whole set as one batch.
    test_batch_size: validation and testing batch size. Default: about a tenth of the validation set, at least 1
        (evaluation uses BatchNorm's stored statistics, so 1 is safe); with batch_size 0, the whole validation set.
    """
    if batch_size is None:
        batch_size = max(2, len(training_data) // 10)
    if batch_size == 0:  # Full-batch training, and evaluation
        batch_size, test_batch_size = len(training_data), len(validation_data)
    elif test_batch_size is None:
        test_batch_size = max(1, len(validation_data) // 10)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=test_batch_size)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    logger.info("Data loaded: %d training batches of up to %d samples per epoch", len(training_loader), batch_size)
    return training_loader, validation_loader, test_loader


def save_checkpoint(
    model_path,
    model,
    *,
    image_stats,
    label_offset,
    label_scale,
    hyperparams,
    hyperparams_text,
    datasets_text,
    run_name,
    input_shape,
    targets,
    radial_bins,
    radial_projection,
):
    """Save the trained network's weights to model_path + ".pt", and to model_path + ".hdf5" everything needed to
    rebuild and use it (see use_model.load_model): the image normalization (image_stats, its mean and standard
    deviation), each target's label scaling, every hyperparameter, and the parameter text of both files."""
    torch.save(model.state_dict(), model_path + ".pt")

    with h5py.File(model_path + ".hdf5", "w") as hf_norm:
        hf_norm.create_dataset("image_stats", data=np.array(image_stats))
        hf_norm.create_dataset(
            "label_stats", data=np.stack([label_offset.ravel(), label_scale.ravel()])
        )  # (2, n_maps): physical label = scaled label * label_scale + label_offset, for each target
        for key, value in hyperparams.items():  # Every hyperparameter of the network, including network itself
            hf_norm.create_dataset(key, data=value)
        hf_norm.create_dataset("run_name", data=run_name)
        hf_norm.create_dataset("input_shape", data=input_shape)  # Needed to rebuild the network for inference
        # In the order of the network's output maps
        hf_norm.create_dataset("target", data=np.array(targets, dtype=h5py.string_dtype()))
        if radial_bins is not None:
            hf_norm.create_dataset("radial_bins", data=radial_bins)
            hf_norm.create_dataset("radial_projection", data=radial_projection)
        # The full parameter files, so every trained model records exactly what it was trained with
        hf_norm.create_dataset("hyperparams_text", data=hyperparams_text)
        hf_norm.create_dataset("datasets_text", data=datasets_text)


def save_history(output, result, training_names, validation_names, testing_names):
    """Save a run's loss history (a TrainingResult) and its sets' sample names to output + "_LossesAndAccuracies.hdf5"."""
    with h5py.File(output + "_LossesAndAccuracies.hdf5", "w") as hf_diagnostic:
        hf_diagnostic.create_dataset("training_losses", data=result.training_losses)
        hf_diagnostic.create_dataset("validation_losses", data=result.validation_losses)
        hf_diagnostic.create_dataset("training_mses", data=result.training_mses)
        hf_diagnostic.create_dataset("validation_mses", data=result.validation_mses)
        hf_diagnostic.create_dataset("validation_accuracies", data=result.validation_accuracies)
        hf_diagnostic.create_dataset("specific_accuracies", data=result.specific_accuracies)
        hf_diagnostic.create_dataset("training_names", data=training_names)
        hf_diagnostic.create_dataset("validation_names", data=validation_names)
        hf_diagnostic.create_dataset("testing_names", data=testing_names)
        hf_diagnostic.create_dataset("final_test_loss", data=result.final_test_loss)
        hf_diagnostic.create_dataset("final_test_mse", data=result.final_test_mse)
        hf_diagnostic.create_dataset("final_test_accuracy", data=result.final_test_accuracy)
        hf_diagnostic.create_dataset("final_test_specific_accuracy", data=result.final_test_specific_accuracy)


def train(
    hyperparams,
    datasets,
    run_name=None,
    network_dir="NetworkWeights",
    output_dir="DiagnosticOutputs",
    save_diagnostics=True,
    batch_size=None,
    test_batch_size=None,
):
    """Train the network hyperparams["network"] on datasets, and return a TrainingResult.

    hyperparams, datasets: dicts with the keys of the files in configs/. Galaxy directories are relative to
        datasets["data_dir"], if given, or the current directory.
    run_name: names the network and every diagnostic output. Default: the targets.
    network_dir: where the trained network is saved.
    output_dir: where diagnostic outputs are saved.
    save_diagnostics: if False, save only the network and the loss history
        (<output_dir>/diagnostics/<run_name>_LossesAndAccuracies.hdf5), not the final-epoch snapshots, comparison
        images, or loss, accuracy and correlation plots.
    batch_size: training batch size. Default: about a tenth of the training set (at least 2); 0 trains on the whole
        set as one batch.
    test_batch_size: validation and testing batch size. Default: about a tenth of the validation set.
    """
    check_hyperparams(hyperparams)
    check_datasets(datasets)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    # Weight initialization, dropout and the order training batches are drawn in all come from these generators
    seed = hyperparams["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds the CPU and every GPU

    network = hyperparams["network"]
    epochs = hyperparams["epochs"]
    predict_every = hyperparams["predict_every"]
    label_scaling = hyperparams["label_scaling"]
    mass_weight_power = hyperparams.get("mass_weight_power", 0)  # 0: unweighted loss
    predict_variance = hyperparams.get("predict_variance", False)  # resblock only

    # One target name, or a list of them: the network predicts a map (or radial curve) of each
    targets = [datasets["target"]] if isinstance(datasets["target"], str) else datasets["target"]
    n_maps = len(targets)
    training_sources = dataset_sources(datasets["training"], "training", datasets.get("data_dir"))
    validation_sources = dataset_sources(datasets["validation"], "validation", datasets.get("data_dir"))
    testing_sources = dataset_sources(datasets["testing"], "testing", datasets.get("data_dir"))
    augment_training = None if datasets["augment"] == "none" else datasets["augment"]  # TOML has no None
    radial_bins = datasets["radial_bins"] or None  # 0 means train on maps
    radial_projection = datasets["radial_projection"]

    run_name = run_name or "_".join(targets)

    # Output directories
    os.makedirs(network_dir, exist_ok=True)
    if save_diagnostics:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "diagnostics"), exist_ok=True)

    image_output_training = os.path.join(output_dir, "images", run_name + "_training")
    image_output_validation = os.path.join(output_dir, "images", run_name + "_validation")
    image_output_final_test = os.path.join(output_dir, "images", run_name + "_test")

    diagnostic_output = os.path.join(output_dir, "diagnostics", run_name)
    model_output_path = os.path.join(network_dir, run_name)

    # Load the data, with the image normalization and label scaling of the training set
    dataset_settings = dict(
        radial_bins=radial_bins,
        radial_projection=radial_projection,
        inclinations=datasets["inclinations"],  # "all" or a list, for every set
        suffix=datasets.get("suffix", ""),  # VOF's outputSuffix in the filenames, for every set
        mass_weight_power=mass_weight_power,
    )
    training_data = CoNNGaFitImageDataset(training_sources, targets, augment=augment_training, **dataset_settings)
    training_mean, training_stdv, labels = training_statistics(training_data)
    label_offset, label_scale, accuracy_tolerance = calc_label_scaling(labels, label_scaling, targets)

    def scale_labels(label):
        """Apply label normalization"""
        return (label - label_offset) / label_scale

    def to_physical(a):
        """
        Revert label normalization from scale_labels back to physical units
        """
        return a * label_scale + label_offset

    image_transform = Compose([ToTensor(), Normalize(training_mean, training_stdv)])
    training_data.transform = image_transform
    training_data.target_transform = scale_labels
    validation_data = CoNNGaFitImageDataset(
        validation_sources, targets, transform=image_transform, target_transform=scale_labels, **dataset_settings
    )
    # Loaded before training, so a mistake in the testing sources is found right away
    test_data = CoNNGaFitImageDataset(
        testing_sources, targets, transform=image_transform, target_transform=scale_labels, **dataset_settings
    )
    train_dataloader, validation_dataloader, test_dataloader = make_dataloaders(
        training_data, validation_data, test_data, batch_size, test_batch_size
    )

    # The network
    input_shape = tuple(
        training_data[0][0].shape
    )  # (nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
    output_shape = (radial_bins,) if radial_bins is not None else None  # A radial curve, or a map on the cube's grid
    model = build_model(network, input_shape, hyperparams, output_shape=output_shape, n_maps=n_maps).to(device)
    logger.info("Network:\n%s", model_summary(model, input_shape))  # Layers, output shapes and parameter counts

    # Loss and optimizer
    n_samples = hyperparams.get("n_samples")  # ddpm only: posterior samples per prediction
    if network == "ddpm":
        loss_label = "Denoising MSE"
    else:
        loss_label = ("Weighted " if mass_weight_power > 0 else "") + ("Gaussian NLL" if predict_variance else "MSE")

    # What save_batch_outputs needs to name every output and convert it back to physical units
    output_settings = dict(targets=targets, to_physical=to_physical, label_scale=label_scale, radial_bins=radial_bins)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"]
    )  # Can use different optimizers

    # Each epoch's losses and accuracies
    training_losses = np.zeros((epochs))
    validation_losses = np.zeros((epochs))
    training_mses = np.zeros((epochs))
    validation_mses = np.zeros((epochs))
    validation_accuracies = np.zeros((epochs))
    specific_accuracies = np.zeros((len(validation_data), epochs))  # Each validation sample's accuracy, every epoch

    # Train
    for t in range(epochs):
        logger.info("Epoch %d/%d", t + 1, epochs)
        final = t == epochs - 1
        predict = final or (t + 1) % predict_every == 0  # The ddpm doesn't predict every epoch: sampling is slow
        save_training = save_validation = None
        validation_outputs = []  # The final epoch's validation predictions and labels, for the correlation plots
        if final and save_diagnostics:  # Save every sample's prediction/label snapshot of the final epoch
            save_training = partial(
                save_batch_outputs,
                dataset=training_data,
                image_output=image_output_training,
                epoch=t,
                comp_images=False,
                **output_settings,
            )
            save_validation = partial(
                save_batch_outputs,
                dataset=validation_data,
                image_output=image_output_validation,
                epoch=t,
                saved=validation_outputs,
                **output_settings,
            )
        training_losses[t], training_mses[t] = train_loop(
            model, train_dataloader, optimizer, device, predict, n_samples, save_training
        )
        validation_losses[t], validation_mses[t], validation_accuracies[t], specific_accuracies[:, t] = evaluate(
            model, validation_dataloader, device, predict, accuracy_tolerance, "Validation", n_samples, save_validation
        )
        if validation_outputs and radial_bins is None:  # Correlation plots compare maps
            plotting.save_correlation_plots(
                validation_outputs, targets, diagnostic_output + "_ValidationCorrelationPlot", t
            )

    save_checkpoint(
        model_output_path,
        model,
        image_stats=(training_mean, training_stdv),
        label_offset=label_offset,
        label_scale=label_scale,
        hyperparams=hyperparams,
        hyperparams_text=to_toml(hyperparams),  # The full parameters, saved with the model
        datasets_text=to_toml(datasets),
        run_name=run_name,
        input_shape=input_shape,
        targets=targets,
        radial_bins=radial_bins,
        radial_projection=radial_projection,
    )

    testing_names = test_data.names

    save_test = None
    test_outputs = []  # The test predictions and labels, for the correlation plots
    if save_diagnostics:
        save_test = partial(
            save_batch_outputs,
            dataset=test_data,
            image_output=image_output_final_test,
            epoch=epochs - 1,
            saved=test_outputs,
            **output_settings,
        )
    final_test_loss, final_test_mse, final_test_accuracy, final_test_specific_accuracy = evaluate(
        model, test_dataloader, device, True, accuracy_tolerance, "Test", n_samples, save_test
    )
    if test_outputs and radial_bins is None:  # Correlation plots compare maps
        plotting.save_correlation_plots(test_outputs, targets, diagnostic_output + "_TestCorrelationPlot", epochs - 1)
    logger.info("Testing set loss = %g", final_test_loss)
    logger.info("Testing set MSE = %g", final_test_mse)
    logger.info("Testing set accuracy = %g", final_test_accuracy)
    logger.info("Testing set specific accuracies:")
    for name, accuracy in zip(testing_names, final_test_specific_accuracy):
        logger.info("  %s: %.1f%%", name, accuracy)

    result = TrainingResult(
        model_path=model_output_path,
        training_losses=training_losses,
        validation_losses=validation_losses,
        training_mses=training_mses,
        validation_mses=validation_mses,
        validation_accuracies=validation_accuracies,
        specific_accuracies=specific_accuracies,
        final_test_loss=final_test_loss,
        final_test_mse=final_test_mse,
        final_test_accuracy=final_test_accuracy,
        final_test_specific_accuracy=final_test_specific_accuracy,
    )
    if save_diagnostics:
        plotting.plot_history(diagnostic_output, result, validation_data.names, loss_label, ACCURACY_TOLERANCE)
    save_history(diagnostic_output, result, training_data.names, validation_data.names, testing_names)
    return result
