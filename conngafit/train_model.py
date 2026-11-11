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

import h5py
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit import plotting
from conngafit.datasets import TARGET_SETTINGS, CoNNGaFitImageDataset, inclination_folders
from conngafit.networks import HYPERPARAM_KEYS, OPTIONAL_HYPERPARAM_KEYS, build_model, model_summary

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402 - after choosing the backend

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


@dataclass
class TrainingResult:
    """What train() returns: the trained network's path (without .pt/.hdf5), each epoch's losses (the training
    objective), MSEs of the predicted mean and validation accuracies (NaN on epochs between predict_every), and the
    test set's final loss, MSE and accuracy."""

    model_path: str
    training_losses: np.ndarray
    validation_losses: np.ndarray
    training_mses: np.ndarray
    validation_mses: np.ndarray
    validation_accuracies: np.ndarray
    final_test_loss: float
    final_test_mse: float
    final_test_accuracy: float


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using %s device", device)

    ####   HYPERPARAMETERS   ####
    network = hyperparams.get("network")  # Sets which keys hyperparams must have
    if network not in HYPERPARAM_KEYS:
        raise ValueError(f"hyperparams: network = {network!r} should be one of {sorted(HYPERPARAM_KEYS)}")
    check_keys(hyperparams, HYPERPARAM_KEYS[network], OPTIONAL_HYPERPARAM_KEYS[network], "hyperparams")
    hyperparams_text = to_toml(hyperparams)  # Saved with the model
    learning_rate = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    epochs = hyperparams["epochs"]
    label_scaling = hyperparams["label_scaling"]
    if label_scaling not in ("none", "symmetric", "one-sided"):
        raise ValueError(f'hyperparams: label_scaling = {label_scaling!r} should be "none", "symmetric" or "one-sided"')
    mass_weight_power = hyperparams.get("mass_weight_power", 0)  # 0: unweighted loss
    if not mass_weight_power >= 0:
        raise ValueError(f"hyperparams: mass_weight_power = {mass_weight_power!r} should be 0 or more")
    predict_variance = hyperparams.get("predict_variance", False)  # resblock only
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
    seed = hyperparams["seed"]
    # Weight initialization, dropout and the order training batches are drawn in all come from these generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds the CPU and every GPU
    #############################

    ####   DATASETS   ####
    check_keys(datasets, DATASET_KEYS, OPTIONAL_DATASET_KEYS, "datasets")
    datasets_text = to_toml(datasets)  # Saved with the model
    # One target name, or a list of them: the network predicts a map (or radial curve) of each
    targets = [datasets["target"]] if isinstance(datasets["target"], str) else datasets["target"]
    n_maps = len(targets)
    training_sources = dataset_sources(datasets["training"], "training", datasets.get("data_dir"))
    validation_sources = dataset_sources(datasets["validation"], "validation", datasets.get("data_dir"))
    testing_sources = dataset_sources(datasets["testing"], "testing", datasets.get("data_dir"))
    augment_training = None if datasets["augment"] == "none" else datasets["augment"]  # TOML has no None
    radial_bins = datasets["radial_bins"] or None  # 0 means train on maps
    radial_projection = datasets["radial_projection"]
    inclinations = datasets["inclinations"]  # "all" or a list, for every set
    inclination_folders(inclinations, "datasets")  # Check it here, so an error names datasets
    suffix = datasets.get("suffix", "")  # VOF's outputSuffix in the filenames, for every set
    if not isinstance(suffix, str):
        raise ValueError(f"datasets: suffix = {suffix!r} should be a string")
    #############################

    run_name = run_name or "_".join(targets)

    ####  Set input/output directories ####
    os.makedirs(network_dir, exist_ok=True)
    if save_diagnostics:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "diagnostics"), exist_ok=True)

    image_output_training = os.path.join(output_dir, "images", run_name + "_training")
    image_output_validation = os.path.join(output_dir, "images", run_name + "_validation")
    image_output_final_test = os.path.join(output_dir, "images", run_name + "_test")

    diagnostic_output = os.path.join(output_dir, "diagnostics", run_name)
    model_output_path = os.path.join(network_dir, run_name)

    #### Load and Normalize Data ####
    training_data = CoNNGaFitImageDataset(
        training_sources,
        targets,
        augment=augment_training,
        radial_bins=radial_bins,
        radial_projection=radial_projection,
        inclinations=inclinations,
        suffix=suffix,
    )

    if batch_size is not None:
        batch_size_default = None if batch_size == 0 else batch_size  # batch_size 0 means full-batch
    else:
        batch_size_default = max(2, len(training_data) // 10)  # Reasonable default division of the training set

    batches_per_training_set = 1
    if batch_size_default is None:
        batch_size = len(training_data)
    else:
        batch_size = batch_size_default
        batches_per_training_set = np.ceil(len(training_data) / batch_size)
        logger.info("Batches per training set = %d", batches_per_training_set)

    #### Image normalization and label scaling, from every training file ####
    # One cube in memory at a time. Every cube has the same size, so the training set's mean (and mean of squares)
    # is the mean of each cube's.
    cube_means, cube_mean_squares, labels = [], [], []
    for i in range(len(training_data.img_paths)):
        cube = training_data.image(i)
        cube_means.append(cube.mean(dtype=np.float64))
        cube_mean_squares.append(np.square(cube, dtype=np.float64).mean())
        labels.append(training_data.label(i))  # (n_maps, nY, nX), or (n_maps, n_rings) for radial curves
    training_mean = np.mean(cube_means)
    training_stdv = np.sqrt(np.mean(cube_mean_squares) - training_mean**2)  # var = E[x²] - E[x]²
    logger.info("Image normalization: mean = %.4g, stdv = %.4g", training_mean, training_stdv)
    all_labels = np.concatenate([label.reshape(n_maps, -1) for label in labels], axis=1)  # (n_maps, every value)

    # Labels are scaled to (label - label_offset) / label_scale for training; predictions and labels are
    # scaled back to physical units for every saved output and plot. Both have shape (n_maps, 1), one per target.
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

    #### Rough accuracy: the percentage of predicted values within accuracy_tolerance of their label, where ####
    #### accuracy_tolerance is ACCURACY_TOLERANCE times the spread (standard deviation) of each target's labels. ####
    #### Not a great metric, but okay for tracking general performance ####
    accuracy_tolerance = ACCURACY_TOLERANCE * np.std(all_labels, axis=1, keepdims=True) / label_scale  # Scaled units
    for target, tolerance in zip(targets, (accuracy_tolerance * label_scale)[:, 0]):
        logger.info("Accuracy tolerance of %s = %.4g (physical units)", target, tolerance)
    del all_labels

    # Shape each target's statistics, (n_maps, 1) so far, to broadcast over its map (or curve): (n_maps, 1, 1) for maps
    per_target = (n_maps,) + (1,) * (labels[0].ndim - 1)
    label_offset, label_scale = label_offset.reshape(per_target), label_scale.reshape(per_target)
    accuracy_tolerance = accuracy_tolerance.reshape(per_target)

    def scale_labels(label):
        """Apply label normalization"""
        return (label - label_offset) / label_scale

    def to_physical(a):
        """
        Revert label normalization from scale_labels back to physical units
        """
        return a * label_scale + label_offset

    training_data.transform = Compose([ToTensor(), Normalize(training_mean, training_stdv)])
    training_data.target_transform = scale_labels

    validation_data = CoNNGaFitImageDataset(
        validation_sources,
        targets,
        radial_bins=radial_bins,
        radial_projection=radial_projection,
        inclinations=inclinations,
        suffix=suffix,
        transform=Compose([ToTensor(), Normalize(training_mean, training_stdv)]),
        target_transform=scale_labels,
    )

    class WithIndices(Dataset):
        """Wraps a dataset so each sample also returns its loss weights (see mass_weight_power) and its index,
        to look up its name after shuffling."""

        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            if mass_weight_power > 0:
                weights = self.dataset.mass_weights(idx, mass_weight_power)
            else:
                weights = np.ones_like(label)
            return image, label, weights, idx

    train_dataloader = DataLoader(
        WithIndices(training_data), batch_size=batch_size, shuffle=True
    )  # Reshuffled every epoch

    if test_batch_size is None:
        test_batch_size = max(1, len(validation_data) // 10)  # Reasonable division of the validation/test
        # sets into ~10 batches; floored at 1 (unlike batch_size_default, batch size 1 is safe here
        # since validation/testing run with model.eval(), which uses BatchNorm's running stats
        # rather than computing per-batch statistics). Override with test_batch_size.
    if batch_size_default is None:
        batch_size = len(validation_data)
    else:
        batch_size = test_batch_size
    validation_dataloader = DataLoader(WithIndices(validation_data), batch_size=batch_size)

    logger.info("Data loaded")

    # Radial curves are plotted as curves rather than maps
    make_comp_image = plotting.make_comp_image_1d if radial_bins is not None else plotting.make_comp_image

    #### Save final-epoch outputs ####
    def save_batch_outputs(x, y, pred, sigma, idx, dataset, output_base, epoch, comp_images=True):
        """Save each sample's map (or curve) of each target in a batch as an hdf5 snapshot (and a comparison image),
        named after its image via its dataset index (idx), so names stay correct with shuffled or multiple batches,
        and after the target. sigma is each prediction's 1-sigma (see compute_loss), saved and plotted too, or None."""
        np_x = x.cpu().float().detach().numpy()
        np_y = to_physical(y.cpu().float().detach().numpy())
        np_pred = to_physical(pred.cpu().detach().numpy())
        # 1-sigma in physical units: a spread, so scaled by label_scale but not offset
        np_sigma = None if sigma is None else sigma.cpu().detach().numpy() * label_scale
        for m, target in enumerate(targets):
            param_label, unit_label, bin_op = TARGET_SETTINGS[target]
            target_sigma = None if np_sigma is None else np_sigma[:, m]
            for tt in range(len(idx)):
                output_name = f"{output_base}_{dataset.names[int(idx[tt])]}_{target}_Epoch{epoch}"
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

    def save_correlation_plots(all_pred, all_y, output_base, epoch):
        """Write a correlation plot of each target's predicted and true maps, from lists of (nBatch, n_maps, nY, nX)
        batches in physical units."""
        all_pred, all_y = np.concatenate(all_pred), np.concatenate(all_y)
        for m, target in enumerate(targets):
            plotting.make_correlation_plot(all_pred[:, m], all_y[:, m], f"{output_base}_{target}_Epoch{epoch}")

    #### Define Training Loop ####
    def train_loop(dataloader, model, loss_fn, optimizer, epoch):
        """Run one training epoch: iterate the dataloader, backprop loss_fn (see compute_loss) each batch, and on
        the final epoch dump every sample's prediction/label hdf5 snapshot for later inspection.
        Returns (loss, MSE of the predicted mean), each averaged over all batches in the epoch; the MSE is NaN if no
        batch was predicted.
        """
        model.train()
        size = len(dataloader.dataset)
        epoch_loss, epoch_mse = 0.0, 0.0
        n_predicted = 0  # Batches with predictions, made every predict_every epochs
        num_batches = len(dataloader)
        is_final_epoch = epoch >= epochs - 1
        if is_final_epoch or (epoch + 1) % predict_every == 0:  # ddpm does not predict every epoch
            predict = True
        else:
            predict = False

        for batch, (x, y, w, idx) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            loss, pred, sigma = loss_fn(model, x.float(), y.float(), w.float(), predict=predict)
            epoch_loss += loss.item()
            if pred is not None:
                n_predicted += 1
                epoch_mse += mse_fn(pred, y.float()).item()
            if is_final_epoch and save_diagnostics:
                save_batch_outputs(
                    x, y, pred, sigma, idx, training_data, image_output_training, epoch, comp_images=False
                )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                current = batch * len(x)
                logger.info("loss: %7f [%5d/%5d]", loss.item(), current, size)

        return epoch_loss / num_batches, (epoch_mse / n_predicted if n_predicted else np.nan)

    def count_accurate(pred, y):
        """Return (each sample's accuracy [%], number of accurate values) for a batch of scaled predictions and labels."""
        within = np.abs(pred - y) < accuracy_tolerance
        return within.mean(axis=tuple(range(1, within.ndim))) * 100, within.sum()  # Over each sample's every value

    #### Define Evaluation Loop ####
    def evaluate(dataloader, model, loss_fn, epoch, name, image_output):
        """Evaluate the model on the validation or test set (no gradient updates) and compute a rough "accuracy".
        name ("Validation" or "Test") labels the log line and the correlation plot. On the final epoch, also writes
        per-sample comparison images/hdf5 snapshots under image_output, and correlation plots.

        Returns (loss, mse, accuracy, specific_accuracy) where mse is the MSE of the predicted mean and
        specific_accuracy is the accuracy of every sample in the set. The mse and accuracies are NaN on the epochs
        between predict_every, which make no predictions.
        """
        model.eval()
        num_batches = len(dataloader)
        summed_loss, summed_mse, correct = 0, 0, 0
        total = 0
        final = epoch >= epochs - 1
        predict = final or (epoch + 1) % predict_every == 0
        all_pred, all_y = [], []  # Every sample's prediction/label on the final epoch, for the correlation plot
        specific_accuracy = np.full(len(dataloader.dataset), np.nan)

        with torch.no_grad():
            for x, y, w, idx in dataloader:
                x = x.to(device)
                y = y.to(device)
                w = w.to(device)

                loss, pred, sigma = loss_fn(model, x.float(), y.float(), w.float(), predict=predict)
                summed_loss += loss.item()
                if pred is None:  # Predictions are made every predict_every epochs
                    continue
                summed_mse += mse_fn(pred, y.float()).item()

                np_pred = pred.cpu().float().detach().numpy()
                np_y = y.cpu().float().detach().numpy()

                specific_accuracy[idx.numpy()], n_correct = count_accurate(np_pred, np_y)
                correct += n_correct
                total += np.size(np_y)

                if final and save_diagnostics:
                    # dataloader.dataset is the WithIndices wrapper; its .dataset has the sample names
                    save_batch_outputs(x, y, pred, sigma, idx, dataloader.dataset.dataset, image_output, epoch)
                    all_pred.append(to_physical(np_pred))
                    all_y.append(to_physical(np_y))

        accuracy = correct / total * 100 if total else np.nan
        mean_loss = summed_loss / num_batches
        mse = summed_mse / num_batches if total else np.nan
        logger.info("%s: accuracy %.1f%%, avg loss %8f, avg MSE %8f", name, accuracy, mean_loss, mse)

        if final and save_diagnostics and radial_bins is None:
            save_correlation_plots(all_pred, all_y, f"{diagnostic_output}_{name}CorrelationPlot", epoch)

        model.train()
        return mean_loss, mse, accuracy, specific_accuracy

    #### Define the Model ####
    input_shape = tuple(
        training_data[0][0].shape
    )  # (nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
    model = build_model(network, input_shape, hyperparams, output_size=radial_bins, n_maps=n_maps).to(device)
    logger.info("Network:\n%s", model_summary(model, input_shape))  # Layers, output shapes and parameter counts

    #### Define Loss Function and Optimizer ####
    def weighted_mse_loss(pred, y, w):
        """Mean squared error, each pixel (or ring) weighted by w; all ones is plain MSE."""
        return torch.mean(w * nn.functional.mse_loss(pred, y, reduction="none"))

    def weighted_gaussian_nll_loss(pred, y, logvar, w):
        """Gaussian NLL of (pred, exp(logvar)), each pixel (or ring) weighted by w; all ones is nn.GaussianNLLLoss."""
        return torch.mean(w * nn.functional.gaussian_nll_loss(pred, y, logvar.exp(), reduction="none"))

    loss_fn = weighted_gaussian_nll_loss if predict_variance else weighted_mse_loss  # The training objective
    mse_fn = nn.MSELoss()  # MSE of the predicted mean, logged for every run so runs can be compared
    if network == "ddpm":
        loss_label = "Denoising MSE"
    else:
        loss_label = ("Weighted " if mass_weight_power > 0 else "") + ("Gaussian NLL" if predict_variance else "MSE")

    def compute_loss(model, x, y, w, predict=True):
        """
        Returns (loss, prediction, 1-sigma) for a batch of cubes x and labels y, with weights by w. The
        prediction and 1-sigma are None unless predict=True.

        resblock: If predict_variance=True, use Gaussian NLL of (mean, exp(logvar)) with sigma = exp(logvar / 2)
                  Otherwise, use MSE loss, with sigma = None.
        ddpm: the denoising loss (unweighted), and the mean and standard deviation of n_samples posterior
        samples, which are only drawn if predict since this has high compute cost.
        """
        if network == "ddpm":
            loss = model(x, y)
            if not predict:
                return loss, None, None
            pred, sigma = model.predict_with_uncertainty(x, hyperparams["n_samples"])
            return loss, pred, sigma

        output = model(x)
        if predict_variance:
            pred, logvar = output
            loss, sigma = loss_fn(pred, y, logvar, w), torch.exp(0.5 * logvar)
        else:
            pred, sigma = output, None
            loss = loss_fn(pred, y, w)
        if not predict:
            return loss, None, None
        return loss, pred, sigma

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # Can use different optimizers

    #### Initialize losses and specific accuracies ####
    training_losses = np.zeros((epochs))
    validation_losses = np.zeros((epochs))
    training_mses = np.zeros((epochs))
    validation_mses = np.zeros((epochs))
    validation_accuracies = np.zeros((epochs))
    specific_accuracies = np.zeros((len(validation_data), epochs))  # Each validation sample's accuracy, every epoch

    #### Perform Training Loop ####
    for t in range(epochs):
        logger.info("Epoch %d/%d", t + 1, epochs)
        training_losses[t], training_mses[t] = train_loop(train_dataloader, model, compute_loss, optimizer, t)
        validation_losses[t], validation_mses[t], validation_accuracies[t], specific_accuracies[:, t] = evaluate(
            validation_dataloader, model, compute_loss, t, "Validation", image_output_validation
        )

    # Save the Trained Model
    torch.save(model.state_dict(), model_output_path + ".pt")

    # Save the normalization information for loading data
    with h5py.File(model_output_path + ".hdf5", "w") as hf_norm:
        hf_norm.create_dataset("image_stats", data=np.array([training_mean, training_stdv]))
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

    # Testing Data

    test_data = CoNNGaFitImageDataset(
        testing_sources,
        targets,
        radial_bins=radial_bins,
        radial_projection=radial_projection,
        inclinations=inclinations,
        suffix=suffix,
        transform=Compose([ToTensor(), Normalize(training_mean, training_stdv)]),
        target_transform=scale_labels,
    )

    test_dataloader = DataLoader(WithIndices(test_data), batch_size=batch_size)
    testing_names = test_data.names

    final_test_loss, final_test_mse, final_test_accuracy, final_test_specific_accuracy = evaluate(
        test_dataloader, model, compute_loss, t, "Test", image_output_final_test
    )
    logger.info("Testing set loss = %g", final_test_loss)
    logger.info("Testing set MSE = %g", final_test_mse)
    logger.info("Testing set accuracy = %g", final_test_accuracy)
    logger.info("Testing set specific accuracies:")
    for name, accuracy in zip(testing_names, final_test_specific_accuracy):
        logger.info("  %s: %.1f%%", name, accuracy)

    #### Make Basic Diagnostic Outputs ####
    def plot_loss(losses, title, output):
        """Plot a loss curve per epoch, on a log axis unless some value isn't positive (the Gaussian NLL can be negative)."""
        plt.figure()
        plt.plot(np.arange(1, len(losses) + 1), losses)
        if np.all(losses > 0):
            plt.yscale("log")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(loss_label)
        plt.savefig(output)
        plt.close()

    if save_diagnostics:
        plot_loss(training_losses, "Training Loss", diagnostic_output + "_trainingLoss.png")
        plot_loss(validation_losses, "Validation Loss", diagnostic_output + "_validationLoss.png")

        # Accuracies are NaN on the epochs between predict_every, so only the predicted epochs are plotted
        epoch_numbers = np.arange(1, epochs + 1)
        predicted = np.isfinite(validation_accuracies)

        plt.figure()
        plt.plot(epoch_numbers[predicted], validation_accuracies[predicted], marker="o", ms=3)
        plt.title(f"Validation Accuracy (within {ACCURACY_TOLERANCE:g} label std. dev.)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.ylim([0, 101])
        plt.savefig(diagnostic_output + "_ValidationAccuracy.png")
        plt.close()

        plt.figure()
        for name, accuracies in zip(validation_data.names, specific_accuracies):
            plt.plot(epoch_numbers[predicted], accuracies[predicted], lw=0.5, marker="o", ms=2, label=name)
        if len(validation_data.names) <= 10:  # Name each line, unless there are too many to read
            plt.legend(fontsize=6)
        plt.title("Validation Accuracy for each Snapshot")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.ylim([0, 101])
        plt.savefig(diagnostic_output + "_ValidationAccuracyPerSnapshot.png")
        plt.close()

    with h5py.File(diagnostic_output + "_LossesAndAccuracies.hdf5", "w") as hf_diagnostic:
        hf_diagnostic.create_dataset("training_losses", data=training_losses)
        hf_diagnostic.create_dataset("validation_losses", data=validation_losses)
        hf_diagnostic.create_dataset("training_mses", data=training_mses)
        hf_diagnostic.create_dataset("validation_mses", data=validation_mses)
        hf_diagnostic.create_dataset("validation_accuracies", data=validation_accuracies)
        hf_diagnostic.create_dataset("specific_accuracies", data=specific_accuracies)
        hf_diagnostic.create_dataset("training_names", data=training_data.names)
        hf_diagnostic.create_dataset("validation_names", data=validation_data.names)
        hf_diagnostic.create_dataset("testing_names", data=testing_names)
        hf_diagnostic.create_dataset("final_test_loss", data=final_test_loss)
        hf_diagnostic.create_dataset("final_test_mse", data=final_test_mse)
        hf_diagnostic.create_dataset("final_test_accuracy", data=final_test_accuracy)
        hf_diagnostic.create_dataset("final_test_specific_accuracy", data=final_test_specific_accuracy)

    return TrainingResult(
        model_path=model_output_path,
        training_losses=training_losses,
        validation_losses=validation_losses,
        training_mses=training_mses,
        validation_mses=validation_mses,
        validation_accuracies=validation_accuracies,
        final_test_loss=final_test_loss,
        final_test_mse=final_test_mse,
        final_test_accuracy=final_test_accuracy,
    )
