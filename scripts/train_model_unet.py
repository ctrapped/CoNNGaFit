import logging
import os
import random

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit.datasets import TARGET_SETTINGS, TARGETS, CoNNGaFitImageDataset, inclination_folders
from conngafit.logging_setup import add_logging_arguments, configure_logging
from conngafit.networks import HYPERPARAM_KEYS, OPTIONAL_HYPERPARAM_KEYS, build_model

matplotlib.use("Agg")
import argparse
import tomllib

import h5py
from matplotlib import pyplot as plt

from conngafit.plotting import (
    make_comp_image,
    make_comp_image_1d,
    make_correlation_plot,
    save_hdf5,
    save_hdf5_1d,
)

####Trains the network chosen by the hyperparameters file's network key (see conngafit/networks.py) on the given training data. Provides additional diagnostic plots and images on the validation and test datasets provided.
####Training, validation, and testing datasets are VOF optically thin outputs, chosen by galaxy and snapshot range (see datasets.py).
####Hyperparameters were chosen based on paramter space optimization+trial and error.
####Hyperparameters and datasets (including the target annotations) are read from two parameter files,
####configs/hyperparams_resblock.toml and configs/datasets.toml by default; plot labels per target are in TARGET_SETTINGS in datasets.py
#
####Run as: python scripts/train_model_unet.py [--hyperparams FILE] [--datasets FILE] [--run-name NAME] [options]
####  Run with --help to see all options. The two parameter files and --run-name are normally the
####  only options you need. The datasets file gives the path to each galaxy's VOF outputs, and by default:
####    NetworkWeights/    receives the trained model checkpoint (<run-name>.pt/.hdf5)
####    DiagnosticOutputs/ receives per-epoch images/snapshots and loss/accuracy/correlation plots
####  NetworkWeights/ and DiagnosticOutputs/ are created automatically if missing.
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 10/21/2026


####   PARAMETER FILES   ####
# Keys every datasets file must have (see configs/*.toml for what each one does; networks.py lists each network's
# hyperparameter keys)
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
OPTIONAL_DATASET_KEYS = {"base_dir", "suffix"}
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_parameter_file(path, keys, optional_keys=frozenset()):
    """Read a TOML parameter file, returning (its parameters, its text to save with the model).
    Raises ValueError if any of keys is missing, or any key is unrecognized, so typos aren't silently ignored."""
    with open(path, "rb") as f:
        text = f.read().decode()
    params = tomllib.loads(text)
    missing, unknown = keys - params.keys(), params.keys() - keys - optional_keys
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}, unrecognized keys {sorted(unknown)}")
    return params, text


def dataset_sources(section, name, path, base_dir=None):
    """Check a datasets-file section ("VOF output directory" = snapshots lines) and resolve each
    directory relative to base_dir (default: the datasets file's folder). Returns the dict datasets.py expects."""
    if not isinstance(section, dict):
        raise ValueError(f'{path}: write [{name}] as "/path/to/vof_outputs" = snapshots lines')
    # base_dir itself may be relative, to the datasets file's folder
    base = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.expanduser(base_dir or ""))
    return {os.path.normpath(os.path.join(base, os.path.expanduser(d))): snaps for d, snaps in section.items()}


#############################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CoNNGaFit 3-d U-Net to predict a gas property map from HI datacubes."
    )
    parser.add_argument(
        "--hyperparams",
        default=os.path.join(CONFIG_DIR, "hyperparams_resblock.toml"),
        help="Hyperparameter file (network, learning rate, epochs, network size, output head). "
        "Default: configs/hyperparams_resblock.toml",
    )
    parser.add_argument(
        "--datasets",
        default=os.path.join(CONFIG_DIR, "datasets.toml"),
        help="Datasets file (targets, augmentation, training/validation/testing galaxies). Default: configs/datasets.toml",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run; the checkpoint and every diagnostic output are named after it. Default: the targets",
    )

    io = parser.add_argument_group(
        "output directories",
        "Override where results are written to (the training data is set in the datasets file). "
        "Both are created automatically if missing.",
    )
    io.add_argument(
        "--network-dir",
        default="NetworkWeights",
        help="Directory the trained model checkpoint (.pt) and its normalization stats (.hdf5) are written to. Default: %(default)s",
    )
    io.add_argument(
        "--output-dir",
        default="DiagnosticOutputs",
        help="Root directory diagnostic outputs are written to: per-epoch comparison images/snapshots "
        "under <output-dir>/images/, and loss/accuracy curves and correlation plots under "
        "<output-dir>/diagnostics/. Default: %(default)s",
    )

    training = parser.add_argument_group(
        "training options",
        "Override values normally computed automatically from the dataset. Not needed for normal runs.",
    )
    training.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for training. Default: a reasonable division of the training set into "
        "~10 batches per epoch (floored at 2 - BatchNorm3d needs more than 1 sample per batch "
        "while training). Pass 0 to instead train on the full dataset as a single batch per epoch.",
    )
    training.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Mini-batch size for validation/testing. Default: a reasonable division of the "
        "validation set into ~10 batches (floored at 1).",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


logger = logging.getLogger(__name__)
args = parse_args()
configure_logging(verbose=args.verbose, quiet=args.quiet)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
logger.info("Using %s device", device)

####   HYPERPARAMETERS   ####
with open(args.hyperparams, "rb") as f:
    network = tomllib.load(f).get("network")  # Sets which keys the file must have
if network not in HYPERPARAM_KEYS:
    raise ValueError(f"{args.hyperparams}: network = {network!r} should be one of {sorted(HYPERPARAM_KEYS)}")
hyperparams, hyperparams_text = load_parameter_file(
    args.hyperparams, HYPERPARAM_KEYS[network], OPTIONAL_HYPERPARAM_KEYS[network]
)
learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
epochs = hyperparams["epochs"]
label_scaling = hyperparams["label_scaling"]
if label_scaling not in ("none", "symmetric", "one-sided"):
    raise ValueError(
        f'{args.hyperparams}: label_scaling = {label_scaling!r} should be "none", "symmetric" or "one-sided"'
    )
mass_weight_power = hyperparams.get("mass_weight_power", 0)  # 0: unweighted loss
if not mass_weight_power >= 0:
    raise ValueError(f"{args.hyperparams}: mass_weight_power = {mass_weight_power!r} should be 0 or more")
predict_variance = hyperparams.get("predict_variance", False)  # resblock only
if not isinstance(predict_variance, bool):
    raise ValueError(f"{args.hyperparams}: predict_variance = {predict_variance!r} should be true or false")
if predict_variance and label_scaling == "none":
    raise ValueError(
        f'{args.hyperparams}: predict_variance = true needs label_scaling = "symmetric" or "one-sided", '
        "since nn.GaussianNLLLoss floors the variance at 1e-6 in the units of the labels it trains on"
    )
ema_decay = hyperparams.get("ema_decay", 0.999)  # On by default for every network; 0 uses the raw weights
if not 0 <= ema_decay < 1:
    raise ValueError(f"{args.hyperparams}: ema_decay = {ema_decay!r} should be in [0, 1)")
max_grad_norm = hyperparams["max_grad_norm"]  # 0: no gradient clipping
if not max_grad_norm >= 0:
    raise ValueError(f"{args.hyperparams}: max_grad_norm = {max_grad_norm!r} should be 0 or more")
predict_every = hyperparams["predict_every"]
if not (isinstance(predict_every, int) and predict_every >= 1):
    raise ValueError(f"{args.hyperparams}: predict_every = {predict_every!r} should be an integer of 1 or more")
seed = hyperparams["seed"]
# Weight initialization, dropout and the order training batches are drawn in all come from these generators
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # Seeds the CPU and every GPU
#############################


####   DATASETS   ####
datasets, datasets_text = load_parameter_file(args.datasets, DATASET_KEYS, OPTIONAL_DATASET_KEYS)
# One target name, or a list of them: the network predicts a map (or radial curve) of each
targets = [datasets["target"]] if isinstance(datasets["target"], str) else datasets["target"]
if not targets:
    raise ValueError(f"{args.datasets}: target is empty; name at least one annotation")
for target in targets:
    if target not in TARGETS:
        raise ValueError(f"{args.datasets}: unknown target {target!r}; choose from {sorted(TARGETS)}")
n_maps = len(targets)
training_sources = dataset_sources(datasets["training"], "training", args.datasets, datasets.get("base_dir"))
validation_sources = dataset_sources(datasets["validation"], "validation", args.datasets, datasets.get("base_dir"))
testing_sources = dataset_sources(datasets["testing"], "testing", args.datasets, datasets.get("base_dir"))
augment_training = None if datasets["augment"] == "none" else datasets["augment"]  # TOML has no None
radial_bins = datasets["radial_bins"] or None  # 0 means train on maps
radial_projection = datasets["radial_projection"]
inclinations = datasets["inclinations"]  # "all" or a list, for every set
inclination_folders(inclinations, args.datasets)  # Check it here, so an error names this file
suffix = datasets.get("suffix", "")  # VOF's outputSuffix in the filenames, for every set
if not isinstance(suffix, str):
    raise ValueError(f"{args.datasets}: suffix = {suffix!r} should be a string")
#############################

run_name = args.run_name or "_".join(targets)


####  Set input/output directories ####
os.makedirs(args.network_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "diagnostics"), exist_ok=True)

image_output_training = os.path.join(args.output_dir, "images", run_name + "_training")
image_output_validation = os.path.join(args.output_dir, "images", run_name + "_validation")
image_output_final_test = os.path.join(args.output_dir, "images", run_name + "_test")

diagnostic_output = os.path.join(args.output_dir, "diagnostics", run_name)
model_output_path = os.path.join(args.network_dir, run_name)


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

if args.batch_size is not None:
    batch_size_default = None if args.batch_size == 0 else args.batch_size  # --batch-size 0 means full-batch
else:
    batch_size_default = max(2, len(training_data) // 10)  # Reasonable division of the training set into ~10
    # batches per epoch; floored at 2 (BatchNorm3d needs more than 1 sample per batch while
    # training). Override with --batch-size (0 for full-batch training).

batches_per_training_set = 1
if batch_size_default is None:
    batch_size = len(training_data)
else:
    batch_size = batch_size_default
    batches_per_training_set = np.ceil(len(training_data) / batch_size)
    logger.info("Batches per training set = %d", batches_per_training_set)

#### Image normalization, from every training image (augmentation doesn't change their mean/spread) ####
# Each cube's mean and variance are combined exactly (Chan et al.'s parallel algorithm), one cube in memory at a time
n_pixels, training_mean, sum_squared_deviations = 0, 0.0, 0.0
for i in range(len(training_data.img_paths)):
    cube = training_data.image(i)
    n, mean = cube.size, cube.mean(dtype=np.float64)
    delta = mean - training_mean
    sum_squared_deviations += cube.var(dtype=np.float64) * n + delta**2 * n_pixels * n / (n_pixels + n)
    training_mean += delta * n / (n_pixels + n)
    n_pixels += n
training_stdv = np.sqrt(sum_squared_deviations / n_pixels)
del cube
logger.info("Image normalization: mean = %.4g, stdv = %.4g", training_mean, training_stdv)

#### Label scaling of each target, from every training label (augmentation doesn't change their range) ####
# Labels are scaled to (label - label_offset) / label_scale for training; predictions and labels are
# scaled back to physical units for every saved output and plot. Both have shape (n_maps, 1), one per target.
label_offset, label_scale = np.zeros((n_maps, 1)), np.ones((n_maps, 1))
# (n_maps, every training pixel or ring)
all_labels = np.concatenate([training_data.label(i) for i in range(len(training_data.img_paths))], axis=1)
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
            raise ValueError(f"label_scaling = {label_scaling!r} needs {target} training labels that aren't constant")
        logger.info("Label scaling of %s (%s): offset = %.4g, scale = %.4g", target, label_scaling, offset, scale)

#### Rough accuracy: the percentage of predicted values within accuracy_tolerance of their label, where ####
#### accuracy_tolerance is ACCURACY_TOLERANCE times the spread (standard deviation) of each target's labels. ####
#### Not a great metric, but okay for tracking general performance ####
ACCURACY_TOLERANCE = 0.1
accuracy_tolerance = ACCURACY_TOLERANCE * np.std(all_labels, axis=1, keepdims=True) / label_scale  # Scaled units
for target, tolerance in zip(targets, (accuracy_tolerance * label_scale)[:, 0]):
    logger.info("Accuracy tolerance of %s = %.4g (physical units)", target, tolerance)
del all_labels


def scale_labels(label):
    return (label - label_offset) / label_scale


def to_physical(a):
    """Scale labels or predictions back to physical units."""
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


train_dataloader = DataLoader(WithIndices(training_data), batch_size=batch_size, shuffle=True)  # Reshuffled every epoch

if args.test_batch_size is not None:
    test_batch_size = args.test_batch_size
else:
    test_batch_size = max(1, len(validation_data) // 10)  # Reasonable division of the validation/test
    # sets into ~10 batches; floored at 1 (unlike batch_size_default, batch size 1 is safe here
    # since validation/testing run with model.eval(), which uses BatchNorm's running stats
    # rather than computing per-batch statistics). Override with --test-batch-size.
if batch_size_default is None:
    batch_size = len(validation_data)
else:
    batch_size = test_batch_size
validation_dataloader = DataLoader(WithIndices(validation_data), batch_size=batch_size)

logger.info("Data loaded")


if radial_bins is not None:  # Labels are radial curves, so plot and save them as curves rather than maps
    make_comp_image = make_comp_image_1d
    save_hdf5 = save_hdf5_1d


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
            save_hdf5(np_y[:, m], np_pred[:, m], output_name + ".hdf5", tt, uncertainty=target_sigma)


def save_correlation_plots(all_pred, all_y, output_base, epoch):
    """Write a correlation plot of each target's predicted and true maps, from lists of (nBatch, n_maps, n_pixels)
    batches in physical units."""
    all_pred, all_y = np.concatenate(all_pred), np.concatenate(all_y)
    for m, target in enumerate(targets):
        make_correlation_plot(all_pred[:, m], all_y[:, m], f"{output_base}_{target}_Epoch{epoch}")


#### Define Training Loop ####
def train_loop(dataloader, model, loss_fn, optimizer, epoch, ema_model):
    """Run one training epoch: iterate the dataloader, backprop loss_fn (see compute_loss) each batch, and on
    the final epoch dump every sample's prediction/label hdf5 snapshot for later inspection. Clips the total
    gradient norm to max_grad_norm (if > 0) and logs the epoch's mean gradient norm and how often it was clipped.
    Returns (loss, MSE of the predicted mean), each averaged over all batches in the epoch; the MSE is NaN if no
    batch was predicted.
    ema_model: an AveragedModel of model, updated after every optimizer step. The ddpm samples its predictions
    from it. The resblock network's training-set predictions come from the raw-weight, while validation and test
    use the EMA.
    """
    model.train()
    size = len(dataloader.dataset)
    epoch_loss, epoch_mse = 0.0, 0.0
    n_predicted = 0  # Batches with predictions, made every predict_every epochs
    epoch_grad_norm, n_clipped = 0.0, 0  # Total gradient norm before clipping, and how many batches were clipped
    num_batches = len(dataloader)
    final = epoch >= epochs - 1
    predict = final or (epoch + 1) % predict_every == 0
    for batch, (x, y, w, idx) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        loss, pred, sigma = loss_fn(
            model, x.float(), y.float(), w.float(), predict=predict, sampling_model=ema_model.module
        )
        epoch_loss += loss.item()
        if pred is not None:
            n_predicted += 1
            epoch_mse += mse_fn(pred, y.float()).item()
        if final:
            save_batch_outputs(x, y, pred, sigma, idx, training_data, image_output_training, epoch, comp_images=False)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_ rescales all gradients together (keeping their direction) when their total norm exceeds
        # clip_at, and returns the norm before clipping; with an infinite limit it only measures it
        clip_at = max_grad_norm if max_grad_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_at).item()
        epoch_grad_norm += grad_norm
        if grad_norm > clip_at:
            n_clipped += 1
        optimizer.step()
        ema_model.update_parameters(model)

        if batch % 100 == 0:
            current = batch * len(x)
            logger.info("loss: %7f [%5d/%5d]", loss.item(), current, size)

    logger.info(
        "Mean gradient norm %.3g, clipped in %d of %d batches", epoch_grad_norm / num_batches, n_clipped, num_batches
    )
    return epoch_loss / num_batches, (epoch_mse / n_predicted if n_predicted else np.nan)


def count_accurate(pred, y):
    """Return (each sample's accuracy [%], number of accurate values) for a batch of scaled predictions and labels."""
    within = np.abs(pred - y) < accuracy_tolerance
    return within.mean(axis=(1, 2)) * 100, within.sum()


#### Define Validation Loop ####
def validation_loop(dataloader, model, loss_fn, epoch, fixed_draws=None):
    """Evaluate the model on the validation set (no gradient updates) and compute a rough
    "accuracy" (the percentage of predicted values within accuracy_tolerance of the label -
    a loose sanity metric, not a real accuracy). On the final epoch, also writes correlation
    plots and per-sample comparison images/hdf5 snapshots. Returns
    (validation_loss, validation_mse, accuracy, specific_accuracy) where validation_mse is the MSE of
    the predicted mean and specific_accuracy is the accuracy of every sample in the validation set,
    in dataset order. fixed_draws: for the ddpm network, one (t, noise) pair per batch, so every epoch's denoising
    loss scores the same noisy maps; None draws new ones.
    """
    model.eval()
    num_batches = len(dataloader)
    validation_loss, validation_mse, correct = 0, 0, 0
    total = 0
    final = epoch >= epochs - 1
    predict = final or (epoch + 1) % predict_every == 0
    all_pred, all_y = [], []  # Every sample's prediction/label on the final epoch, for the correlation plot
    specific_accuracy = np.full(len(dataloader.dataset), np.nan)

    with torch.no_grad():
        for batch, (x, y, w, idx) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            t, noise = fixed_draws[batch] if fixed_draws is not None else (None, None)
            loss, pred, sigma = loss_fn(model, x.float(), y.float(), w.float(), predict=predict, t=t, noise=noise)
            validation_loss += loss.item()
            if pred is None:  # Predictions are made every predict_every epochs
                continue
            validation_mse += mse_fn(pred, y.float()).item()

            np_pred = pred.cpu().float().detach().numpy()
            np_y = y.cpu().float().detach().numpy()

            specific_accuracy[idx.numpy()], n_correct = count_accurate(np_pred, np_y)
            correct += n_correct
            total += np.size(np_y)

            if final:
                save_batch_outputs(x, y, pred, sigma, idx, validation_data, image_output_validation, epoch)
                all_pred.append(to_physical(np_pred))
                all_y.append(to_physical(np_y))

    # No predictions, and so no accuracy or MSE, on the epochs between predict_every
    accuracy = correct / total * 100 if total else np.nan
    validation_loss /= num_batches
    validation_mse = validation_mse / num_batches if total else np.nan
    logger.info("Validation: accuracy %.1f%%, avg loss %8f, avg MSE %8f", accuracy, validation_loss, validation_mse)

    if final and radial_bins is None:
        save_correlation_plots(all_pred, all_y, diagnostic_output + "_ValidationCorrelationPlot", epoch)

    return validation_loss, validation_mse, accuracy, specific_accuracy


#### Define Test Loop ####
def test_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the held-out test set. Same rough accuracy metric and final-epoch
    diagnostic outputs as validation_loop above, written under image_output_final_test instead.
    Returns (test_loss, test_mse, accuracy, specific_accuracy).
    """
    model.eval()
    num_batches = len(dataloader)
    test_loss, test_mse, correct = 0, 0, 0
    total = 0
    final = epoch >= epochs - 1
    all_pred, all_y = [], []  # Every sample's prediction/label on the final epoch, for the correlation plot
    specific_accuracy = np.full(len(dataloader.dataset), np.nan)

    with torch.no_grad():
        for x, y, w, idx in dataloader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            loss, pred, sigma = loss_fn(model, x.float(), y.float(), w.float(), predict=final)
            test_loss += loss.item()
            test_mse += mse_fn(pred, y.float()).item()

            np_pred = pred.cpu().float().detach().numpy()
            np_y = y.cpu().float().detach().numpy()

            specific_accuracy[idx.numpy()], n_correct = count_accurate(np_pred, np_y)
            correct += n_correct
            total += np.size(np_y)

            if final:
                save_batch_outputs(x, y, pred, sigma, idx, test_data, image_output_final_test, epoch)
                all_pred.append(to_physical(np_pred))
                all_y.append(to_physical(np_y))

    accuracy = correct / total * 100
    test_loss /= num_batches
    test_mse /= num_batches
    logger.info("Test: accuracy %.1f%%, avg loss %8f, avg MSE %8f", accuracy, test_loss, test_mse)

    if final and radial_bins is None:
        save_correlation_plots(all_pred, all_y, diagnostic_output + "_TestCorrelationPlot", epoch)

    return test_loss, test_mse, accuracy, specific_accuracy


#### Define the Model ####
input_shape = tuple(
    training_data[0][0].shape
)  # (nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
model = build_model(network, input_shape, hyperparams, output_size=radial_bins, n_maps=n_maps).to(device)
model.summary(print_fn=logger.info, layers=False)  # stage table and parameter counts
validation_names = validation_data.names
training_names = training_data.names


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
    LOSS_LABEL = "Denoising MSE"
else:
    LOSS_LABEL = ("Weighted " if mass_weight_power > 0 else "") + ("Gaussian NLL" if predict_variance else "MSE")


def compute_loss(model, x, y, w, predict=True, t=None, noise=None, sampling_model=None):
    """
    (loss, prediction, 1-sigma) for a batch of cubes x and labels y, each pixel (or ring) weighted by w; the
    prediction and 1-sigma are None unless predict.
    resblock: Gaussian NLL of (mean, exp(logvar)) if predict_variance, with 1-sigma exp(logvar / 2), else MSE, with
    1-sigma None.
    ddpm: the denoising loss (unweighted) at timesteps t with noise (drawn at random if None), and the mean and
    standard deviation of n_samples posterior samples, which are only drawn if predict since this has high compute
    cost. t, noise and sampling_model are ignored by the resblock network.
    sampling_model: the ddpm to draw posterior samples from (e.g. the EMA weights while the loss trains the
    raw ones); None samples from model.
    """
    if network == "ddpm":
        loss = model(x, y, t, noise)
        if not predict:
            return loss, None, None
        sampler = sampling_model if sampling_model is not None else model
        pred, sigma = sampler.predict_with_uncertainty(x, hyperparams["n_samples"])
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
)  # Steepest gradient descent, can use different optimizers

# Alternative: AdamW. To use it, uncomment the block below in place of Adam above, and raise weight_decay in the
# hyperparameter file (1e-2, PyTorch's AdamW default).
# AdamW decouples weight decay from the gradient step (Loshchilov & Hutter 2019): every decayed weight shrinks by
# the same factor (1 - learning_rate * weight_decay) per step. Adam's weight_decay instead adds an L2 term to the
# gradient, which its per-parameter scaling then distorts, so weights with small gradients are decayed far harder.
# Only conv and linear weights (ndim > 1), which set how strongly each layer responds to its input, are decayed.
# Biases and the GroupNorm/BatchNorm scales and shifts (ndim <= 1) set offsets and activation scales rather than
# model complexity, so decaying them would bias the outputs and fight the normalization instead of regularizing.
# decay_params = [p for p in model.parameters() if p.ndim > 1]
# no_decay_params = [p for p in model.parameters() if p.ndim <= 1]
# optimizer = torch.optim.AdamW(
#     [
#         {"params": decay_params, "weight_decay": weight_decay},
#         {"params": no_decay_params, "weight_decay": 0.0},
#     ],
#     lr=learning_rate,
# )

#### Exponential moving average (EMA) of the weights ####
# With a constant learning rate the raw weights jitter around a minimum, by an amount set by the gradient
# noise of each batch (for the ddpm also the random t and noise of every sample). The EMA,
# w_ema <- ema_decay * w_ema + (1 - ema_decay) * w   after every optimizer step,
# averages that jitter out over ~1 / (1 - ema_decay) steps. Training (loss, gradients) still updates the raw
# weights, but validation, testing, the ddpm's posterior samples and the saved model all use the EMA.
# ema_decay = 0 copies the raw weights, which reproduces training without an EMA.
# use_buffers=True also averages buffers with the same EMA: the resblock network's BatchNorm running mean and
# variance then follow the averaged weights, rather than being the raw model's statistics (use_buffers=False),
# which come from the raw weights' activations. The ddpm's only buffers are its fixed noise schedule, which
# averages to itself.
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
eval_model = ema_model.module  # The weights every evaluation uses and that are saved
# The EMA copy is never trained, so it stays in eval mode for good: dropout is off whenever it is evaluated or
# sampled from, including the ddpm's training-set predictions inside train_loop. The raw model's mode is set
# by train_loop (model.train()) each epoch and is not changed by the validation or test loops.
eval_model.eval()

#### Fixed validation noise (ddpm) ####
# One (t, noise) draw per validation batch, made once here, so every epoch's validation loss scores the same noisy
# maps and its changes reflect the model rather than the draw. The validation dataloader is not shuffled, so batch i
# holds the same cubes every epoch.
validation_draws = None
if network == "ddpm":
    # One timestep per validation cube, stratified over the whole set (see GaussianDiffusion.sample_timesteps), so
    # the fixed draws cover the noise levels evenly; batch i takes the next len(batch) of them, in dataset order.
    validation_t = model.sample_timesteps(len(validation_data), device=device)
    validation_draws = []
    start = 0
    for _, y, _, _ in validation_dataloader:
        n = y.shape[0]
        validation_draws.append((validation_t[start : start + n], torch.randn(y.shape, device=device)))
        start += n

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
    training_losses[t], training_mses[t] = train_loop(train_dataloader, model, compute_loss, optimizer, t, ema_model)
    validation_losses[t], validation_mses[t], validation_accuracies[t], specific_accuracies[:, t] = validation_loop(
        validation_dataloader, eval_model, compute_loss, t, validation_draws
    )


# Save the Trained Model
torch.save(eval_model.state_dict(), model_output_path + ".pt")  # The EMA weights, with the same keys as model

# Save the normalization information for loading data
hf_norm = h5py.File(model_output_path + ".hdf5", "w")
hf_norm.create_dataset("image_stats", data=np.array([training_mean, training_stdv]))
hf_norm.create_dataset(
    "label_stats", data=np.stack([label_offset[:, 0], label_scale[:, 0]])
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
hf_norm.close()


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

final_test_loss, final_test_mse, final_test_accuracy, final_test_specific_accuracy = test_loop(
    test_dataloader, eval_model, compute_loss, t
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
    plt.ylabel(LOSS_LABEL)
    plt.savefig(output)
    plt.close()


plot_loss(training_losses, "Training Loss", diagnostic_output + "_trainingLoss.png")
plot_loss(validation_losses, "Validation Loss", diagnostic_output + "_validationLoss.png")

# Accuracies are NaN on the epochs between predict_every, so only the predicted epochs are plotted
epoch_numbers = np.arange(1, epochs + 1)
predicted = np.isfinite(validation_accuracies)

fig3 = plt.figure()
plt.plot(epoch_numbers[predicted], validation_accuracies[predicted], marker="o", ms=3)
plt.title(f"Validation Accuracy (within {ACCURACY_TOLERANCE:g} label std. dev.)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([0, 101])
plt.savefig(diagnostic_output + "_ValidationAccuracy.png")


fig4 = plt.figure()
for name, accuracies in zip(validation_names, specific_accuracies):
    plt.plot(epoch_numbers[predicted], accuracies[predicted], lw=0.5, marker="o", ms=2, label=name)
if len(validation_names) <= 10:  # Name each line, unless there are too many to read
    plt.legend(fontsize=6)
plt.title("Validation Accuracy for each Snapshot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([0, 101])
plt.savefig(diagnostic_output + "_ValidationAccuracyPerSnapshot.png")


hf_diagnostic = h5py.File(diagnostic_output + "_LossesAndAccuracies.hdf5", "w")
hf_diagnostic.create_dataset("training_losses", data=training_losses)
hf_diagnostic.create_dataset("validation_losses", data=validation_losses)
hf_diagnostic.create_dataset("training_mses", data=training_mses)
hf_diagnostic.create_dataset("validation_mses", data=validation_mses)
hf_diagnostic.create_dataset("validation_accuracies", data=validation_accuracies)
hf_diagnostic.create_dataset("specific_accuracies", data=specific_accuracies)
hf_diagnostic.create_dataset("training_names", data=training_names)
hf_diagnostic.create_dataset("validation_names", data=validation_names)
hf_diagnostic.create_dataset("testing_names", data=testing_names)
hf_diagnostic.create_dataset("final_test_loss", data=final_test_loss)
hf_diagnostic.create_dataset("final_test_mse", data=final_test_mse)
hf_diagnostic.create_dataset("final_test_accuracy", data=final_test_accuracy)
hf_diagnostic.create_dataset("final_test_specific_accuracy", data=final_test_specific_accuracy)

hf_diagnostic.close()
