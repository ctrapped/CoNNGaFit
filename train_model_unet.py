import logging
import os

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from conngafit_datasets import TARGETS, CoNNGaFitImageDataset, inclination_folders
from logging_setup import add_logging_arguments, configure_logging
from unet3d_resblock import NeuralNetwork

matplotlib.use("Agg")
import argparse
import tomllib

import h5py
from matplotlib import pyplot as plt

from plotting_functions import make_comp_image, make_comp_image_1d, make_correlation_plot, save_hdf5, save_hdf5_1d

####Trains the Unet read from unet3d_resblock.py on the given training data. Provides additional diagnostic plots and images on the validation and test datasets provided.
####Training, validation, and testing datasets are VOF optically thin outputs, chosen by galaxy and snapshot range (see conngafit_datasets.py).
####Hyperparameters were chosen based on paramter space optimization+trial and error.
####Hyperparameters and datasets (including the target annotation) are read from two parameter files,
####configs/hyperparams.toml and configs/datasets.toml by default; plot labels per target are in TARGET_SETTINGS below
#
####Run as: python train_model_unet.py [--hyperparams FILE] [--datasets FILE] [--run-name NAME] [options]
####  Run with --help to see all options. The two parameter files and --run-name are normally the
####  only options you need. The datasets file gives the path to each galaxy's VOF outputs, and by default:
####    NetworkWeights/    receives the trained model checkpoint (<run-name>.pt/.hdf5)
####    DiagnosticOutputs/ receives per-epoch images/snapshots and loss/accuracy/correlation plots
####  NetworkWeights/ and DiagnosticOutputs/ are created automatically if missing.
#
####Written By Cameron Trapp (ctrapped@gmail.com)
####Updated 10/21/2026


####   PER-TARGET SETTINGS   ####
# target -> (plot label, plot units, how pixels are combined when binning plots)
TARGET_SETTINGS = {
    "mass": ("Mass", r"[M$_{\odot}$]", "sum"),
    "radial_velocity": ("Rad. Vel.", r"[km s$^{-1}$]", "mean"),
    "cylindrical_radial_velocity": ("Cyl. Rad. Vel.", r"[km s$^{-1}$]", "mean"),
    "rotational_velocity": ("Rot. Vel.", r"[km s$^{-1}$]", "mean"),
    "radial_mass_flux": ("Radial Mass Flux", r"[M$_{\odot}$ yr$^{-1}$]", "sum"),
    "cylindrical_radial_mass_flux": ("Cyl. Radial Mass Flux", r"[M$_{\odot}$ yr$^{-1}$]", "sum"),
}
#############################

####   PARAMETER FILES   ####
# Keys every parameter file must have (see configs/*.toml for what each one does)
HYPERPARAM_KEYS = {
    "learning_rate",
    "weight_decay",
    "epochs",
    "n_fc",
    "n_filt0",
    "k0",
    "k1",
    "dropout_rate",
    "block_dropout_rate",
    "output_head",
    "label_scaling",
}
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
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


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
    directory relative to base_dir (default: the datasets file's folder). Returns the dict conngafit_datasets.py expects."""
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
        default=os.path.join(CONFIG_DIR, "hyperparams.toml"),
        help="Hyperparameter file (learning rate, epochs, network size, output head). Default: configs/hyperparams.toml",
    )
    parser.add_argument(
        "--datasets",
        default=os.path.join(CONFIG_DIR, "datasets.toml"),
        help="Datasets file (target, augmentation, training/validation/testing galaxies). Default: configs/datasets.toml",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run; the checkpoint and every diagnostic output are named after it. Default: the target",
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
hyperparams, hyperparams_text = load_parameter_file(args.hyperparams, HYPERPARAM_KEYS)
learning_rate = hyperparams["learning_rate"]
weight_decay = hyperparams["weight_decay"]
epochs = hyperparams["epochs"]
n_fc = hyperparams["n_fc"]
n_filt0 = hyperparams["n_filt0"]
k0 = hyperparams["k0"]
k1 = hyperparams["k1"]
dropout_rate = hyperparams["dropout_rate"]
block_dropout_rate = hyperparams["block_dropout_rate"]
output_head = hyperparams["output_head"]
label_scaling = hyperparams["label_scaling"]
if label_scaling not in ("none", "symmetric", "one-sided"):
    raise ValueError(
        f'{args.hyperparams}: label_scaling = {label_scaling!r} should be "none", "symmetric" or "one-sided"'
    )
#############################


####   DATASETS   ####
datasets, datasets_text = load_parameter_file(args.datasets, DATASET_KEYS, OPTIONAL_DATASET_KEYS)
target = datasets["target"]
if target not in TARGETS:
    raise ValueError(f"{args.datasets}: unknown target {target!r}; choose from {sorted(TARGETS)}")
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

param_label, unit_label, bin_op = TARGET_SETTINGS[target]
run_name = args.run_name or target


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
    target,
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

#### Label scaling, from every training label (augmentation doesn't change their range) ####
# Labels are scaled to (label - label_offset) / label_scale for training; predictions and labels are
# scaled back to physical units for every saved output and plot
label_offset, label_scale = 0.0, 1.0
if label_scaling != "none":
    all_labels = np.concatenate([training_data.label(i) for i in range(len(training_data.img_paths))])
    if label_scaling == "symmetric":  # e.g. mass flux: divide by max(|label|), to [-1, 1]
        label_scale = np.max(np.abs(all_labels))
    else:  # "one-sided", e.g. rotational velocity: [min(label), max(label)] to [-1, 1]
        label_offset = (np.max(all_labels) + np.min(all_labels)) / 2
        label_scale = (np.max(all_labels) - np.min(all_labels)) / 2
    if not label_scale > 0:
        raise ValueError(
            f"label_scaling = {label_scaling!r} needs training labels that aren't all {np.max(all_labels)}"
        )
    del all_labels
    logger.info("Label scaling (%s): offset = %.4g, scale = %.4g", label_scaling, label_offset, label_scale)


def scale_labels(label):
    return (label - label_offset) / label_scale


def to_physical(a):
    """Scale labels or predictions back to physical units."""
    return a * label_scale + label_offset


training_data.transform = Compose([ToTensor(), Normalize(training_mean, training_stdv)])
training_data.target_transform = scale_labels

validation_data = CoNNGaFitImageDataset(
    validation_sources,
    target,
    radial_bins=radial_bins,
    radial_projection=radial_projection,
    inclinations=inclinations,
    suffix=suffix,
    transform=Compose([ToTensor(), Normalize(training_mean, training_stdv)]),
    target_transform=scale_labels,
)


class WithIndices(Dataset):
    """Wraps a dataset so each sample also returns its index, to look up its name after shuffling."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)


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
def save_batch_outputs(x, y, pred, idx, names, output_base, epoch, comp_images=True):
    """Save each sample in a batch as an hdf5 snapshot (and a comparison image), named after its
    image via its dataset index (idx), so names stay correct with shuffled or multiple batches."""
    np_x = x.cpu().float().detach().numpy()
    np_y = to_physical(y.cpu().float().detach().numpy())
    np_pred = to_physical(pred.cpu().detach().numpy())
    for tt in range(len(idx)):
        output_name = output_base + "_" + names[int(idx[tt])] + "_Epoch" + str(epoch)
        if comp_images:
            make_comp_image(
                np_pred,
                np_y,
                np_x,
                output_name + ".png",
                tt,
                sim_to_physical_units=1,
                param_label=param_label,
                unit_label=unit_label,
                bin_op=bin_op,
            )  # Labels are already in physical units
        save_hdf5(np_y, np_pred, output_name + ".hdf5", tt)


#### Define Training Loop ####
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """Run one training epoch: iterate the dataloader, backprop MSE loss each batch, and on
    the final epoch dump every sample's prediction/label hdf5 snapshot for later inspection.
    Returns the loss averaged over all batches in the epoch.
    """
    model.train()
    size = len(dataloader.dataset)
    epoch_loss = 0.0
    num_batches = len(dataloader)
    for batch, (x, y, idx) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x.float())
        loss = loss_fn(pred, y.float())
        epoch_loss += loss.item()
        if epoch >= epochs - 1:
            save_batch_outputs(x, y, pred, idx, training_names, image_output_training, epoch, comp_images=False)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current = batch * len(x)
            logger.info("loss: %7f [%5d/%5d]", loss.item(), current, size)

    return epoch_loss / num_batches


#### Define Validation Loop ####
def validation_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the validation set (no gradient updates) and compute a rough
    "accuracy" (fraction of pixels whose prediction falls within one std-dev of the label -
    a loose sanity metric, not a real accuracy). On the final epoch, also writes correlation
    plots and per-sample comparison images/hdf5 snapshots. Returns
    (validation_loss, accuracy, specific_accuracy) where specific_accuracy is the per-sample
    accuracy for every sample in the validation batch.
    """
    model.eval()
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0
    total = 0
    final = epoch >= epochs - 1
    all_pred, all_y = [], []  # Every sample's prediction/label on the final epoch, for the correlation plot

    with torch.no_grad():
        for x, y, idx in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x.float())
            validation_loss += loss_fn(pred, y.float()).item()

            np_pred = pred.cpu().float().detach().numpy()
            np_y = y.cpu().float().detach().numpy()

            #### Roughly estimate accuracy of network. Not a great metric but okay for tracking general peformance ####
            deviation = np.std(np_y)
            correct += np.size(np.where(((np_pred > np_y - deviation) & (np_pred < np_y + deviation)))[0])
            total += np.size(np_y)

            n_snaps, n_pix = np.shape(np_y)
            specific_accuracy = np.zeros((n_snaps))
            for i in range(0, n_snaps):
                correct_for_snap = np.size(
                    np.where(((np_pred[i, :] > np_y[i, :] - deviation) & (np_pred[i, :] < np_y[i, :] + deviation)))[0]
                )
                specific_accuracy[i] = correct_for_snap / np.size(np_y[i, :]) * 100

            if final:
                save_batch_outputs(x, y, pred, idx, validation_names, image_output_validation, epoch)
                all_pred.append(to_physical(np_pred))
                all_y.append(to_physical(np_y))

    accuracy = correct / total * 100
    validation_loss /= num_batches
    logger.info("Validation: accuracy %.1f%%, avg loss %8f", accuracy, validation_loss)

    if final and radial_bins is None:
        make_correlation_plot(
            np.concatenate(all_pred),
            np.concatenate(all_y),
            sum(len(a) for a in all_pred),
            diagnostic_output + "_ValidationCorrelationPlot_Epoch" + str(epoch),
        )

    model.train()
    return validation_loss, accuracy, specific_accuracy


#### Define Test Loop ####
def test_loop(dataloader, model, loss_fn, epoch):
    """Evaluate the model on the held-out test set. Same rough accuracy metric and final-epoch
    diagnostic outputs as validation_loop above, written under image_output_final_test instead.
    Returns (test_loss, accuracy, specific_accuracy).
    """
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    total = 0
    final = epoch >= epochs - 1
    all_pred, all_y = [], []  # Every sample's prediction/label on the final epoch, for the correlation plot

    with torch.no_grad():
        for x, y, idx in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x.float())
            test_loss += loss_fn(pred, y.float()).item()

            np_pred = pred.cpu().float().detach().numpy()
            np_y = y.cpu().float().detach().numpy()

            #### Roughly estimate accuracy of network. Not a great metric but okay for tracking general peformance ####
            deviation = np.std(np_y)
            correct += np.size(np.where(((np_pred > np_y - deviation) & (np_pred < np_y + deviation)))[0])

            total += np.size(np_y)

            n_snaps, n_pix = np.shape(np_y)
            specific_accuracy = np.zeros((n_snaps))
            for i in range(0, n_snaps):
                correct_for_snap = np.size(
                    np.where(((np_pred[i, :] > np_y[i, :] - deviation) & (np_pred[i, :] < np_y[i, :] + deviation)))[0]
                )
                specific_accuracy[i] = correct_for_snap / np.size(np_y[i, :]) * 100

            if final:
                save_batch_outputs(x, y, pred, idx, testing_names, image_output_final_test, epoch)
                all_pred.append(to_physical(np_pred))
                all_y.append(to_physical(np_y))

    accuracy = correct / total * 100
    test_loss /= num_batches
    logger.info("Test: accuracy %.1f%%, avg loss %8f", accuracy, test_loss)

    if final and radial_bins is None:
        make_correlation_plot(
            np.concatenate(all_pred),
            np.concatenate(all_y),
            sum(len(a) for a in all_pred),
            diagnostic_output + "_TestCorrelationPlot_Epoch" + str(epoch),
        )

    model.train()
    return test_loss, accuracy, specific_accuracy


#### Define the Model ####
input_shape = tuple(
    training_data[0][0].shape
)  # (nSpec, nY, nX) after ToTensor; the output map has one value per (nY, nX) pixel
model = NeuralNetwork(
    input_shape,
    n_filt0,
    (k0, k0, k0),
    (k1, k1, k1),
    n_fc,
    fc_dropout_rate=dropout_rate,
    block_dropout_rate=block_dropout_rate,
    head=output_head,
    output_size=radial_bins,
).to(device)
model.summary(print_fn=logger.info, layers=False)  # stage table and parameter counts
validation_names = validation_data.names
training_names = training_data.names

#### Define Loss Function and Optimizer ####
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)  # Steepest gradient descent, can use different optimizers

#### Initialize losses and specific accuracies ####
training_losses = np.zeros((epochs))
validation_losses = np.zeros((epochs))
validation_accuracies = np.zeros((epochs))

if batch_size_default is not None:
    # specific_accuracies is only ever filled in per-sample when batch_size_default is None (see the
    # main training loop below), so it stays unused in batch mode - sized to 0 rows rather than
    # referencing test_batch_size here.
    specific_accuracies = np.zeros((0, epochs))
else:
    specific_accuracies = np.zeros((len(validation_data), epochs))


#### Perform Training Loop ####
for t in range(epochs):
    logger.info("Epoch %d/%d", t + 1, epochs)
    training_losses[t] = train_loop(train_dataloader, model, loss_fn, optimizer, t)
    if batch_size_default is None:
        validation_losses[t], validation_accuracies[t], specific_accuracies[:, t] = validation_loop(
            validation_dataloader, model, loss_fn, t
        )
    else:
        validation_losses[t], validation_accuracies[t], tmp = validation_loop(validation_dataloader, model, loss_fn, t)


# Save the Trained Model
torch.save(model.state_dict(), model_output_path + ".pt")

# Save the normalization information for loading data
hf_norm = h5py.File(model_output_path + ".hdf5", "w")
hf_norm.create_dataset("image_stats", data=np.array([training_mean, training_stdv]))
hf_norm.create_dataset("label_scaling", data=label_scaling)
hf_norm.create_dataset(
    "label_stats", data=np.array([label_offset, label_scale])
)  # physical label = scaled label * label_scale + label_offset
hf_norm.create_dataset("learning_rate", data=learning_rate)
hf_norm.create_dataset("n_fc", data=n_fc)
hf_norm.create_dataset("weight_decay", data=weight_decay)
hf_norm.create_dataset("n_filt0", data=n_filt0)
hf_norm.create_dataset("dropout_rate", data=dropout_rate)
hf_norm.create_dataset("block_dropout_rate", data=block_dropout_rate)
hf_norm.create_dataset("run_name", data=run_name)
hf_norm.create_dataset("input_shape", data=input_shape)  # Needed to rebuild the network for inference
hf_norm.create_dataset("target", data=target)
hf_norm.create_dataset("output_head", data=output_head)
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
    target,
    radial_bins=radial_bins,
    radial_projection=radial_projection,
    inclinations=inclinations,
    suffix=suffix,
    transform=Compose([ToTensor(), Normalize(training_mean, training_stdv)]),
    target_transform=scale_labels,
)

test_dataloader = DataLoader(WithIndices(test_data), batch_size=batch_size)
testing_names = test_data.names

final_test_loss, final_test_accuracy, final_test_specific_accuracy = test_loop(test_dataloader, model, loss_fn, t)
logger.info("Testing set loss = %g", final_test_loss)
logger.info("Testing set accuracy = %g", final_test_accuracy)
logger.info("Testing set specific accuracies:")
for accuracy in final_test_specific_accuracy:
    logger.info("  %s", accuracy)


#### Make Basic Diagnostic Outputs ####
fig1 = plt.figure()
plt.plot(np.log10(training_losses))
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Log(Loss)")
plt.savefig(diagnostic_output + "_trainingLoss.png")

fig2 = plt.figure()
plt.plot(np.log10(validation_losses))
plt.title("Valdiation Loss")
plt.xlabel("Epoch")
plt.ylabel("Log(Loss)")

plt.savefig(diagnostic_output + "_validationLoss.png")

fig3 = plt.figure()
plt.plot(validation_accuracies)
plt.title("Valdiation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([60, 101])
plt.savefig(diagnostic_output + "_ValdiationAccuracy.png")


fig4 = plt.figure()
for i in range(0, np.shape(specific_accuracies)[0]):
    plt.plot(specific_accuracies[i, :], lw=0.5)
plt.title("Valdiation Accuracy for each Snapshot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.ylim([60, 101])
plt.savefig(diagnostic_output + "_ValdiationAccuracyPerSnapshot.png")


hf_diagnostic = h5py.File(diagnostic_output + "_LossesAndAccuracies.hdf5", "w")
hf_diagnostic.create_dataset("training_losses", data=training_losses)
hf_diagnostic.create_dataset("validation_losses", data=validation_losses)
hf_diagnostic.create_dataset("validation_accuracies", data=validation_accuracies)
hf_diagnostic.create_dataset("specific_accuracies", data=specific_accuracies)
hf_diagnostic.create_dataset("training_names", data=training_names)
hf_diagnostic.create_dataset("validation_names", data=validation_names)
hf_diagnostic.create_dataset("testing_names", data=testing_names)
hf_diagnostic.create_dataset("final_test_loss", data=final_test_loss)
hf_diagnostic.create_dataset("final_test_accuracy", data=final_test_accuracy)
hf_diagnostic.create_dataset("final_test_specific_accuracy", data=final_test_specific_accuracy)

hf_diagnostic.close()
