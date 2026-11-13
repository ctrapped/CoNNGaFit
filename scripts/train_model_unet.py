"""Train a CoNNGaFit network from the command line, on a hyperparameters file and a datasets file (see configs/).

    python scripts/train_model_unet.py --hyperparams FILE --datasets FILE --run-name NAME [options]

Run with --help for every option. The training itself is conngafit.train_model.train, which can also be called from
Python with the parameters as dicts (see examples/demo.py). By default the trained network is saved to
NetworkWeights/ and the diagnostic images and plots to DiagnosticOutputs/.
"""

import argparse
import os

from conngafit.logging_setup import add_logging_arguments, configure_logging
from conngafit.train_model import load_datasets, load_hyperparams, train

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


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
    io.add_argument(
        "--no-diagnostics",
        dest="save_diagnostics",
        action="store_false",
        help="Save only the checkpoint and the loss history (<output-dir>/diagnostics/<run-name>_LossesAndAccuracies.hdf5), "
        "not the final-epoch snapshots, comparison images, or loss, accuracy and correlation plots.",
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
    training.add_argument(
        "--diagnostic-mode",
        action="store_true",
        help="Also record the ddpm diagnostics (conngafit/models/ddpm_diagnostics/) every predict_every epochs and "
        "on the final epoch, to <output-dir>/diagnostics/. Ignored for the other networks.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    train(
        load_hyperparams(args.hyperparams),
        load_datasets(args.datasets),
        run_name=args.run_name,
        network_dir=args.network_dir,
        output_dir=args.output_dir,
        save_diagnostics=args.save_diagnostics,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        diagnostic_mode=args.diagnostic_mode,
    )


if __name__ == "__main__":
    main()
