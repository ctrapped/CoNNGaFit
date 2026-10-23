import argparse
import os

from logging_setup import add_logging_arguments, configure_logging
from use_model import run_inferences

####Runs inference with a previously trained 'unet18' network on the validation set for a
####given sample, so predictions can be spot-checked against the known validation labels.
#
####Run as: python inference_test_validation_set.py [options]
####  Run with --help to see all options.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a previously trained 'unet18' network on the validation set, to spot-check predictions against known labels."
    )
    parser.add_argument(
        "--sample-suffix",
        default="finalSnapNoM12m",
        help="Suffix used to build the default validation annotation CSV filename, ignored if --validation-csv is given explicitly. Default: %(default)s",
    )
    parser.add_argument(
        "--data-dir",
        default="CoNNGaFitData",
        help="Root data directory the validation annotation CSV is read from (<data-dir>/annotation_datasets/). Default: %(default)s",
    )
    parser.add_argument(
        "--validation-csv",
        default=None,
        help="Validation annotations CSV filename, read from <data-dir>/annotation_datasets/. Default: validation_annotations_MassFlux_All_Inclinations_<sample-suffix>.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="CoNNGaFitData/observations/validation_tests/",
        help="Directory to place outputs in. Default: %(default)s",
    )
    parser.add_argument(
        "--network-type", default="unet18", help="Which trained network configuration to use. Default: %(default)s"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path (without the trailing .pt/.hdf5 extension) to the trained model checkpoint to load. Default: the network-type's built-in default checkpoint.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


args = parse_args()
configure_logging(verbose=args.verbose, quiet=args.quiet)

sample_suffix = args.sample_suffix
validation_set_filename = (
    args.validation_csv or "validation_annotations_MassFlux_All_Inclinations_" + sample_suffix + ".csv"
)
validation_dir = os.path.join(args.data_dir, "annotation_datasets", validation_set_filename)

run_inferences(args.network_type, args.output_dir, validation_dir, model_path=args.model_path, save_latent_images=True)
