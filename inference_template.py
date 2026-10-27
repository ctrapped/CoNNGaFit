import argparse

from logging_setup import add_logging_arguments, configure_logging
from use_model import read_image_list, run_inferences

####Template for running inference with a previously trained CoNNGaFit network on a list of
####input datacubes. See use_model.run_inferences for the full parameter description.
#
####Run as: python inference_template.py [options]
####  Run with --help to see all options.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a previously trained CoNNGaFit network on a list of input datacubes."
    )
    parser.add_argument(
        "--image-list",
        default="CoNNGaFitData/observations/inferencesToRun.csv",
        help="Text or CSV file listing the datacubes to run inference on, one path per line (the first field of each CSV row). Default: %(default)s",
    )
    parser.add_argument(
        "--output-prefix",
        default="CoNNGaFitData/observations/output_example",
        help="Prefix (directory + filename base) outputs are written under. Default: %(default)s",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a checkpoint written by train_model_unet.py, without the .pt/.hdf5 extension, e.g. NetworkWeights/my_run.",
    )
    add_logging_arguments(parser)
    return parser.parse_args()


args = parse_args()
configure_logging(verbose=args.verbose, quiet=args.quiet)

run_inferences(args.model_path, read_image_list(args.image_list), args.output_prefix)
