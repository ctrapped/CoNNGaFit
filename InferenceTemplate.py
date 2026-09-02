import argparse
from UseModel import RunInferences
####Template for running inference with a previously trained CoNNGaFit network on a list of
####input datacubes. See UseModel.RunInferences for the full parameter description.
#
####Run as: python InferenceTemplate.py [options]
####  Run with --help to see all options.

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a previously trained CoNNGaFit network on a list of input datacubes.")
    parser.add_argument('--network-type', default='unet18',
                         help="Which trained network configuration to use. Default: %(default)s")
    parser.add_argument('--image-list', default="CoNNGaFitData/observations/inferencesToRun.csv",
                         help="Path to a CSV listing the input datacubes to run inference on. Default: %(default)s")
    parser.add_argument('--output-prefix', default="CoNNGaFitData/observations/output_example",
                         help="Prefix (directory + filename base) outputs are written under. Default: %(default)s")
    parser.add_argument('--model-path', default=None,
                         help="Path (without the trailing .pt/.hdf5 extension) to the trained model checkpoint to load. Default: the network-type's built-in default checkpoint.")
    return parser.parse_args()

args = parse_args()

RunInferences(args.network_type, args.output_prefix, args.image_list, modelPath=args.model_path)
