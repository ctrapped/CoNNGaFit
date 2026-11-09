"""Shared logging configuration for the command-line scripts.

Used by train_model_unet.py and the inference/FITS scripts. Library modules only create a logger
(logging.getLogger(__name__)) and never configure output themselves.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers kept at INFO under -v, so debug output is CoNNGaFit's own (torch sets its own levels)
QUIET_LIBRARIES = ("matplotlib", "PIL", "h5py")


def add_logging_arguments(parser):
    """Add the -v/--verbose and -q/--quiet flags to an argparse parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show debug messages (array shapes, per-step details)",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show warnings and errors",
    )


def configure_logging(verbose=False, quiet=False):
    """Set up timestamped log output at the level chosen by -v/-q."""
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    # Not a package yet, so there's no shared "conngafit" logger to set; the root logger is set instead
    if verbose:
        for name in QUIET_LIBRARIES:
            logging.getLogger(name).setLevel(logging.INFO)
