"""
Utility functions for SME

command line argument parsing
logging
"""
import argparse
import builtins
import logging
import os
from platform import python_version

from numpy import __version__ as npversion
from pandas import __version__ as pdversion
from scipy import __version__ as spversion

try:
    from .src.sme.sme_synth import SMELibraryVersion
    from .version import version as __version__
except ImportError:
    from src.sme.sme_synth import SMELibraryVersion
    from version import version as __version__

try:
    from IPython import get_ipython

    cfg = get_ipython()
    in_notebook = cfg is not None
except (AttributeError, ImportError):
    in_notebook = False


def has_logger(log_file="log.log"):
    logger = logging.getLogger()
    return len(logger.handlers) == 0


def start_logging(log_file="log.log"):
    """Start logging to log file and command line

    Parameters
    ----------
    log_file : str, optional
        name of the logging file (default: "log.log")
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove existing File handles
    hasStream = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Command Line output
    # only if not running in notebook
    if not in_notebook and not hasStream:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    # Log file settings
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir == "":
            log_dir = "./"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file = logging.FileHandler(log_file)
        file.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file.setFormatter(file_formatter)
        logger.addHandler(file)

    # Turns print into logging.info
    # But messes with the debugger
    # builtins.print = lambda msg, *args, **kwargs: logging.info(msg, *args)
    logging.captureWarnings(True)

    logging.debug("----------------------")
    logging.debug("Python version: %s", python_version())
    logging.debug("SME CLib version: %s", SMELibraryVersion())
    logging.debug("PySME version: %s", __version__)
    logging.debug("Numpy version: %s", npversion)
    logging.debug("Scipy version: %s", spversion)
    logging.debug("Pandas version: %s", pdversion)


def parse_args():
    """Parse command line arguments

    Returns
    -------
    sme : str
        filename to input sme structure
    vald : str
        filename of input linelist or None
    fitparameters : list(str)
        names of the parameters to fit, empty list if none are specified
    """

    parser = argparse.ArgumentParser(description="SME solve")
    parser.add_argument(
        "sme",
        type=str,
        help="an sme input file (either in IDL sav or Numpy npy format)",
    )
    parser.add_argument("--vald", type=str, default=None, help="the vald linelist file")
    parser.add_argument(
        "fitparameters",
        type=str,
        nargs="*",
        help="Parameters to fit, abundances are 'Mg Abund'",
    )
    args = parser.parse_args()
    return args.sme, args.vald, args.fitparameters
