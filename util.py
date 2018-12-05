"""
Utility functions for SME

command line argument parsing
logging
"""

import os
import argparse
import logging

# import builtins

try:
    from IPython import get_ipython

    cfg = get_ipython().config
    in_notebook = cfg["IPKernelApp"]["parent_appname"] == "ipython-notebook"
except AttributeError:
    in_notebook = False
except ImportError:
    in_notebook = False


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
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
        if isinstance(h, logging.StreamHandler):
            hasStream = True

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
    # builtins.print = lambda msg, *args, **kwargs: logging.info(msg, *args)
    logging.captureWarnings(True)

    logging.debug("----------------------")


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
