"""
Utility functions for SME

command line argument parsing
logging
"""

import os
import argparse
import logging

import numpy as np
from scipy.interpolate import interp1d

from ...version import version as __version__
from .sme_synth import SMELibraryVersion
from platform import python_version
from numpy import __version__ as npversion
from scipy import __version__ as spversion
from pandas import __version__ as pdversion


import builtins

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


def safe_interpolation(x_old, y_old, x_new=None):
    """
    'Safe' interpolation method that should avoid
    the common pitfalls of spline interpolation

    masked arrays are compressed, i.e. only non masked entries are used
    remove NaN input in x_old and y_old
    only unique x values are used, corresponding y values are 'random'
    if all else fails, revert to linear interpolation

    Parameters
    ----------
    x_old : array of size (n,)
        x values of the data
    y_old : array of size (n,)
        y values of the data
    x_new : array of size (m, ) or None, optional
        x values of the interpolated values
        if None will return the interpolator object
        (default: None)

    Returns
    -------
    y_new: array of size (m, ) or interpolator
        if x_new was given, return the interpolated values
        otherwise return the interpolator object
    """

    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)

    mask = np.isfinite(x_old) & np.isfinite(y_old)
    x_old = x_old[mask]
    y_old = y_old[mask]

    # avoid duplicate entries in x
    # also sorts data, which allows us to use assume_sorted below
    x_old, index = np.unique(x_old, return_index=True)
    y_old = y_old[index]

    try:
        interpolator = interp1d(
            x_old,
            y_old,
            kind="cubic",
            fill_value=0,
            bounds_error=False,
            assume_sorted=True,
        )
    except ValueError:
        logging.warning(
            "Could not instantiate cubic spline interpolation, using linear instead"
        )
        interpolator = interp1d(
            x_old,
            y_old,
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=True,
        )

    if x_new is not None:
        return interpolator(x_new)
    else:
        return interpolator
