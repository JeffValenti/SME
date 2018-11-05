import sys
import os.path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
from scipy.optimize import OptimizeWarning

from src.gui import plotting
from src.sme import sme as SME
from src.sme.solve import sme_func, solve
from src.sme.vald import ValdFile


def parse_args():
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


if len(sys.argv) > 1:
    in_file, vald_file, fitparameters = parse_args()
else:
    # in_file = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
    in_file = "./sme_3param_extra_errors.npy"
    vald_file = "/home/ansgar/Documents/IDL/SME/harps_red.lin"
    fitparameters = []

sme = SME.SME_Struct.load(in_file)

if vald_file is not None:
    vald = ValdFile(vald_file)
    sme.linelist = vald.linelist

# Choose free parameters, i.e. sme.pname
if len(fitparameters) == 0:
    # ["teff", "logg", "feh", "Mg Abund", "Y Abund"]
    fitparameters = sme.pname

sme = solve(sme, fitparameters)

# Plot results
mask_plot = plotting.MaskPlot(sme)
# Update mask
# new_mask = mask_plot.mask
# sme.mob = new_mask.__values__

# Calculate stellar age based on abundances
x = sme["Y Abund"] / sme["Mg Abund"]
sx = (sme.punc["Y Abund"] / sme["Mg Abund"]) ** 2 + (
    sme.punc["Mg Abund"] * sme["Y Abund"] / sme["Mg Abund"] ** 2
) ** 2
sx = np.sqrt(sx)

# Values from paper
a = 0.175
sa = 0.011
b = -0.0404
sb = 0.0019
age = (x - a) / b
sigma_age = 1 / b * np.sqrt(sx ** 2 + sa ** 2 + ((x - a) / b) ** 2 * sb ** 2)

print("Age = %.3f Gyr", age)
