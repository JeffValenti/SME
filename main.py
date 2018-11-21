import sys
import os.path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.gui import plot_pyplot

from src.sme import sme as SME
from src.sme.vald import ValdFile

from src.sme.solve import solve


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
    in_file = "./wasp117_short.inp"
    vald_file = "./5475-5548hps_vdw.lin"
    fitparameters = []

sme = SME.SME_Struct.load(in_file)

if vald_file is not None:
    vald = ValdFile(vald_file)
    sme.linelist = vald.linelist

# Choose free parameters, i.e. sme.pname
if len(fitparameters) == 0:
    # ["teff", "logg", "monh", "Mg Abund", "Y Abund"]
    if sme.pname is not None:
        fitparameters = sme.pname
    else:
        fitparameters = ["teff", "logg", "monh"]

# TODO: DEBUG
fitparameters = ["teff", "logg", "monh"]
sme = solve(sme, fitparameters)


# # Plot results
mask_plot = plot_pyplot.MaskPlot(sme)
input("Wait a second...")

# # Update mask
# # new_mask = mask_plot.mask
# # sme.mob = new_mask.__values__

# # Calculate stellar age based on abundances
# x = sme.abund["Y"] / sme.abund["Mg"]
# sx = (sme.fitresults.punc["Y Abund"] / sme.abund["Mg"]) ** 2 + (
#     sme.fitresults.punc["Mg Abund"] * sme.abund["Y"] / sme.abund["Mg"] ** 2
# ) ** 2
# sx = np.sqrt(sx)

# # Values from paper
# a = 0.175
# sa = 0.011
# b = -0.0404
# sb = 0.0019
# age = (x - a) / b
# sigma_age = 1 / b * np.sqrt(sx ** 2 + sa ** 2 + ((x - a) / b) ** 2 * sb ** 2)

# print("Age = %.3f Gyr", age)
