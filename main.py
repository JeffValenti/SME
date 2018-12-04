""" Main entry point for an SME script """
import sys

import numpy as np
import matplotlib.pyplot as plt

import util
from src.gui import plot_jupyter, plot_pyplot
from src.sme import sme as SME
from src.sme.vald import ValdFile
from src.sme.solve import solve

util.start_logging()

# Get input files
if len(sys.argv) > 1:
    in_file, vald_file, fitparameters = util.parse_args()
else:
    # in_file = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
    in_file = "./sun_6440_grid.out"
    # in_file = "./wasp117_short.inp"
    # in_file = "./wasp117.npy"
    # vald_file = "sun.lin"
    vald_file = None
    fitparameters = []

# Load files
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

resid = (sme.smod - sme.sob) / (sme.uob / sme.sob)
z = resid ** 2
z = 2 * ((1 + z) ** 0.5 - 1)
initial_cost = 0.5 * np.sum(z)
print(f"Initial Cost: {initial_cost:1.4e}")

fitparameters = ["teff", "logg", "monh"]  # , "Y Abund", "Mg Abund"]
target = "sun"
# sme.nlte.set_nlte("Ca")

# Start SME solver
sme = solve(sme, fitparameters, filename=f"{target}.npy")


# # Calculate stellar age based on abundances
# solar = Abund.solar()
# y, mg = sme.abund["Y"], sme.abund["Mg"]
# sy, smg = sme.fitresults.punc["Y abund"], sme.fitresults.punc["Mg abund"]
# x = y - mg - (solar["Y"] - solar["Mg"])
# sx = np.sqrt(sy ** 2 + smg ** 2)

# # Values from paper
# a = 0.175
# sa = 0.011
# b = -0.0404
# sb = 0.0019
# age = (x - a) / b
# sigma_age = 1 / b * np.sqrt(sx ** 2 + sa ** 2 + ((x - a) / b) ** 2 * sb ** 2)
# sigma_age = abs(sigma_age)
# print(f"Age {age:.3f} +- {sigma_age:.3f} Gyr")

# p = np.linspace(0, 10, 1000)
# g = norm.pdf(p, loc=age, scale=sigma_age)
# # Rescale to area = 1
# area = np.sum(g * np.gradient(p))  # Cheap integral
# g *= 1 / area
# plt.plot(p, g)
# plt.xlabel("Age [Gyr]")
# plt.ylabel("Probability")
# plt.show()

# # Plot results
fig = plot_jupyter.FinalPlot(sme)
fig.save(filename=f"{target}.html")

plt.plot(sme.wave, sme.sob - sme.smod)
plt.show()

mask_plot = plot_pyplot.MaskPlot(sme)
input("Wait a second...")
