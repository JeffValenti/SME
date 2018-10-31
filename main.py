import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
from scipy.optimize import OptimizeWarning

from src.gui import plotting
from src.sme import sme as SME
from src.sme.solve import sme_func, solve
from src.sme.vald import ValdFile


in_file = "/home/ansgar/wasp39_11.out"
vald_file = "/home/ansgar/linelist.lin"
vald = ValdFile(vald_file)
sme = SME.SME_Struct.load(in_file)
orig = readsav(in_file)["sme"]

sme.linelist = vald.linelist

# Choose free parameters, i.e. sme.pname
parameter_names = ["teff", "logg", "feh", "Mg Abund", "Y Abund"]
sme = solve(sme, parameter_names)

sme.linelist = vald.linelist

# sme = SME.SME_Struct.load("sme.npy")
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
