import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
from scipy.optimize import OptimizeWarning

from src.gui import plotting
from src.sme import sme as SME
from src.sme.solve import sme_func, solve
from src.sme.vald import ValdFile


in_file = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
vald_file = "/home/ansgar/Documents/IDL/SME/harps_red.lin"
vald = ValdFile(vald_file)
sme = SME.SME_Struct.load(in_file)
orig = readsav(in_file)["sme"]

# make linelist errors
rel_error = vald.linelist.error
wlcent = vald.linelist.wlcent
width = 1  # TODO
sig_syst = np.zeros(len(sme.uob), dtype=float)
wave = sme.wave

for i, line in enumerate(vald.linelist):
    # find closest wavelength region
    w = (wave >= wlcent[i] - width) & (wave <= wlcent[i] + width)
    sig_syst[w] += rel_error[i]

sig_syst *= np.clip(1 - sme.sob, 0, 1)
sme.uob += sig_syst

# Choose free parameters, i.e. sme.pname
parameter_names = ["teff", "logg", "feh", "Mn Abund", "Y Abund"]
sme = solve(sme, parameter_names)

# sme = SME.SME_Struct.load("sme.npy")
mask_plot = plotting.MaskPlot(sme)
# Update mask
# new_mask = mask_plot.mask
# sme.mob = new_mask.__values__
