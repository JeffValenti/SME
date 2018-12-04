import os

import numpy as np
from scipy.io import readsav

from .sme import Atmo


class sav_file(np.recarray):
    """ IDL savefile atmosphere grid """

    def __new__(cls, filename):
        prefix = os.path.dirname(__file__)
        path = os.path.join(prefix, "atmospheres", filename)
        krz2 = readsav(path)
        atmo_grid = krz2["atmo_grid"]
        atmo_grid_maxdep = krz2["atmo_grid_maxdep"]
        atmo_grid_natmo = krz2["atmo_grid_natmo"]
        atmo_grid_vers = krz2["atmo_grid_vers"]
        atmo_grid_file = filename

        # Keep values around for next run
        data = atmo_grid.view(cls)
        data.maxdep = atmo_grid_maxdep
        data.natmo = atmo_grid_natmo
        data.vers = atmo_grid_vers
        data.file = atmo_grid_file

        return data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.maxdep = getattr(self, "maxdep", None)
        self.natmo = getattr(self, "natmo", None)
        self.vers = getattr(self, "vers", None)
        self.file = getattr(self, "file", None)

    def load(self, filename, reload=False):
        changed = filename != self.file
        if reload or changed:
            new = sav_file(filename)
        else:
            new = self
        return new


class krz_file(Atmo):
    def __init__(self, filename):
        self.source = filename
        self.method = "embedded"
        self.load(filename)

    def load(self, filename):
        # TODO: this only works for some krz files
        # 1..2 lines header
        # 3 line opacity
        # 4..13 elemntal abundances
        # 14.. Table data for each layer
        #    Rhox Temp XNE XNA RHO

        with open(filename, "r") as file:
            header1 = file.readline()
            header2 = file.readline()
            opacity = file.readline()
            abund = [file.readline() for _ in range(10)]
            table = file.readlines()

            # Parse header
            # vturb
        i = header1.find("VTURB")
        self.vturb = float(header1[i + 5 : i + 9])
        # L/H, metalicity
        i = header1.find("L/H")
        self.lonh = float(header1[i + 3 :])

        k = len("T EFF=")
        i = header2.find("T EFF=")
        j = header2.find("GRAV=", i + k)
        self.teff = float(header2[i + k : j])

        i = j
        k = len("GRAV=")
        j = header2.find("MODEL TYPE=", i + k)
        self.logg = float(header2[i + k : j])

        i, k = j, len("MODEL TYPE=")
        j = header2.find("WLSTD=", i + k)
        model_type_key = {0: "rhox", 1: "tau", 3: "sph"}
        self.model_type = int(header2[i + k : j])
        self.depth = model_type_key[self.model_type]
        self.geom = "pp"

        i = j
        k = len("WLSTD=")
        self.wlstd = float(header2[i + k :])

        # parse opacity
        i = opacity.find("-")
        opacity = opacity[:i].split()
        self.opflag = np.array([int(k) for k in opacity])

        # parse abundance
        pattern = np.genfromtxt(abund).flatten()[:-1]
        pattern[1] = 10 ** pattern[1]
        self.set_abund(0, pattern, "sme")

        # parse table
        self.table = np.genfromtxt(table, delimiter=",", usecols=(0, 1, 2, 3, 4))
        self.rhox = self.table[:, 0]
        self.temp = self.table[:, 1]
        self.xne = self.table[:, 2]
        self.xna = self.table[:, 3]
        self.rho = self.table[:, 4]


if __name__ == "__main__":
    filename = "/home/ansgar/Documents/Python/SME/src/sme/atmospheres/sun.krz"
    krz_file(filename)
