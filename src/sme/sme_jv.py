import sys
import platform
import numpy as np
from scipy.io import readsav

try:
    from .abund import Abund
except ModuleNotFoundError:
    from abund import Abund


class Collection:
    """
    A dictionary that is case insensitive (always lowercase) and
    that can be accessed both by attribute or index (for names that don't start with "_")
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattribute__(self, name):
        return super().__getattribute__(name.casefold())

    def __setattr__(self, name, value):
        return super().__setattr__(name.casefold(), value)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __contains__(self, key):
        return key.casefold() in dir(self)

    @property
    def names(self):
        return dir(self)

    @property
    def dtype(self):
        """ emulate numpt recarray dtype names """
        dummy = lambda: None
        dummy.names = self.names
        return dummy

    def get(self, key, alt=None):
        if key in self:
            return self[key]
        else:
            return alt


class Param(Collection):
    """Handle model parameters for a Spectroscopy Made Easy (SME) job.
    """

    def __init__(self, monh=None, abund=None, abund_pattern="sme", **kwargs):
        if monh is None:
            monh = kwargs.pop("feh", 0)
        if "grav" in kwargs.keys():
            kwargs["logg"] = kwargs["grav"]
            kwargs.pop("grav")

        self.teff = 0
        self.logg = 0
        self.vsini = 0
        self.vmac = 0
        self.vmic = 0
        self.monh = monh
        self.set_abund(monh, abund, abund_pattern)
        super().__init__(**kwargs)

    def __str__(self):
        return self.summary()

    @property
    def abund(self):
        return self._abund

    def set_abund(self, monh, abpatt, abtype):
        if abpatt is None:
            self._abund = None
        else:
            self._abund = Abund(monh, abpatt, abtype)

    def summary(self):
        fmt = "Teff={} K,  logg={:.3f},  [M/H]={:.3f},  Vmic={:.2f},  Vmac={:.2f},  Vsini={:.1f}"
        print(
            fmt.format(
                self.teff, self.logg, self.monh, self.vmic, self.vmac, self.vsini
            )
        )
        self._abund.print()

class NLTE(Collection):
    def __init__(self, *args, **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.nlte_pro = "sme_nlte"
        self.nlte_elem_flags_byte = []
        self.nlte_subgrid_size = []
        self.nlte_grids = []
        super().__init__(**kwargs)

class Version(Collection):
    def __init__(self, *args, **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.arch = platform.machine()
        self.os = sys.platform
        self.os_family = sys.platform
        self.os_name = platform.version()
        self.release = platform.python_version()
        self.build_date = platform.python_build()[1]
        self.memory_bits = int(platform.architecture()[0][:2])
        self.field_offset_bits = int(platform.architecture()[0][:2])
        self.host = platform.node()
        self.info = sys.version
        super().__init__(**kwargs)


class Atmo(Param):
    def __init__(self, *args,  **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.rhox = np.zeros(0)
        self.tau = np.zeros(0)
        self.temp = np.zeros(0)
        self.xna = np.zeros(0)
        self.xne = np.zeros(0)
        self.vturb = 0
        self.lonh = 0
        self.method = ""
        self.source = ""
        self.depth = ""
        self.interp = ""
        self.geom = ""
        super().__init__(**kwargs)


class SME_Struct(Param):
    def __init__(self, atmo=None, nlte=None, idlver=None, **kwargs):
        # Meta information
        self.version = 5.1
        self.md5 = ""
        self.id = "today"
        # additional parameters
        self.vrad = 0
        self.vrad_flag = -3
        self.cscale = [1]
        self.cscale_flag = 0
        self.gam6 = 1
        self.h2broad = 0
        self.accwi = 0
        self.accrt = 0
        self.clim = 0.01
        self.nmu = 7
        self.mu = []
        # linelist
        self.species = []
        self.atomic = []
        self.lande = []
        self.depth = []
        self.lineref = []
        self.short_line_format = 2
        self.line_extra = []
        self.line_lulande = []
        self.line_term_low = []
        self.line_term_upp = []
        self.wran = []
        # free parameters
        self.glob_free = []
        self.ab_free = []
        # wavelength grid
        # Illiffe vector?
        self.nseg = 1
        self.wave = []
        self.wind = []
        # Observation
        self.sob = []
        self.uob = []
        self.mob = []
        self.obs_name = ""
        self.obs_type = 3
        # Instrument broadening
        self.iptype = "gauss"
        self.ipres = 110000
        self.vmac_pro = ""
        self.cintb = []
        self.cintr = []
        # Fit results
        self.maxiter = 100
        self.chirat = 0
        self.smod_orig = []
        self.smod = []
        self.cmod_orig = []
        self.cmod = []
        self.jint = []
        self.wint = []
        self.sint = []
        self.chisq = 0
        self.rchisq = 0
        self.crms = 0
        self.lrms = 0
        self.pfree = []
        self.punc = []
        self.psig_l = []
        self.psig_r = []
        self.pname = []
        self.covar = [[]]
        # Substructures
        self.idlver = Version(idlver)
        self.atmo = Atmo(atmo)
        self.nlte = NLTE(nlte)
        super().__init__(**kwargs)

    @staticmethod
    def load(filename="sme.npy"):
        """ load SME data from disk """
        if filename[-3:] == "npy":
            s = np.load(filename)
            s = np.atleast_1d(s)[0]
        else:
            s = readsav(filename)["sme"]
            s = {name.casefold(): s[name][0] for name in s.dtype.names}
            s = SME_Struct(**s)

        return s

    def save(self, filename="sme.npy"):
        """ save SME data to disk """
        np.save(filename, self)

    def spectrum(self, syn=False):
        """
        load the wavelength and spectrum, with wavelength sets seperated into seperate arrays

        syn : bool, optional
            wether to load the synthetic spectrum instead (the default is False, which means the observed spectrum is used)

        Returns
        -------
        wave, spec : 1d-array(1d-array), or 2d if all segements have the same size
            As the size of each wavelength set is not equal in general, numpy can't usually create a 2d array from the results
        """

        # wavelength grid
        wave = self.wave
        # wavelength indices of the various sections
        # +1 because of different indexing between idl and python
        section_index = self.wind + 1

        if syn:
            # synthetic spectrum
            obs_flux = self.smod
        else:
            # observed spectrum
            obs_flux = self.sob

        w, s = [], []
        for i, j in zip(section_index[:-1], section_index[1:]):
            w += [wave[i:j]]
            s += [obs_flux[i:j]]

        return np.array(w), np.array(s)


if __name__ == "__main__":
    filename = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
    test = SME_Struct.load(filename)
    temp = SME_Struct()
    print("Teff", test.teff)
    print("Species", test.species)
