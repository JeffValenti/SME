"""
This module contains all Classes needed to load, save and handle SME structures
Notably all SME objects will be Collections, which can accessed both by attribute and by index
"""

import sys
import platform
import inspect
import numpy as np
from scipy.io import readsav

try:
    from .abund import Abund
except ModuleNotFoundError:
    from abund import Abund


class Iliffe_vector:
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, sizes, index=None, values=None, dtype=float):
        # sizes = size of the individual parts
        # the indices are then [0, s1, s1+s2, s1+s2+s3, ...]
        if index is None:
            self.__idx__ = np.concatenate([[0], np.cumsum(sizes, dtype=int)])
        else:
            if index[0] != 0:
                index = [0, *index]
            self.__idx__ = np.asarray(index).ravel()
            sizes = index[-1]
        # this stores the actual data
        if values is None:
            self.__values__ = np.zeros(np.sum(sizes), dtype=dtype)
        else:
            if np.size(values) != np.sum(sizes):
                raise ValueError("Size of values and sizes do not fit")
            self.__values__ = np.asarray(values).ravel()

    def __len__(self):
        return len(self.__idx__) - 1

    def __getitem__(self, index):
        if not hasattr(index, "__len__"):
            index = (index,)
        if len(index) == 0:
            return self.__values__
        i0 = self.__idx__[index[0]]
        i1 = self.__idx__[index[0] + 1]
        if len(index) == 1:
            return self.__values__[i0:i1]
        if len(index) == 2:
            return self.__values__[i0:i1][index[1]]
        raise KeyError("Key must be maximum 2D")

    def __setitem__(self, index, value):
        if not hasattr(index, "__len__"):
            index = (index,)

        if len(index) == 0:
            self.__values__ = value
        elif len(index) in [1, 2]:
            i0 = self.__idx__[index[0]]
            i1 = self.__idx__[index[0] + 1]
            if len(index) == 1:
                self.__values__[i0:i1] = value
            elif len(index) == 2:
                self.__values__[i0:i1][index[1]] = value
        else:
            raise KeyError("Key must be maximum 2D")

    @property
    def size(self):
        return self.__idx__[-1]

    @property
    def shape(self):
        return len(self.__idx__) - 1, np.diff(self.__idx__)

    @property
    def dtype(self):
        return self.__values__.dtype


class Collection(object):
    """
    A dictionary that is case insensitive (always lowercase) and
    that can be accessed both by attribute or index (for names that don't start with "_")
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, bytes):
                value = value.decode()
            if isinstance(value, np.ndarray) and value.dtype == np.dtype("O"):
                value = value.astype(str)
            if isinstance(value, np.ndarray):
                value = np.require(value, requirements="WO")
            setattr(self, key, value)

    def __getattribute__(self, name):
        return object.__getattribute__(self, name.casefold())

    def __setattr__(self, name, value):
        return object.__setattr__(self, name.casefold(), value)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __contains__(self, key):
        return key.casefold() in dir(self)

    @property
    def names(self):
        exclude = ["names", "dtype"]
        exclude += [
            s[0]
            for s in inspect.getmembers(self.__class__, predicate=inspect.isroutine)
        ]
        return [
            s
            for s in dir(self)
            if s[0] != "_" and s not in exclude and getattr(self, s) is not None
        ]

    @property
    def dtype(self):
        """ emulate numpt recarray dtype names """
        dummy = lambda: None
        dummy.names = [s.upper() for s in self.names]
        return dummy

    def get(self, key, alt=None):
        if key in self:
            return self[key]
        else:
            return alt


class Param(Collection):
    """ Handle model parameters for a Spectroscopy Made Easy (SME) job. """

    def __init__(self, monh=None, abund=None, abund_pattern="sme", **kwargs):
        if monh is None:
            monh = kwargs.pop("feh", None)
        if "grav" in kwargs.keys():
            kwargs["logg"] = kwargs["grav"]
            kwargs.pop("grav")

        self.teff = None
        self.logg = None
        self.vsini = None
        self.vmac = None
        self.vmic = None
        self.monh = monh

        # TODO: in the SME structure, the abundance values are in a different scheme than described in Abund
        if abund is not None and abund[1] > 0:
            abund = np.copy(abund)
            abund[1] = np.log10(abund[1])

        self.set_abund(monh, abund, abund_pattern)

        # asp = Abund(0, "asplund2009").get_pattern("sme", raw=True)
        # temp = self.abund.get_pattern(abund_pattern, raw=True)

        # if not np.allclose(abund, temp, 0.001):
        #     raise ValueError("Abund messed up")

        super().__init__(**kwargs)

    def __str__(self):
        return self.summary()

    @property
    def abund(self):
        return self._abund

    @abund.setter
    def abund(self, value):
        # assume that its sme type
        if isinstance(value, Abund):
            self._abund = value
        elif isinstance(value, np.ndarray) and value.size == 99:
            self.set_abund(self.monh, value, "sme")
        else:
            raise TypeError(
                "Abundance can only be set by Abund object, use set_abund otherwise"
            )

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
    """ NLTE data """

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
    """ Describes the Python version and information about the computer host """

    def __init__(self, *args, **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.arch = None
        self.os = None
        self.os_family = None
        self.os_name = None
        self.release = None
        self.build_date = None
        self.memory_bits = None
        self.field_offset_bits = None
        self.host = None
        # self.info = sys.version
        if len(kwargs) == 0:
            self.update()
        super().__init__(**kwargs)

    def update(self):
        """ update version info with current machine data """
        self.arch = platform.machine()
        self.os = sys.platform
        self.os_family = sys.platform
        self.os_name = platform.version()
        self.release = platform.python_version()
        self.build_date = platform.python_build()[1]
        self.memory_bits = int(platform.architecture()[0][:2])
        self.field_offset_bits = int(platform.architecture()[0][:2])
        self.host = platform.node()
        # self.info = sys.version

    def __str__(self):
        return "%s %s" % (self.os_name, self.release)


class Atmo(Param):
    def __init__(self, *args, **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.rhox = None
        self.tau = None
        self.temp = None
        self.xna = None
        self.xne = None
        self.vturb = None
        self.lonh = None
        self.method = None
        self.source = None
        self.depth = None
        self.interp = None
        self.geom = None
        super().__init__(**kwargs)


class SME_Struct(Param):
    def __init__(self, atmo=None, nlte=None, idlver=None, **kwargs):
        # Meta information
        self.version = None
        self.md5 = None
        self.id = None
        # additional parameters
        self.vrad = None
        self.vrad_flag = None
        self.cscale = None
        self.cscale_flag = None
        self.gam6 = None
        self.h2broad = None
        self.accwi = None
        self.accrt = None
        self.clim = None
        self.nmu = None
        self.mu = None
        # linelist
        self.species = None
        self.atomic = None
        self.lande = None
        self.depth = None
        self.lineref = None
        self.short_line_format = None
        self.line_extra = None
        self.line_lulande = None
        self.line_term_low = None
        self.line_term_upp = None
        self.wran = None
        # free parameters
        self.glob_free = None
        self.ab_free = None
        # wavelength grid
        # Illiffe vector?
        self.nseg = None
        self.wave = None
        self.wind = None
        # Observation
        self.sob = None
        self.uob = None
        self.mob = None
        self.obs_name = None
        self.obs_type = None
        # Instrument broadening
        self.iptype = None
        self.ipres = None
        self.vmac_pro = None
        self.cintb = None
        self.cintr = None
        # Fit results
        self.maxiter = None
        self.chirat = None
        self.smod_orig = None
        self.smod = None
        self.cmod_orig = None
        self.cmod = None
        self.jint = None
        self.wint = None
        self.sint = None
        self.chisq = None
        self.rchisq = None
        self.crms = None
        self.lrms = None
        self.pfree = None
        self.punc = None
        self.psig_l = None
        self.psig_r = None
        self.pname = None
        self.covar = None
        # Substructures
        self.idlver = Version(idlver)
        self.atmo = Atmo(atmo)
        self.nlte = NLTE(nlte)
        super().__init__(**kwargs)

    @property
    def feh(self):
        return self.monh

    @feh.setter
    def feh(self, value):
        self.monh = value

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

    def spectrum(self, syn=False, cont=False):
        """
        load the wavelength and spectrum, with wavelength sets seperated into seperate arrays

        syn : bool, optional
            wether to load the synthetic spectrum instead (the default is False, which means the observed spectrum is used)

        Returns
        -------
        wave, spec : Iliffe_vector
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

        # return as Iliffe vectors, where the arrays can have different sizes
        # this is pretty much the same as it was before, just with fancier indexing
        w = Iliffe_vector(None, index=section_index, values=wave)
        s = Iliffe_vector(None, index=section_index, values=obs_flux)

        # Apply continuum normalization
        # but only to the real observations
        if cont and not syn:
            if self.cscale.ndim == 1:
                x = np.arange(len(wave))
                s /= np.polyval(self.cscale[::-1], x)
            elif self.cscale.ndim == 2:
                for i in range(len(s)):
                    x = np.arange(len(s[i]))
                    s[i] /= np.polyval(self.cscale[i][::-1], x)

        return w, s


if __name__ == "__main__":
    filename = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
    test = SME_Struct.load(filename)
    w, s = test.spectrum()
    test.teff = 2
    test["logg"] = 3
    test.idlver.update()

    test.save("test.npy")
    test = SME_Struct.load("test.npy")

    print("Teff", test.teff)
    print("logg", test["LOGG"])
    print("Py_version", test.system)
