"""
This module contains all Classes needed to load, save and handle SME structures
Notably all SME objects will be Collections, which can accessed both by attribute and by index
"""


import sys
import logging
import os.path
import platform
import inspect
import numpy as np
from scipy.io import readsav

try:
    from .abund import Abund
    from .vald import LineList
except ModuleNotFoundError:
    from abund import Abund
    from vald import LineList


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
            self.__idx__ = np.asarray(index)
            sizes = index[-1]
        # this stores the actual data
        if values is None:
            self.__values__ = np.zeros(np.sum(sizes), dtype=dtype)
        else:
            self.__values__ = np.asarray(values)

    def __len__(self):
        return len(self.__idx__) - 1

    def __getitem__(self, index):
        if not hasattr(index, "__len__"):
            index = (index,)

        if len(index) == 0:
            return self.__values__

        if isinstance(index, str):
            # This happens for example for np.recarrays
            return Iliffe_vector(
                None, index=self.__idx__, values=self.__values__[index]
            )

        if isinstance(index[0], slice):
            start = index[0].start if index[0].start is not None else 0
            stop = index[0].stop if index[0].stop is not None else len(self)
            step = index[0].step if index[0].step is not None else 1

            if stop > len(self):
                stop = len(self)

            idx = self.__idx__
            if step == 1:
                values = self.__values__[idx[start] : idx[stop]]
            else:
                values = []
                for i in range(start, stop, step):
                    values += [self.__values__[idx[i] : idx[i + 1]]]
                values = np.concatenate(values)
            sizes = np.diff(idx)[index[0]]

            return Iliffe_vector(sizes, values=values)

        if index[0] >= 0:
            i0 = self.__idx__[index[0]]
            i1 = self.__idx__[index[0] + 1]
        else:
            i0 = self.__idx__[index[0] - 1]
            i1 = self.__idx__[index[0]]
        if len(index) == 1:
            return self.__values__[i0:i1]
        if len(index) == 2:
            return self.__values__[i0:i1][index[1]]
        raise KeyError("Key must be maximum 2D")

    def __setitem__(self, index, value):
        if not hasattr(index, "__len__"):
            index = (index,)

        if isinstance(index, str):
            self.__values__[index] = value

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

    def max(self):
        return np.max(self.__values__)

    def min(self):
        return np.min(self.__values__)

    @property
    def size(self):
        """ number of elements in vector """
        return self.__idx__[-1]

    @property
    def shape(self):
        """ number of segments, array with size of each segment """
        return len(self.__idx__) - 1, np.diff(self.__idx__)

    @property
    def ndim(self):
        """ its always 2D """
        return 2

    @property
    def dtype(self):
        """ numpy datatype of the values """
        return self.__values__.dtype

    @property
    def flat(self):
        return self.__values__.flat

    def flatten(self):
        """
        Returns a new(!) flattened version of the vector
        Values are identical to __values__ iff the segments don't overlap
        """
        return np.concatenate([self[i] for i in range(len(self))])

    def copy(self):
        """ Create a copy of the current vector """
        idx = np.copy(self.__idx__)
        values = np.copy(self.__values__)
        return Iliffe_vector(None, index=idx, values=values)


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

            self.__temps__ = []

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
        return key.casefold() in dir(self) and self[key] is not None

    @property
    def names(self):
        """ Names of all not None parameters in the Collection """
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
        """ Get a value with name key if it exists and is not None or alt if not """
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
        self._vsini = 0
        self._vmac = 0
        self._vmic = 0
        self._abund = None

        # TODO: in the SME structure, the abundance values are in a different scheme than described in Abund
        # Helium is also fractional (sometimes?)
        if abund is not None and abund[1] > 0:
            abund = np.copy(abund)
            abund[1] = np.log10(abund[1])

        if abund is not None:
            self.set_abund(monh, abund, abund_pattern)
        else:
            self.set_abund(monh, "asplund2009", "str")

        super().__init__(**kwargs)

    def __str__(self):
        text = (
            f"Teff={self.teff} K, logg={self.logg:.3f}, "
            f"[M/H]={self.monh:.3f}, Vmic={self.vmic:.2f}, "
            f"Vmac={self.vmac:.2f}, Vsini={self.vsini:.1f}\n"
        )
        text += str(self._abund)
        return text

    @property
    def monh(self):
        """ Metallicity """
        return self.abund.monh

    @monh.setter
    def monh(self, value):
        self.abund.monh = value

    @property
    def abund(self):
        """ Elemental abundances """
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
        """ Set elemental abundances together with the metallicity """
        self._abund = Abund(monh, abpatt, abtype)

    @property
    def vmac(self):
        """ Macro Turbulence Velocity """
        return self._vmac

    @vmac.setter
    def vmac(self, value):
        self._vmac = abs(value)

    @property
    def vmic(self):
        """ Micro Turbulence Velocity """
        return self._vmic

    @vmic.setter
    def vmic(self, value):
        self._vmic = abs(value)

    @property
    def vsini(self):
        return self._vsini

    @vsini.setter
    def vsini(self, value):
        self._vsini = abs(value)


class NLTE(Collection):
    """ NLTE data """

    def __init__(self, *args, **kwargs):
        if len(args) != 0 and args[0] is not None:
            args = {name.casefold(): args[0][name][0] for name in args[0].dtype.names}
            args.update(kwargs)
            kwargs = args
        self.nlte_pro = kwargs.pop("sme_nlte", None)
        self.nlte_pro = "nlte"
        elements = kwargs.pop("nlte_elem_flags", [])
        elements = [Abund._elem[i] for i, j in enumerate(elements) if j == 1]
        self.elements = elements
        self.subgrid_size = kwargs.pop("nlte_subgrid_size", [2, 2, 2, 2])

        grids = kwargs.pop("nlte_grids", {})

        if isinstance(grids, (list, np.ndarray)):
            grids = {
                Abund._elem[i]: name.decode()
                for i, name in enumerate(grids)
                if name != ""
            }

        self.grids = grids
        super().__init__(**kwargs)

    _default_grids = {
        "Al": "marcs2012_Al2017.grd",
        "Fe": "marcs2012_Fe2016.grd",
        "Li": "marcs2012_Li.grd",
        "Mg": "marcs2012_Mg2016.grd",
        "Na": "marcs2012p_t1.0_Na.grd",
        "O": "marcs2012_O2015.grd",
        "Ba": "marcs2012p_t1.0_Ba.grd",
        "Ca": "marcs2012p_t1.0_Ca.grd",
        "Si": "marcs2012_SI2016.grd",
        "Ti": "marcs2012s_t2.0_Ti.grd",
    }

    def set_nlte(self, element, grid=None):
        """ add an element to the NLTE calculations """
        if element in self.elements:
            return

        if grid is None:
            # Use default grid
            grid = NLTE._default_grids[element]
            logging.info("Using default grid %s for element %s", grid, element)

        self.elements += [element]
        self.grids[element] = grid

    def remove_nlte(self, element):
        """ remove an element from the NLTE calculations """
        if element not in self.elements:
            return

        self.elements.remove(element)
        self.grids.pop(element)


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
    """
    Atmosphere structure
    contains all information to describe the solar atmosphere
    i.e. temperature etc in the different layers
    as well as stellar parameters and abundances
    """

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


class Fitresults(Collection):
    """
    Fitresults collection for all parameters
    that are created by the SME fit
    i.e. parameter uncertainties
    and Goodness of Fit parameters
    """

    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop("maxiter", None)
        self.chirat = kwargs.pop("chirat", None)
        self.chisq = kwargs.pop("chisq", None)
        self.rchisq = kwargs.pop("rchisq", None)
        self.crms = kwargs.pop("crms", None)
        self.lrms = kwargs.pop("lrms", None)
        self.punc = kwargs.pop("punc", None)
        self.psig_l = kwargs.pop("psig_l", None)
        self.psig_r = kwargs.pop("psig_r", None)
        self.covar = kwargs.pop("covar", None)
        super().__init__(**kwargs)

    def clear(self):
        """ reset all values to None """
        self.maxiter = None
        self.chirat = None
        self.chisq = None
        self.rchisq = None
        self.crms = None
        self.lrms = None
        self.punc = None
        self.psig_l = None
        self.psig_r = None
        self.covar = None


class SME_Struct(Param):
    """
    The all important SME structure
    contains all information necessary to create a synthetic spectrum
    and perform a fit to existing data
    """

    def __init__(self, atmo=None, nlte=None, idlver=None, **kwargs):
        # Meta information
        self.version = None
        self.md5 = None
        self.id = None
        # additional parameters
        self.vrad = None
        self.vrad_flag = None
        self.cscale = kwargs.pop("cscale", None)
        if self.cscale is not None:
            self.cscale = np.atleast_2d(self.cscale)
        self.cscale_flag = None
        self.gam6 = None
        self.h2broad = None
        self.accwi = None
        self.accrt = None
        self.clim = None
        self.nmu = None
        self.mu = np.atleast_1d(kwargs.pop("mu"))
        # linelist
        self.linelist = LineList(
            None,
            species=kwargs.pop("species"),
            atomic=kwargs.pop("atomic"),
            lande=kwargs.pop("lande"),
            depth=kwargs.pop("depth"),
            reference=kwargs.pop("lineref"),
            short_line_format=kwargs.pop("short_line_format"),
            line_extra=kwargs.pop("line_extra", None),
            line_lulande=kwargs.pop("line_lulande", None),
            line_term_low=kwargs.pop("line_term_low", None),
            line_term_upp=kwargs.pop("line_term_upp", None),
        )
        # free parameters
        self.pfree = None
        self.pname = None
        self.glob_free = None
        self.ab_free = None
        # wavelength grid
        # Illiffe vector?
        self.nseg = None
        self.wob = kwargs.pop("wave")
        self.wind = kwargs.pop("wind")
        if self.wind is not None:
            self.wind = np.array([0, *(self.wind + 1)])
        # Wavelength range of each section
        self.wran = kwargs.pop("wran", None)
        if self.wran is not None:
            self.wran = np.atleast_2d(self.wran)
        # Observation
        self.sob = None
        self.uob = kwargs.pop("uob")
        self.mob = kwargs.pop("mob")
        if self.mob is not None:
            self.mob = np.require(self.mob, requirements="W")
        self.obs_name = None
        self.obs_type = None
        # Instrument broadening
        self.iptype = None
        self.ipres = None
        self.vmac_pro = None
        self.cintb = None
        self.cintr = None
        # Fit results
        self.fitresults = Fitresults(
            maxiter=kwargs.pop("maxiter", None),
            chirat=kwargs.pop("chirat", None),
            chisq=kwargs.pop("chisq", None),
            rchisq=kwargs.pop("rchisq", None),
            crms=kwargs.pop("crms", None),
            lrms=kwargs.pop("lrms", None),
            punc=kwargs.pop("punc", None),
            psig_l=kwargs.pop("psig_l", None),
            psig_r=kwargs.pop("psig_r", None),
            covar=kwargs.pop("covar", None),
        )
        self.smod_orig = None
        self.cmod_orig = None
        self.smod = None
        self.cmod = None
        self.jint = None
        self.wint = None
        self.sint = None
        # Substructures
        self.idlver = Version(idlver)
        self.atmo = Atmo(atmo)
        self.nlte = NLTE(nlte)
        super().__init__(**kwargs)

        # Apply final conversions from IDL to Python version
        if "wave" in self:
            self.__convert_cscale__()

    @property
    def atomic(self):
        """ Atomic linelist data, usually passed to the C library
        Use sme.linelist instead for other purposes """
        return self.linelist.atomic

    @property
    def species(self):
        """ Names of the species of each spectral line """
        return self.linelist.species

    @property
    def wave(self):
        """ Wavelength """
        if self.wob is None:
            return None
        w = Iliffe_vector(None, index=self.wind, values=self.wob)
        return w

    @wave.setter
    def wave(self, value):
        if isinstance(value, Iliffe_vector):
            value = value.__values__
        self.wob = value

    @property
    def spec(self):
        """ Observed Spectrum """
        if self.sob is None:
            return None
        s = Iliffe_vector(None, index=self.wind, values=self.sob)
        return s

    @spec.setter
    def spec(self, value):
        if isinstance(value, Iliffe_vector):
            value = value.__values__
        self.sob = value

    @property
    def uncs(self):
        """ Uncertainties of the observed spectrum """
        if self.uob is None:
            return None
        u = Iliffe_vector(None, index=self.wind, values=self.uob)
        return u

    @uncs.setter
    def uncs(self, value):
        if isinstance(value, Iliffe_vector):
            value = value.__values__
        self.uob = value

    @property
    def synth(self):
        """ Synthetic Spectrum """
        if self.smod is None:
            return None
        s = Iliffe_vector(None, index=self.wind, values=self.smod)
        return s

    @synth.setter
    def synth(self, value):
        if isinstance(value, Iliffe_vector):
            value = value.__values__
        self.smod = value

    @property
    def mask(self):
        """ Line and Continuum Mask """
        if self.mob is None:
            return None
        if self.mob is not None and self.uob is not None:
            self.mob[self.uob == 0] = 0
        m = Iliffe_vector(None, index=self.wind, values=self.mob)
        return m

    @mask.setter
    def mask(self, value):
        if isinstance(value, Iliffe_vector):
            value = value.__values__
        self.mob = value

    @property
    def line_mask(self):
        """ Line Mask """
        return self.mask == 1

    @property
    def continuum_mask(self):
        """ Continuum Mask """
        return self.mask == 2

    @property
    def good_mask(self):
        """ Good Pixel Mask """
        return self.mask != 0

    @property
    def bad_mask(self):
        """ Bad Pixel Mask """
        return self.mask == 0

    def __getitem__(self, key):
        if key[-5:].casefold() == "abund":
            element = key.split(" ", 1)[0]
            element = element.capitalize()
            return self.abund[element]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key[-5:].casefold() == "abund":
            element = key.split(" ", 1)[0]
            element = element.capitalize()
            self.abund.update_pattern({element: value})
            return
        return super().__setitem__(key, value)

    def __convert_cscale__(self):
        """
        Convert IDL SME continuum scale to regular polynomial coefficients
        Uses Taylor series approximation, as IDL version used the inverse of the continuum
        """
        wave, _ = self.spectrum()
        self.cscale = np.require(self.cscale, requirements="W")
        for i in range(len(self.cscale)):
            c, d = self.cscale[i]
            a, b = max(wave[i]), min(wave[i])
            c0 = (a - b) * (c - d) / (a * c - b * d) ** 2
            c1 = (a - b) / (a * c - b * d)
            self.cscale[i] = [c0, c1]

    @staticmethod
    def load(filename="sme.npy"):
        """ load SME data from disk """
        logging.info("Loading SME file %s", filename)
        _, ext = os.path.splitext(filename)
        if ext == ".npy":
            s = np.load(filename)
            s = np.atleast_1d(s)[0]
        else:
            s = readsav(filename)["sme"]
            s = {name.casefold(): s[name][0] for name in s.dtype.names}
            s = SME_Struct(**s)

        return s

    def save(self, filename="sme.npy", verbose=True):
        """ save SME data to disk """
        if verbose:
            logging.info("Saving SME structure %s", filename)
        np.save(filename, self)

    def spectrum(self, syn=False, return_mask=False, return_uncertainty=False):
        """
        load the wavelength and spectrum, with wavelength sets seperated into seperate arrays

        syn : bool, optional
            wether to load the synthetic spectrum instead (the default is False, which means the observed spectrum is used)

        Returns
        -------
        wave, spec : Iliffe_vector
            As the size of each wavelength set is not equal in general, numpy can't usually create a 2d array from the results
        """

        w = self.wave
        s = self.spec if not syn else self.synth

        args = [w, s]
        if return_mask:
            args += [self.mask]
        if return_uncertainty:
            args += [self.uncs]
        return args


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
