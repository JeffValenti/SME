"""
This module contains all Classes needed to load, save and handle SME structures
Notably all SME objects will be Collections, which can accessed both by attribute and by index
"""


import inspect
import logging
import os.path
import platform
import sys
from datetime import datetime as dt
import hashlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

from . import echelle
from .abund import Abund
from .vald import LineList


class Iliffe_vector:
    """
    Illiffe vectors are multidimensional (here 2D) but not necessarily rectangular
    Instead the index is a pointer to segments of a 1D array with varying sizes
    """

    def __init__(self, sizes, values=None, index=None, dtype=float):
        # sizes = size of the individual parts
        # the indices are then [0, s1, s1+s2, s1+s2+s3, ...]
        if index is None:
            self.__idx__ = np.concatenate([[0], np.cumsum(sizes, dtype=int)])
        else:
            if index[0] != 0:
                index = [0, *index]
            self.__idx__ = np.asarray(index, dtype=int)
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


        if isinstance(index, range):
            index = list(index)

        if isinstance(index, (list, np.ndarray)):
            values = [self[i] for i in index]
            sizes = [len(v) for v in values]
            return Iliffe_vector(sizes, values=values)

        if isinstance(index, str):
            # This happens for example for np.recarrays
            return Iliffe_vector(
                None, index=self.__idx__, values=self.__values__[index]
            )


        if isinstance(index, Iliffe_vector):
            if not self.__equal_size__(index):
                raise ValueError("Index vector has a different shape")
            return self.__values__[index.__values__]

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

    # Math operators
    # If both are Iliffe vectors of the same size, use element wise operations
    # Otherwise apply the operator to __values__
    def __equal_size__(self, other):
        if not isinstance(other, Iliffe_vector):
            return NotImplemented

        if self.shape[0] != other.shape[0]:
            return False

        return np.array_equal(self.__idx__, other.__idx__)

    def __operator__(self, other, operator):
        if isinstance(other, Iliffe_vector):
            if not self.__equal_size__(other):
                return NotImplemented
            other = other.__values__

        operator = getattr(self.__values__, operator)
        values = operator(other)
        if values is NotImplemented:
            return NotImplemented
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __eq__(self, other):
        return self.__operator__(other, "__eq__")

    def __ne__(self, other):
        return self.__operator__(other, "__ne__")

    def __lt__(self, other):
        return self.__operator__(other, "__lt__")

    def __gt__(self, other):
        return self.__operator__(other, "__gt__")

    def __le__(self, other):
        return self.__operator__(other, "__le__")

    def __ge__(self, other):
        return self.__operator__(other, "__ge__")

    def __add__(self, other):
        return self.__operator__(other, "__add__")

    def __sub__(self, other):
        return self.__operator__(other, "__sub__")

    def __mul__(self, other):
        return self.__operator__(other, "__mul__")

    def __truediv__(self, other):
        return self.__operator__(other, "__truediv__")

    def __floordiv__(self, other):
        return self.__operator__(other, "__floordiv__")

    def __mod__(self, other):
        return self.__operator__(other, "__mod__")

    def __divmod__(self, other):
        return self.__operator__(other, "__divmod__")

    def __pow__(self, other):
        return self.__operator__(other, "__pow__")

    def __lshift__(self, other):
        return self.__operator__(other, "__lshift__")

    def __rshift__(self, other):
        return self.__operator__(other, "__rshift__")

    def __and__(self, other):
        return self.__operator__(other, "__and__")

    def __or__(self, other):
        return self.__operator__(other, "__or__")

    def __xor__(self, other):
        return self.__operator__(other, "__xor__")

    def __radd__(self, other):
        return self.__operator__(other, "__radd__")

    def __rsub__(self, other):
        return self.__operator__(other, "__rsub__")

    def __rmul__(self, other):
        return self.__operator__(other, "__rmul__")

    def __rtruediv__(self, other):
        return self.__operator__(other, "__rtruediv__")

    def __rfloordiv__(self, other):
        return self.__operator__(other, "__rfloordiv__")

    def __rmod__(self, other):
        return self.__operator__(other, "__rmod__")

    def __rdivmod__(self, other):
        return self.__operator__(other, "__rdivmod__")

    def __rpow__(self, other):
        return self.__operator__(other, "__rpow__")

    def __rlshift__(self, other):
        return self.__operator__(other, "__rlshift__")

    def __rrshift__(self, other):
        return self.__operator__(other, "__rrshift__")

    def __rand__(self, other):
        return self.__operator__(other, "__rand__")

    def __ror__(self, other):
        return self.__operator__(other, "__ror__")

    def __rxor__(self, other):
        return self.__operator__(other, "__rxor__")

    def __iadd__(self, other):
        return self.__operator__(other, "__iadd__")

    def __isub__(self, other):
        return self.__operator__(other, "__isub__")

    def __imul__(self, other):
        return self.__operator__(other, "__imul__")

    def __itruediv__(self, other):
        return self.__operator__(other, "__itruediv__")

    def __ifloordiv__(self, other):
        return self.__operator__(other, "__ifloordiv__")

    def __imod__(self, other):
        return self.__operator__(other, "__imod__")

    def __ipow__(self, other):
        return self.__operator__(other, "__ipow__")

    def __iand__(self, other):
        return self.__operator__(other, "__iand__")

    def __ior__(self, other):
        return self.__operator__(other, "__ior__")

    def __ixor__(self, other):
        return self.__operator__(other, "__ixor__")

    def __neg__(self):
        values = -self.__values__
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __pos__(self):
        return self

    def __abs__(self):
        values = abs(self.__values__)
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __invert__(self):
        values = ~self.__values__
        iv = Iliffe_vector(None, index=self.__idx__, values=values)
        return iv

    def __str__(self):
        s = [str(i) for i in self]
        s = str(s).replace("'", "")
        return s

    def __repr__(self):
        return f"Iliffe_vector({self.sizes}, {self.__values__})"

    def max(self):
        """ Maximum value in all segments """
        return np.max(self.__values__)

    def min(self):
        """ Minimum value in all segments """
        return np.min(self.__values__)

    @property
    def size(self):
        """int: number of elements in vector """
        return self.__idx__[-1]

    @property
    def shape(self):
        """tuple(int, list(int)): number of segments, array with size of each segment """
        return len(self), self.sizes

    @property
    def sizes(self):
        """list(int): Sizes of the different segments """
        return list(np.diff(self.__idx__))

    @property
    def ndim(self):
        """int: its always 2D """
        return 2

    @property
    def dtype(self):
        """dtype: numpy datatype of the values """
        return self.__values__.dtype

    @property
    def flat(self):
        """iter: Flat iterator through the values """
        return self.__values__.flat

    def flatten(self):
        """
        Returns a new(!) flattened version of the vector
        Values are identical to __values__ if the size
        of all segements equals the size of __values__

        Returns
        -------
        flatten: array
            new flat (1d) array of the values within this Iliffe vector
        """
        return np.concatenate([self[i] for i in range(len(self))])

    def ravel(self):
        """
        View of the contained values as a 1D array.
        Not a copy

        Returns
        -------
        raveled: array
            1d array of the contained values
        """

        return self.__values__

    def copy(self):
        """
        Create a copy of the current vector

        Returns
        -------
        copy : Iliffe_vector
            A copy of this vector
        """
        idx = np.copy(self.__idx__)
        values = np.copy(self.__values__)
        return Iliffe_vector(None, index=idx, values=values)

    def append(self, other):
        """
        Append a new segment to the end of the vector
        This creates new memory arrays for the values and the index
        """
        self.__values__ = np.concatenate((self.__values__, other))
        self.__idx__ = np.concatenate((self.__idx__, len(other)))


class Collection:
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
        """list(str): Names of all not None parameters in the Collection """
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
        """:obj: emulate numpt recarray dtype names """
        dummy = lambda: None
        dummy.names = [s.upper() for s in self.names]
        return dummy

    def get(self, key, alt=None):
        """
        Get a value with name key if it exists and is not None or alt if not

        Parameters
        ----------
        key: str
            Name of the value to get
        alt: obj, optional
            alternative value to get if key does not exist (default: None)

        Returns
        -------
        obj
        """
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

        #:float: effective Temperature in K
        self.teff = None
        #:float: surface gravity in log10(cgs)
        self.logg = None

        self._vsini = None
        self._vmac = None
        self._vmic = None
        self._abund = None

        # Helium is also fractional (sometimes?)
        if abund is not None and abund[1] > 0:
            abund = np.copy(abund)
            abund[1] = np.log10(abund[1])

        if abund is not None:
            self.set_abund(monh, abund, abund_pattern)
        else:
            self.set_abund(monh, "empty", "")

        super().__init__(**kwargs)

    def __str__(self):
        text = (
            f"Teff={self.teff} K, logg={self.logg}, "
            f"[M/H]={self.monh}, Vmic={self.vmic}, "
            f"Vmac={self.vmac}, Vsini={self.vsini}\n"
        )
        text += str(self._abund)
        return text

    @property
    def monh(self):
        """float: Metallicity """
        return self.abund.monh

    @monh.setter
    def monh(self, value):
        self.abund.monh = value

    @property
    def abund(self):
        """Abund: Elemental abundances """
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
        """
        Set elemental abundances together with the metallicity

        Parameters
        ----------
        monh : float
            Metallicity
        abpatt : str or array of size (99,)
            Abundance pattern. If string one of the valid presets, otherwise a list of the individual abundances
        abtype : str
            Abundance description format of the input. One of the valid values as described in Abund
        """
        self._abund = Abund(monh, abpatt, abtype)

    @property
    def vmac(self):
        """float: Macro Turbulence Velocity in km/s """
        return self._vmac

    @vmac.setter
    def vmac(self, value):
        self._vmac = abs(value)

    @property
    def vmic(self):
        """float: Micro Turbulence Velocity in km/s """
        return self._vmic

    @vmic.setter
    def vmic(self, value):
        self._vmic = abs(value)

    @property
    def vsini(self):
        """float: Rotational Velocity in km/s, times sine of the inclination """
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
        #:str: OBSOLETE name of the nlte function to use
        self.nlte_pro = kwargs.pop("sme_nlte", None)
        self.nlte_pro = "nlte"
        elements = kwargs.pop("nlte_elem_flags", [])
        elements = [Abund._elem[i] for i, j in enumerate(elements) if j == 1]
        #:list(str): NLTE elements in use
        self.elements = elements
        #:list of size (4,): OBSOLETE defines subgrid size that is kept in memory
        self.subgrid_size = kwargs.pop("nlte_subgrid_size", [2, 2, 2, 2])

        grids = kwargs.pop("nlte_grids", {})
        if isinstance(grids, (list, np.ndarray)):
            grids = {
                Abund._elem[i]: name.decode()
                for i, name in enumerate(grids)
                if name != ""
            }
        #:dict(str, str): NLTE grids to use for any given element
        self.grids = grids
        self.flags = None
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
        """
        Add an element to the NLTE calculations

        Parameters
        ----------
        element : str
            The abbreviation of the element to add to the NLTE calculations
        grid : str, optional
            Filename of the NLTE data grid to use for this element
            the file must be in nlte_grids directory
            Defaults to a set of "known" files for some elements
        """
        if element in self.elements:
            # Element already in NLTE
            # Change grid if given
            if grid is not None:
                self.grids[element] = grid
            return

        if grid is None:
            # Use default grid
            if element not in NLTE._default_grids.keys():
                raise ValueError(f"No default grid known for element {element}")
            grid = NLTE._default_grids[element]
            logging.info("Using default grid %s for element %s", grid, element)

        self.elements += [element]
        self.grids[element] = grid

    def remove_nlte(self, element):
        """
        Remove an element from the NLTE calculations

        Parameters
        ----------
        element : str
            Abbreviation of the element to remove from NLTE
        """
        if element not in self.elements:
            # Element not included in NLTE anyways
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
        #:str: System architecture
        self.arch = None
        #:str: Operating System
        self.os = None
        #:str: Operating System Family
        self.os_family = None
        #:str: OS Name
        self.os_name = None
        #:str: Python Version
        self.release = None
        #:str: Build date of the Python version used
        self.build_date = None
        #:int: Platform architecture bit size (usually 32 or 64)
        self.memory_bits = None
        #:int: OBSOLETE File offset bits (same as memory bits) ???
        self.file_offset_bits = None
        #:str: Name of the machine that created the SME Structure
        self.host = None
        # if len(kwargs) == 0:
        #     self.update()
        super().__init__(**kwargs)

    def update(self):
        """ Update version info with current machine data """
        self.arch = platform.machine()
        self.os = sys.platform
        self.os_family = platform.system()
        self.os_name = platform.version()
        self.release = platform.python_version()
        self.build_date = platform.python_build()[1]
        self.memory_bits = int(platform.architecture()[0][:2])
        self.file_offset_bits = int(platform.architecture()[0][:2])
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
        #:array of size (ndepth,): Mass column density, only rhox or tau needs to be specified
        self.rhox = None
        #:array of size (ndepth,): Continuum optical depth, only rhox or tau needs to be specified
        self.tau = None
        #:array of size (ndepth,): Temperatures in K of each layer
        self.temp = None
        #:array of size (ndepth,): Number density of atoms in 1/cm**3
        self.xna = None
        #:array of size (ndepth,): Number density of electrons in 1/cm**3
        self.xne = None
        #:float: Turbulence velocity in km/s
        self.vturb = None
        #:float: ???
        self.lonh = None
        #:str: Method to use for interpolating atmospheres. Valid values "grid", "embedded"
        self.method = None
        #:str: filename of the atmosphere grid
        self.source = None
        #:str: Flag that determines wether to use RHOX or TAU for calculations. Values are "RHOX", "TAU"
        self.depth = None
        #:str: Flag that determines wether RHOX or TAU are used for interpolation. Values are "RHOX", "TAU"
        self.interp = None
        #:str: Flag that describes the geometry of the atmosphere model. Values are "PP" Plane Parallel, "SPH" Spherical
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
        #:int: Maximum number of iterations in the solver
        self.maxiter = kwargs.pop("maxiter", None)
        #:float: Reduced Chi square of the solution
        self.chisq = kwargs.pop("chisq", None)
        #:array of size (nfree,): Uncertainties of the free parameters
        self.punc = kwargs.pop("punc", None)
        #:array of size (nfree, nfree): Covariance matrix
        self.covar = kwargs.pop("covar", None)
        super().__init__(**kwargs)

    def clear(self):
        """ Reset all values to None """
        self.maxiter = None
        self.chisq = None
        self.punc = None
        self.covar = None


class SME_Struct(Param):
    """
    The all important SME structure
    contains all information necessary to create a synthetic spectrum
    and perform a fit to existing data
    """

    #:dict(str, int): Mask value specifier used in mob
    mask_values = {"bad": 0, "line": 1, "continuum": 2}

    def __init__(self, atmo=None, nlte=None, idlver=None, **kwargs):
        """
        Create a new SME Structure

        Some properties have default values but most will be empty (i.e. None)
        if not set specifically

        When possible will convert values from IDL description to Python equivalent

        Parameters
        ----------
        atmo : Atmo, optional
            Atmopshere structure
        nlte : NLTE, optional
            NLTE structure
        idlver : Version, optional
            system information structure
        **kwargs
            additional values to set
        """
        # Meta information
        #:str: Name of the observation target
        self.object = kwargs.pop("obs_name", None)
        #:str: Version of SME used to create this structure
        self.version = "5.1"
        #:str: DateTime when this structure was created
        self.id = str(dt.now())

        # additional parameters
        self.vrad = 0
        self.vrad_flag = "none"
        self.cscale = 1
        self.cscale_flag = "none"

        #:float: van der Waals scaling factor
        self.gam6 = 1
        #:bool: flag determing wether to use H2 broadening or not
        self.h2broad = False
        #:float: Minimum accuracy for linear spectrum interpolation vs. wavelength. Values below 1e-4 are not meaningful.
        self.accwi = 0.003
        #:float: Minimum accuracy for synthethized spectrum at wavelength grid points in sme.wint. Values below 1e-4 are not meaningful.
        self.accrt = 0.001
        self.mu = 1
        # linelist
        try:
            #:LineList: spectral line information
            self.linelist = LineList(
                species=kwargs.pop("species"),
                atomic=kwargs.pop("atomic"),
                lande=kwargs.pop("lande"),
                depth=kwargs.pop("depth"),
                reference=kwargs.pop("lineref"),
                short_line_format=kwargs.pop("short_line_format", None),
                line_extra=kwargs.pop("line_extra", None),
                line_lulande=kwargs.pop("line_lulande", None),
                line_term_low=kwargs.pop("line_term_low", None),
                line_term_upp=kwargs.pop("line_term_upp", None),
            )
        except KeyError:
            # some data is unavailable
            logging.warning("No or incomplete linelist data present")
            self.linelist = None
        # free parameters
        #:list of float: values of free parameters
        self.pfree = []
        pname = kwargs.pop("pname", [])
        glob_free = kwargs.pop("glob_free", [])
        ab_free = kwargs.pop("ab_free", [])
        if len(ab_free) != 0:
            ab_free = [f"{el} ABUND" for i, el in zip(ab_free, Abund._elem) if i == 1]
        fitparameters = np.concatenate((pname, glob_free, ab_free)).astype("U")
        #:array of size (nfree): Names of the free parameters
        self.fitparameters = np.unique(fitparameters)

        # wavelength grid
        self.wob = None
        #:array of size(nseg+1,): indices of the wavelength segments within wob
        self.wind = kwargs.pop("wind", None)
        if self.wind is not None:
            self.wind = np.array([0, *(self.wind + 1)])
        # Wavelength range of each section
        self.wran = None
        # Observation
        self.sob = None
        self.uob = None
        self.mob = None

        # Instrument broadening
        #:str: Instrumental broadening type, values are "table", "gauss", "sinc"
        self.iptype = None
        #:int: Instrumental resolution for instrumental broadening
        self.ipres = None
        #:Fitresults: results from the latest fit
        self.fitresults = Fitresults(
            maxiter=kwargs.pop("maxiter", None),
            chisq=kwargs.pop("chisq", None),
            punc=kwargs.pop("punc", None),
            covar=kwargs.pop("covar", None),
        )

        #:list of arrays: calculated adaptive wavelength grids
        self.wint = None
        self.smod = None
        # remove old keywords
        _ = kwargs.pop("smod_orig", None)
        _ = kwargs.pop("cmod_orig", None)
        _ = kwargs.pop("cmod", None)
        _ = kwargs.pop("jint", None)
        _ = kwargs.pop("sint", None)
        _ = kwargs.pop("psig_l", None)
        _ = kwargs.pop("psig_r", None)
        _ = kwargs.pop("rchisq", None)
        _ = kwargs.pop("crms", None)
        _ = kwargs.pop("lrms", None)
        _ = kwargs.pop("chirat", None)
        _ = kwargs.pop("vmac_pro", None)
        _ = kwargs.pop("cintb", None)
        _ = kwargs.pop("cintr", None)
        _ = kwargs.pop("obs_type", None)
        _ = kwargs.pop("clim", None)
        _ = kwargs.pop("nmu", None)
        _ = kwargs.pop("nseg", None)
        _ = kwargs.pop("md5", None)


        # Substructures
        #:Version: System information
        self.idlver = Version(idlver)
        #:Atmo: Stellar atmosphere
        self.atmo = Atmo(atmo)
        #:NLTE: NLTE settings
        self.nlte = NLTE(nlte)
        super().__init__(**kwargs)

        # Apply final conversions from IDL to Python version
        if "wave" in self:
            self.__convert_cscale__()

    @property
    def atomic(self):
        """array of size (nlines, 8): Atomic linelist data, usually passed to the C library
        Use sme.linelist instead for other purposes """
        if self.linelist is None:
            return None
        return self.linelist.atomic

    @property
    def species(self):
        """array of size (nlines,): Names of the species of each spectral line """
        if self.linelist is None:
            return None
        return self.linelist.species

    @property
    def nmu(self):
        """int: Number of mu values in mu property """
        if self.mu is None:
            return 0
        else:
            return np.size(self.mu)

    @property
    def nseg(self):
        """int: Number of wavelength segments """
        if self.wran is None:
            return None
        else:
            return len(self.wran)

    @property
    def md5(self):
        """hash: md5 hash of this SME structure """
        m = hashlib.md5(str(self).encode())
        if self.wob is not None:
            m.update(self.wob)
        if self.sob is not None:
            m.update(self.sob)
        if self.uob is not None:
            m.update(self.uob)
        if self.mob is not None:
            m.update(self.mob)
        if self.smod is not None:
            m.update(self.smod)

        return m.hexdigest()

    @property
    def wob(self):
        """array: Wavelength array """
        return self._wob

    @wob.setter
    def wob(self, value):
        if value is not None:
            value = np.require(value, requirements="W")
        self._wob = value

    @property
    def sob(self):
        """array: Observed spectrum """
        return self._sob

    @sob.setter
    def sob(self, value):
        if value is not None:
            value = np.require(value, requirements="W")
        self._sob = value

    @property
    def uob(self):
        """array: Uncertainties of the observed spectrum """
        return self._uob

    @uob.setter
    def uob(self, value):
        if value is not None:
            value = np.require(value, requirements="W")
        self._uob = value

    @property
    def mob(self):
        """array: bad/good/line/continuum Mask to apply to observations """
        return self._mob

    @mob.setter
    def mob(self, value):
        if value is not None:
            value = np.require(value, requirements="W")
        self._mob = value

    @property
    def smod(self):
        """array: Synthetic spectrum """
        return self._smod

    @smod.setter
    def smod(self, value):
        if value is not None:
            value = np.require(value, requirements="W")
        self._smod = value

    @property
    def wran(self):
        """array of size (nseg, 2): Beginning and end Wavelength points of each segment"""
        if self._wran is None and self.wob is not None:
            # Default to just one wavelength range with all points if not specified
            return [self.wob[[0, -1]]]
        return self._wran

    @wran.setter
    def wran(self, value):
        if value is not None:
            value = np.atleast_2d(value)
        self._wran = value

    @property
    def mu(self):
        """array of size (nmu,): Mu values to calculate radiative transfer at
        mu values describe the distance from the center of the stellar disk to the edge
        with mu = cos(theta), where theta is the angle of the observation,
        i.e. mu = 1 at the center of the disk and 0 at the edge"""
        return self._mu

    @mu.setter
    def mu(self, value):
        if value is not None:
            value = np.atleast_1d(value)
        self._mu = value

    @property
    def vrad(self):
        """array of size (nseg,): Radial velocity in km/s for each wavelength region"""
        if self._vrad is None or self.nseg is None:
            return None
        if self.vrad_flag == "none":
            return np.zeros(self.nseg)
        else:
            nseg = self._vrad.shape[0]
            if nseg == self.nseg:
                return self._vrad

            rv = np.zeros(self.nseg)
            rv[:nseg] = self._vrad[:nseg]
            rv[nseg:] = self._vrad[-1]
            return rv

        return self._vrad

    @vrad.setter
    def vrad(self, value):
        if value is not None:
            value = np.atleast_1d(value)
        self._vrad = value

    @property
    def cscale(self):
        """array of size (nseg, ndegree): Continumm polynomial coefficients for each wavelength segment
        The x coordinates of each polynomial are chosen so that x = 0, at the first wavelength point,
        i.e. x is shifted by wave[segment][0]
        """
        if self._cscale is None:
            return None

        nseg = self.nseg if self.nseg is not None else 1
        if self.cscale_flag == "none":
            return np.ones((nseg, 1))

        ndeg = {"fix": 1, "constant":1, "linear":2, "quadratic":3}[self.cscale_flag]
        n, length = self._cscale.shape

        if length == ndeg and n == nseg:
            return self._cscale

        cs = np.ones((nseg, ndeg))
        if length == n:
            cs[:n, :] = self._cscale[:n, :]
        elif length < ndeg:
            cs[:n, -length:] = self._cscale[:n, :]
        else:
            cs[:n, :] = self._cscale[:n, -ndeg :]

        cs[n:, -1] = self._cscale[-1, -1]

        return cs

    @cscale.setter
    def cscale(self, value):
        if value is not None:
            value = np.atleast_2d(value)
        self._cscale = value

    @property
    def cscale_flag(self):
        """str: Flag that describes how to correct for the continuum

        allowed values are:
            * "none": No continuum correction
            * "fix": Use whatever continuum scale has been set, but don't change it
            * "constant": Zeroth order polynomial, i.e. scale everything by a factor
            * "linear": First order polynomial, i.e. approximate continuum by a straight line
            * "quadratic": Second order polynomial, i.e. approximate continuum by a quadratic polynomial
        """
        return self._cscale_flag

    @cscale_flag.setter
    def cscale_flag(self, value):
        if isinstance(value, (int, np.integer)):
            value = {-3:"none", -2:"fix", -1:"fix", 0:"constant", 1:"linear", 2:"quadratic"}[value]

        options = ["none", "fix", "constant", "linear", "quadratic"]
        if value not in options:
            raise ValueError(f"Expected one of {options} got {value}")

        self._cscale_flag = value

    @property
    def cscale_degree(self):
        """int: Polynomial degree of the continuum as determined by cscale_flag """
        if self.cscale_flag == "constant":
            return 0
        if self.cscale_flag == "linear":
            return 1
        if self.cscale_flag == "quadratic":
            return 2
        if self.cscale_flag == "fix":
            return self._cscale.shape[1] - 1

        # "none"
        return 0

    @property
    def vrad_flag(self):
        """str: Flag that determines how the radial velocity is determined

        allowed values are:
            * "none": No radial velocity correction
            * "each": Determine radial velocity for each segment individually
            * "whole": Determine one radial velocity for the whole spectrum
        """
        return self._vrad_flag

    @vrad_flag.setter
    def vrad_flag(self, value):
        if isinstance(value, (int, np.integer)):
            value = {-2:"none", -1:"whole", 0:"each"}[value]
        self._vrad_flag = value

    @property
    def wind(self):
        """array of shape (nseg + 1,): Indices of the wavelength segments in the overall arrays """
        if self._wind is None:
            if self.wob is None:
                return None
            return [0, len(self.wob)]
        return self._wind

    @wind.setter
    def wind(self, value):
        self._wind = value

    @property
    def wave(self):
        """Iliffe_vector of shape (nseg, ...): Wavelength """
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
        """Iliffe_vector of shape (nseg, ...): Observed Spectrum """
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
        """Iliffe_vector of shape (nseg, ...): Uncertainties of the observed spectrum """
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
        """Iliffe_vector of shape (nseg, ...): Synthetic Spectrum """
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
        """Iliffe_vector of shape (nseg, ...): Line and Continuum Mask """
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
    def mask_line(self):
        """Iliffe_vector of shape (nseg, ...): Line Mask """
        if self.mask is None:
            return None
        return self.mask == self.mask_values["line"]

    @property
    def mask_continuum(self):
        """Iliffe_vector of shape (nseg, ...): Continuum Mask """
        if self.mask is None:
            return None
        return self.mask == self.mask_values["continuum"]

    @property
    def mask_good(self):
        """Iliffe_vector of shape (nseg, ...): Good Pixel Mask """
        if self.mask is None:
            return None
        return self.mask != self.mask_values["bad"]

    @property
    def mask_bad(self):
        """Iliffe_vector of shape (nseg, ...): Bad Pixel Mask """
        if self.mask is None:
            return None
        return self.mask == self.mask_values["bad"]

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
        wave = self.wave
        self.cscale = np.require(self.cscale, requirements="W")

        if self.cscale_flag == "linear":
            for i in range(len(self.cscale)):
                c, d = self.cscale[i]
                a, b = max(wave[i]), min(wave[i])
                c0 = (a - b) * (c - d) / (a * c - b * d) ** 2
                c1 = (a - b) / (a * c - b * d)

                # Shift zero point to first wavelength of the segment
                c1 += c0 * self.spec[i][0]

                self.cscale[i] = [c0, c1]
        elif self.cscale_flag == "fix":
            self.cscale = self.cscale / np.sqrt(2)
        elif self.cscale_flag == "constant":
            self.cscale = np.sqrt(1 / self.cscale)



    @staticmethod
    def load(filename="sme.npy"):
        """
        Load SME data from disk

        Currently supported file formats:
            * ".npy": Numpy save file of an SME_Struct
            * ".sav", ".inp", ".out": IDL save file with an sme structure
            * ".ech": Echelle file from (Py)REDUCE

        Parameters
        ----------
        filename : str, optional
            name of the file to load (default: 'sme.npy')

        Returns
        -------
        sme : SME_Struct
            Loaded SME structure

        Raises
        ------
        ValueError
            If the file format extension is not recognized
        """
        logging.info("Loading SME file %s", filename)
        _, ext = os.path.splitext(filename)
        if ext == ".npy":
            # Numpy Save file
            s = np.load(filename)
            s = np.atleast_1d(s)[0]
        elif ext in [".sav", ".out", ".inp"]:
            # IDL save file (from SME)
            s = readsav(filename)["sme"]
            s = {name.casefold(): s[name][0] for name in s.dtype.names}
            s = SME_Struct(**s)
        elif ext == ".ech":
            # Echelle file (from REDUCE)
            ech = echelle.read(filename)
            s = SME_Struct()
            s.wind = np.cumsum([0, *np.diff(ech.columns, axis=1).ravel()])
            s.wave = np.ma.compressed(ech.wave)
            s.spec = np.ma.compressed(ech.spec)
            s.uncs = np.ma.compressed(ech.sig)
            s.mask = np.full(s.sob.size, 1)
            s.wran = [[w[0], w[-1]] for w in s.wave]
            try:
                s.object = ech.head["OBJECT"]
            except KeyError:
                pass
        else:
            options = [".npy", ".sav", ".out", ".inp", ".ech"]
            raise ValueError(f"File format not recognised, expected one of {options} but got {ext}")

        return s

    def save(self, filename="sme.npy", verbose=True):
        """
        Save SME data to disk

        Parameters
        ----------
        filename : str, optional
            location to save the SME structure at (default: "sme.npy")
            Should have ending ".npy", otherwise it will be appended to whatever was passed
        verbose : bool, optional
            if True will log the event
        """
        if verbose:
            logging.info("Saving SME structure %s", filename)
        np.save(filename, self)


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
