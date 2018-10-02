from scipy.io import readsav
from .abund import Abund


class Collection:
    """
    A collection of predefined fields (in __fields__, all lowercase)
    that can be accessed either as attributes or via the index
    """
    def __init__(self):
        self.__fields__ = []

    def __getattribute__(self, name):
        return super().__getattribute__(name.lower())

    def __setattr__(self, key, value):
        if key == "__fields__" or key in self.__fields__:
            return super().__setattr__(key.lower(), value)
        else:
            raise KeyError("Key %s not found" % key)

    def __getitem__(self, key):
        return getattr(self, key.lower())

    def __setitem__(self, key, value):
        if key in self.__fields__:
            setattr(self, key.lower(), value)
        else:
            raise KeyError("Key %s not found" % key)

    def __contains__(self, key):
        return key in self.__fields__

    @property
    def tags(self):
        return [tag for tag in self.__fields__ if self[tag] is not None]


class Param(Collection):
    """Handle model parameters for a Spectroscopy Made Easy (SME) job.
    """

    def __init__(
        self,
        teff=None,
        logg=None,
        monh=None,
        vmic=None,
        vmac=None,
        vsini=None,
        abund=None,
        abund_pattern="sme",
    ):
        self.__fields__ = ["teff", "logg", "feh", "vmic", "vmac", "vsini"]

        self.teff = teff
        self.logg = logg
        self.monh = monh
        self.vmic = vmic
        self.vmac = vmac
        self.vsini = vsini

        self.set_abund(monh, abund, abund_pattern)

    def __str__(self):
        return self.summary()

    @property
    def teff(self):
        return self._teff

    @teff.setter
    def teff(self, teff):
        self._teff = float(teff)

    @property
    def logg(self):
        return self._logg

    @logg.setter
    def logg(self, logg):
        self._logg = float(logg)

    @property
    def monh(self):
        return self._monh

    @monh.setter
    def monh(self, monh):
        self._monh = float(monh)

    @property
    def vmic(self):
        return self._vmic

    @vmic.setter
    def vmic(self, vmic):
        self._vmic = float(vmic)

    @property
    def vmac(self):
        return self._vmac

    @vmac.setter
    def vmac(self, vmac):
        self._vmac = float(vmac)

    @property
    def vsini(self):
        return self._vsini

    @vsini.setter
    def vsini(self, vsini):
        self._vsini = float(vsini)

    @property
    def abund(self):
        return self._abund

    def set_abund(self, monh, abpatt, abtype):
        self._abund = Abund(monh, abpatt, abtype)

    def summary(self):
        fmt = "Teff={} K,  logg={:.3f},  [M/H]={:.3f},  Vmic={:.2f},  Vmac={:.2f},  Vsini={:.1f}"
        print(
            fmt.format(
                self.teff, self.logg, self.monh, self.vmic, self.vmac, self.vsini
            )
        )
        self._abund.print()


class Atmo(Collection):
    def __init__(self):
        self.__fields__ = [
            "teff",
            "logg",
            "feh",
            "vmac",
            "vmic",
            "vsini",
            "depth",
            "rhox",
            "tau",
            "interp",
            "source",
            "geom",
        ]
        self._param = Param()
        self.temp = None
        self.depth = None
        self.rhox = None
        self.tau = None
        self.source = None
        self.interp = None
        self.geom = None

    @property
    def teff(self):
        return self._param.teff

    @teff.setter
    def teff(self, value):
        self._param.teff = value

    @property
    def logg(self):
        return self._param.logg

    @logg.setter
    def logg(self, value):
        self._param.logg = value

    @property
    def monh(self):
        return self._param.monh

    @monh.setter
    def monh(self, value):
        self._param.monh = value


class SME_Struct(Collection):
    def __init__(self):
        self.__fields__ = [
            "version",
            "id",
            "teff",
            "logg",
            "feh",
            "vmac",
            "vmic",
            "vsini",
            "atmo",
            "atomic",
            "species",
            "jint",
            "cint",
            "wint",
            "sint",
            "cintb",
            "cintr",
            "sob",
            "uob",
            "mod",
            "cscale",
            "cscale_flag",
            "wave",
            "wind",
            "vrad",
            "vrad_flag",
            "gam6",
            "h2broad",
            "accwi",
            "accrt",
            "clim",
            "maxiter",
            "chirat",
            "nmu",
            "nseg",
            "abund",
            "nlte",
            "lande",
            "lineref",
            "short_line_format",
            "line_extra",
            "line_lulande",
            "line_term_low",
            "line_term_upp",
            "wran",
            "mu",
            "glob_free",
            "ab_free",
        ]
        self.version = None
        self.id = None

        self.atmo = Atmo()

        self.atomic = None
        self.species = None

        self.jint = None
        self.cint = None
        self.wint = None
        self.sint = None

        self.sob = None
        self.uob = None
        self.mod = None

        self.cscale = None
        self.cscale_flag = None
        self.wave = None
        self.wind = None
        self.vrad = None
        self.vrad_flag = None
        self.gam6 = None
        self.h2broad = False
        self.cintr = None
        self.cintb = None

    @property
    def vmic(self):
        return self.atmo.vmic

    @vmic.setter
    def vmic(self, value):
        self.atmo.vmic = value

    @property
    def vmac(self):
        return self.atmo.vmac

    @vmac.setter
    def vmac(self, value):
        self.atmo.vmac = value

    @property
    def vsini(self):
        return self.atmo.vsini

    @vsini.setter
    def vsini(self, value):
        self.atmo.vsini = value

    @property
    def teff(self):
        return self.atmo.teff

    @teff.setter
    def teff(self, value):
        self.atmo.teff = value

    @property
    def logg(self):
        return self.atmo.logg

    @logg.setter
    def logg(self, value):
        self.atmo.logg = value

    @property
    def monh(self):
        return self.atmo.monh

    @monh.setter
    def monh(self, value):
        self.atmo.monh = value

    @property
    def abund(self):
        return self.atmo.abund


def idlfile(file):
    """Read parameters and any results from an IDL save file created by
    the IDL version of Spectroscopy Made Easy (SME).
    """

    def loadfile(file):
        """Load an 'sme' structure from an IDL save file created by SME.
        """
        try:
            return readsav(file)["sme"]
        except FileNotFoundError:
            print("file not found: {}".format(file))
            return
        except KeyError:
            print("no 'sme' structure in {}".format(file))
            return

    def createpar():
        """Create an SME model parameter object from an IDL structure
        previously read from an IDL save file.
        """
        monh = getval("feh")
        par = Param()
        par.set_teff(getval("teff"))
        par.set_logg(getval("grav"))
        par.set_monh(monh)
        par.set_vmic(getval("vmic"))
        par.set_vmac(getval("vmac"))
        par.set_vsini(getval("vsini"))
        par.set_abund(monh, getval("abund"), "sme")
        return par

    def getval(key):
        """Extract value for the specified key from an IDL structure.
        """
        try:
            return idl[key][0]
        except ValueError:
            print("no {} key in IDL file/structure".format(key))
            return

    idl = loadfile(file)
    return createpar()
