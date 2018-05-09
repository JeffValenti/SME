from scipy.io import readsav
from abund import Abund

class Param:
    """Handle model parameters for a Spectroscopy Made Easy (SME) job.
    """

    @property
    def teff(self):
        return self._teff

    def set_teff(self, teff):
        self._teff = float(teff)

    @property
    def logg(self):
        return self._logg

    def set_logg(self, logg):
        self._logg = float(logg)

    def set_monh(self, monh):
        self._monh = float(monh)

    @property
    def vmic(self):
        return self._vmic

    def set_vmic(self, vmic):
        self._vmic = float(vmic)

    @property
    def vmac(self):
        return self._vmac

    def set_vmac(self, vmac):
        self._vmac = float(vmac)

    @property
    def vsini(self):
        return self._vsini

    def set_vsini(self, vsini):
        self._vsini = float(vsini)

    @property
    def abund(self):
        return self._abund

    def set_abund(self, monh, abpatt, abtype):
        self._abund = Abund(monh, abpatt, abtype)

    def summary(self):
        fmt = 'Teff={} K,  logg={:.3f},  [M/H]={:.3f},  ' \
                'Vmic={:.2f},  Vmac={:.2f},  Vsini={:.1f}'
        print(fmt.format(self._teff, self._logg, self._monh, \
               self._vmic, self._vmac, self._vsini))
        self._abund.print()

def idlfile(file):
    """Read parameters and any results from an IDL save file created by
    the IDL version of Spectroscopy Made Easy (SME).
    """

    def loadfile(file):
        """Load an 'sme' structure from an IDL save file created by SME.
        """
        try:
            return readsav(file)['sme']
        except FileNotFoundError:
            print('file not found: {}'.format(file))
            return
        except KeyError:
            print("no 'sme' structure in {}".format(file))
            return

    def createpar():
        """Create an SME model parameter object from an IDL structure
        previously read from an IDL save file.
        """
        monh = getval('feh')
        par = Param()
        par.set_teff(getval('teff'))
        par.set_logg(getval('grav'))
        par.set_monh(monh)
        par.set_vmic(getval('vmic'))
        par.set_vmac(getval('vmac'))
        par.set_vsini(getval('vsini'))
        par.set_abund(monh, getval('abund'), 'sme')
        return par

    def getval(key):
        """Extract value for the specified key from an IDL structure.
        """
        try:
            return idl[key][0]
        except ValueError:
            print('no {} key in IDL file/structure'.format(key))
            return

    idl = loadfile(file)
    return createpar()
