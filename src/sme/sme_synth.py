import os
import numpy as np
from .cwrapper import idl_call_external


class check_error:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __call__(self, *args, **kwargs):
        error = self.func(*args, **kwargs)
        if error != b"":
            raise ValueError(f"{self.name} (call external): {error.decode()}")
        return error


def SMELibraryVersion():
    """ Retern SME library version """
    return idl_call_external("SMELibraryVersion")


@check_error
def SetLibraryPath():
    prefix = os.path.dirname(__file__)
    libpath = os.path.join(prefix, "dll") + os.sep
    return idl_call_external("SetLibraryPath", libpath)


@check_error
def InputWaveRange(wfirst, wlast):
    """ Read in Wavelength range """
    return idl_call_external("InputWaveRange", wfirst, wlast, type="double")


@check_error
def SetVWscale(gamma6):
    """ Set van der Waals scaling factor """
    return idl_call_external("SetVWscale", gamma6, type="double")


@check_error
def SetH2broad(h2_flag=True):
    """ Set flag for H2 molecule """
    if h2_flag:
        return idl_call_external("SetH2broad")
    else:
        return ClearH2broad()


@check_error
def ClearH2broad():
    """ Clear flag for H2 molecule """
    return idl_call_external("ClearH2broad")


@check_error
def InputLineList(atomic, species):
    """ Read in line list """
    nlines = species.size
    atomic = atomic.T
    return idl_call_external(
        "InputLineList", nlines, species, atomic, type=("int", "double", "double")
    )


def OutputLineList(atomic, copy=False):
    """ Return line list """
    nlines = atomic.shape[0]
    if copy:
        atomic = np.copy(atomic)
    error = idl_call_external(
        "OutputLineList", nlines, atomic.T, type=("int", "double")
    )
    if error != b"":
        raise ValueError(f"{__name__} (call external): {error.decode()}")
    return atomic


@check_error
def UpdateLineList(atomic, species, index):
    """ Change line list parameters """
    nlines = atomic.shape[0]
    return idl_call_external(
        "UpdateLineList",
        nlines,
        species,
        atomic.T,
        index,
        type=("int", "str", "double", "short"),
    )


@check_error
def InputModel(teff, grav, vturb, atmo):
    """ Read in model atmosphere """
    motype = atmo.depth
    depth = atmo[motype]
    ndepth = len(depth)
    t = atmo.temp
    xne = atmo.xne
    xna = atmo.xna
    rho = atmo.rho
    vt = np.full(ndepth, vturb) if vturb.size == 1 else vturb
    wlstd = atmo.get("wlstd", 5000.0)
    opflag = atmo.get(
        "opflag", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    args = [ndepth, teff, grav, wlstd, motype, opflag, depth, t, xne, xna, rho, vt]
    type = "sdddusdddddd"  # s : short, d: double, u: unicode (string)

    if atmo.geom == "SPH":
        radius = atmo.radius
        height = atmo.height
        motype = "SPH"
        args = args[:5] + [radius] + args[5:] + [height]
        type = type[:5] + "d" + type[5:] + "d"

    return idl_call_external("InputModel", *args, type=type)


@check_error
def InputAbund(abund, feh):
    """
    Pass abundances to radiative transfer code.

    Calculate elemental abundances (abund) from abundance pattern (sme.abund)
    and metallicity (sme.feh). Metallicity adjustment is not applied to H or He.
    Renormalize abundances after applying metallicity [added 2012-Mar-08].
    Introduced limiter in case the proposed step in abundance is too large [2018-Apr-25].
    """
    # Convert abundances to the right format
    # metallicity is included in the abundance class, ignored in function call
    abund = abund("n/nTot", raw=True)
    return idl_call_external("InputAbund", abund, type="double")


@check_error
def Opacity():
    """ Calculate opacities """
    return idl_call_external("Opacity")


def GetOpacity():
    """ Returns specific cont. opacity """
    # j=*(short *)arg[0];   # IFOP number */
    # i=*(short *)arg[1];   # Length of IDL arrays */
    # a1=(double *)arg[2];
    raise NotImplementedError()
    return idl_call_external("GetOpacity")


# @check_error
def Ionization(ion=0):
    """
    Calculate ionization balance for current atmosphere and abundances.
    Ionization state is stored in the external library.
    Set adopt_eos bit mask to 7 = 1 + 2 + 4 to:

      (1) adopt particle number densities from EOS,
      (2) adopt electron number densities from EOS,
      (4) and adopt gas densities (g/cm^3) from EOS,

    instead of using values from model atmosphere. Different abundance patterns
    in the model atmosphere (usually scaled solar) and SME (may be non-solar)
    can affect line shape, e.g. shape of hydrogen lines.
    """
    error = idl_call_external("Ionization", ion, type="short")
    if error != b"":
        print(f"{__name__} (call external): {error.decode()}")


def GetDensity():
    """ """
    raise NotImplementedError()


def GetNatom():
    """ """
    raise NotImplementedError()


def GetNelec():
    """ """
    raise NotImplementedError()


def Transf(mu, accrt, accwi, keep_lineop=0, long_continuum=1, nwmax=400000, wave=None):
    """ Radiative Transfer Calculation """

    if wave is None:
        nw = 0
        wint_seg = np.zeros((nwmax))
    else:
        nw = len(wave)
        nwmax = nw
        wint_seg = np.asarray(wave, float)

    nmu = len(mu)

    # Prepare data:
    sint_seg = np.zeros((nwmax, nmu))  # line+continuum intensities
    cint_seg = np.zeros((nwmax, nmu))  # all continuum intensities
    cintr_seg = np.zeros((nmu))  # red continuum intensity

    type = "sdddiiddddssu"  # s: short, d:double, i:int, u:unicode (string)

    error = idl_call_external(
        "Transf",
        nmu,
        mu,
        cint_seg,
        cintr_seg,
        nwmax,
        nw,
        wint_seg,
        sint_seg,
        accrt,
        accwi,
        keep_lineop,
        long_continuum,
        type=type,
    )
    if error != b"":
        raise ValueError(f"Transf (call external): {error.decode()}")
    nw = np.count_nonzero(wint_seg)

    wint_seg = wint_seg[:nw]
    sint_seg = sint_seg[:nw, :].T
    cint_seg = cint_seg[:nw, :].T

    return nw, wint_seg, sint_seg, cint_seg


def CentralDepth():
    """ """
    raise NotImplementedError()


def GetLineRange(nlines):
    """ """
    linerange = np.zeros((nlines, 2))

    error = idl_call_external("GetLineRange", linerange, nlines, type=("double", "int"))
    if error != b"":
        raise ValueError(f"GetLineRange (call external): {error.decode()}")

    return linerange


@check_error
def InputNLTE(bmat, lineindices):
    """ Input NLTE departure coefficients """
    return idl_call_external(
        "InputDepartureCoefficients", bmat, lineindices, type=("double", "int")
    )


def GetNLTE(nrhox, line):
    bmat = np.full((2, nrhox), -1., dtype=float)
    error = idl_call_external(
        "GetDepartureCoefficients", bmat, nrhox, line, type=("double", "int", "int")
    )
    if error != b"":
        raise ValueError(f"ERROR {error.decode()}")
    return bmat


@check_error
def ResetNLTE():
    """ Reset departure coefficients from any previous call, to ensure LTE as default """
    return idl_call_external("ResetDepartureCoefficients")


def GetNLTEflags(linelist):
    """ """
    nlines = len(linelist)
    nlte_flags = np.zeros(nlines, dtype=np.int16)

    error = idl_call_external("GetNLTEflags", nlte_flags, nlines, type=("short", "int"))
    if error != b"":
        raise ValueError(f"GetNLTEflags (call external): {error.decode()}")

    return nlte_flags.astype(bool)
