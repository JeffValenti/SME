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
            raise ValueError("%s (call external): %s" % (self.name, error.decode()))
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
    wfirst, wlast = float(wfirst), float(wlast)
    return idl_call_external("InputWaveRange", wfirst, wlast)


@check_error
def SetVWscale(gamma6):
    """ Set van der Waals scaling factor """
    gamma6 = float(gamma6)
    return idl_call_external("SetVWscale", gamma6)


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
    nlines = atomic.shape[0]
    atomic = atomic.T.astype(float, copy=False)
    species = species.astype(str, copy=False)
    return idl_call_external("InputLineList", nlines, species, atomic)


def OutputLineList(atomic, copy=False):
    """ Return line list """
    nlines = atomic.shape[0]
    atomic = atomic.astype(float)
    if copy:
        atomic = np.copy(atomic)
    error = idl_call_external("OutputLineList", nlines, atomic.T)
    if error != b"":
        raise ValueError("%s (call external): %s" % (__name__, error.decode()))
    return atomic


@check_error
def UpdateLineList(atomic, species, index):
    """ Change line list parameters """
    nlines = atomic.shape[0]
    atomic = atomic.astype(float)
    species = species.astype("S")
    return idl_call_external("UpdateLineList", nlines, species, atomic.T, index)


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

    if atmo.geom == "SPH":
        radius = atmo.radius
        height = float(atmo.height)
        motype = "SPH"
        args = args[:5] + [radius] + args[5:] + [height]

    return idl_call_external("InputModel", *args)


def InputDepartureCoefficients(n, *args):
    """ """
    raise NotImplementedError()


def GetDepartureCoefficients(n, *args):
    """ Get NLTE b's for specific line """
    raise NotImplementedError()


def ResetDepartureCoefficients(n, *args):
    """ Reset LTE """
    raise NotImplementedError()


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
    # TODO might be obsolete with new SME structure?
    abund = np.copy(abund)
    abund[2:] = abund[2:] + feh  # apply metallicity, except to H He
    abund[2:] = np.clip(abund[2:], None, -1.)  # make sure we do not go berserk here
    abund[1:] = 10.0 ** abund[1:]  # convert log(fraction) to fraction
    abund /= np.sum(abund)  # normalize sum of fractions
    abund[1:] = np.log10(abund[1:])  # convert fraction to log(fraction)

    return idl_call_external("InputAbund", abund)


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


@check_error
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
    return idl_call_external("Ionization", ion, inttype="short")


def GetDensity():
    """ """
    raise NotImplementedError()


def GetNatom():
    """ """
    raise NotImplementedError()


def GetNelec():
    """ """
    raise NotImplementedError()


def Transf(mu, accrt, accwi, keep_lineop=0, long_continuum=1, nwmax=400000):
    """ """
    nw = 0
    nmu = len(mu)
    # Prepare data:
    wint_seg = np.zeros((nwmax))
    sint_seg = np.zeros((nwmax, nmu))  # line+continuum intensities
    cint_seg = np.zeros((nwmax, nmu))  # all continuum intensities
    cintr_seg = np.zeros((nmu))  # red continuum intensity

    error = idl_call_external(
        "Transf",
        np.int16(nmu),
        mu,
        cint_seg,
        cintr_seg,
        nwmax,
        nw,
        wint_seg,
        sint_seg,
        accrt,
        accwi,
        np.int16(keep_lineop),
        np.int16(long_continuum),
    )
    if error != b"":
        raise ValueError("Transf (call external): %s" % error.decode())
    nw = np.count_nonzero(wint_seg)

    wint_seg = wint_seg[:nw]
    sint_seg = sint_seg[:nw, :].T
    cint_seg = cint_seg[:nw, :].T

    return nw, wint_seg, sint_seg, cint_seg


def CentralDepth():
    """ """
    raise NotImplementedError()
