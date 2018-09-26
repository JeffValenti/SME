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
    atomic = atomic.astype(float)
    species = species.astype(str)
    return idl_call_external("InputLineList", nlines, species, atomic.T)


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
def UpdateLineList(atomic, species):
    """ Change line list parameters """
    nlines = atomic.shape[0]
    atomic = atomic.astype(float)
    species = species.astype("S")
    return idl_call_external("UpdateLineList", nlines, species, atomic.T)


@check_error
def InputModel(teff, grav, vturb, atmo):
    """ Read in model atmosphere """
    motype = atmo.depth
    depth = atmo[motype]
    ndepth = len(depth)
    t = atmo.teff
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
    """ Read in abundances """
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
    """ """
    return idl_call_external("Ionization", ion)


def GetDensity():
    """ """
    raise NotImplementedError()


def GetNatom():
    """ """
    raise NotImplementedError()


def GetNelec():
    """ """
    raise NotImplementedError()


def Transf():
    """ """
    raise NotImplementedError()


def CentralDepth():
    """ """
    raise NotImplementedError()
