""" Wrapper for sme_synth.so C library """
import os
import warnings

import numpy as np

from .cwrapper import idl_call_external


class check_error:
    """
    decorator that raises an error if a
    function does not return b"", i.e. empty bytes string
    """

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
    """ Set the path to the library """
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

    species = np.asarray(species, "U8")

    # Sort list by wavelength
    sort = np.argsort(atomic[:, 2])
    species = species[sort]
    atomic = atomic[sort, :]

    atomic = atomic.T
    return idl_call_external(
        "InputLineList", nlines, species, atomic, type=("int", "string", "double")
    )


def OutputLineList(nlines):
    """ Return line list """
    atomic = np.zeros((nlines, 6))
    error = idl_call_external("OutputLineList", nlines, atomic, type=("int", "double"))
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
        "opflag", np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
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
def InputAbund(abund):
    """
    Pass abundances to radiative transfer code.

    Calculate elemental abundances (abund) from abundance pattern (sme.abund)
    and metallicity (sme.feh). Metallicity adjustment is not applied to H or He.
    Renormalize abundances after applying metallicity [added 2012-Mar-08].
    Introduced limiter in case the proposed step in abundance is too large [2018-Apr-25].
    """
    # Convert abundances to the right format
    # metallicity is included in the abundance class, ignored in function call
    abund = abund("sme", raw=True)
    return idl_call_external("InputAbund", abund, type="double")


def Opacity(nmu=None, motype=1):
    """ Calculate opacities """
    args = []
    type = ""
    if nmu is not None:
        copblu = np.zeros(nmu)
        copred = np.zeros(nmu)
        args = [nmu, copblu, copred]
        type = ["s", "d", "d"]

        if motype == 0:
            copstd = np.zeros(nmu)
            args += [copstd]
            type += ["d"]

    check_error(idl_call_external)("Opacity", *args, type=type)

    return args[1:]


def GetOpacity(sme, switch, species=None, key=None):
    """
    Returns specific cont. opacity

    switch : int
        -3 : COPSTD
        -2 : COPRED
        -1 : COPBLU
         0 : AHYD
         1 : AH2P
         2 : AHMIN
         3 : SIGH
         4 : AHE1
         5 : AHE2
         6 : AHEMIN
         7 : SIGHE
         8 : ACOOL, continuous opacity C1, Mg1, Al1, Si1, Fe1, CH, NH, OH
         9 : ALUKE, continuous opacity N1, O1, Mg2, Si2, Ca2
         10: AHOT
         11: SIGEL
         12: SIGH2
    """
    # j=*(short *)arg[0];   # IFOP number */
    # i=*(short *)arg[1];   # Length of IDL arrays */
    # a1=(double *)arg[2];
    length = np.size(sme.mu)  # nmu
    result = np.ones(length)
    args = [switch, length, result]
    type = ["s", "s", "d"]

    if switch == 8:
        if species is not None:
            if key is None:
                raise AttributeError(
                    "Both species and key keywords need to be set with switch 8, continous opacity"
                )
            else:
                args += [species, key]
                type += ["u", "u"]
    elif switch == 9:
        if species is not None:
            args += [species]
            type += ["u"]

    error = idl_call_external("GetOpacity", *args, type=type)

    if error != b"":
        raise ValueError(f"GetOpacity (call external): {error.decode()}")
    return result


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
        warnings.warn(f"{__name__} (call external): {error.decode()}")


def GetDensity():
    """ """
    raise NotImplementedError()


def GetNatom():
    """ """
    raise NotImplementedError()


def GetNelec():
    """ """
    raise NotImplementedError()


def Transf(
    mu, accrt, accwi, keep_lineop=False, long_continuum=True, nwmax=400000, wave=None
):
    """
    Radiative Transfer Calculation

    Parameters
    ---------
    mu : array
        mu angles (1 - cos(phi)) of different limb points along the stellar surface
    accrt : float
        accuracy of the radiative transfer integration
    accwi : float
        accuracy of the interpolation on the wavelength grid
    keep_lineop : bool, optional
        if True do not recompute the line opacities (default: False)
    long_continuum : bool, optional
        if True the continuum is calculated at every wavelength (default: True)
    nwmax : int, optional
        maximum number of wavelength points if wavelength grid is not set with wave (default: 400000)
    wave : array, optional
        wavelength grid to use for the calculation,
        if not set will use an adaptive wavelength grid with no constant step size (default: None)

    Returns
    -------
    nw : int
        number of actual wavelength points, i.e. size of wint_seg
    wint_seg : array[nw]
        wavelength grid
    sint_seg : array[nw]
        spectrum
    cint_seg : array[nw]
        continuum
    """
    keep_lineop = 1 if keep_lineop else 0
    long_continuum = 1 if long_continuum else 0

    if wave is None:
        nw = 0
        wint_seg = np.zeros(nwmax, float)
    else:
        nw = len(wave)
        nwmax = nw
        wint_seg = np.asarray(wave, float)

    nmu = np.size(mu)

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


def GetLineOpacity(wave, nmu):
    """
    Retrieve line opacity data from the C library

    Parameters
    ----------
    wave : float
        Wavelength of the line opacity to retrieve
    nmu  : int
        number of depth points in the atmosphere

    Returns
    ---------
    lop : array
        line opacity
    cop : array
        continuum opacity including scatter
    scr : array
        Scatter
    tsf : array
        Total source function
    csf : array
        Continuum source function
    """
    lop = np.zeros(nmu)
    cop = np.zeros(nmu)
    scr = np.zeros(nmu)
    tsf = np.zeros(nmu)
    csf = np.zeros(nmu)
    type = "dsddddd"
    error = idl_call_external(
        "GetLineOpacity", wave, nmu, lop, cop, scr, tsf, csf, type=type
    )
    if error != b"":
        raise ValueError(f"GetLineOpacity (call external): {error.decode()}")
    return lop, cop, scr, tsf, csf


def GetLineRange(nlines):
    """ Get the effective wavelength range for each line
    i.e. the wavelengths for which the line has significant impact
    
    Parameters
    ----------
    nlines : int
        number of lines in the linelist
    
    Returns
    -------
    linerange : array of size (nlines, 2)
        lower and upper wavelength for each spectral line
    """

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
    """ Get the NLTE departure coefficients as stored in the C library

    Parameters
    ----------
    nrhox : int
        number of layers
    line : int
        requested line index, i.e. between 0 and number of lines

    Returns
    -------
    bmat : array of size (2, nrhox)
        departure coefficients for the given line index
    """

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


def GetNLTEflags(nlines):
    """Get an array that tells us which lines have been used with NLTE correction

    Parameters
    ----------
    linelist : int
        number of lines

    Returns
    -------
    nlte_flags : array(bool) of size (nlines,)
        True if line was used with NLTE, False if line is only LTE
    """

    nlte_flags = np.zeros(nlines, dtype=np.int16)

    error = idl_call_external("GetNLTEflags", nlte_flags, nlines, type=("short", "int"))
    if error != b"":
        raise ValueError(f"GetNLTEflags (call external): {error.decode()}")

    return nlte_flags.astype(bool)
