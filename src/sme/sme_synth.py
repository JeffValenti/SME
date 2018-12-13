""" Wrapper for sme_synth.so C library """
import os
import warnings

import numpy as np

from .cwrapper import idl_call_external


class DLL:
    """ Stores the expected sizes of arrays, e.g. number of lines """

    def __init__(self):
        self._ndepth = None
        self._nmu = None
        self._nlines = None

    @property
    def ndepth(self):
        if self._ndepth is None:
            raise ValueError("No model atmosphere has been set")
        return self._ndepth

    @ndepth.setter
    def ndepth(self, value):
        self._ndepth = value

    @property
    def nmu(self):
        if self._nmu is None:
            raise ValueError("No radiative transfer has been calculated")
        return self._nmu

    @nmu.setter
    def nmu(self, value):
        self._nmu = value

    @property
    def nlines(self):
        if self._nlines is None:
            raise ValueError("No line list has been set")
        return self._nlines

    @nlines.setter
    def nlines(self, value):
        self._nlines = value


dll = DLL()


def check_error(name, *args, **kwargs):
    """ run idl_call_external and check for errors in the output """
    error = idl_call_external(name, *args, **kwargs)
    error = error.decode()
    if error != "":
        raise ValueError(f"{name} (call external): {error}")
    return error


def SMELibraryVersion():
    """ Retern SME library version """
    version = idl_call_external("SMELibraryVersion")
    return version.decode()


def SetLibraryPath():
    """ Set the path to the library """
    prefix = os.path.dirname(__file__)
    libpath = os.path.join(prefix, "dll") + os.sep
    check_error("SetLibraryPath", libpath)


def InputWaveRange(wfirst, wlast):
    """ Read in Wavelength range """
    check_error("InputWaveRange", wfirst, wlast, type="double")


def SetVWscale(gamma6):
    """ Set van der Waals scaling factor """
    check_error("SetVWscale", gamma6, type="double")


def SetH2broad(h2_flag=True):
    """ Set flag for H2 molecule """
    if h2_flag:
        check_error("SetH2broad")
    else:
        ClearH2broad()


def ClearH2broad():
    """ Clear flag for H2 molecule """
    check_error("ClearH2broad")


def InputLineList(atomic, species):
    """ Read in line list """
    nlines = species.size
    species = np.asarray(species, "U8")

    # Sort list by wavelength
    sort = np.argsort(atomic[:, 2])
    species = species[sort]
    atomic = atomic[sort, :]

    atomic = atomic.T
    check_error(
        "InputLineList", nlines, species, atomic, type=("int", "string", "double")
    )

    dll.nlines = nlines


def OutputLineList():
    """ Return line list """
    nlines = dll.nlines
    atomic = np.zeros((nlines, 6))
    check_error("OutputLineList", nlines, atomic, type=("int", "double"))
    return atomic


def UpdateLineList(atomic, species, index):
    """ Change line list parameters """
    nlines = atomic.shape[0]
    atomic = atomic.T
    check_error(
        "UpdateLineList",
        nlines,
        species,
        atomic,
        index,
        type=("int", "str", "double", "short"),
    )


def InputModel(teff, grav, vturb, atmo):
    """ Read in model atmosphere """
    motype = atmo.depth
    depth = atmo[motype]
    ndepth = len(depth)
    t = atmo.temp
    xne = atmo.xne
    xna = atmo.xna
    rho = atmo.rho
    vt = np.full(ndepth, vturb) if np.size(vturb) == 1 else vturb
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

    dll.ndepth = ndepth

    check_error("InputModel", *args, type=type)


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
    check_error("InputAbund", abund, type="double")


def Opacity(getData=False, motype=1):
    """ Calculate opacities """
    args = []
    type = ""
    if getData:
        nmu = dll.nmu
        copblu = np.zeros(nmu)
        copred = np.zeros(nmu)
        args = [nmu, copblu, copred]
        type = ["s", "d", "d"]

        if motype == 0:
            copstd = np.zeros(nmu)
            args += [copstd]
            type += ["d"]

    check_error("Opacity", *args, type=type)

    return args[1:]


def GetOpacity(switch, species=None, key=None):
    """
    Returns specific cont. opacity

    Parameters
    ----------
    switch : int
        -3  = COPSTD
        -2  = COPRED
        -1  = COPBLU
         0  = AHYD
         1  = AH2P
         2  = AHMIN
         3  = SIGH
         4  = AHE1
         5  = AHE2
         6  = AHEMIN
         7  = SIGHE
         8  = ACOOL, continuous opacity C1, Mg1, Al1, Si1, Fe1, CH, NH, OH
         9  = ALUKE, continuous opacity N1, O1, Mg2, Si2, Ca2
         10 = AHOT
         11 = SIGEL
         12 = SIGH2
    """
    length = dll.nmu
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

    check_error("GetOpacity", *args, type=type)
    return result


def Ionization(ion=0):
    """
    Calculate ionization balance for current atmosphere and abundances.
    Ionization state is stored in the external library.
    Set adopt_eos bit mask to 7 = 1 + 2 + 4 to:

    1: adopt particle number densities from EOS
    2: adopt electron number densities from EOS
    4: and adopt gas densities (g/cm^3) from EOS

    instead of using values from model atmosphere. Different abundance patterns
    in the model atmosphere (usually scaled solar) and SME (may be non-solar)
    can affect line shape, e.g. shape of hydrogen lines.
    """
    error = idl_call_external("Ionization", ion, type="short")
    if error != b"":
        warnings.warn(f"{__name__} (call external): {error.decode()}")


def GetDensity():
    """ Retrieve density in each layer """
    length = dll.ndepth
    array = np.zeros(length, dtype=float)
    check_error("GetDensity", length, array, type="sd")
    return array


def GetNatom():
    """ Get XNA """
    length = dll.ndepth
    array = np.zeros(length, dtype=float)
    check_error("GetNatom", length, array, type="sd")
    return array


def GetNelec():
    """ Get XNE """
    length = dll.ndepth
    array = np.zeros(length, dtype=float)
    check_error("GetNelec", length, array, type="sd")
    return array


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

    check_error(
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
    nw = np.count_nonzero(wint_seg)

    wint_seg = wint_seg[:nw]
    sint_seg = sint_seg[:nw, :].T
    cint_seg = cint_seg[:nw, :].T

    dll.nmu = nmu

    return nw, wint_seg, sint_seg, cint_seg


def CentralDepth():
    """ """
    raise NotImplementedError()


def GetLineOpacity(wave):
    """
    Retrieve line opacity data from the C library

    Parameters
    ----------
    wave : float
        Wavelength of the line opacity to retrieve

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
    nmu = dll.nmu
    lop = np.zeros(nmu)
    cop = np.zeros(nmu)
    scr = np.zeros(nmu)
    tsf = np.zeros(nmu)
    csf = np.zeros(nmu)
    type = "dsddddd"
    check_error("GetLineOpacity", wave, nmu, lop, cop, scr, tsf, csf, type=type)
    return lop, cop, scr, tsf, csf


def GetLineRange():
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
    nlines = dll.nlines
    linerange = np.zeros((nlines, 2))

    check_error("GetLineRange", linerange, nlines, type=("double", "int"))

    return linerange


def InputNLTE(bmat, lineindices):
    """ Input NLTE departure coefficients """
    check_error("InputDepartureCoefficients", bmat, lineindices, type=("double", "int"))


def GetNLTE(line):
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
    nrhox = dll.ndepth

    bmat = np.full((2, nrhox), -1., dtype=float)
    check_error(
        "GetDepartureCoefficients", bmat, nrhox, line, type=("double", "int", "int")
    )
    return bmat


def ResetNLTE():
    """ Reset departure coefficients from any previous call, to ensure LTE as default """
    check_error("ResetDepartureCoefficients")


def GetNLTEflags():
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
    nlines = dll.nlines
    nlte_flags = np.zeros(nlines, dtype=np.int16)

    check_error("GetNLTEflags", nlte_flags, nlines, type=("short", "int"))

    return nlte_flags.astype(bool)
