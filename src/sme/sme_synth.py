""" Wrapper for sme_synth.so C library """
import os
import warnings

import numpy as np

from .cwrapper import idl_call_external, get_lib_name


def check_error(name, *args, **kwargs):
    """
    run idl_call_external and check for errors in the output

    Parameters
    ----------
    name : str
        name of the external C function to call
    args
        parameters for the function
    kwargs
        keywords for the function

    Raises
    --------
    ValueError
        If the returned string is not empty, it means an error occured in the C library
    """
    error = idl_call_external(name, *args, **kwargs)
    error = error.decode()
    if error != "":
        raise ValueError(f"{name} (call external): {error}")


class SME_DLL:
    """ Object Oriented interface for the SME C library """

    def __init__(self):
        #:LineList: Linelist passed to the library
        self.linelist = None
        #:int: Number of mu points passed to the library
        self.nmu = None
        #:Abund: Elemental abundances passed to the library
        self.abund = None
        #:float: First wavelength of the current segment in Angstrom
        self.wfirst = None
        #:float: Last wavelength of the current segment in Angstrom
        self.wlast = None
        #:float: Van der Waals broadening parameter set in the library
        self.vw_scale = None
        #:bool: Wether the library uses H2 broadening or not
        self.H2broad = False
        #:float: Effective temperature set in the model
        self.teff = None
        #:float: Surface gravity set in the model
        self.grav = None
        #:float: Turbulence velocity in the model in km/s
        self.vturb = None
        #:Atmo: Atmosphere structure in the model
        self.atmo = None

    @property
    def ndepth(self):
        """int: Number of depth layers in the atmosphere model"""
        assert self.atmo is not None, f"No model atmosphere has been set"
        motype = self.atmo.depth
        return len(self.atmo[motype])

    @property
    def nlines(self):
        """int: number of lines in the linelist"""
        assert self.linelist is not None, f"No line list has been set"
        return len(self.linelist)

    @property
    def file(self):
        """str: (Expected) Location of the library file"""
        return get_lib_name()

    def SMELibraryVersion(self):
        """
        Return SME library version

        Returns
        -------
        version : str
            SME library version
        """
        version = idl_call_external("SMELibraryVersion")
        return version.decode()

    def SetLibraryPath(self):
        """ Set the path to the library """
        prefix = os.path.dirname(__file__)
        libpath = os.path.join(prefix, "dll") + os.sep
        check_error("SetLibraryPath", libpath)

    def InputWaveRange(self, wfirst, wlast):
        """
        Read in Wavelength range

        Will raise an exception if wfirst is larger than wlast

        Parameters
        ----------
        wfirst : float
            first wavelength of the segment
        wlast : float
            last wavelength of the segment
        """
        assert (
            wfirst < wlast
        ), "Input Wavelength range is wrong, first wavelength is larger than last"
        check_error("InputWaveRange", wfirst, wlast, type="double")

        self.wfirst = wfirst
        self.wlast = wlast

    def SetVWscale(self, gamma6):
        """
        Set van der Waals scaling factor

        Parameters
        ----------
        gamma6 : float
            van der Waals scaling factor
        """
        check_error("SetVWscale", gamma6, type="double")
        self.vw_scale = gamma6

    def SetH2broad(self, h2_flag=True):
        """ Set flag for H2 molecule """
        if h2_flag:
            check_error("SetH2broad")
            self.H2broad = True
        else:
            self.ClearH2broad()

    def ClearH2broad(self):
        """ Clear flag for H2 molecule """
        check_error("ClearH2broad")
        self.H2broad = False

    def InputLineList(self, linelist):
        """
        Read in line list

        Parameters
        ---------
        atomic : array of size (nlines, 8)
            atomic linelist data for each line
            fields are: atom_number, ionization, wlcent, excit, gflog, gamrad, gamqst, gamvw
        species : array(string) of size (nlines,)
            names of the elements (with Ionization level)
        """
        atomic = linelist.atomic.T
        species = linelist.species

        nlines = len(linelist)
        species = np.asarray(species, "U8")

        assert (
            atomic.shape[1] == nlines
        ), f"Got wrong Linelist shape, expected ({nlines}, 8) but got {atomic.shape}"
        assert (
            atomic.shape[0] == 8
        ), f"Got wrong Linelist shape, expected ({nlines}, 8) but got {atomic.shape}"

        check_error(
            "InputLineList", nlines, species, atomic, type=("int", "string", "double")
        )

        self.linelist = linelist

    def OutputLineList(self):
        """
        Return line list

        Returns
        -------
        atomic : array of size (nlines, 6)
            relevant data of the linelist
            wlcent, excit, gflog, gamrad, gamqst, gamvw
        """
        nlines = self.nlines
        atomic = np.zeros((nlines, 6))
        check_error("OutputLineList", nlines, atomic, type=("int", "double"))
        return atomic

    def UpdateLineList(self, atomic, species, index):
        """
        Change line list parameters

        Parameters
        ---------
        atomic : array of size (nlines, 8)
            atomic linelist data for each line
            fields are: atom_number, ionization, wlcent, excit, gflog, gamrad, gamqst, gamvw
        species : array(string) of size (nlines,)
            names of the elements (with Ionization level)
        index : array(int) of size (nlines,)
            indices of the lines to update relative to the overall linelist
        """
        nlines = atomic.shape[0]
        assert (
            atomic.shape[1] == 8
        ), f"Got wrong Linelist shape, expected ({nlines}, 8) but got {atomic.shape}"

        assert (
            len(index) == nlines
        ), "Inconsistent number if lines, between index and linelist"
        assert (
            len(species) == nlines
        ), "Inconsistent number if lines, between index and linelist"

        atomic = atomic.T

        check_error(
            "UpdateLineList",
            nlines,
            species,
            atomic,
            index,
            type=("int", "str", "double", "short"),
        )

    def InputModel(self, teff, grav, vturb, atmo):
        """ Read in model atmosphere

        Parameters
        ---------
        teff : float
            effective Temperature in Kelvin
        grav : float
            surface gravity in log10(cgs)
        vturb : float
            turbulence velocity in km/s
        atmo : Atmo
            atmosphere structure (see Atmo for details)
        """
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
            "opflag",
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]),
        )
        args = [ndepth, teff, grav, wlstd, motype, opflag, depth, t, xne, xna, rho, vt]
        type = "sdddusdddddd"  # s : short, d: double, u: unicode (string)

        if atmo.geom == "SPH":
            radius = atmo.radius
            height = atmo.height
            motype = "SPH"
            args = args[:5] + [radius] + args[5:] + [height]
            type = type[:5] + "d" + type[5:] + "d"

        check_error("InputModel", *args, type=type)

        self.teff = teff
        self.grav = grav
        self.vturb = vturb
        self.atmo = atmo

    def InputAbund(self, abund):
        """
        Pass abundances to radiative transfer code.

        Calculate elemental abundances from abundance pattern and metallicity.
        Metallicity adjustment is not applied to H or He.
        Renormalize abundances after applying metallicity.
        Introduced limiter in case the proposed step in abundance is too large.

        Parameters
        ---------
        abund : Abund
            abundance structure to be passed (see Abund for more details)
        """
        # Convert abundances to the right format
        # metallicity is included in the abundance class, ignored in function call
        abund = abund("sme", raw=True)
        check_error("InputAbund", abund, type="double")

        self.abund = abund

    def Opacity(self, getData=False, motype=1):
        """ Calculate opacities

        Parameters
        ---------
        getData : bool
            if True copblu and copred (and copstd) will be returned
            requires that radiative transfer was run
        motype : int
            if getData is True and motype is 0 then copstd will also be returned

        Returns
        -------
        copblu : array of size (nmu,)
            only if getData is True
        copred : array of size (nmu,)
            only if getData is True
        copstd : array of size (nmu,)
            only if getData is True and motype is 0
        """
        args = []
        type = ""
        if getData:
            nmu = self.nmu
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

    def GetOpacity(self, switch, species=None, key=None):
        """
        Returns specific cont. opacity

        Parameters
        ----------
        switch : int
            | -3: COPSTD, -2: COPRED, -1: COPBLU, 0: AHYD,
            | 1: AH2P, 2: AHMIN, 3: SIGH, 4: AHE1, 5: AHE2,
            | 6: AHEMIN, 7: SIGHE,
            | 8: ACOOL, continuous opacity C1, Mg1, Al1, Si1, Fe1, CH, NH, OH,
            | 9: ALUKE, continuous opacity N1, O1, Mg2, Si2, Ca2,
            | 10: AHOT, 11: SIGEL, 12: SIGH2
        """
        length = self.nmu
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

    def Ionization(self, ion=0):
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

        Parameters
        ----------
        ion : int
            flag that determines the behaviour of the C function
        """
        error = idl_call_external("Ionization", ion, type="short")
        if error != b"":
            warnings.warn(f"{__name__} (call external): {error.decode()}")

        self.ion = ion

    def GetDensity(self):
        """
        Retrieve density in each layer

        Returns
        -------
        density : array of size (ndepth,)
            Density of the atmosphere in each layer
        """
        length = self.ndepth
        array = np.zeros(length, dtype=float)
        check_error("GetDensity", length, array, type="sd")
        return array

    def GetNatom(self):
        """
        Get XNA

        Returns
        -------
        XNA : array of size (ndepth,)
            XNA in each layer
        """
        length = self.ndepth
        array = np.zeros(length, dtype=float)
        check_error("GetNatom", length, array, type="sd")
        return array

    def GetNelec(self):
        """
        Get XNE

        Returns
        -------
        XNE : array of size (ndepth,)
            XNE in each layer
        """
        length = self.ndepth
        array = np.zeros(length, dtype=float)
        check_error("GetNelec", length, array, type="sd")
        return array

    def Transf(
        self,
        mu,
        accrt,
        accwi,
        keep_lineop=False,
        long_continuum=True,
        nwmax=400000,
        wave=None,
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

        self.nmu = nmu

        return nw, wint_seg, sint_seg, cint_seg

    def CentralDepth(self, mu, accrt):
        """
        This subroutine explicitly solves the transfer equation
        for a set of nodes on the star disk in the centers of spectral
        lines. The results are specific intensities.

        Parameters
        ----------
        mu : array of size (nmu,)
            mu values along the stellar disk to calculate
        accrt : float
            precision of the radiative transfer calculation

        Returns
        -------
        table : array of size (nlines,)
            Centeral depth (i.e. specific intensity) of each line
        """

        nmu = np.size(mu)
        nwsize = self.nlines
        table = np.zeros(nwsize)

        check_error("CentralDepth", nmu, mu, nwsize, table, accrt, type="idifd")

        self.nmu = nmu

        return table

    def GetLineOpacity(self, wave):
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
        nmu = self.nmu
        lop = np.zeros(nmu)
        cop = np.zeros(nmu)
        scr = np.zeros(nmu)
        tsf = np.zeros(nmu)
        csf = np.zeros(nmu)
        type = "dsddddd"
        check_error("GetLineOpacity", wave, nmu, lop, cop, scr, tsf, csf, type=type)
        return lop, cop, scr, tsf, csf

    def GetLineRange(self):
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
        nlines = self.nlines
        linerange = np.zeros((nlines, 2))

        check_error("GetLineRange", linerange, nlines, type=("double", "int"))

        return linerange

    def InputNLTE(self, bmat, lineindex):
        """
        Input NLTE departure coefficients

        Parameters
        ----------
        bmat : array of size (2, ndepth,)
            departure coefficient matrix
        lineindex : float
            index of the line in the linelist
        """
        ndepth = self.ndepth
        nlines = self.nlines
        assert (
            bmat.shape[0] == 2
        ), f"Departure coefficient matrix has the wrong shape, expected (2, {ndepth}) but got {bmat.shape} instead"
        assert (
            bmat.shape[1] == ndepth
        ), f"Departure coefficient matrix has the wrong shape, expected (2, {ndepth}) but got {bmat.shape} instead"

        assert (
            0 <= lineindex < nlines
        ), f"Lineindex out of range, expected value between 0 and {nlines}, but got {lineindex} instead"

        check_error(
            "InputDepartureCoefficients", bmat, lineindex, type=("double", "int")
        )

    def GetNLTE(self, line):
        """ Get the NLTE departure coefficients as stored in the C library

        Parameters
        ----------
        line : int
            requested line index, i.e. between 0 and number of lines

        Returns
        -------
        bmat : array of size (2, nrhox)
            departure coefficients for the given line index
        """
        nrhox = self.ndepth

        bmat = np.full((2, nrhox), -1., dtype=float)
        check_error(
            "GetDepartureCoefficients", bmat, nrhox, line, type=("double", "int", "int")
        )
        return bmat

    def ResetNLTE(self):
        """ Reset departure coefficients from any previous call, to ensure LTE as default """
        check_error("ResetDepartureCoefficients")

    def GetNLTEflags(self):
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
        nlines = self.nlines
        nlte_flags = np.zeros(nlines, dtype=np.int16)

        check_error("GetNLTEflags", nlte_flags, nlines, type=("short", "int"))

        return nlte_flags.astype(bool)
