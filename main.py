import src.sme.sme as sme
import src.sme.abund as abund

from scipy.io import readsav
import numpy as np
import os.path

from awlib.sme import sme as SME
from src.sme.cwrapper import idl_call_external
from src.sme import sme_synth

from rtint import rtint

from src.sme.interp_atmo_grid import interp_atmo_grid

def pass_nlte(sme):
    nlines = len(sme.species)
    ndep = len(sme.atmo.temp)
    b_nlte = np.ones((ndep, nlines, 2))  # initialize the departure coefficient array
    modname = os.path.basename(sme.atm_file[0] + ".krz")
    poppath = os.path.dirname(sme.atm_file[0])
    for iline in range(nlines):
        bnlte = rdpop(
            sme.species[iline],
            sme.atomic[2, iline],
            sme.atomic[3, iline],
            modname,
            POP_DIR=poppath,
        )
        if len(bnlte) == 2 * ndep:
            b_nlte[:, iline, :] = bnlte

    error = idl_call_external("InputNLTE", b_nlte)
    if error != b"":
        raise ValueError(
            "InputDepartureCoefficients (call_external): %s" % error.decode()
        )
    return error


def sme_func_atmo(sme):
    """
    Purpose:
     Return an atmosphere based on specification in an SME structure

    Inputs:
     SME (structure) atmosphere specification

    Outputs:
     ATMO (structure) atmosphere structure
     [.WLSTD] (scalar) wavelength for continuum optical depth scale [A]
     [.TAU] (vector[ndep]) optical depth scale,
     [.RHOX] (vector[ndep]) mass column scale
      .TEMP (vector[ndep]) temperature vs. depth
      .XNE (vector[ndep]) electron number density vs. depth
      .XNA (vector[ndep]) atomic number density vs. depth
      .RHO (vector[ndep]) mass density vs. depth

    History:
     2013-Sep-23 Valenti Extracted and adapted from sme_func.pro
     2013-Dec-13 Valenti Bundle atmosphere variables in ATMO structure
    """

    # Static storage
    #   common common_sme_func_atmo, prev_msdi

    # Handle atmosphere grid or user routine.
    atmo = sme.atmo

    if atmo.method == "grid":
        # reload = msdi_save and atmo.source != prev_msdi[1]
        reload = True 
        interp_atmo_grid(sme.teff, sme.grav, sme.feh, sme.atmo, atmo, reload=reload)
        prev_msdi = [atmo.method, atmo.source, atmo.depth, atmo.interp]
    elif atmo.method == "routine":
        atmo.source(sme, atmo)
    elif atmo.method == "embedded":
        # atmo structure already extracted in sme_main
        pass
    else:
        raise AttributeError("Source must be 'grid', 'routine', or 'file'")

    return sme


def sme_func(
    sme,
    only_seg=None,
    file=None,
    no_atmo=False,
    save=False,
    check=False,
    same_wl_grid=False,
):
    """
    Calculate an output synthetic spectrum for specified input parameters.

    Parameters
    -------
    sme : SME_struct
        input parameters and output spectrum

    no_atmo : bool
        reuses the most recent stellar atmosphere, abundances,
        spectral line data, gamma_6 enhancement, chemical equilibrium, and
        ionization equilibrium. These data are stored in the external library,
        not in IDL variables local to sme_func. Calculation of these data is
        relatively expensive.
        Sme_solve uses /noatmo when calculating partial derivatives of intensity
        with respect to spectral line parameters, unless the preceding parameter
        was not a line parameter. In the latter case, the quantities above must
        be reloaded.

    save : bool
        save the calculated intensity spectrum in a local common
        block for possible reuse in a later call (see /check). Reuse is faster
        than a new calculation. Sme_solve uses /save upon entry and at the end
        of each major iteration (which is the starting point for the next major
        iteration).

    check : bool
        reuse the intensity spectrum saved in a local common
        block (see /save), if the previous /save call and the current call
        have identical values for the following global parameters:
        sme.teff, sme.grav, sme.feh, sme.vmic, sme.gam6.
        The calling routine must verify that the previous /save call and the
        current call use identical abundances and line data.
        Calls with and without /check may be interspersed after a /save.
        The /check option may be used multiple times for one /save.
        Sme_solve uses /check when calculating partial derivatives for global
        parameters. In this case, sme_func reuses saved intensities for vmac
        and vsini, but not for the five parameters listed above, which affect
        the intensities.
        Sme_solve also uses /check when evaluating new parameters proposed by
        the Marquardt solver. In this case, sme_func reuses saved intensities
        if vmac and/or vsini are the only free parameters.

    same_wl_grid : bool
        reuse the most recent wavelength set for the
        intensities. No preceding "save" is required because wavelengths from
        the preceding call are stored in a common block and remain available
        for reuse.
        Sme_solve uses /same_wl_grid when calculating partial derivatives of
        intensity with respect to the free parameters. Sme_solve computes the
        initial intensities without /same_wl_grid and then perturbed intensities
        with /same_wl_grid to obtain more precise partial derivatives.

    onlyseg : int
        restricts calculation to one spectral segment.
        Sme_solve uses onlyseg= when calculating partial derivatives of the
        intensity with respect to spectral line parameters, which are assumed
        to affect only one wavelength interval. [What about broad lines or
        overlapping wavelength segments?]

    Returns
    --------
    sme : SME_struct
        same object as input sme structure, but now updated with results

    Notes
    -------
    Calls dynamically-linked external module.

    History
    ---------
    13-Jun-95 JAV Adapted from piskf.pro
    12-May-97 JAV Changed sme.res to sme.ipres in IP broadening section.
    26-Mar-02 JAV Update sme.feh, if solving for abundances of more than 3
                individual elements and if NOT solving for global FEH.
                Disable log(Fe/H) interpretation of sme.feh.
    06-Apr-02 JAV Added support for passing [M/H] to interpkrz2.
    06-Jun-02 JAV Tied Vmac to current value of Teff if Vmac eq -99
    03-Jan-03 JAV Tied [Mg/H] to mean of [Si/H] and [Ti/H]
    2010-May-12 Valenti  Disallow sme.vmac=-99. Allow sme.vmac to be a vector
                        of coefficients that specify vmac as function of Teff.
    2010-May-13 Valenti  Restore old behavior when sme.vmac eq -99. Add logic
                        to handle new sme.vmac_coef structure tag.
    2011-Jun-25 Valenti Make long_continuum=1 the default. Created sme.cmod.
    2012-Mar-08 Valenti Major editing to remove redundant arrays and rename
                        remaining arrays more clearly. Added many comments.
                        Renormalize abundances for current metallicity. 
                        Require atmosphere routines to return argument list.
                        Allow user-specified macroturbulence procedure.
                        Extend radiative transfer beyond segment boundary.
                        Divide model by continuum flux before fitting data.
    2012-Nov-30 TN Update to use the latest atmospheric grids krz3 and marcs2012,
                which use either rhox (column mass depth) or tau (reference
                optical depth) as the independent atmospheric depth variable.
    2013-Apr-29 TN krz3 structure renamed to generic atmo_grid.
    2013-Dec-13 Valenti Bundled atmosphere variables into ATMO structure.
    2013-Dec-27 Valenti Allow sme.ipres to be scalar or have one element per
                        segment.
    2014-Feb-26 Piskunov Modified intermediate wavelength grid from equispaced
                in wavelength to equispaced in velocity to allow correct
                handling of long wavelength ranges
    2015-Feb-13 Valenti Add segment endpoints to output wavelength scale,
                when the user does not provide an observed wavelength scale.
                This guarantees at least two output wavelength points.

    *** NOTES ABOUT VARIABLE NAMES IN THIS PROCEDURE ***
    -----------------------------

    Variables that include _INT contain data (e.g. "intensity spectra") sampled
    on the adaptive wavelength grid created by the external module, e.g. WINT,
    SINT, CINT, JINT.

    Variables ending with _SEG contain data for one spectral "segment", e.g.
    VRAD_SEG, VFACT_SEG, WINT_SEG, SINT_SEG, CINT_SEG, WAVE_SEG, SOB_SEG,
    UOB_SEG, MOB_SEG, SMOD_SEG, CMOD_SEG.

    Analogs of _SEG variables without the _SEG suffix contain analogous data
    for all segments, e.g. WINT, SINT, CINT, WAVE, SME.VRAD, SME.SOB, SME.UOB,
    SME.MOB, SMOD, CMOD. These variables are generally used to communicate data
    to the calling routine (sme_main and/or sme_solve).

    Variables ending with _SAVE contain output from the radiative transfer that
    is "saved" in a common blocks for possible reuse in subsequent calls, e.g.
    RTPAR_SAVE, WINT_SAVE, SINT_SAVE, CINT_SAVE, JINT_SAVE.

    Variables beginning with X or Y contain data interpolated to a uniform
    wavelength grid that facilitates convolutions, e.g. X_SEG, Y_SEG, YC_SEG,
    YI_SEG. """

    clight = 2.997925e5  # km/s

    tags = sme.names
    f_opro = "sob" in tags
    f_wave = "wave" in tags
    f_NLTE = False

    #*** LOAD NEW ATMOSPHERE AND/OR LINE DATA ***

    # If sme_func is called with the /noatmo switch set, then the following
    # block of code will be skipped. This means the external module will
    # reuse the most recent atmosphere, abundances, ionization equilibrium
    # from the equation of state, and the collisional damping parameter
    # enhancement factor (sme.gam6).
    #
    # If the /noatmo switch is not set, then the following code block will:
    # [1] get an atmosphere from a user-specified procedure or explicitly
    # from the sme structure, [2] pass the atmosphere to the external module,
    # [3] pass abundances modified by the current value of sme.feh to the
    # external module, [4] instruct the external module to recompute chemical
    # equilibrium and ionization balance, and [5] pass the current value of
    # sme.gam6 to the external module.

    if not no_atmo:
        sme = sme_func_atmo(sme)
        sme_synth.InputModel(sme.teff, sme.grav, sme.vmic, sme.atmo)
        sme_synth.InputLineList(sme.atomic, sme.species)
        # Compile the table of departure coefficients if NLTE flag is set
        if f_NLTE and "atmo_pro" in tags:
            pass_nlte(sme)

        sme_synth.InputAbund(sme.abund, sme.feh)
        sme_synth.Ionization()
        sme_synth.SetVWscale(sme.gam6)
        sme_synth.SetH2broad(sme.h2broad)

    # TODO sme_update_depcoeff

    jint = np.empty(sme.nseg)
    wint = np.empty(sme.nseg)
    wave = []
    smod = []
    cmod = []
    if os.path.exists("rtpar_save.dat"):
        rtpar_save = np.load("rtpar_save.dat")
        check = 1
    else:
        check = 0

    firstseg = 0
    lastseg = sme.nseg - 1

    nmu = np.array(len(sme.mu), dtype=np.int16)
    vmac = sme.vmac

    callrt = 1  # default is to call radiative transfer
    if check:
        rtpar = [sme.teff, sme.grav, sme.feh, sme.vmic, sme.gam6]
        if max(abs(rtpar_save - rtpar)) == 0:
            callrt = 0

    for il in range(firstseg, lastseg + 1):

        # *** CALCULATE OR REUSE INTENSITIES FOR CURRENT SEGMENT ***
        # If not reusing the saved intensities, then prepare to calculate intensities.

        if callrt:
            # Calculate shift and padding of wavelength range for current segment. The
            # radial velocity shift may be a global scalar or one scalar per segment.
            # The padding includes 30 km/s to account for the maximum barycentric shift
            # and vsini/2 to handle the wings of rotationally broadened lines that just
            # touch the nominal segment boundary.

            vrad_seg = sme.vrad if sme.vrad.ndim == 0 else sme.vrad[il]  # km/s
            vfact_seg = np.sqrt(
                (1 + vrad_seg / clight) / (1 - vrad_seg / clight)
            )  # factor
            vrad_pad = 30.0 + 0.5 * np.clip(sme.vsini, 0, None)  # km/s
            vbeg = vrad_pad + np.clip(vrad_seg, 0, None)  # km/s
            vend = vrad_pad - np.clip(vrad_seg, None, 0)  # km/s

            # Pass wavelenth range for current segment to external module, including
            # the shift and padding just calculated. The external module calculates
            # intensities in the laboratory frame because the line data in sme.atomic
            # are in the laboratory frame. Thus, we shift wavelengths in sme.wran from
            # the observed frame to the laboratory frame by applying an inverse velocity
            # shift (use the negative of vrad_seg).

            wran_seg = sme.wran[il]
            wbeg = wran_seg[0] * (1 - vbeg / clight)
            wend = wran_seg[1] * (1 + vend / clight)
            sme_synth.InputWaveRange(wbeg, wend)

            # Call external module to compute continuous opacities for current segment.
            # The external module has a GetOpacity entry point for reading opacities.
            sme_synth.Opacity()

            # Use previous wavelength grid for intensities, if SAME_WL_GRID=1. Previous
            # wavelengths should still be available in common block used to communicate
            # results to sme_solve and sme_main. Update NWMAX so that intensity arrays
            # will be allocated (immediately below) with the exact number of points
            # required for the current segment.

            if same_wl_grid:
                jbeg = 0 if il == 0 else 1 + jint[il - 1]  # first point in segment
                jend = jint[il]  # last point in segment
                nw = jend - jbeg + 1  # number of points in seg
                nwmax = nw  # size of intensity arrays
                wint_seg = wint[jbeg : jend + 1]  # use previous wavelengths
            else:
                nw = 0  # flag value -> new wavelengths
                wint_seg = np.array(nwmax)  # allocate new wavelengths

            # Allocate *_SEG arrays used by the external module to return intensities
            # for the current segment. The *_SEG variables change dimensionality and
            # value each time through the segment loop.
            sint_seg = np.zeros((nmu, nwmax))  # line+continuum intensities
            cint_seg = np.zeros((nmu, nwmax))  # all continuum intensities
            cintr_seg = np.zeros((nmu))  # red continuum intensity

            # Set flag that controls calculation of line center opacity for every line
            # in the line list. Calculate line center opacities for the first segment.
            # No need to recalculate opacities for later segments because one call
            # handles all lines for all segments.
            #
            #  KEEP_LINEOP=0, need to calculate line center opacities (first segment)
            #  KEEP_LINEOP=1, use existing line center opacities (subsequent segments)

            keep_lineop = 0 if il == firstseg else 1

            # Flag that controls continuum intensities returned by external module. Older
            # versions of the external module did not have the LONG_CONTINUUM flag and
            # returned continuum intensities only at the first and last wavelength.
            #
            #  LONG_CONTINUUM=0, continuum intensities at first and last wavelength
            #  LONG_CONTINUUM=1, continuum intensities at every wavelength [default]

            long_continuum = 1  # continuum at every wavelength

            # Enforce data type of output arrays used to communicate with external module.
            # It is worth checking here because incorrect type can cause cryptic errors.

            # if not isinstance(wint_seg) != 'DOUBLE' then message, 'make wint_seg double'
            # if size(sint_seg) != 'DOUBLE' then message, 'make sint_seg double'
            # if size(cint_seg) != 'DOUBLE' then message, 'make cint_seg double'
            # if size(cintr_seg) != 'DOUBLE' then message, 'make cintr_seg double'
            # if size(nw      ) != 'LONG'   then message, 'make nw long'

            # Call external module to calculate intensities for the current segment.
            # Enforce data type of input-only parameters to avoid cryptic errors, i.e.
            # wrap each input argument in fix(), long(), and double() functions.
            #
            # If wavelengths are being reused (CALLRT=0), then NW and WINT_SEG are inputs
            # to the external module. NW is the number of valid wavelengths in WINT_SEG.
            # If new wavelengths are being calculated, then the input value of NW is 0L
            # and the output value is the number of valid wavelength returned in WINT_SEG.
            #
            # Below, CINT_SEG[*,NW-1] is used instead of CINTR_SEG. The two are equal.

            sme_synth.Transf(
                nmu,
                sme.mu,
                cint_seg,
                cintr_seg,
                nwmax,
                nw,
                wint_seg,
                sint_seg,
                sme.accrt,
                sme.accwi,
                keep_lineop,
                long_continuum,
            )

            # Transpose and trim the intensity arrays. Only points 0 to NW-1 are valid.
            # External module returns arrays that are (nmu,nw) while SME expects (nw,nmu).

            wint_seg = wint_seg[:nw]
            sint_seg = sint_seg[:, :nw].T
            cint_seg = cint_seg[:, :nw].T

            # If these are new wavelengths (SAME_WL_GRID=0), then store boundaries and
            # wavelengths for current segment in arrays that contain all segments. Always
            # store intensities for current segment in arrays that contain all segments.
            # These arrays (JINT, WINT, SINT, CINT) are shared with sme_solve and sme_main,
            # so they should persist unchanged between calls to sme_func.

            if il == firstseg:
                if same_wl_grid is None:
                    jint = nw - 1
                    wint = wint_seg
                sint = sint_seg
                cint = cint_seg
            else:
                if same_wl_grid is None:
                    jint += [max(jint) + nw]
                    wint += [wint_seg]
                sint += [sint_seg]
                cint += [cint_seg]

            # For compatibility with previous versions of SME, return continuum intensity
            # at the segment boundaries defined in sme.wran. The wavelength range for
            # intensities (WINT_SEG) extends beyond sme.wran, so we need to interpolate.

            for imu in range(nmu):
                cint_at_wran = np.interp(cint_seg[imu, :], wint_seg, wran_seg)
                sme.cintb[il, imu] = cint_at_wran[0]
                sme.cintr[il, imu] = cint_at_wran[1]

        # If reusing saved intensities (CALLRT=0), copy intensities from save arrays.
        else:
            jbeg_save = 0 if il == 0 else 1 + np.load("jint_save")[il - 1]
            jend_save = np.load("jint_save")[il]
            nw = jend_save - jbeg_save + 1
            wint_seg = np.load("wint_save")[jbeg_save : jend_save + 1]
            sint_seg = np.load("sint_save")[:, jbeg_save : jend_save + 1]
            cint_seg = np.load("cint_save")[:, jbeg_save : jend_save + 1]

        # *** CALCULATE FLUX SPECTRUM FROM INTENSITIES ***

        # Calculate the continuum flux spectrum from the continuum intensities
        # on the adaptive wavelength grid used by the external module.

        # TODO
        cflx_seg = rtint(sme.mu, cint_seg, 1, 0, 0)

        # ***** NP: SWITCHED FROM WAVELENGTH TO VELOCITY SCALE FOR USING rtint ******

        ##Determine step size for a new model wavelength scale, which must be uniform
        ##to facilitate convolution with broadening kernels. The uniform step size
        ##is the larger of:
        ##
        ## [1] smallest wavelength step in WINT_SEG, which has variable step size
        ## [2] 10% the mean dispersion of WINT_SEG
        ## [3] 0.05 km/s, which is 1% the width of solar line profiles
        ##
        #    wmid = 0.5 * (wint_seg[nw-1] + wint_seg[0])          #midpoint of segment
        #    wspan = wint_seg[nw-1] - wint_seg[0]                 #width of segment
        #    xstep1 = min(wint_seg[1:nw-1] - wint_seg[0:nw-2])    #smallest step
        #    xstep2 = 0.1 * wspan / (nw-1)                        #10% mean dispersion
        #    xstep3 = wmid * (0.05 / clight)                      #0.05 km/s step
        #    xstep = xstep1 > xstep2 > xstep3                     #select the largest
        #
        ##Generate model wavelength scale X, with uniform wavelength step.
        ##
        #    nx = (round(wspan / xstep) + 1) < 10L * nw           #number of wavelengths
        #    if nx mod 2 eq 0 then nx += 1                        #force nx to be odd
        #    xstep = wspan / (nx - 1)                             #uniform dispersion
        #    x_seg = wint_seg[0] + wspan * dindgen(nx) / (nx - 1) #new wavelength scale

        # ***** NP: HERE IS THE VELOCITY VERSION ******

        # Determine step size for a new model wavelength scale, which must be uniform
        # in velocity to facilitate convolution with broadening kernels. The uniform
        # step size is the largest of:
        #
        # [1] smallest wavelength step in the adaptiv grid WINT_SEG constructed by RT
        # [2] 10% the mean dispersion of WINT_SEG
        # [3] 0.05 km/s, which is 1% the width of solar line profiles
        #
        wmid = 0.5 * (wint_seg[nw - 1] + wint_seg[0])  # midpoint of segment
        wspan = wint_seg[nw - 1] - wint_seg[0]  # width of segment
        jmin = np.argmin(wint_seg[1:nw] - wint_seg[: nw - 1])
        vstep1 = (wint_seg[1:nw] - wint_seg[: nw - 1])[jmin]
        vstep1 = vstep1 / wint_seg[jmin] * clight  # smallest step
        vstep2 = 0.1 * wspan / (nw - 1) / wmid * clight  # 10% mean dispersion
        vstep3 = 0.05  # 0.05 km/s step
        vstep = max(vstep1, vstep2, vstep3)  # select the largest

        # Generate model wavelength scale X, with uniform wavelength step.
        #
        nx = (
            np.floor(
                np.log10(wint_seg[nw - 1] / wint_seg[0]) / np.log10(1 + vstep / clight)
            )
            + 1
        )  # number of wavelengths
        if nx % 2 == 0:
            nx += 1  # force nx to be odd
        resol_out = 1 / ((wint_seg[nw - 1] / wint_seg[0]) ** (1 / (nx - 1)) - 1)
        vstep = clight / resol_out
        x_seg = wint_seg[0] * (1 + 1 / resol_out) ** np.arange(
            nx
        )  # new wavelength scale

        # Interpolate intensity spectra onto new model wavelength scale.
        yi_seg = np.empty((nx, nmu))
        for imu in range(nmu):
            yi_seg[imu] = np.interp(sint_seg[imu], wint_seg, x_seg)

        # Interpolate continuum flux spectrum onto new model wavelength scale.
        yc_seg = np.interp(cflx_seg, wint_seg, x_seg)

        # Apply macroturbulent and rotational broadening while integrating intensities
        # over the stellar disk to produce flux spectrum Y. Use absolute value as an
        # imperfect way to preventing unphysical negative broadening parameters.

        y_seg = rtint(sme.mu, yi_seg, vstep, abs(sme.vsini), abs(vmac))

        # Apply instrumental broadening.

        iwhr = sme.tags == "IPTYPE"
        f_ipro = np.count_nonzero(iwhr)
        if f_ipro:
            nipres = len(sme.ipres)
            if nipres != 1 and nipres != sme.nseg:
                raise ValueError(
                    "sme.ipres must be a scalar or have 1 element per segment"
                )
            ipres = sme.ipres[0] if nipres == 1 else sme.ipres[il]

            # Using the log-linear wavelength grid requres using the first point
            # for specifying the the width of the instrumental profile
            hwhm = 0.5 * x_seg[0] / ipres if ipres > 0 else 0

            if sme.iptype == "table":
                y = tablebroad(x_seg, y_seg, sme.ip_x, sme.ip_y)
            elif sme.iptype == "gauss":
                if hwhm > 0:
                    y_seg = gaussbroad(x_seg, y_seg, hwhm)
            elif sme.iptype == "sinc":
                if hwhm > 0:
                    y_seg = sincbroad(x_seg, y_seg, hwhm)
            else:
                raise AttributeError("Unknown IP type - " + sme.iptype)

        # Define final wavelength scale for current segment. Use observed wavelengths
        # passed in by common block, if defined. Otherwise, use the model wavelength
        # scale with uniform wavelength steps, trimmed to requested wavelength range,
        # ensuring that segment endpoints are included.

        if f_wave:  # wavelengths already defined
            ibeg = (
                0 if il == 0 else sme.wind[il - 1] + 1
            )  # first pixel in current segment
            iend = sme.wind[il]  # last pixel in current segment
            wave_seg = wave[ibeg : iend + 1]  # wavelengths for current segment
        else:  # else must build wavelengths
            itrim = (x_seg > wran_seg[0]) & (x_seg < wran_seg[1])  # trim padding
            wave_seg = [wran_seg[0], x_seg[itrim], wran_seg[1]]  # use model scale

        # *** DETERMINE RADIAL VELOCITY AND CONTINUUM SCALING,  ***
        # *** WHILE BINNING MODEL SPECTRUM INTO OBSERVED PIXELS ***

        # Extract observed spectrum, uncertainties, and mask for the current segment.
        # Having an observation (F_OPRO=1) implies having wavelengths (F_WAVE=1), so
        # IBEG and IEND should already be defined if needed.
        if f_opro:
            sob_seg = sme.sob[ibeg : iend + 1]  # observed spectrum
            uob_seg = sme.uob[ibeg : iend + 1]  # associated uncertainties
            mob_seg = sme.mob[ibeg : iend + 1]  # ignore/line/cont mask
            clim = sme.clim  # continuum threshold

        # Extract flag and value that specifies radial velocity.
        #
        #  VALUE  IMPLICATION
        #  -2     Do not solve for radial velocity. Use input value(s).
        #  -1     Determine global radial velocity that applies to all segments
        #   0     Determine a separate radial velocity for each spectral segment
        #
        # Can't solve for radial velocities if there is no observation.
        # Express radial velocities as dimensionless wavelength scale factor.
        # Formula includes special relativity, though correction is negligible.

        vrad_flag = sme.vrad_flag if f_opro else -2
        fixv = (vrad_flag < 0) or same_wl_grid

        vrad_seg = sme.vrad if sme.vrad.ndim == 0 else sme.vrad[il]  # km/s
        vfact_seg = np.sqrt((1 + vrad_seg / clight) / (1 - vrad_seg / clight))

        # Extract flag and value that specifies continuum normalization.
        #
        #  VALUE  IMPLICATION
        #  -3     Return residual intensity. Continuum is unity. Ignore sme.cscale
        #  -2     Return physical flux at stellar surface (units? erg/s/cm^2/A?)
        #  -1     Determine one scalar normalization that applies to all segments
        #   0     Determine separate scalar normalization for each spectral segment
        #   1     Determine separate linear normalization for each spectral segment
        #
        # Don't solve for single scalar normalization (-1) if there is no observation
        # CSCALE_FLAG is polynomial degree of continuum scaling, when fitting segments.

        cscale_flag = sme.cscale_flag
        if not f_opro and cscale_flag >= -1:
            cscale_flag = -3
        fixc = (cscale_flag < 0) or same_wl_grid

        if cscale_flag == -3:
            cscale = 1
        elif cscale_flag in [-2, -1]:
            cscale = sme.cscale
        elif cscale_flag == 0:
            cscale = sme.cscale[il]
        elif cscale_flag == 1:
            cscale = sme.cscale[il, :]
        else:
            raise AttributeError("invalid cscale_flag: %i" % cscale_flag)
        if cscale_flag >= 0:
            ndeg = cscale_flag

        # Divide the model flux spectrum by the continuum flux spectrum to produce
        # residual intensity, except when fitting observations with a global scale
        # factor (CSCALE_FLAG=-2), which usually implies observations have physical
        # units.

        if cscale_flag != -2:
            y_seg /= yc_seg

        # Handle cases where radial velocity and continuum scaling are either ignored,
        # fixed values, or free parameters determined at a higher level by sme_solve.
        # For all of these cases, we apply the radial velocity shift here. Continuum
        # scaling is applied only if it is a global free parameter (CSCALE_FLAG=-1).

        if vrad_flag < 0 and cscale_flag < 0:
            smod_seg = resamp(x_seg * vfact_seg, y_seg, wave_seg)
            if cscale_flag == -1:
                smod_seg /= cscale

            # Handle cases where radial velocity and/or continuum scaling must be
            # determined for the current segment by fitting the observed spectrum.

        else:
            if f_opro:
                smod_seg = sme_crvmatch(
                    x_seg,
                    y_seg,
                    wave_seg,
                    sob_seg,
                    uob_seg,
                    mob_seg,
                    ndeg,
                    clim,
                    wmid,
                    vrad_seg,
                    cscale,
                    fixv=fixv,
                    fixc=fixc,
                )

            # Handle cases where there is no observed spectrum.
            else:
                cscale = 1 if cscale_flag == 0 else [1, 1]

        # Update radial velocity, continuum scaling, and mask for current segment.

        if vrad_flag == 0:
            sme.vrad[il] = vrad_seg
        if cscale_flag == 0:
            sme.cscale[il] = cscale
        if cscale_flag == 1:
            sme.cscale[il, :] = cscale
        if f_opro and cscale_flag >= 0:
            sme.mob[ibeg : iend + 1] = mob_seg

        # Bin continuum flux onto observed wavelengths.
        cmod_seg = resamp(x_seg * vfact_seg, yc_seg, wave_seg)

        # *** SAVE RESULTS ***

        # Insert model spectrum for current segment into existing array that contains
        # all segments. Clear array before inserting spectrum data for first segment.

        if f_wave:
            if il == 0:
                smod[:] = 1
            smod[ibeg : iend + 1] = smod_seg
            cmod[ibeg : iend + 1] = cmod_seg

        # Build arrays containing wavelengths, model spectrum, and segment boundaries
        # for all segments, when there is no observed spectrum. Include data for the
        # current segment.
        else:
            wave += [wave_seg]
            smod += [smod_seg]
            cmod += [cmod_seg]
            sme.wind[il] = len(smod) - 1  # last pixel of current segment

    # *** DONE LOOPING THROUGH SPECTRAL SEGMENTS ***

    # Save results of synthesis, if SAVE=1 was specified.

    if save and callrt:
        rtpar_save = [sme.teff, sme.grav, sme.feh, sme.vmic, sme.gam6]
        np.save("rtpar_save.dat", rtpar_save)
        wint_save = wint
        sint_save = sint
        cint_save = cint
        jint_save = jint

    # Write diagnostic output to disk file, if requested.
    if file is not None:
        with open(file, "a") as f:
            f.write("%i %i %i" % (sme.nseg, len(sme.cintb), len(wint)))
            f.write(
                "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f"
                % (
                    sme.teff,
                    sme.grav,
                    sme.feh,
                    sme.vmic,
                    sme.cintb,
                    sme.cintr,
                    jint,
                    wint,
                    sint,
                )
            )

    pass


in_file = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
sme = SME.read(in_file)

file = None
no_atmo = True
save = True
check = False
same_wl_grid = False



sme_synth.SetLibraryPath()

sme_func(sme, no_atmo=True, save=True, check=False, same_wl_grid=False)