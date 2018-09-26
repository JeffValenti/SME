import os
import numpy as np
from scipy.io import readsav

from interp_atmo_pair import interp_atmo_pair


def interp_atmo_grid(atmo_in, Teff, logg, MonH, trace=0, plot=False, reload=False):
    # General routine to interpolate in 3D grid of model atmospheres
    #
    # Inputs:
    # Teff (scalar) effective temperature of desired model (K).
    # logg (scalar) logarithmic gravity of desired model (log cm/s/s).
    # MonH (scalar) metalicity of desired model.
    #
    # Outputs:
    # atmo= (structure) interpolated atmosphere data in structure format.
    #  .teff (scalar) effective temperature of model (in K)
    #  .logg (scalar) logarithm of surface gravity (in cm/s^2)
    #  .monh (scalar) metalicity, i.e. logarithm of abundances, relative to solar
    #  .vturb (scalar) turbulent velocity assumed in model (in km/s)
    #  .lonh (mixing length parameter relative to pressure scale height
    #  .wlstd (scalar) wavelength at which continuum TAU is computed, if present
    #  .modtyp (scalar) model type: 0='RHOX', 1='TAU'
    #  .opflag (vector(20)) opacity flags: 0=OFF, 1=ON
    #  .abund (vector(99)) abundance relative to total atomic nuclei
    #  .tau (vector(ndep)) reference optical depth
    #  .rhox (vector(ndep)) mass column density (in g/cm^2)
    #  .temp  (vector(ndep)) temperature (in K)
    #  .xne (vector(ndep)) electron number density (in 1/cm^3)
    #  .xne (vector(ndep)) atomic number density (in 1/cm^3)
    #  .rho (vector(ndep)) mass density (in g/cm^3)
    #  .ndep (scalar) number of depths in atmosphere
    #
    # History:
    # 28-Oct-94 JAV	Create.
    # 05-Apr-96 JAV Update syntax description for "atmo=" mode.
    # 05-Apr-02 JAV Complete rewrite of interpkrz to allow interpolation in [M/H].
    # 28-Jan-04 JAV Significant code update to remove jumps in output atmosphere
    #                for small changes in Teff, logg, or [M/H] that cross grid
    #                points. Now the output depth scale is also interpolated.
    #                Also switched from spline to linear interpolation so that
    #                extrapolation at top and bottom of atmosphere behaves better.
    # 16-Apr-04 JAV Complete overhaul of the interpolation algorithm to account
    #                for shifts in mass column scale between models. Two new
    #                auxiliary routines are now required: interp_atmo_pair.pro
    #                and interp_atmo_func.pro. Previous results were flawed.
    # 05-Mar-12 JAV Added arglist= and _extra to support changes to sme_func.
    # 20-May-12 UH  Use _extra to pass grid file name.
    # 30-Oct-12 TN  Rewritten to interpolate on either the tau (optical depth)
    #                or the rhox (column mass) scales. Renamed from interpkrz2/3
    #                to be used as a general routine, called via interfaces.
    # 29-Apr-13 TN  krz3 structure renamed to the generic name atmo_grid.
    #                Savefiles have also been renamed: atlas12.sav and marcs2012.sav.
    # 09-Sep-13 JAV The gridfile= keyword argument is now mandatory. There is no
    #                default value. The gridfile= keyword argument is now defined
    #                explicitly, rather than as part of the _extra= argument.
    #                Execution halts with an explicit error, if gridfile is not
    #                set. If a gridfile is loaded and contains data in variables
    #                with old-style names (e.g., krz2), then the data are copied
    #                to variables with new-style names (e.g., atmo_grid). This
    #                allow the routine to work with old or new variable names.
    #
    # 10-Sep-13 JAV Added interpvar= keyword argument to control interpolation
    #                variable. Default value is 'TAU' if available, otherwise
    #                'RHOX'. Added depthvar= keyword argument to control depth
    #                scale for radiative transfer. Default value is value of
    #                atmo_grid.modtyp. These keywords can have values of 'RHOX'
    #                or 'TAU'. Deprecated type= keyword argument, which sets
    #                both interpvar and depthvar to the specified value. Added
    #                logic to handle receiving a structure from interp_atmo_pair,
    #                rather than a [5,NDEP] array.
    #
    # 20-Sep-13 JAV For spherical models, save mean radius in existing radius
    #                field, rather than trying to add a new radius field. Code
    #                block reformatted.
    # 13-Dec-13 JAV Bundle atmosphere variables into an ATMO structure.
    #
    # 11-Nov-14 JAV Calculate the actual fractional step between each pair of
    #                atmospheres, rather than assuming that all interpolations
    #                of a particular flavor (monh, logg, or teff) have the same
    #                fractional step. This fixes a bug when the atmosphere grid
    #                is irregular, e.g. due to a missing atmosphere.

    # Define common block used to hold 17 MB grid of pre-computed atmospheres,
    # and the common block used by sme_entrypoints to store execution directory.
    # common atmo_grid_common, atmo_grid, atmo_grid_maxdep, atmo_grid_natmo $
    # , atmo_grid_vers, atmo_grid_file, atmo_grid_pro
    # common entrypoints, entry, prefix, sme_library

    # Internal parameters.
    nb = 2  # number of bracket points
    itop = 1  # index of top depth to use on rhox scale

    # Load common block variable "prefix" with directory containing atmospheric grids
    # sme_entrypoints()				#locate atmospheric grids
    prefix = os.path.dirname(__file__)
    atmo_grid_file = None
    atmo_grid = None

    # Load atmosphere grid from disk, if not already loaded.
    changed = atmo_grid_file is not None and atmo_grid_file != atmo_in.source
    if changed or atmo_grid is None or reload:
        path = os.path.join(prefix, "atmospheres ", atmo_in.source)
        krz2 = readsav(path)
        atmo_grid = krz2["atmo_grid"]
        atmo_grid_maxdep = krz2["atmo_grid_maxdep"]
        atmo_grid_natmo = krz2["atmo_grid_natmo"]
        atmo_grid_vers = krz2["atmo_grid_vers"]
        atmo_grid_file = atmo_in.source

    # Get field names in ATMO and ATMO_GRID structures.
    atags = dir(atmo_in)
    gtags = dir(atmo_grid)

    # Determine ATMO.DEPTH radiative transfer depth variable. Order of precedence:
    # (1) Value of ATMO_IN.DEPTH, if it exists and is set
    # (2) Value of ATMO_GRID[0].DEPTH, if it exists and is set
    # (3) 'RHOX', if ATMO_GRID.RHOX exists (preferred over 'TAU' for depth)
    # (4) 'TAU', if ATMO_GRID.TAU exists
    # Check that INTERP is valid and the corresponding field exists in ATMO.
    #
    if "DEPTH" in atags and atmo_in.depth is not None:
        depth = str.upper(atmo_in.depth)
    elif "DEPTH" in gtags and atmo_grid[0].depth is not None:
        depth = str.upper(atmo_grid.depth)
    elif "RHOX" in gtags:
        depth = "RHOX"
    elif "TAU" in gtags:
        depth = "TAU"
    else:
        raise ValueError("no value for ATMO.DEPTH")
    if depth != "TAU" and depth != "RHOX":
        raise ValueError("ATMO.DEPTH must be 'TAU' or 'RHOX', not '%s'" % depth)
    if depth not in gtags:
        raise ValueError(
            "ATMO.DEPTH='%s', but ATMO. %s does not exist" % (depth, depth)
        )

    # Determine ATMO.INTERP interpolation variable. Order of precedence:
    # (1) Value of ATMO_IN.INTERP, if it exists and is set
    # (2) Value of ATMO_GRID[0].INTERP, if it exists and is set
    # (3) 'TAU', if ATMO_GRID.TAU exists (preferred over 'RHOX' for interpolation)
    # (4) 'RHOX', if ATMO_GRID.RHOX exists
    # Check that INTERP is valid and the corresponding field exists in ATMO.
    #
    if "INTERP" in atags and atmo_in.interp is not None:
        interp = str.upper(atmo_in.interp)
    elif "INTERP" in gtags and atmo_grid[0].interp is not None:
        interp = str.upper(atmo_grid.interp)
    elif "TAU" in gtags:
        interp = "TAU"
    elif "RHOX" in gtags:
        interp = "RHOX"
    else:
        raise ValueError("no value for ATMO.INTERP")
    if interp != "TAU" and interp != "RHOX":
        raise ValueError("ATMO.INTERP must be 'TAU' or 'RHOX', not '%s'" % interp)
    if interp not in gtags:
        raise ValueError(
            "ATMO.INTERP='%s', but ATMO. %s does not exist" % (interp, interp)
        )
    interpvar = interp

    # The purpose of the first major set of code blocks is to find values
    # of [M/H] in the grid that bracket the requested [M/H]. Then in this
    # subset of models, find values of log(g) in the subgrid that bracket
    # the requested log(g). Then in this subset of models, find values of
    # Teff in the subgrid that bracket the requested Teff. The net result
    # is a set of 8 models in the grid that bracket the requested stellar
    # parameters. Only these 8 "corner" models will be used in the
    # interpolation that constitutes the remainder of the program.

    # *** DETERMINATION OF METALICITY BRACKET ***
    # Find unique set of [M/H] values in grid.
    muniq = np.unique(atmo_grid.monh, return_index=True)  # indices of unique [M/H]
    Mlist = atmo_grid(muniq).monh  # list of unique [M/H]

    # Test whether requested metalicity is in grid.
    Mmin = np.min(Mlist)  # range of [M/H] in grid
    Mmax = np.max(Mlist)
    if MonH > Mmax:  # true: [M/H] too large
        raise ValueError(
            "interp_atmo_grid: requested [M/H] (%f20.3) larger than max grid value (%f20.3). extrapolating."
            % (MonH, Mmax)
        )
    if MonH < Mmin:  # true: logg too small
        raise ValueError(
            "interp_atmo_grid: requested [M/H] (%f20.3) smaller than min grid value (%f20.3). returning."
            % (MonH, Mmin)
        )

    # Find closest two [M/H] values in grid that bracket requested [M/H].
    if MonH <= Mmax:
        Mlo = np.max(Mlist[Mlist <= MonH])
        Mup = np.min(Mlist[Mlist >= MonH])
    else:
        Mup = Mmax
        Mlo = np.max(Mlist[Mlist < Mup])
    Mb = [Mlo, Mup]  # bounding [M/H] values

    # Trace diagnostics.
    if trace >= 5:
        print("[M/H]: %f20.3 < %f20.3 < %f20.3" % (Mlo, MonH, Mup))

    # *** DETERMINATION OF LOG(G) BRACKETS AT [M/H] BRACKET VALUES ***
    # Set up for loop through [M/H] bounds.
    Gb = np.zeros((nb, nb))  # bounding gravities
    for iMb in range(nb):
        # Find unique set of gravities at boundary below [M/H] value.
        im = atmo_grid.monh == Mb[iMb]  # models on [M/H] boundary
        Glist = np.unique(atmo_grid[im].logg)  # list of unique gravities

        # Test whether requested logarithmic gravity is in grid.
        Gmin = np.min(Glist)  # range of gravities in grid
        Gmax = np.max(Glist)
        if logg > Gmax:  # true: logg too large
            print(
                "interp_atmo_grid: requested log(g) (%f20.3) larger than max grid value (%f20.3). extrapolating."
                % (logg, Gmax)
            )

        if logg < Gmin:  # true: logg too small
            raise ValueError(
                "interp_atmo_grid: requested log(g) (%f20.3) smaller than min grid value (%f20.3). returning."
                % (logg, Gmin)
            )

        # Find closest two gravities in Mlo subgrid that bracket requested gravity.
        if logg <= Gmax:
            Glo = np.max(Glist[Glist <= logg])
            Gup = np.min(Glist[Glist >= logg])
        else:
            Gup = Gmax
            Glo = np.max(Glist[Glist < Gup])
        Gb[iMb] = [Glo, Gup]  # store boundary values.

        # Trace diagnostics.
        if trace >= 5:
            if iMb == 0:
                print()
                print(
                    "log(g) at [M/H]=%f20.3: %f20.3 < %f20.3 < %f20.3"
                    % (Mb[iMb], Glo, logg, Gup)
                )

    # End of loop through [M/H] bracket values.
    # *** DETERMINATION OF TEFF BRACKETS AT [M/H] and LOG(G) BRACKET VALUES ***
    # Set up for loop through [M/H] and log(g) bounds.
    Tb = np.zeros((nb, nb, nb))  # bounding temperatures
    for iGb in range(nb):
        for iMb in range(nb):
            # Find unique set of gravities at boundary below [M/H] value.
            it = (atmo_grid.monh == Mb[iMb]) & (
                atmo_grid.logg == Gb[iMb, iGb]
            )  # models on joint boundary
            Tlist = np.unique(atmo_grid[it].teff)  # list of unique temperatures

            # Test whether requested temperature is in grid.
            Tmin = np.min(Tlist)  # range of temperatures in grid
            Tmax = np.max(Tlist)
            if Teff > Tmax:  # true: Teff too large
                raise ValueError(
                    "interp_atmo_grid: requested Teff (%i) larger than max grid value (%i). returning."
                    % (Teff, Tmax)
                )
            if Teff < Tmin:  # true: logg too small
                print(
                    "interp_atmo_grid: requested Teff (%i) smaller than min grid value (%i). extrapolating."
                    % (Teff, Tmin)
                )

            # Find closest two temperatures in subgrid that bracket requested Teff.
            if Teff > Tmin:
                Tlo = np.max(Tlist[Tlist <= Teff])
                Tup = np.min(Tlist[Tlist >= Teff])
            else:
                Tlo = Tmin
                Tup = np.min(Tlist[Tlist > Tlo])
            Tb[iMb, iGb, :] = [Tlo, Tup]  # store boundary values.

            # Trace diagnostics.
            if trace >= 5:
                if iGb == 0 and iMb == 0:
                    print()
                print(
                    "Teff at log(g)=%f20.3 and [M/H]=%f20.3: %i < %i < %i"
                    % (Gb[iMb, iGb], Mb[iMb], Tlo, Teff, Tup)
                )

    # End of loop through log(g) and [M/H] bracket values.

    # Find and save atmo_grid indices for the 8 corner models.
    icor = np.zeros((nb, nb, nb), dtype=int)
    ncor = len(icor)
    for iTb in range(nb):
        for iGb in range(nb):
            for iMb in range(nb):
                iwhr = np.where(
                    (atmo_grid.teff == Tb[iMb, iGb, iTb])
                    & (atmo_grid.logg == Gb[iMb, iGb])
                    & (atmo_grid.monh == Mb[iMb])
                )[0]
                nwhr = iwhr.size
                if nwhr != 1:
                    print(
                        "interp_atmo_grid: %i models in grid with [M/H]=%f20.1, log(g)=%f20.1, and Teff=%i"
                        % (nwhr, Mb[iMb], Gb[iMb, iGb], Tb[iMb, iGb, iTb])
                    )
                icor[iMb, iGb, iTb] = iwhr[0]

    # Trace diagnostics.
    if trace >= 1:
        print()
        print("     Teff=%i,  log(g)=%f20.3,  [M/H]=%f20.3:" % (Teff, logg, MonH))
        print()
        print("   indx  M/H  g   Teff     indx  M/H  g   Teff")
        for iMb in range(nb):
            for iGb in range(nb):
                i0 = icor[iMb, iGb, 0]
                i1 = icor[iMb, iGb, 1]
                print(
                    i0,
                    atmo_grid(i0).monh,
                    atmo_grid(i0).logg,
                    atmo_grid(i0).teff,
                    i1,
                    atmo_grid(i1).monh,
                    atmo_grid(i1).logg,
                    atmo_grid(i1).teff,
                )
            print()

    # The code below interpolates between 8 corner models to produce
    # the output atmosphere. In the first step, pairs of models at each
    # of the 4 combinations of log(g) and Teff are interpolated to the
    # desired value of [M/H]. These 4 new models are then interpolated
    # to the desired value of log(g), yielding 2 models at the requested
    # [M/H] and log(g). Finally, this pair of models is interpolated
    # to the desired Teff, producing a single output model.

    # Interpolation is done on the logarithm of all quantities to improve
    # linearity of the fitted data. Kurucz models sometimes have very small
    # fractional steps in mass column at the top of the atmosphere. These
    # cause wild oscillations in splines fitted to facilitate interpolation
    # onto a common depth scale. To circumvent this problem, we ignore the
    # top point in the atmosphere by setting itop=1.

    # Interpolate 8 corner models to create 4 models at the desired [M/H].
    m0 = atmo_grid[icor[0, 0, 0]].monh
    m1 = atmo_grid[icor[1, 0, 0]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo00 = interp_atmo_pair(
        atmo_grid[icor[0, 0, 0]],
        atmo_grid[icor[1, 0, 0]],
        mfrac,
        interpvar=interpvar,
        itop=itop,
        trace=trace - 1,
        plot=plot * (mfrac != 0),
    )
    m0 = atmo_grid[icor[0, 1, 0]].monh
    m1 = atmo_grid[icor[1, 1, 0]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo01 = interp_atmo_pair(
        atmo_grid[icor[0, 1, 0]],
        atmo_grid[icor[1, 1, 0]],
        mfrac,
        interpvar=interpvar,
        itop=itop,
        trace=trace - 1,
        plot=plot * (mfrac != 0),
    )
    m0 = atmo_grid[icor[0, 0, 1]].monh
    m1 = atmo_grid[icor[1, 0, 1]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo10 = interp_atmo_pair(
        atmo_grid[icor[0, 0, 1]],
        atmo_grid[icor[1, 0, 1]],
        mfrac,
        interpvar=interpvar,
        itop=itop,
        trace=trace - 1,
        plot=plot * (mfrac != 0),
    )
    m0 = atmo_grid[icor[0, 1, 1]].monh
    m1 = atmo_grid[icor[1, 1, 1]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo11 = interp_atmo_pair(
        atmo_grid[icor[0, 1, 1]],
        atmo_grid[icor[1, 1, 1]],
        mfrac,
        interpvar=interpvar,
        itop=itop,
        trace=trace - 1,
        plot=plot * (mfrac != 0),
    )

    # Interpolate 4 models at the desired [M/H] to create 2 models at desired
    # [M/H] and log(g).
    g0 = atmo00.logg
    g1 = atmo01.logg
    gfrac = 0 if g0 == g1 else (logg - g0) / (g1 - g0)
    atmo0 = interp_atmo_pair(
        atmo00,
        atmo01,
        gfrac,
        interpvar=interpvar,
        trace=trace - 1,
        plot=plot * (gfrac != 0),
    )
    g0 = atmo10.logg
    g1 = atmo11.logg
    gfrac = 0 if g0 == g1 else (logg - g0) / (g1 - g0)
    atmo1 = interp_atmo_pair(
        atmo10,
        atmo11,
        gfrac,
        interpvar=interpvar,
        trace=trace - 1,
        plot=plot * (gfrac != 0),
    )

    # Interpolate the 2 models at desired [M/H] and log(g) to create final
    # model at desired [M/H], log(g), and Teff
    t0 = atmo0.teff
    t1 = atmo1.teff
    tfrac = 0 if t0 == t1 else (Teff - t0) / (t1 - t0)
    krz = interp_atmo_pair(
        atmo0,
        atmo1,
        tfrac,
        interpvar=interpvar,
        trace=trace - 1,
        plot=plot * (tfrac != 0),
    )
    ktags = dir(krz)

    # Set model type to depth variable that should be used for radiative transfer.
    krz.modtyp = depth

    # If all interpolated models were spherical, the interpolated model should
    # also be reported as spherical. This enables spherical-symmetric radiative
    # transfer in the spectral synthesis.
    #
    # Formulae for mass and radius at the corners of interpolation cube:
    #  log(M/Msol) = log g - log g_sol - 2*log(R_sol / R)
    #  2 * log(R / R_sol) = log g_sol - log g + log(M / M_sol)
    #
    if "RADIUS" in gtags and np.min(atmo_grid[icor].radius) > 1 and "HEIGHT" in gtags:
        solR = 69.550e9  # radius of sun in cm
        sollogg = 4.44  # solar log g [cm s^-2]
        mass_cor = (
            atmo_grid[icor].logg - sollogg - 2 * np.log10(solR / atmo_grid[icor].radius)
        )
        mass = 10 ** np.mean(mass_cor)
        radius = solR * 10 ** ((sollogg - logg + np.log10(mass)) * 0.5)
        krz.radius = radius
        geom = "SPH"
    else:
        geom = "PP"

    # Add standard ATMO input fields, if they are missing from ATMO_IN.
    atmo = atmo_in

    # Create ATMO.DEPTH, if necessary, and set value.
    atmo.depth = depth

    # Create ATMO.INTERP, if necessary, and set value.
    atmo.interp = interp

    # Create ATMO.GEOM, if necessary, and set value.
    if "GEOM" in atags:
        if atmo.geom != "" and atmo.geom != geom:
            if atmo.geom == "SPH":
                raise ValueError(
                    "Input ATMO.GEOM='%s' not valid for requested model." % atmo.geom
                )
            else:
                print(
                    "Input ATMO.GEOM='%s' overrides '%s' from grid." % (atmo.geom, geom)
                )
    atmo.geom = geom

    # Copy most fields from KRZ interpolation result to output ATMO.
    discard = ["MODTYP", "SPHERE", "NDEP"]
    for krzi, tag in zip(krz, ktags):
        if tag not in discard:
            setattr(atmo, tag, krzi)

    return atmo
