""" Handles reading and interpolation of atmopshere (grid) data """
import itertools
import logging
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import readsav
from scipy.optimize import curve_fit

from .sme import Atmo


class AtmosphereError(Exception):
    """ Something went wrong with the atmosphere interpolation """


class sav_file(np.recarray):
    """ IDL savefile atmosphere grid """

    def __new__(cls, filename):
        prefix = os.path.dirname(__file__)
        path = os.path.join(prefix, "atmospheres", filename)
        krz2 = readsav(path)
        atmo_grid = krz2["atmo_grid"]
        atmo_grid_maxdep = krz2["atmo_grid_maxdep"]
        atmo_grid_natmo = krz2["atmo_grid_natmo"]
        atmo_grid_vers = krz2["atmo_grid_vers"]
        atmo_grid_file = filename

        # Keep values around for next run
        data = atmo_grid.view(cls)
        data.maxdep = atmo_grid_maxdep
        data.natmo = atmo_grid_natmo
        data.vers = atmo_grid_vers
        data.file = atmo_grid_file

        return data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.maxdep = getattr(self, "maxdep", None)
        self.natmo = getattr(self, "natmo", None)
        self.vers = getattr(self, "vers", None)
        self.file = getattr(self, "file", None)

    def load(self, filename, reload=False):
        """ load a new file if necessary """
        changed = filename != self.file
        if reload or changed:
            new = sav_file(filename)
        else:
            new = self
        return new


class krz_file(Atmo):
    """ Read .krz atmosphere files """

    def __init__(self, filename):
        super().__init__()
        self.source = filename
        self.method = "embedded"
        self.load(filename)

    def load(self, filename):
        """
        Load data from disk

        Parameters
        ----------
        filename : str
            name of the file to load
        """
        # TODO: this only works for some krz files
        # 1..2 lines header
        # 3 line opacity
        # 4..13 elemntal abundances
        # 14.. Table data for each layer
        #    Rhox Temp XNE XNA RHO

        with open(filename, "r") as file:
            header1 = file.readline()
            header2 = file.readline()
            opacity = file.readline()
            abund = [file.readline() for _ in range(10)]
            table = file.readlines()

            # Parse header
            # vturb
        i = header1.find("VTURB")
        self.vturb = float(header1[i + 5 : i + 9])
        # L/H, metalicity
        i = header1.find("L/H")
        self.lonh = float(header1[i + 3 :])

        k = len("T EFF=")
        i = header2.find("T EFF=")
        j = header2.find("GRAV=", i + k)
        self.teff = float(header2[i + k : j])

        i = j
        k = len("GRAV=")
        j = header2.find("MODEL TYPE=", i + k)
        self.logg = float(header2[i + k : j])

        i, k = j, len("MODEL TYPE=")
        j = header2.find("WLSTD=", i + k)
        model_type_key = {0: "rhox", 1: "tau", 3: "sph"}
        self.model_type = int(header2[i + k : j])
        self.depth = model_type_key[self.model_type]
        self.geom = "pp"

        i = j
        k = len("WLSTD=")
        self.wlstd = float(header2[i + k :])

        # parse opacity
        i = opacity.find("-")
        opacity = opacity[:i].split()
        self.opflag = np.array([int(k) for k in opacity])

        # parse abundance
        pattern = np.genfromtxt(abund).flatten()[:-1]
        pattern[1] = 10 ** pattern[1]
        self.set_abund(0, pattern, "sme")

        # parse table
        self.table = np.genfromtxt(table, delimiter=",", usecols=(0, 1, 2, 3, 4))
        self.rhox = self.table[:, 0]
        self.temp = self.table[:, 1]
        self.xne = self.table[:, 2]
        self.xna = self.table[:, 3]
        self.rho = self.table[:, 4]


def interp_atmo_pair(atmo1, atmo2, frac, interpvar="RHOX", itop=0):
    """
    Interpolate between two model atmospheres, accounting for shifts in
    the mass column density or optical depth scale.

    How it works:
    1) The second atmosphere is fitted onto the first, individually for
    each of the four atmospheric quantitites: T, xne, xna, rho.
    The fitting uses a linear shift in both the (log) depth parameter and
    in the (log) atmospheric quantity. For T, the midpoint of the two
    atmospheres is aligned for the initial guess. The result of this fit
    is used as initial guess for the other quantities. A penalty function
    is applied to each fit, to avoid excessively large shifts on the
    depth scale.
    2) The mean of the horizontal shift in each parameter is used to
    construct the common output depth scale.
    3) Each atmospheric quantity is interpolated after shifting the two
    corner models by the amount determined in step 1), rescaled by the
    interpolation fraction (frac).

    Parameters
    ------
    atmo1 : Atmosphere
        first atmosphere to interpolate
    atmo2 : Atmosphere
        second atmosphere to interpolate
    frac : float
        interpolation fraction: 0.0 -> atmo1 and 1.0 -> atmo2
    interpvar : {"RHOX", "TAU"}, optional
        atmosphere interpolation variable (default:"RHOX").
    itop : int, optional
        index of top point in the atmosphere to use. default
        is to use all points (itop=0). use itop=1 to clip top depth point.
    atmop : array[5, ndep], optional
        interpolated atmosphere prediction (for plots)
        Not needed if atmospheres are provided as structures. (default: None)
    verbose : {0, 5}, optional
        diagnostic print level (default 0: no printing)
    plot : {-1, 0, 1}, optional
        diagnostic plot level. Larger absolute
        value yields more plots. Negative values cause a wait for keypress
        after each plot. (default: 0, no plots)
    old : bool, optional
        also plot result from the old interpkrz2 algorithm. (default: False)

    Returns
    ------
    atmo : Atmosphere
        interpolated atmosphere
        .rhox (vector[ndep]) mass column density (g/cm^2)
        .tau  (vector[ndep]) reference optical depth (at 5000 Å)
        .temp (vector[ndep]) temperature (K)
        .xne  (vector[ndep]) electron number density (1/cm^3)
        .xna  (vector[ndep]) atomic number density (1/cm^3)
        .rho  (vector[ndep]) mass density (g/cm^3)
    """
    """
    History
    -------
    2004-Apr-15 Valenti
        Initial coding.
    MB
        interpolation on tau scale
    2012-Oct-30 TN
        Rewritten to use either column mass (RHOX) or
        reference optical depth (TAU) as vertical scale. Shift-interpolation
        algorithms have been improved for stability in cool dwarfs (<=3500 K).
        The reference optical depth scale is preferred in terms of interpolation
        accuracy across most of parameter space, with significant improvement for
        both cool models (where depth vs temperature is rather flat) and hot
        models (where depth vs temperature exhibits steep transitions).
        Column mass depth is used by default for backward compatibility.
    2013-May-17 Valenti
        Use frac to weight the two shifted depth scales,
        rather than simply averaging them. This change fixes discontinuities
        when crossing grid nodes.
    2013-Sep-10 Valenti
        Now returns an atmosphere structure instead of a
        [5,NDEP] atmosphere array. This was necessary to support interpolation
        using one variable (e.g., TAU) and radiative transfer using a different
        variable (e.g. RHOX). The atmosphere array could only store one depth
        variable, meaning the radiative transfer variable had to be the same
        as the interpolation variable. Returns atmo.rhox if available and also
        atmo.tau if available. Since both depth variables are returned, if
        available, this routine no longer needs to know which depth variable
        will be used for radiative transfer. Only the interpolation variable
        is important. Thus, the interpvar= keyword argument replaces the
        type= keyword argument. Very similar code blocks for each atmospheric
        quantity have been unified into a single code block inside a loop over
        atmospheric quantities.
    2013-Sep-21 Valenti
        Fixed an indexing bug that affected the output depth
        scale but not other atmosphere vectors. Itop clipping was not being
        applied to the depth scale ('RHOX' or 'TAU'). Bug fixed by adding
        interpvar to vtags. Now atmospheres interpolated with interp_atmo_grid
        match output from revision 398. Revisions back to 399 were development
        only, so no users should be affected.
    2014-Mar-05 Piskunov
        Replicated the removal of the bad top layers in models
        for interpvar eq 'TAU'
    """

    # Internal program parameters.
    min_drhox = 0.01  # minimum fractional step in rhox
    min_dtau = 0.01  # minimum fractional step in tau

    ##
    ## Select interpolation variable (RHOX vs. TAU)
    ##

    # Check which depth scales are available in both input atmospheres.
    tags1 = atmo1.dtype.names
    tags2 = atmo2.dtype.names
    ok_tau = "TAU" in tags1 and "TAU" in tags2
    ok_rhox = "RHOX" in tags1 and "RHOX" in tags2
    if not ok_tau and not ok_rhox:
        raise AtmosphereError(
            "atmo1 and atmo2 structures must both contain RHOX or TAU"
        )

    # Set interpolation variable, if not specified by keyword argument.
    if interpvar is None:
        if ok_tau:
            interpvar = "TAU"
        else:
            interpvar = "RHOX"
    if interpvar != "TAU" and interpvar != "RHOX":
        raise AtmosphereError("interpvar must be 'TAU' (default) or 'RHOX'")

    ##
    ## Define depth scale for both atmospheres
    ##

    # Define depth scale for atmosphere #1
    itop1 = itop
    if interpvar == "RHOX":
        while atmo1.rhox[itop1 + 1] / atmo1.rhox[itop1] - 1 <= min_drhox:
            itop1 += 1
    elif interpvar == "TAU":
        while atmo1.tau[itop1 + 1] / atmo1.tau[itop1] - 1 <= min_dtau:
            itop1 += 1

    ibot1 = atmo1.ndep - 1
    ndep1 = ibot1 - itop1 + 1
    if interpvar == "RHOX":
        depth1 = np.log10(atmo1.rhox[itop1 : ibot1 + 1])
    elif interpvar == "TAU":
        depth1 = np.log10(atmo1.tau[itop1 : ibot1 + 1])

    # Define depth scale for atmosphere #2
    itop2 = itop
    if interpvar == "RHOX":
        while atmo2.rhox[itop2 + 1] / atmo2.rhox[itop2] - 1 <= min_drhox:
            itop2 += 1
    elif interpvar == "TAU":
        while atmo2.tau[itop2 + 1] / atmo2.tau[itop2] - 1 <= min_dtau:
            itop2 += 1

    ibot2 = atmo2.ndep - 1
    ndep2 = ibot2 - itop2 + 1
    if interpvar == "RHOX":
        depth2 = np.log10(atmo2.rhox[itop2 : ibot2 + 1])
    elif interpvar == "TAU":
        depth2 = np.log10(atmo2.tau[itop2 : ibot2 + 1])

    ##
    ## Prepare to find best shift parameters for each atmosphere vector.
    ##

    # List of atmosphere vectors that need to be shifted.
    # The code below assumes 'TEMP' is the first vtag in the list.
    vtags = ["TEMP", "XNE", "XNA", "RHO", interpvar]
    if interpvar == "RHOX" and ok_tau:
        vtags += ["TAU"]
    if interpvar == "TAU" and ok_rhox:
        vtags += ["RHOX"]
    nvtag = len(vtags)

    # Adopt arbitrary uncertainties for shift determinations.
    err1 = np.full(ndep1, 0.05)

    # Initial guess for TEMP shift parameters.
    # Put depth and TEMP midpoints for atmo1 and atmo2 on top of one another.
    npar = 4
    ipar = np.zeros(npar, dtype="f4")
    temp1 = np.log10(atmo1.temp[itop1 : ibot1 + 1])
    temp2 = np.log10(atmo2.temp[itop2 : ibot2 + 1])
    mid1 = np.argmin(np.abs(temp1 - 0.5 * (temp1[1] + temp1[ndep1 - 2])))
    mid2 = np.argmin(np.abs(temp2 - 0.5 * (temp2[1] + temp2[ndep2 - 2])))
    ipar[0] = depth1[mid1] - depth2[mid2]  # horizontal
    ipar[1] = temp1[mid1] - temp2[mid2]  # vertical

    # Apply a constraint on the fit, to avoid runaway for cool models, where
    # the temperature structure is nearly linear with both TAU and RHOX.
    constraints = np.zeros(npar)
    constraints[0] = 0.5  # weakly constrain the horizontal shift

    # For first pass ('TEMP'), use all available depth points.
    ngd = ndep1
    igd = np.arange(ngd)

    ##
    ## Find best shift parameters for each atmosphere vector.
    ##

    # Loop through atmosphere vectors.
    pars = np.zeros((nvtag, npar))
    for ivtag in range(nvtag):
        vtag = vtags[ivtag]

        # Find vector in each structure.
        if vtag not in tags1:
            raise AtmosphereError("atmo1 does not contain " + vtag)
        if vtag not in tags2:
            raise AtmosphereError("atmo2 does not contain " + vtag)

        vect1 = np.log10(atmo1[vtag][itop1 : ibot1 + 1])
        vect2 = np.log10(atmo2[vtag][itop2 : ibot2 + 1])

        # Fit the second atmosphere onto the first by finding the best horizontal
        # shift in depth2 and the best vertical shift in vect2.
        pars[ivtag], _ = interp_atmo_constrained(
            depth1[igd],
            vect1[igd],
            err1[igd],
            ipar,
            x2=depth2,
            y2=vect2,
            y1=vect1,
            ndep=ngd,
            constraints=constraints,
        )

        # After first pass ('TEMP'), adjust initial guess and restrict depth points.
        if ivtag == 0:
            ipar = [pars[0, 0], 0.0, 0.0, 0.0]
            igd = np.where(
                (depth1 >= min(depth2) + ipar[0]) & (depth1 <= max(depth2) + ipar[0])
            )[0]
            if igd.size < 2:
                raise AtmosphereError("unstable shift in temperature")

    ##
    ## Use mean shift to construct output depth scale.
    ##

    # Calculate the mean depth2 shift for all atmosphere vectors.
    xsh = np.sum(pars[:, 0]) / nvtag

    # Base the output depth scale on the input scale with the fewest depth points.
    # Combine the two input scales, if they have the same number of depth points.
    depth1f = depth1 - xsh * frac
    depth2f = depth2 + xsh * (1 - frac)
    if ndep1 > ndep2:
        depth = depth2f
    elif ndep1 == ndep2:
        depth = depth1f * (1 - frac) + depth2f * frac
    elif ndep1 < ndep2:
        depth = depth1f
    ndep = len(depth)

    ##
    ## Interpolate input atmosphere vectors onto output depth scale.
    ##

    # Loop through atmosphere vectors.
    vects = np.zeros((nvtag, ndep))
    for ivtag in range(nvtag):
        vtag = vtags[ivtag]
        par = pars[ivtag]

        # Extract data
        vect1 = np.log10(atmo1[vtag][itop1 : ibot1 + 1])
        vect2 = np.log10(atmo2[vtag][itop2 : ibot2 + 1])

        # Identify output depth points that require extrapolation of atmosphere vector.
        depth1f = depth1 - par[0] * frac
        depth2f = depth2 + par[0] * (1 - frac)
        x1max = np.max(depth1f)
        x2max = np.max(depth2f)
        iup = (depth > x1max) | (depth > x2max)
        nup = np.count_nonzero(iup)
        checkup = (nup >= 1) and abs(frac - 0.5) <= 0.5 and ndep1 == ndep2

        # Combine shifted vect1 and vect2 structures to get output vect.
        vect1f = interp_atmo_func(depth, -frac * par, x2=depth1, y2=vect1)
        vect2f = interp_atmo_func(depth, (1 - frac) * par, x2=depth2, y2=vect2)
        vect = (1 - frac) * vect1f + frac * vect2f
        ends = [vect1[ndep1 - 1], vect[ndep - 1], vect2[ndep2 - 1]]
        if (
            checkup
            and np.median(ends) != vect[ndep - 1]
            and (abs(vect1[ndep1 - 1] - 4.2) < 0.1 or abs(vect2[ndep2 - 1] - 4.2) < 0.1)
        ):
            vect[iup] = vect2f[iup] if x1max < x2max else vect1f[iup]
        vects[ivtag] = vect

    ##
    ## Construct output structure
    ##

    # Construct output structure with interpolated atmosphere.
    # Might be wise to interpolate abundances, in case those ever change.
    atmo = Atmo()
    stags = ["TEFF", "LOGG", "MONH", "VTURB", "LONH", "ABUND"]
    ndep_orig = len(atmo1.temp)
    for tag in tags1:

        # Default is to copy value from atmo1. Trim vectors.
        value = atmo1[tag]
        if np.size(value) == ndep_orig and tag != "ABUND":
            value = value[:ndep]

        # Vector quantities that have already been interpolated.
        if tag in vtags:
            ivtag = [i for i in range(nvtag) if tag == vtags[i]][0]
            value = 10.0 ** vects[ivtag]

        # Scalar quantities that should be interpolated using frac.
        if tag in stags:
            if tag in tags2:
                value = (1 - frac) * atmo1[tag] + frac * atmo2[tag]
            else:
                value = atmo1[tag]

        # Remaining cases.
        if tag == "NDEP":
            value = ndep

        # Abundances
        if tag == "ABUND":
            value = (1 - frac) * atmo1[tag] + frac * atmo2[tag]

        # Create or add to output structure.
        atmo[tag] = value
    return atmo


def interp_atmo_grid(Teff, logg, MonH, atmo_in, verbose=0, reload=False):
    """
    General routine to interpolate in 3D grid of model atmospheres

    Parameters
    -----
    Teff : float
        effective temperature of desired model (K).
    logg : float
        logarithmic gravity of desired model (log cm/s/s).
    MonH : float
        metalicity of desired model.
    atmo_in : Atmosphere
        Input atmosphere
    verbose : {0, 1}, optional
        how much information to plot/print (default: 0)
    plot : bool, optional
        wether to plot debug information (default: False)
    reload : bool
        wether to reload atmosphere information from disk (default: False)

    Returns
    -------
    atmo : Atmosphere
        interpolated atmosphere data
    """
    """
    History
    -------
    28-Oct-94 JAV
        Created
    05-Apr-96 JAV
        Update syntax description for "atmo=" mode.
    05-Apr-02 JAV
        Complete rewrite of interpkrz to allow interpolation in [M/H].
    28-Jan-04 JAV
        Significant code update to remove jumps in output atmosphere
        for small changes in Teff, logg, or [M/H] that cross grid
        points. Now the output depth scale is also interpolated.
        Also switched from spline to linear interpolation so that
        extrapolation at top and bottom of atmosphere behaves better.
    16-Apr-04 JAV
        Complete overhaul of the interpolation algorithm to account
        for shifts in mass column scale between models. Two new
        auxiliary routines are now required: interp_atmo_pair.pro
        and interp_atmo_func.pro. Previous results were flawed.
    05-Mar-12 JAV
        Added arglist= and _extra to support changes to sme_func.
    20-May-12 UH
        Use _extra to pass grid file name.
    30-Oct-12 TN
        Rewritten to interpolate on either the tau (optical depth)
        or the rhox (column mass) scales. Renamed from interpkrz2/3
        to be used as a general routine, called via interfaces.
    29-Apr-13 TN
        krz3 structure renamed to the generic name atmo_grid.
        Savefiles have also been renamed: atlas12.sav and marcs2012.sav.
    09-Sep-13 JAV
        The gridfile= keyword argument is now mandatory. There is no
        default value. The gridfile= keyword argument is now defined
        explicitly, rather than as part of the _extra= argument.
        Execution halts with an explicit error, if gridfile is not
        set. If a gridfile is loaded and contains data in variables
        with old-style names (e.g., krz2), then the data are copied
        to variables with new-style names (e.g., atmo_grid). This
        allow the routine to work with old or new variable names.
    10-Sep-13 JAV
        Added interpvar= keyword argument to control interpolation
        variable. Default value is 'TAU' if available, otherwise
        'RHOX'. Added depthvar= keyword argument to control depth
        scale for radiative transfer. Default value is value of
        atmo_grid.modtyp. These keywords can have values of 'RHOX'
        or 'TAU'. Deprecated type= keyword argument, which sets
        both interpvar and depthvar to the specified value. Added
        logic to handle receiving a structure from interp_atmo_pair,
        rather than a [5,NDEP] array.
    20-Sep-13 JAV
        For spherical models, save mean radius in existing radius
        field, rather than trying to add a new radius field. Code
        block reformatted.
    13-Dec-13 JAV
        Bundle atmosphere variables into an ATMO structure.
    11-Nov-14 JAV
        Calculate the actual fractional step between each pair of
        atmospheres, rather than assuming that all interpolations
        of a particular flavor (monh, logg, or teff) have the same
        fractional step. This fixes a bug when the atmosphere grid
        is irregular, e.g. due to a missing atmosphere.
    """

    # Internal parameters.
    nb = 2  # number of bracket points
    itop = 1  # index of top depth to use on rhox scale

    atmo_file = atmo_in.source
    self = interp_atmo_grid
    if not hasattr(self, "atmo_grid"):
        self.atmo_grid = atmo_grid = sav_file(atmo_file)
    else:
        self.atmo_grid = atmo_grid = self.atmo_grid.load(atmo_file, reload=reload)

    # Get field names in ATMO and ATMO_GRID structures.
    atags = [s.upper() for s in atmo_in.names]
    gtags = [s for s in atmo_grid.dtype.names]

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
        raise AtmosphereError("no value for ATMO.DEPTH")
    if depth != "TAU" and depth != "RHOX":
        raise AtmosphereError("ATMO.DEPTH must be 'TAU' or 'RHOX', not '%s'" % depth)
    if depth not in gtags:
        raise AtmosphereError(
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
        raise AtmosphereError("no value for ATMO.INTERP")
    if interp not in ["TAU", "RHOX"]:
        raise AtmosphereError("ATMO.INTERP must be 'TAU' or 'RHOX', not '%s'" % interp)
    if interp not in gtags:
        raise AtmosphereError(
            "ATMO.INTERP='%s', but ATMO. %s does not exist" % (interp, interp)
        )

    # print("CHECK ATMOSPHERE INTERPOLATION")
    # atmo = interpolate_atmosphere_grid(
    #     atmo_grid, Teff, logg, MonH, interp, depth, atmo_in, atmo_file
    # )
    # return atmo

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
    Mlist = np.unique(atmo_grid.monh)  # list of unique [M/H]

    # Test whether requested metalicity is in grid.
    Mmin = np.min(Mlist)  # range of [M/H] in grid
    Mmax = np.max(Mlist)
    if MonH > Mmax:  # true: [M/H] too large
        logging.info(
            "interp_atmo_grid: requested [M/H] (%.3f) larger than max grid value (%.3f). extrapolating.",
            MonH,
            Mmax,
        )
    if MonH < Mmin:  # true: logg too small
        raise AtmosphereError(
            "interp_atmo_grid: requested [M/H] (%.3f) smaller than min grid value (%.3f). returning."
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
    if verbose >= 5:
        logging.info("[M/H]: %.3f < %.3f < %.3f", Mlo, MonH, Mup)

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
            logging.info(
                "interp_atmo_grid: requested log(g) (%.3f) larger than max grid value (%.3f). extrapolating.",
                logg,
                Gmax,
            )

        if logg < Gmin:  # true: logg too small
            raise AtmosphereError(
                "interp_atmo_grid: requested log(g) (%.3f) smaller than min grid value (%.3f). returning."
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
        if verbose >= 5:
            logging.info(
                "log(g) at [M/H]=%.3f: %.3f < %.3f < %.3f", Mb[iMb], Glo, logg, Gup
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
                raise AtmosphereError(
                    "interp_atmo_grid: requested Teff (%i) larger than max grid value (%i). returning."
                    % (Teff, Tmax)
                )
            if Teff < Tmin:  # true: logg too small
                logging.info(
                    "interp_atmo_grid: requested Teff (%i) smaller than min grid value (%i). extrapolating.",
                    Teff,
                    Tmin,
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
            if verbose >= 5:
                logging.info(
                    "Teff at log(g)=%.3f and [M/H]=%.3f: %i < %i < %i",
                    Gb[iMb, iGb],
                    Mb[iMb],
                    Tlo,
                    Teff,
                    Tup,
                )

    # End of loop through log(g) and [M/H] bracket values.

    # Find and save atmo_grid indices for the 8 corner models.
    icor = np.zeros((nb, nb, nb), dtype=int)
    for iTb, iGb, iMb in itertools.product(range(nb), repeat=3):
        iwhr = np.where(
            (atmo_grid.teff == Tb[iMb, iGb, iTb])
            & (atmo_grid.logg == Gb[iMb, iGb])
            & (atmo_grid.monh == Mb[iMb])
        )[0]
        nwhr = iwhr.size
        if nwhr != 1:
            logging.info(
                "interp_atmo_grid: %i models in grid with [M/H]=%.1f, log(g)=%.1f, and Teff=%i",
                nwhr,
                Mb[iMb],
                Gb[iMb, iGb],
                Tb[iMb, iGb, iTb],
            )
        icor[iMb, iGb, iTb] = iwhr[0]

    # Trace diagnostics.
    if verbose >= 1:
        logging.info("Teff=%i,  log(g)=%.3f,  [M/H]=%.3f:", Teff, logg, MonH)
        logging.info("indx  M/H  g   Teff     indx  M/H  g   Teff")
        for iMb in range(nb):
            for iGb in range(nb):
                i0 = icor[iMb, iGb, 0]
                i1 = icor[iMb, iGb, 1]
                logging.info(
                    i0,
                    atmo_grid[i0].monh,
                    atmo_grid[i0].logg,
                    atmo_grid[i0].teff,
                    i1,
                    atmo_grid[i1].monh,
                    atmo_grid[i1].logg,
                    atmo_grid[i1].teff,
                )

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
        interpvar=interp,
        itop=itop,
    )
    m0 = atmo_grid[icor[0, 1, 0]].monh
    m1 = atmo_grid[icor[1, 1, 0]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo01 = interp_atmo_pair(
        atmo_grid[icor[0, 1, 0]],
        atmo_grid[icor[1, 1, 0]],
        mfrac,
        interpvar=interp,
        itop=itop,
    )
    m0 = atmo_grid[icor[0, 0, 1]].monh
    m1 = atmo_grid[icor[1, 0, 1]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo10 = interp_atmo_pair(
        atmo_grid[icor[0, 0, 1]],
        atmo_grid[icor[1, 0, 1]],
        mfrac,
        interpvar=interp,
        itop=itop,
    )
    m0 = atmo_grid[icor[0, 1, 1]].monh
    m1 = atmo_grid[icor[1, 1, 1]].monh
    mfrac = 0 if m0 == m1 else (MonH - m0) / (m1 - m0)
    atmo11 = interp_atmo_pair(
        atmo_grid[icor[0, 1, 1]],
        atmo_grid[icor[1, 1, 1]],
        mfrac,
        interpvar=interp,
        itop=itop,
    )

    # Interpolate 4 models at the desired [M/H] to create 2 models at desired
    # [M/H] and log(g).
    g0 = atmo00.logg
    g1 = atmo01.logg
    gfrac = 0 if g0 == g1 else (logg - g0) / (g1 - g0)
    atmo0 = interp_atmo_pair(atmo00, atmo01, gfrac, interpvar=interp)
    g0 = atmo10.logg
    g1 = atmo11.logg
    gfrac = 0 if g0 == g1 else (logg - g0) / (g1 - g0)
    atmo1 = interp_atmo_pair(atmo10, atmo11, gfrac, interpvar=interp)

    # Interpolate the 2 models at desired [M/H] and log(g) to create final
    # model at desired [M/H], log(g), and Teff
    t0 = atmo0.teff
    t1 = atmo1.teff
    tfrac = 0 if t0 == t1 else (Teff - t0) / (t1 - t0)
    krz = interp_atmo_pair(atmo0, atmo1, tfrac, interpvar=interp)
    ktags = krz.names

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
                raise AtmosphereError(
                    "Input ATMO.GEOM='%s' not valid for requested model." % atmo.geom
                )
            else:
                logging.info(
                    "Input ATMO.GEOM='%s' overrides '%s' from grid.", atmo.geom, geom
                )
    atmo.geom = geom

    # Copy most fields from KRZ interpolation result to output ATMO.
    discard = ["MODTYP", "SPHERE", "NDEP"]
    for tag in ktags:
        if tag not in discard:
            setattr(atmo, tag, krz[tag])

    return atmo


def interp_atmo_constrained(x, y, err, par, x2, y2, constraints=None, **kwargs):
    """
    Apply a constraint on each parameter, to have it approach zero

    Parameters
    -------
    x : array[n]
        x data
    y : array[n]
        y data
    err : array[n]
        errors on y data
    par : list[4]
        initial guess for fit parameters - see interp_atmo_func
    x2 : array
        independent variable for tabulated input function
    y2 : array
        dependent variable for tabulated input function
    ** kwargs : dict
        passes keyword arguments to interp_atmo_func

        ndep : int
            number of depth points in supplied quantities
        constraints : array[nconstraint], optional
            error vector for constrained parameters.
            Use errors of 0 for unconstrained parameters.

    Returns
    --------
    ret : list of floats
        best fit parameters
    yfit : array of size (n,)
        best fit to data
    """

    # Evaluate with fixed paramters 3, 4
    # ret = mpfitfun("interp_atmo_func", x, y, err, par, extra=extra, yfit=yfit)
    # TODO: what does the constrain do?
    func = lambda x, *p: interp_atmo_func(x, [*p, *par[2:]], x2, y2, **kwargs)
    popt, _ = curve_fit(func, x, y, sigma=err, p0=par[:2])

    ret = [*popt, *par[2:]]
    yfit = func(x, *popt)
    return ret, yfit


def interp_atmo_func(x1, par, x2, y2, ndep=None, y1=None):
    """
    Apply a horizontal shift to x2.
    Apply a vertical shift to y2.
    Interpolate onto x1 the shifted y2 as a function of shifted x2.

    Parameters
    ---------
    x1 : array[ndep1]
        independent variable for output function
    par : array[3]
        shift parameters
        par[0] - horizontal shift for x2
        par[1] - vertical shift for y2
        par[2] - vertical scale factor for y2
    x2 : array[ndep2]
        independent variable for tabulated input function
    y2 : array[ndep2]
        dependent variable for tabulated input function
    ndep : int, optional
        number of depth points in atmospheric structure (default is use all)
    y1 : array[ndep2], optional
        data values being fitted

    Note
    -------
    Only pass y1 if you want to restrict the y-values of extrapolated
    data points. This is useful when solving for the shifts, but should
    not be used when calculating shifted functions for actual use, since
    this restriction can cause discontinuities.
    """
    # Constrained fits may append non-atmospheric quantities to the end of
    #  input vector.
    # Extract the output depth scale:
    if ndep is None:
        ndep = len(x1)

    # Shift input x-values.
    # Interpolate input y-values onto output x-values.
    # Shift output y-values.
    x2sh = x2 + par[0]
    y2sh = y2 + par[1]
    y = np.zeros_like(x1)
    y[:ndep] = interp1d(x2sh, y2sh, kind="linear", fill_value="extrapolate")(
        x1[:ndep]
    )  # Note, this implicitly extrapolates

    # Scale output y-values about output y-center.
    ymin = np.min(y[:ndep])
    ymax = np.max(y[:ndep])
    ycen = 0.5 * (ymin + ymax)
    y[:ndep] = ycen + (1.0 + par[2]) * (y[:ndep] - ycen)

    # If extra.y1 was passed, then clip minimum and maximum of output y1.
    if y1 is not None:
        y[:ndep] = np.clip(y[:ndep], np.min(y1), np.max(y1))

    # Set all leftover values to zero
    y[ndep:] = 0
    return y
