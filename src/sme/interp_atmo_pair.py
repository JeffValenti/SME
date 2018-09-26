import numpy as np
from awlib.sme.sme import SME_Structure

from interp_atmo_constrained import interp_atmo_constrained
from interp_atmo_func import interp_atmo_func

def interp_atmo_pair(
    atmo1, atmo2, frac, interpvar="", itop=0, atmop=None, trace=0, plot=False, old=False
):
    # Interpolate between two model atmospheres, accounting for shifts in
    # the mass column density or optical depth scale.
    #
    # Inputs:
    # atmo1 (structure) first atmosphere to interpolate
    # atmo2 (structure) second atmosphere to interpolate
    # frac (scalar) interpolation fraction: 0.0 -> atmo1 and 1.0 -> atmo2
    # [interpvar=] (string) atmosphere interpolation variable (RHOX or TAU).
    # [itop=] (scalar) index of top point in the atmosphere to use. default
    #   is to use all points (itop=0). use itop=1 to clip top depth point.
    # [atmop=] (array[5,ndep]) interpolated atmosphere prediction (for plots)
    #   Not needed if atmospheres are provided as structures.
    # [trace=] (scalar) diagnostic print level (0: no printing)
    # [plot=] (scalar) diagnostic plot level (0: no plots). Larger absolute
    #   value yields more plots. Negative values cause a wait for keypress
    #   after each plot.
    # [/old] (switch) also plot result from the old interpkrz2 algorithm.
    #
    # Outputs:
    # atmo (structure) interpolated atmosphere
    #
    # Atmosphere structures:
    # atmo atmosphere specification
    #  .rhox (vector[ndep]) mass column density (g/cm^2)
    #  .tau  (vector[ndep]) reference optical depth (at 5000 Å)
    #  .temp (vector[ndep]) temperature (K)
    #  .xne  (vector[ndep]) electron number density (1/cm^3)
    #  .xna  (vector[ndep]) atomic number density (1/cm^3)
    #  .rho  (vector[ndep]) mass density (g/cm^3)
    #
    # Algorithm:
    # 1) The second atmosphere is fitted onto the first, individually for
    #  each of the four atmospheric quantitites: T, xne, xna, rho.
    # The fitting uses a linear shift in both the (log) depth parameter and
    #  in the (log) atmospheric quantity. For T, the midpoint of the two
    #  atmospheres is aligned for the initial guess. The result of this fit
    #  is used as initial guess for the other quantities. A penalty function
    #  is applied to each fit, to avoid excessively large shifts on the
    #  depth scale.
    # 2) The mean of the horizontal shift in each parameter is used to
    #  construct the common output depth scale.
    # 3) Each atmospheric quantity is interpolated after shifting the two
    #  corner models by the amount determined in step 1), rescaled by the
    #  interpolation fraction (frac).
    #
    # History:
    # 2004-Apr-15 Valenti  Initial coding.
    # MB  - interpolation on tau scale
    # 2012-Oct-30 TN - Rewritten to use either column mass (RHOX) or
    #   reference optical depth (TAU) as vertical scale. Shift-interpolation
    #   algorithms have been improved for stability in cool dwarfs (<=3500 K).
    #   The reference optical depth scale is preferred in terms of interpolation
    #   accuracy across most of parameter space, with significant improvement for
    #   both cool models (where depth vs temperature is rather flat) and hot
    #   models (where depth vs temperature exhibits steep transitions).
    #   Column mass depth is used by default for backward compatibility.
    #
    # 2013-May-17 Valenti  Use frac to weight the two shifted depth scales,
    #   rather than simply averaging them. This change fixes discontinuities
    #   when crossing grid nodes.
    #
    # 2013-Sep-10 Valenti  Now returns an atmosphere structure instead of a
    #   [5,NDEP] atmosphere array. This was necessary to support interpolation
    #   using one variable (e.g., TAU) and radiative transfer using a different
    #   variable (e.g. RHOX). The atmosphere array could only store one depth
    #   variable, meaning the radiative transfer variable had to be the same
    #   as the interpolation variable. Returns atmo.rhox if available and also
    #   atmo.tau if available. Since both depth variables are returned, if
    #   available, this routine no longer needs to know which depth variable
    #   will be used for radiative transfer. Only the interpolation variable
    #   is important. Thus, the interpvar= keyword argument replaces the
    #   type= keyword argument. Very similar code blocks for each atmospheric
    #   quantity have been unified into a single code block inside a loop over
    #   atmospheric quantities.
    #
    # 2013-Sep-21 Valenti  Fixed an indexing bug that affected the output depth
    #   scale but not other atmosphere vectors. Itop clipping was not being
    #   applied to the depth scale ('RHOX' or 'TAU'). Bug fixed by adding
    #   interpvar to vtags. Now atmospheres interpolated with interp_atmo_grid
    #   match output from revision 398. Revisions back to 399 were development
    #   only, so no users should be affected.
    #
    # 2014-Mar-05 Piskunov  Replicated the removal of the bad top layers in
    #                       models for interpvar eq 'TAU'

    # Internal program parameters.
    min_drhox = 0.01  # minimum fractional step in rhox
    min_dtau = 0.01  # minimum fractional step in tau

    # Parameters used to annotate plots.
    tgm1 = [atmo1.teff, atmo1.logg, atmo1.monh]
    tgm2 = [atmo2.teff, atmo2.logg, atmo2.monh]
    tgm = (1 - frac) * tgm1 + frac * tgm2

    ##
    ## Select interpolation variable (RHOX vs. TAU)
    ##

    # Check which depth scales are available in both input atmospheres.
    tags1 = dir(atmo1)
    tags2 = dir(atmo2)
    ok_tau = "TAU" in tags1 and "TAU" in tags2
    ok_rhox = "RHOX" in tags1 and "RHOX" in tags2
    if not ok_tau and not ok_rhox:
        raise ValueError("atmo1 and atmo2 structures must both contain RHOX or TAU")

    # Set interpolation variable, if not specified by keyword argument.
    if interpvar is None:
        if ok_tau:
            interpvar = "TAU"
        else:
            interpvar = "RHOX"
    if interpvar != "TAU" and interpvar != "RHOX":
        raise ValueError("interpvar must be 'TAU' (default) or 'RHOX'")

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
    ipar = np.zeros(npar)
    temp1 = np.log10(atmo1.temp[itop1 : ibot1 + 1])
    temp2 = np.log10(atmo2.temp[itop2 : ibot2 + 1])
    mid1 = np.argmin(np.abs(temp1 - 0.5 * (temp1[1] + temp1[ndep1 - 2])))
    mid2 = np.argmin(np.abs(temp2 - 0.5 * (temp2[1] + temp2[ndep2 - 2])))
    ipar[0] = depth1[mid1] - depth2[mid2]  # horizontal
    ipar[1] = temp1[mid1] - temp2[mid2]  # vertical

    # Apply a constraint on the fit, to avoid runaway for cool models, where
    # the temperature structure is nearly linear with both TAU and RHOX.
    constraints = np.zeros(npar)
    constraints[0] = 0.5  # weakly contrain the horizontal shift

    # Fix the remaining two parameters.
    # parinfo = replicate({fixed:0}, npar)
    parinfo = np.zeros(npar, dtype=[("fixed", int)])
    parinfo[2:3].fixed = 1

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
            raise ValueError("atmo1 does not contain " + vtag)
        if vtag not in tags2:
            raise ValueError("atmo2 does not contain " + vtag)

        vect1 = np.log10(atmo1[vtag][itop1 : ibot1 + 1])
        vect2 = np.log10(atmo2[vtag][itop2 : ibot2 + 1])

        # Fit the second atmosphere onto the first by finding the best horizontal
        # shift in depth2 and the best vertical shift in vect2.
        functargs = {"x2": depth2, "y2": vect2, "y1": vect1, "ndep": ngd, "ipar": ipar}
        pars[ivtag], vect21 = interp_atmo_constrained(
            depth1[igd],
            vect1[igd],
            err1[igd],
            ipar,
            functargs=functargs,
            parinfo=parinfo,
            constraints=constraints,
            nprint=(trace >= 1),
            quiet=True,
        )

        # Make a diagnostic plot.
        if abs(plot) >= 4:
            pass
            # xr = minmax([depth1, depth2])
            # yr = minmax([vect1, vect2, vect21])
            # fmt = '(2(a,":",i0,",",f0.2,",",f0.2,"  "),a)'
            # tit = string(form=fmt, 'red', tgm1, 'blue', tgm2, 'white:blue->red')
            # plot, depth1, vect1, xr=xr, xsty=3, yr=yr, ysty=3, /nodata $
            #     , xtit='log '+interpvar, ytit='log '+vtag, tit=tit, chars=1.4
            # colors
            # oplot, depth1, vect1, co=c24(2)
            # oplot, depth2, vect2, co=c24(4)
            # oplot, depth1, vect21
            # x = !x.crange[0] + 0.05*(!x.crange[1]-!x.crange[0])
            # y = !y.crange[0] + [0.92,0.85]*(!y.crange[1]-!y.crange[0])
            # xyouts, x, y[0], size=1.4, 'xshift='+string(pars[0,ivtag],'(f+0.3)')
            # xyouts, x, y[1], size=1.4, 'yshift='+string(pars[1,ivtag],'(f+0.3)')
            # if plot lt 0 then junk = get_kbrd(1)

        # After first pass ('TEMP'), adjust initial guess and restrict depth points.
        if ivtag == 0:
            ipar = [pars[0, 0], 0.0, 0.0, 0.0]
            igd = np.where(
                (depth1 >= min(depth2) + ipar[0]) & (depth1 <= max(depth2) + ipar[0])
            )[0]
            ndg = igd.size
            if ngd < 2:
                raise ValueError("unstable shift in temperature")

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
    xmin = np.min(depth)
    xmax = np.max(depth)

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
        x1min = np.min(depth1f)
        x1max = np.max(depth1f)
        x2min = np.min(depth2f)
        x2max = np.max(depth2f)
        ilo = (depth < x1min) | (depth < x2min)
        iup = (depth > x1max) | (depth > x2max)
        nlo = np.count_nonzero(ilo)
        nup = np.count_nonzero(iup)
        checklo = (nlo >= 1) and abs(frac - 0.5) <= 0.5 and ndep1 == ndep2
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

        # Make diagnostic plot.
        if abs(plot) >= 1:
            pass
            # if keyword_set(atmop) then depthp = alog10(atmop[0,*])
            # if keyword_set(old) then deptho = (1-frac)*depth1 + frac*depth2
            # xr = minmax([depth1, depth2, depth])
            # yr = minmax([vect1, vect2, vect])
            # fmt = '(3(a,":",i0,",",f0.3,",",f0.3,"  "))'
            # tit = string(form=fmt, 'red', tgm1, 'white', tgm, 'blue', tgm2)
            # plot, depth, vect, xr=xr, xsty=3, yr=yr, ysty=3, /nodata $
            #     , xtit='log '+interpvar, ytit='log '+vtag, tit=tit, chars=1.4
            # oplot, depth, vect1f, co=c24(2), li=2
            # oplot, depth, vect2f, co=c24(4), li=2
            # oplot, depth1, vect1, co=c24(2)
            # oplot, depth2, vect2, co=c24(4)
            # oplot, depth, vect
            # if keyword_set(old) :
            #     vecto = (1-frac) * interpol(vect1, depth1, deptho) $
            #         + frac * interpol(vect2, depth2, deptho)
            #     oplot, deptho, vecto, co=c24(6)
            # endif
            # if keyword_set(atmop) then oplot, depthp, alog10(atmop[1,*]), co=c24(4)
            # if plot lt 0 then junk = get_kbrd(1)

    ##
    ## Construct output structure
    ##

    # Construct output structure with interpolated atmosphere.
    # Might be wise to interpolate abundances, in case those ever change.
    atmo = SME_Structure(None)
    stags = ["TEFF", "LOGG", "MONH", "VTURB", "LONH", "ABUND"]
    ndep_orig = len(atmo1.temp)
    for i1 in range(len(tags1)):
        tag = tags1[i1]

        # Default is to copy value from atmo1. Trim vectors.
        value = atmo1[tag]
        if len(value) == ndep_orig and tag != "ABUND":
            value = value[0:ndep]

        # Vector quantities that have already been interpolated.
        if tag in vtags:
            value = 10.0 ** vects[ivtag]

        # Scalar quantities that should be interpolated using frac.
        if tag in stags:
            value1 = atmo1[tag]
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
