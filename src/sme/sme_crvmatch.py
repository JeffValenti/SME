import numpy as np
from scipy.optimize import least_squares
from scipy.signal import correlate
from scipy.optimize import minimize
from scipy.constants import speed_of_light

from .bezier import bezier_interp


def match_rv_continuum(
    x_obs,
    y_obs,
    u_obs,
    x_syn,
    y_syn,
    mask,
    ndeg=1,
    rvel=0,
    cscale=None,
    fix_c=False,
    fix_rv=False,
):
    """
    Match both the continuum and the radial velocity of observed/synthetic spectrum

    Note that the parameterization of the continuum is different to old SME !!!

    Parameters
    ----------
    x_obs : array
        observed wavelength
    y_obs : array
        observed flux
    u_obs : array
        uncertainties of observed flux
    x_syn : array
        synthetic wavel
    y_syn : array
        synthetic flux
    mask : array
        pixel mask, determining continuum and lines (continuum == 2, line == 1, bad == 0)
    ndeg : int, optional
        number of degrees of the continuum polynomial (default: 1, i.e linear)
    rvel : float, optional
        radial velocity guess (not used unless fix_rv is True) (default: 0)
    cscale : array[ndeg], optional
        continuum polynomial coefficients (not used unless fix_c is True) (default: None)
    fix_c : bool, optional
        use old continuum instead of recalculating it (default: False)
    fix_rv : bool, optional
        use old radial velocity instead of recalculating it (default: False)

    Returns
    -------
    rvel : float
        new radial velocity
    cscale : array[ndeg]
        new continuum coefficients
    """

    c_light = speed_of_light * 1e-3  # speed of light in km/s

    if not fix_c:
        # fit a line to the continuum points
        cont = mask == 2
        cscale_new = np.polyfit(x_obs[cont], y_obs[cont], deg=ndeg, w=1 / u_obs[cont])
        cscale = cscale_new[::-1]

    if not fix_rv:
        # apply continuum
        if cscale is not None:
            cont = np.polyval(cscale[::-1], x_obs)
        else:
            print("Warning: No continuum scale passed to radial velocity determination")
            cont = np.ones_like(y_obs)

        y_obs = y_obs / cont
        tmp = np.interp(x_obs, x_syn, y_syn)

        # Get a first rough estimate from cross correlation
        # Subtract continuum level of 1, for better correlation
        corr = correlate(y_obs - np.mean(y_obs), tmp - np.mean(tmp), mode="same")
        offset = np.argmax(corr)

        x1 = x_obs[offset]
        x2 = x_obs[len(x_obs) // 2]

        rvel = c_light * (1 - x2 / x1)

        lines = mask == 1

        # Then minimize the least squares
        def func(x):
            tmp = np.interp(x_obs[lines] * (1 - x / c_light), x_syn, y_syn)
            return np.sum((y_obs[lines] - tmp) ** 2 * u_obs[lines] ** -2)

        res = minimize(func, x0=rvel)
        rvel = res.x[0]

    return rvel, cscale


# def discrep(xc, yc, xo, yo, no, wt, mask, ndeg, clim, mnx, rvel, cscale, fixc=False):
#     # Spline (or bin, eventually) model onto observed wavelength scale.
#     # Fit continuum points and scale model accordingly.
#     # 14-May-97 JAV  Added logic to avoid math errors when continuum consists
#     # 		 of just one point.

#     # Warn if ndeg is unsupported.
#     # if ndeg ne 0 and ndeg ne 1 :
#     #   message, 'Continuum scaling for each segment must be uniform or linear.'
#     # endif

#     # Spline model onto observed wavelength scale. Eventually allow binning.
#     vfact = 1 + rvel / 2.99792e5  # velocity scaling factor
#     xc_sh = (xc + mnx) * vfact - mnx  # shift velocities
#     yco = bezier_interp(xc_sh, yc, xo)  # spline onto xo

#     # Done, if only allowing radial velocity to vary.
#     il = mask == 1  # find line points
#     nl = np.count_nonzero(il)
#     if fixc:
#         if len(cscale) == 1:  # continuum is a scaling factor
#             yfit = cscale * yco
#         elif len(cscale) == 2:  # continuum is a line
#             xmin = min(xo)
#             xmax = max(xo)
#             yfit = (
#                 cscale[0] * (xmax - xo) / (xmax - xmin)
#                 + cscale[1] * (xo - xmin) / (xmax - xmin)
#             ) * yco
#         chisq = np.sum(wt[il] * (yfit[il] - yo[il]) ** 2) / nl  # chi squared (line pts)
#         return chisq, yfit

#     # Find continuum points.
#     ic = mask == 2  # find continuum points
#     nc = np.count_nonzero(ic)
#     if nc < 0:  # true: no continuum yet
#         ic = (mask == 1) & (yo > 1)
#         nc = np.count_nonzero(ic)
#         if nc <= 1:
#             ic = mask == 1
#         if max(ic) - min(ic) < len(mask) / 3:
#             ic = mask == 1
#         mask[ic] = 2  # mark for posterity

#     # Extract continuum points to be fit.
#     wtc = wt[ic]  # weights for continuum
#     xoc = xo[ic]  # continuum wavelengths
#     yoc = yo[ic]  # observed continuum
#     ycoc = yco[ic]  # model continuum

#     # Do least-squares fit to continuum points.
#     # JEFF HACK
#     if ndeg == 0:
#         A11 = np.sum(wtc * ycoc * ycoc)  # least squares matrix
#         B1 = np.sum(wtc * ycoc * yoc)  # fit to observed continuum
#         if A11 == 0:  # true: bad fit - single point?
#             cscale = np.sum(yoc) / np.sum(ycoc)  # just scale
#         else:
#             cscale = B1 / A11  # scale using weights
#         cfit = np.full(len(yco), cscale)
#     elif ndeg == 1:
#         A11 = np.sum(wtc * ycoc * ycoc)  # least squares matrix
#         A12 = np.sum(wtc * ycoc * ycoc * xoc)
#         A21 = np.sum(wtc * ycoc * ycoc * xoc)
#         A22 = np.sum(wtc * ycoc * ycoc * xoc ** 2)
#         B1 = np.sum(wtc * ycoc * yoc)  # fit to observed continuum
#         B2 = np.sum(wtc * ycoc * yoc * xoc)
#         DET = A11 * A22 - A21 * A12  # solve system
#         if DET == 0:  # true: bad fit - single point?
#             tmp = np.sum(yoc) / np.sum(ycoc)  # just scale
#             cscale = np.full(2, tmp)
#             cfit = np.full(len(yco), tmp)
#         else:
#             coef = [
#                 (B1 * A22 - B2 * A21) / DET,
#                 (-1 * B1 * A12 + B2 * A11) / DET,
#             ]  # scale using weights
#             cscale = np.polyval(
#                 coef, [min(xo), max(xo)]
#             )  # continuum value at endpoints
#             cfit = np.polyval(coef, xo)
#     #
#     # BEGIN JEFF HACK for the smecat project
#     #
#     elif ndeg == 2:
#         clip = 2.5  # sigma-clipping threshold
#         ratio = yoc / ycoc

#         # linear function
#         def fun(coef, t, y):
#             return coef[0] * t + coef[1] - y

#         x0 = np.ones(2)
#         coef = least_squares(fun, x0, loss="soft_l1", args=(xoc, ratio))
#         # coef=ladfit(xoc,ratio)		#robust linear fit first
#         yfit = np.polyval(coef, xoc)
#         resid = ratio - yfit
#         thr = np.clip(clip * np.std(resid), 0.01, None)  # within 1% is always fine
#         ib = abs(resid) > thr
#         nb = np.count_nonzero(ib)
#         ik = ~ib
#         nk = len(ib) - nb
#         while nb > 0 and nk > 10:  # keep at least 10 cont pts

#             # Diagnostics.
#             debug = False
#             if debug:
#                 pass
#                 # yr=minmax([resid,thr*[-1,1]])
#                 # plot,xoc,resid,ps=7,xsty=3,yr=yr,ysty=3
#                 # oplot,xoc[ib],resid[ib],ps=7,co=2
#                 # oplot,!x.crange,[0,0]+thr,li=2,co=2
#                 # oplot,!x.crange,[0,0]-thr,li=2,co=2

#             # Trim set of continuum points.
#             wtc = wtc[ik]  # weights for continuum
#             xoc = xoc[ik]  # update vectors
#             yoc = yoc[ik]
#             ycoc = ycoc[ik]
#             ratio = ratio[ik]

#             # Fit quadratic.
#             prev_coef = coef
#             coef = np.polyfit(xoc, ratio, 2)  # now fit quadratic
#             yfit = np.polyval(coef, xoc)
#             coef = coef.flatten()
#             resid = ratio - yfit

#             # Diagnostics.
#             if debug:
#                 pass
#                 # oplot,xo,poly(xo,coef-[prev_coef,0]),li=1
#                 # oplot,xoc,resid,ps=1,co=3
#                 # oplot,!x.crange,[0,0],li=1,co=3

#             # Search for new outliers.
#             thr = np.clip(clip * np.std(resid), 0.01, None)  # within 1% is always fine
#             ib = abs(resid) > thr
#             ik = ~ib
#             nb = np.count_nonzero(ib)
#             nk = np.count_nonzero(ik)

#             # Diagnostics.
#             if debug:
#                 pass
#                 # oplot,!x.crange,[0,0]+thr,li=2,co=5
#                 # oplot,!x.crange,[0,0]-thr,li=2,co=5
#                 # if nb gt 0 then oplot,xoc[ib],resid[ib],ps=1,co=5

#             # Pause for keypress.
#             if debug:
#                 pass
#                 # junk=get_kbrd(1)
#                 # if junk eq 's' then stop
#                 # if junk eq 'q' then retall

#         # Determine global scale factor.
#         rat2 = yoc / (ycoc * yfit)
#         nrat2 = np.size(rat2)
#         isort = np.argsort(rat2)
#         renorm = rat2[isort[round(0.666 * nrat2)]]
#         coef = renorm * coef
#         cscale = np.polyval(coef, [min(xo), max(xo)])  # continuum value at endpoints
#         cfit = np.polyval(coef, xo)
#         #
#         # END JEFF HACK for the smecat project
#         #

#     # Calculate return variables.
#     yfit = yco * cfit  # new fit to observation
#     chisq = np.sum(wt[il] * (yfit[il] - yo[il]) ** 2) / nl  # chi squared (line pts)
#     return chisq, yfit


# def sme_crvmatch(
#     xclc_in,
#     yclc_in,
#     xobs_in,
#     yobs_in,
#     unc,
#     mask,
#     ndeg,
#     wmid,
#     rvel,
#     cscale,
#     clim=0.01,
#     fixc=False,
#     fixv=False,
# ):

#     #TODO
#     # in: Synthetic spectrum (xclc_in, yclc_in)
#     #     Observed spectrum (xobs_in, yobs_in, unc)
#     #     Mask mask (2 = continuum, 1 = line, 0 = bad)
#     #     degree of continuum fit ndeg
#     #     Initial radial velocity guess rvel

#     # out: Radial velocity rvel
#     #      Continuum parameters cscale


#     # Find radial velocity shift (rvel) and continuum scaling (cscale) of
#     #  degree ndeg required to map the calculated spectrum ([xy]clc_in) onto
#     #  an observed spectrum ([xy]obs_in), taking into account uncertainties
#     #  in the observed spectrum (unc) and a mask that may distinguish bad
#     #  and continuum points from good line points. The switch /fixc uses
#     #  input values of cscale, rather than solving.

#     # Check if the mask was generated automatically or was hand-tuned
#     i = mask == 2
#     valid_mask = np.count_nonzero(i)  # Automatic mask has no continuum

#     # Normalize input spectra for better accuracy.
#     nclc = len(xclc_in)
#     nobs = len(xobs_in)
#     mnx = np.mean(xclc_in)  # calculate mean values
#     mnyc = np.mean(yclc_in)
#     mnyo = np.mean(yobs_in)
#     xclc = xclc_in - mnx  # symmetric about zero
#     xobs = xobs_in - mnx
#     yclc = yclc_in / mnyc  # set mean to 1
#     yobs = yobs_in / mnyo

#     # Get spline fit coefficients. Only needs to be done once.
#     # y2=bezier_interp(xclc,yclc)

#     # Set initial parameters.
#     rvlim = 200  # maximum shift (km/s)
#     rva = rvel  # initial shift (km/s)
#     drv = 1  # velocity step size (km/s)
#     csnorm = cscale * mnyc / mnyo

#     # Set fit weights. Formally, the weights would be (1.0/unc)^2, but (yobs/unc)^2
#     # does a much better job of matching the (higher) continuum points.
#     wt = np.zeros(nobs)  # init with no weight
#     inz = (mask > 0) & (unc > 0)  # find valid points
#     nnz = np.count_nonzero(inz)
#     wt[inz] = yobs[inz] / unc[inz] ** 2  # least squares weights

#     # Find model shift and slope adjustments that best matches observation.
#     niter = 0
#     # A test with 24 spectra of dwarf and giant stars with several segments over a wide spectral range in the optical resulted in a maximum number of iterations of 34 (the routine had been called 33935 times). We exit the iterations after a maximum of 3 times this value, in order to bypass segments with bad RV information, which cause endless loops at this point. UH 2012-08-10, Bordeaux.
#     nitermax = 100
#     i = 1
#     while True:

#         # Evaluate points A and B.
#         fa, yfit = discrep(
#             xclc,
#             yclc,
#             xobs,
#             yobs,
#             nobs,
#             wt,
#             mask,
#             ndeg,
#             clim,
#             wmid,
#             rva,
#             csnorm,
#             fixc=fixc,
#         )

#         if valid_mask == 0:
#             knz = yobs[inz] > (0. * np.max(yobs[inz]) + 3. * np.median(yobs[inz])) / 3.
#             kknz = ~knz
#             nknz = np.count_nonzero(knz)
#             nkknz = np.count_nonzero(kknz)
#             if nknz > 0:
#                 mask[inz[knz]] = 2
#             if nkknz > 0:
#                 mask[inz[kknz]] = 1

#         debug = False
#         if debug:
#             pass
#             #     print,form='(a,10g12.4)','chisq,csnorm:',fa,csnorm
#             #     yr=[min(yobs_in)<min(yfit*mnyo) $
#             #     ,max(yobs_in)>max(yfit*mnyo) ]
#             #     plot,wmid+xobs_in,yobs_in,ps=10 $
#             #         ,xsty=3,ysty=3,yr=yr
#             #     colors
#             #     oplot,!x.crange,[1,1],li=2,co=7
#             #     ytmp=yobs_in
#             #     ytmp[inz]=-1000
#             #     oplot,wmid+xobs,ytmp,ps=10,min=-999,co=2	#mark region not being fit
#             #     itmp=where(mask ne 2)
#             #     ytmp=yobs_in
#             #     ytmp[itmp]=-1000
#             #     oplot,wmid+xobs,ytmp,ps=10,min=-999,co=3	#mark continuum regions
#             # #CHECK SIGN IN VFACT EXPRESSION
#             #     vfact=1.0d0+rva/2.99792d5			#wavelength scale factor
#             # #     xclc_sh = (wmid+xclc_in)*vfact		#shifted model wavelengths
#             #     oplot,(wmid+xobs)*vfact,yfit*mnyo,co=4	#plot model spectrum
#             #     ic=where(mask eq 2,nc)
#             #     ratio=yobs_in[ic]/(yfit[ic]*mnyo)
#             #     isort=sort(ratio)
#             #     print,'ratio[0.666]='+strtrim(ratio[isort[0.666*nc]],2)
#             #     junk=get_kbrd(0)
#             #     if junk eq 's' then stop
#             #     if junk eq 'q' then retall
#             #     endif

#         # If radial velocity is fixed, return immediately after first normalization.
#         if fixv:
#             cscale = csnorm * mnyo / mnyc
#             return yfit * mnyo

#         rvb = rva + drv
#         fb, yfit = discrep(
#             xclc,
#             yclc,
#             xobs,
#             yobs,
#             nobs,
#             wt,
#             mask,
#             ndeg,
#             clim,
#             wmid,
#             rvb,
#             csnorm,
#             fixc=fixc,
#         )

#         # Choose C in downhill direction.
#         if fb > fa:  # true: reverse direction
#             rvc, rvb, rva = rvb, rva, rvb - drv
#             fc, fb = fb, fa
#             fa, yfit = discrep(
#                 xclc,
#                 yclc,
#                 xobs,
#                 yobs,
#                 nobs,
#                 wt,
#                 mask,
#                 ndeg,
#                 clim,
#                 wmid,
#                 rva,
#                 csnorm,
#                 fixc=fixc,
#             )
#         else:  # else: continue onwards
#             rvc = rvb + drv
#             fc, yfit = discrep(
#                 xclc,
#                 yclc,
#                 xobs,
#                 yobs,
#                 nobs,
#                 wt,
#                 mask,
#                 ndeg,
#                 clim,
#                 wmid,
#                 rvc,
#                 csnorm,
#                 fixc=fixc,
#             )

#         # Test whether we are done.
#         if fa > fb and fb < fc:  # true: bounded both sides
#             rv = (
#                 0.5
#                 * (
#                     (fb - fc) * (rva - rvc) * (rva + rvc)
#                     - (fa - fc) * (rvb - rvc) * (rvb + rvc)
#                 )
#                 / ((fb - fc) * (rva - rvc) - (fa - fc) * (rvb - rvc))
#             )
#             if rv > rvlim:
#                 rv = rvlim
#             elif rv < -rvlim:
#                 rv = -rvlim
#             drv = min([rva, rvb, rvc]) - rv
#             if drv < 0.05:
#                 drv = 0.05
#             rva = rv
#             i = i + 1
#         elif fa > fb and fb >= fc:  # true: bounded by A
#             drv = 2 * drv
#             rva = rvc
#             if rva > rvlim:
#                 rva = rvlim
#                 i = i + 1
#             elif rva < -rvlim:
#                 rva = -rvlim
#                 i = i + 1
#         elif fa <= fb and fb < fc:  # true: bounded by C
#             drv = 2 * drv
#             if rva > rvlim:
#                 rva = rvlim
#                 i = i + 1
#             elif rva < -rvlim:
#                 rva = -rvlim
#                 i = i + 1
#         niter += 1
#         if niter == nitermax or i == 5:
#             break

#     # Done. Load return arguments.
#     if niter == nitermax:
#         # rvel remains unchanged
#         raise ValueError(
#             "sme_crvmatch: Radial velocity determination did not converge for segment centered on %i A. Vrad remains equal to the initial value for this segment."
#             % wmid
#         )
#     else:
#         rvel = rva  # load return variable
#         cscale = csnorm * mnyo / mnyc
#     return rvel, cscale

