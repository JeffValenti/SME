"""
Determine continuum based on continuum mask
and fit best radial velocity to observation
"""

import logging
import warnings

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.ndimage.filters import median_filter
from scipy.optimize import least_squares
from scipy.linalg import lu_factor, lu_solve
from scipy.constants import speed_of_light


from . import sme_synth
from . import util


c_light = speed_of_light * 1e-3  # speed of light in km/s


def determine_continuum(sme, segment):
    """
    Fit a polynomial to the spectrum points marked as continuum
    The degree of the polynomial fit is determined by sme.cscale_flag

    Parameters
    ----------
    sme : SME_Struct
        input sme structure with sme.sob, sme.wave, and sme.mask
    segment : int
        index of the wavelength segment to use, or -1 when dealing with the whole spectrum

    Returns
    -------
    cscale : array of size (ndeg + 1,)
        polynomial coefficients of the continuum fit, in numpy order, i.e. largest exponent first
    """

    if segment < 0:
        return sme.cscale

    if "spec" not in sme or "mask" not in sme or "wave" not in sme or "uncs" not in sme:
        # If there is no observation, we have no continuum scale
        warnings.warn("Missing data for continuum fit")
        cscale = None
    elif sme.cscale_flag in ["none", -3]:
        cscale = [1]
    elif sme.cscale_flag in ["fix", -1, -2]:
        # Continuum flag is set to no continuum
        cscale = sme.cscale[segment]
    else:
        # fit a line to the continuum points
        if sme.cscale_flag in ["constant", 0]:
            ndeg = 0
        elif sme.cscale_flag in ["linear", 1]:
            ndeg = 1
        elif sme.cscale_flag in ["quadratic", 2]:
            ndeg = 2
        else:
            ndeg = sme.cscale_flag

        # Extract points in this segment
        x, y, m, u = sme.wave, sme.spec, sme.mask, sme.uncs
        x, y, u, m = x[segment], y[segment], u[segment], m[segment]

        # Set continuum mask
        if np.all(m != 2):
            # If no continuum mask has been set
            # Use the effective wavelength ranges of the lines to determine continuum points
            logging.info(
                "No Continuum mask was set, "
                "Using effective wavelength range of lines to find continuum instead"
            )
            cont = get_continuum_mask(x, sme.linelist, mask=m)
            # Save mask for next iteration
            m[cont == 2] = 2
            logging.debug("Continuum mask points: %i", np.count_nonzero(cont == 2))

        cont = m == 2
        x = x - x[0]
        x, y, u = x[cont], y[cont], u[cont]

        # Fit polynomial
        try:
            func = lambda coef: (np.polyval(coef, x) - y) / u
            c0 = [0] * ndeg + [1]
            res = least_squares(func, x0=c0, loss="soft_l1")
            cscale = res.x
        except TypeError:
            warnings.warn("Could not fit continuum, set continuum mask?")
            cscale = [1]

    return cscale


def get_continuum_mask(wave, linelist, threshold=0.1, mask=None):
    """
    Use the effective wavelength range of the lines,
    to find wavelength points that should be unaffected by lines
    However one usually has to ignore the weak lines, as most points are affected by one line or another
    Therefore keep increasing the threshold until enough lines have been found (>10%)

    Parameters
    ----------
    wave : array of size (n,)
        wavelength points
    linelist : LineList
        LineList object that was input into the Radiative Transfer
    threshold : float, optional
        starting threshold, lines with depth below this value are ignored
        the actual threshold is increased until enough points are found (default: 0.1)

    Returns
    -------
    mask : array(bool) of size (n,)
        True for points between lines and False for points within lines
    """

    if threshold <= 0:
        threshold = 0.01

    if mask is None:
        mask = np.full(len(wave), 1)

    width = sme_synth.GetLineRange(len(linelist))
    temp = False
    while np.count_nonzero(temp) < len(wave) * 0.1:
        temp = np.full(len(wave), True)
        for i, line in enumerate(width):
            if linelist["depth"][i] > threshold:
                w = (wave >= line[0]) & (wave <= line[1])
                temp[w] = False

        # TODO: Good value to increase threshold by?
        temp[mask == 0] = False
        threshold *= 1.1

    mask[temp] = 2

    logging.debug("Ignoring lines with depth < %f", threshold)
    return mask


def determine_radial_velocity(sme, segment, cscale, x_syn, y_syn):
    """
    Calculate radial velocity by using cross correlation and
    least-squares between observation and synthetic spectrum

    Parameters
    ----------
    sme : SME_Struct
        sme structure with observed spectrum and flags
    segment : int
        which wavelength segment to handle, -1 if its using the whole spectrum
    cscale : array of size (ndeg,)
        continuum coefficients, as determined by e.g. determine_continuum
    x_syn : array of size (n,)
        wavelength of the synthetic spectrum
    y_syn : array of size (n,)
        intensity of the synthetic spectrum

    Raises
    ------
    ValueError
        if sme.vrad_flag is not recognized

    Returns
    -------
    rvel : float
        best fit radial velocity for this segment/whole spectrum
        or None if no observation is present
    """

    if "spec" not in sme or "mask" not in sme or "wave" not in sme or "uncs" not in sme:
        # No observation no radial velocity
        warnings.warn("Missing data for radial velocity determination")
        rvel = None
    elif sme.vrad_flag in [-2, "none"]:
        # vrad_flag says don't determine radial velocity
        rvel = sme.vrad
        rvel = rvel[segment] if len(rvel) > 1 else rvel[0]
    elif sme.vrad_flag in [-1, "whole"] and segment >= 0:
        # We are inside a segment, but only want to determine rv at the end
        rvel = 0
    else:
        # Fit radial velocity
        # Extract data
        x, y, m, u = sme.wave, sme.spec, sme.mask, sme.uncs
        if sme.vrad_flag in [0, "each"]:
            # Only this one segment
            x_obs = x[segment]
            y_obs = y[segment]
            u_obs = u[segment]
            mask = m[segment]

            # apply continuum
            if cscale is not None:
                cont = np.polyval(cscale, x_obs)
            else:
                warnings.warn(
                    "No continuum scale passed to radial velocity determination"
                )
                cont = np.ones_like(y_obs)

            y_obs = y_obs / cont

        elif sme.vrad_flag in [-1, "whole"]:
            # All segments
            if cscale is not None:
                cscale = np.atleast_2d(cscale)
                cont = [np.polyval(c, x[i]) for i, c in enumerate(cscale)]
            else:
                warnings.warn(
                    "No continuum scale passed to radial velocity determination"
                )
                cont = [1 for _ in range(len(x))]

            y = y.copy()
            for i in range(len(x)):
                y[i] = y[i] / cont[i]

            x_obs = x.__values__
            y_obs = y.__values__
            u_obs = u.__values__
            mask = m.__values__
        else:
            raise ValueError(
                f"Radial velocity flag {sme.vrad_flag} not recognised, expected one of 'each', 'whole', 'none'"
            )

        y_tmp = np.interp(x_obs, x_syn, y_syn)

        # Get a first rough estimate from cross correlation
        # Subtract continuum level of 1, for better correlation
        corr = correlate(
            y_obs - np.median(y_obs), y_tmp - np.median(y_tmp), mode="same"
        )
        offset = np.argmax(corr)

        x1 = x_obs[offset]
        x2 = x_obs[len(x_obs) // 2]
        rvel = c_light * (1 - x2 / x1)

        rv_bounds = (-100, 100)
        rvel = np.clip(rvel, *rv_bounds)

        lines = mask != 0

        # Then minimize the least squares for a better fit
        # as cross correlation can only find
        def func(rv):
            rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
            shifted = interpolator(x_obs[lines] * rv_factor)
            # shifted = np.interp(x_obs[lines], x_syn * rv_factor, y_syn)
            resid = (y_obs[lines] - shifted) / u_obs[lines]
            resid = np.nan_to_num(resid, copy=False)
            return resid

        interpolator = util.safe_interpolation(x_syn, y_syn, None)
        res = least_squares(func, x0=rvel, loss="soft_l1", bounds=rv_bounds)
        rvel = res.x[0]

    return rvel


def determine_rv_and_cont(sme, segment, x_syn, y_syn):
    """
    Fits both radial velocity and continuum level simultaneously
    by comparing the synthetic spectrum to the observation

    The best fit is determined using robust least squares between
    a shifted and scaled synthetic spectrum and the observation

    Parameters
    ----------
    sme : SME_Struct
        contains the observation
    segment : int
        wavelength segment to fit
    x_syn : array of size (ngrid,)
        wavelength of the synthetic spectrum
    y_syn : array of size (ngrid,)
        intensity of the synthetic spectrum

    Returns
    -------
    rvel : float
        radial velocity in km/s
    cscale : array of size (ndeg+1,)
        polynomial coefficients of the continuum
    """

    if "spec" not in sme or "mask" not in sme or "wave" not in sme or "uncs" not in sme:
        # No observation no radial velocity
        warnings.warn("Missing data for radial velocity/continuum determination")
        return 0, [1]
    mask = sme.mask_good[segment]
    x_obs = sme.wave[segment][mask]
    y_obs = sme.spec[segment][mask]
    u_obs = sme.uncs[segment][mask]
    x_num = x_obs - sme.wave[segment][0]

    if sme.cscale_flag in [-3, "none"]:
        cflag = False
        cscale = [1]
    elif sme.cscale_flag in [-1, -2, "fix"]:
        cflag = False
        cscale = sme.cscale[segment]
    elif sme.cscale_flag in [0, "constant"]:
        ndeg = 0
        cflag = True
    elif sme.cscale_flag in [1, "linear"]:
        ndeg = 1
        cflag = True
    elif sme.cscale_flag in [2, "quadratic"]:
        ndeg = 2
        cflag = True
    else:
        raise ValueError("cscale_flag not recognized")

    if cflag:
        cscale = np.zeros(ndeg + 1)
        if sme.cscale is not None:
            cscale = sme.cscale[segment]
        else:
            cscale[-1] = np.median(y_obs)

    if sme.vrad_flag in ["none", -2]:
        rvel = 0
        vflag = False
    elif sme.vrad_flag in ["whole", -1]:
        rvel = sme.vrad[0]
        vflag = segment == -1
    elif sme.vrad_flag in ["each", 0]:
        rvel = sme.vrad[segment]
        vflag = True
    else:
        raise ValueError(f"Radial velocity Flag not understood {sme.vrad_flag}")

    # Get a first rough estimate from cross correlation
    # Subtract median (rough continuum estimate) for better correlation
    y_tmp = np.interp(x_obs, x_syn, y_syn)
    corr = correlate(y_obs - np.median(y_obs), y_tmp - np.median(y_tmp), mode="same")
    offset = np.argmax(corr)

    x1 = x_obs[offset]
    x2 = x_obs[len(x_obs) // 2]
    rvel = c_light * (1 - x2 / x1)

    interpolator = util.safe_interpolation(x_syn, y_syn, None)

    def func(par):
        # The radial velocity shift is inversed
        # i.e. the wavelength grid of the observation is shifted to match the synthetic spectrum
        # but thats ok, because the shift is symmetric
        if vflag:
            rv = par[0]
            rv_factor = np.sqrt((1 - rv / c_light) / (1 + rv / c_light))
            shifted = interpolator(x_obs * rv_factor)
        else:
            shifted = x_obs

        if cflag:
            coef = par[1:]
            shifted *= np.polyval(coef, x_num)

        resid = (y_obs - shifted) / u_obs
        resid = np.nan_to_num(resid, copy=False)
        return resid

    x0 = [rvel, *cscale]
    res = least_squares(func, x0=x0, loss="soft_l1")

    rvel = res.x[0]
    cscale = res.x[1:]

    return rvel, cscale


def cont_fit(sme, segment, x_syn, y_syn, rvel=0):
    """
    Fit a continuum when no continuum points exist

    Parameters
    ----------
    sme : SME_Struct
        sme structure with observation data
    segment : int
        index of the wavelength segment to fit
    x_syn : array of size (n,)
        wavelengths of the synthetic spectrum
    y_syn : array of size (n,)
        intensity of the synthetic spectrum
    rvel : float, optional
        radial velocity in km/s to apply to the wavelength (default: 0)

    Returns
    -------
    continuum : array of size (ndeg,)
        continuum fit polynomial coefficients
    """

    eps = np.mean(sme.uncs[segment])
    weights = sme.spec[segment] / sme.uncs[segment] ** 2
    weights[sme.mask_bad[segment]] = 0

    order = sme.cscale_degree

    xarg = sme.wave[segment]
    yarg = sme.spec[segment]
    yc = np.interp(xarg * (1 - rvel / c_light), x_syn, y_syn)
    yarg = yarg / yc

    if order <= 0 or order > 2:
        # Only polynomial orders up to 2 are supported
        # Return a constant scale otherwise (same as order == 0)
        scl = np.sum(weights * yarg) / np.sum(weights)
        return [scl]

    iterations = 10
    xx = (xarg - (np.max(xarg) - np.min(xarg)) * 0.5) / (
        (np.max(xarg) - np.min(xarg)) * 0.5
    )
    fmin = np.min(yarg) - 1
    fmax = np.max(yarg) + 1
    ff = (yarg - fmin) / (fmax - fmin)
    ff_old = ff

    def linear(a, b):
        a[1, 1] -= a[0, 1] ** 2 / a[0, 0]
        b -= b[::-1] * a[0, 1] / np.diag(a)[::-1]
        cfit = b / np.diag(a)
        return cfit[::-1]

    def quadratic(a, b):
        lu, index = lu_factor(a)
        cfit = lu_solve((lu, index), b)
        return cfit[::-1]

    if order == 1:
        func = linear
    elif order == 2:
        func = quadratic

    for _ in range(iterations):
        n = order + 1
        a = np.zeros((n, n))
        b = np.zeros(n)

        for j, k in product(range(order + 1), repeat=2):
            a[j, k] = np.sum(weights * xx ** (j + k))

        for j in range(order + 1):
            b[j] = np.sum(weights * ff * xx ** j)

        cfit = func(a, b)

        dev = np.polyval(cfit, xx)
        t = median_filter(dev, size=3)
        tt = (t - ff) ** 2

        for j in range(n):
            b[j] = np.sum(weights * tt * xx ** j)

        dev = np.polyval(cfit, xx)
        dev = np.clip(dev, 0, None)
        dev = np.sqrt(dev)

        ff = np.clip(t - dev, ff, t + dev)
        dev2 = np.max(weights * np.abs(ff - ff_old))
        ff_old = ff

        if dev2 < eps:
            break

    coef = np.polyfit(xx, ff, order)
    t = np.polyval(coef, xx)

    # Get coefficients in the wavelength scale
    t = t * (fmax - fmin) + fmin
    coef = np.polyfit(xarg - xarg[0], t, order)

    return coef


def match_rv_continuum(sme, segment, x_syn, y_syn):
    """
    Match both the continuum and the radial velocity of observed/synthetic spectrum

    Note that the parameterization of the continuum is different to old SME !!!

    Parameters
    ----------
    sme : SME_Struct
        input sme structure with all the parameters
    segment : int
        index of the wavelength segment to match, or -1 when dealing with the whole spectrum
    x_syn : array of size (n,)
        wavelength of the synthetic spectrum
    y_syn : array of size (n,)
        intensitz of the synthetic spectrum

    Returns
    -------
    rvel : float
        new radial velocity
    cscale : array of size (ndeg + 1,)
        new continuum coefficients
    """

    rvel, cscale = determine_rv_and_cont(sme, segment, x_syn, y_syn)

    # Scale using relative depth (from Nikolai)
    cscale = cont_fit(sme, segment, x_syn, y_syn, rvel=rvel)

    # cscale = determine_continuum(sme, segment)
    # rvel = determine_radial_velocity(sme, segment, cscale, x_syn, y_syn)

    return cscale, rvel
