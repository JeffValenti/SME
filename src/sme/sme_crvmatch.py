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

        # Then minimize the least squares for a better fit
        # as cross correlation can only find
        def func(x):
            tmp = np.interp(x_obs[lines] * (1 - x / c_light), x_syn, y_syn)
            return np.sum((y_obs[lines] - tmp) ** 2 * u_obs[lines] ** -2)

        res = minimize(func, x0=rvel)
        rvel = res.x[0]

    return rvel, cscale
