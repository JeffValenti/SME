import numpy as np
from scipy.optimize import curve_fit

from interp_atmo_func import interp_atmo_func


def interp_atmo_constrained(x_in, y_in, err_in, par, constraints=None, **kwargs):
    """
    Apply a constraint on each parameter, to have it approach zero:
    constraint (vector[nconstraint]) - error vector for constrained parameters.
    Use errors of 0 for unconstrained parameters.
    Input structure functargs (forwarded via _extra) _must_ contain parameters
    ndep (scalar) - number of depth points in supplied quantities
    ipar (vector[3]) - initial values of fitted parameters
    Other input parameters are forwarded to interp_atmo_func, below, via mpfitfun.
    """

    ndep = kwargs.get("functargs", {}).get("ndep")
    # ndep = extra.functargs.ndep
    if constraints is not None:
        i = constraints != 0
        nconstraints = np.count_nonzero(i)
    else:
        nconstraints = 0
    if nconstraints > 0:
        x = np.concatenate([x_in, np.zeros(nconstraints)])
        y = np.concatenate([y_in, np.zeros(nconstraints)])
        err = np.concatenate([err_in, constraints[i]])
    else:
        x = x_in
        y = y_in
        err = err_in

    # Evaluate
    # ret = mpfitfun("interp_atmo_func", x, y, err, par, extra=extra, yfit=yfit)
    func = lambda x, *args: interp_atmo_func(x, args, **kwargs)
    popt, _ = curve_fit(func, x, y, sigma=err, p0=par)

    ret = popt
    yfit = func(x, *popt)

    # Remove constraints from the list of fitted points:
    if nconstraints > 0:
        yfit = yfit[0:ndep]

    return ret, yfit
