"""
Handle Bezier interpolation just like bezier_init and bezier_interp in IDL(SME)
"""

import numpy as np
import scipy.interpolate

def interpolate(x_old, y_old, x_new):
    return bezier_interp(x_old, y_old, x_new)

def bezier_interp(x_old, y_old, x_new):
    """Do Bezier spline interpolation
    
    Parameters
    ----------
    x_old : array
        old x values
    y_old : array
        old y values
    x_new : array
        new x values
    
    Returns
    -------
    y_new
        interpolated y values
    """

    # Handle masked arrays
    if np.ma.is_masked(x_old):
        x_old = np.ma.compressed(x_old)
        y_old = np.ma.compressed(y_old)

    if np.ma.is_masked(x_new):
        #x_new_tmp = np.ma.compressed(x_new)
        mask = x_new.mask
    else:
        mask = None

    x_old, index = np.unique(x_old, return_index=True)
    y_old = y_old[index]

    knots, coef, order = scipy.interpolate.splrep(x_old, y_old)
    y_new = scipy.interpolate.BSpline(knots, coef, order)(x_new)

    if mask is not None:
        y_new = np.ma.masked_array(y_new, mask=mask)
    return y_new

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, endpoint=False)
    y = np.sin(x)
    xa = np.linspace(0, 9, 20)
    ya = interpolate(x, y, xa)
    plt.plot(x, y, label="old")
    plt.plot(xa, ya, label="new")
    plt.legend(loc="best")
    plt.show()
