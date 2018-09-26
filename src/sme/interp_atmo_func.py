import numpy as np
from scipy.interpolate import interp1d


def interp_atmo_func(x1, par, ndep=None, **kwargs):
    # Apply a horizontal shift to x2 (passed in _extra).
    # Apply a vertical shift to y2 (passed in _extra).
    # Interpolate onto x1 the shifted y2 as a function of shifted x2.
    #
    # Inputs:
    # x1 (vector[ndep1]) independent variable for output function
    # par (vector[3]) shift parameters
    #  par[0] - horizontal shift for x2
    #  par[1] - vertical shift for y2
    #  par[2] - vertical scale factor for y2
    # extra (structure) input atmosphere
    #  .x2 (vector[ndep2]) independent variable for tabulated input function
    #  .y2 (vector[ndep2]) dependent variable for tabulated input function
    #  .y1 (vector[ndep2]) data values being fitted
    #  .ndep (scalar) - number of depth points in atmospheric structure.
    #      Additional input parameters are bogus values for constrained fitting.
    #
    # Note:
    # Only pass extra.y1 if you want to restrict the y-values of extrapolated
    #  data points. This is useful when solving for the shifts, but should
    #  not be used when calculating shifted functions for actual use, since
    #  this restriction can cause discontinuities.

    # Constrained fits may append non-atmospheric quantities to the end of
    #  input vector.
    # Extract the output depth scale:
    if ndep is not None:
        x = x1[:ndep]
    else:
        x = x1

    # Shift input x-values.
    # Interpolate input y-values onto output x-values.
    # Shift output y-values.
    x2sh = kwargs.get("x2", 0) + par[0]
    y2sh = kwargs.get("y2", 0) + par[1]
    y1 = interp1d(x2sh, y2sh, kind="linear", fill_value="extrapolate")(
        x
    )  # Note, this implicitly extrapolates

    # Scale output y-values about output y-center.
    y1min = np.min(y1)
    y1max = np.max(y1)
    y1cen = 0.5 * (y1min + y1max)
    y1 = y1cen + (1.0 + par[2]) * (y1 - y1cen)

    # If extra.y1 was passed, then clip minimum and maximum of output y1.
    if "y1" in kwargs.keys():
        y1 = np.clip(y1, np.max(kwargs["y1"]), np.min(kwargs["y1"]))

    # Append information for constrained fits:
    y = y1
    if ndep is not None:
        y = np.concatenate([y1, par - kwargs.get("ipar", 0)])
    # Return function value.
    return y
