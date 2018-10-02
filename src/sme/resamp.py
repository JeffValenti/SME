import numpy as np
from .bezier import bezier_interp


def resamp(wold, sold, wnew):
    """
    #Interpolates OR integrates a spectrum onto a new wavelength scale, depending
    #  on whether number of pixels per angstrom increases or decreases. Integration
    #  is effectively done analytically under a cubic spline fit to old spectrum.
    # wold (input vector) old wavelngth scale.
    # sold (input vector) old spectrum to be binned.
    # wnew (input vector) new wavelength spectrum.
    # snew (output vector) newly binned spectrum.
    #10-Oct-90 JAV	Create.
    #22-Sep-91 JAV	Translated from IDL to ANA.
    #05-Aug-92 JAV	Changed f/ procedure to function. Renamed f/ rebin to resamp.
    #		 Switched f/ intrinsic rebin() to total() - faster.
    #25-Jan-15 JAV  Fixed logic that tests whether new wavelength range is a subset
    #                of the old. Code was only catching the case when the new range
    #                extended below the old range on both ends. Now code stops if
    #                new range extends beyond old range on either end.
    """

    # Program flags.
    trace = 0  # (0)1: (don't) print trace info

    # Verify that wold, sold, and wnew are real*8
    # wold = wold.astype(float)
    # sold = sold.astype(float)
    # wnew = wnew.astype(float)

    # Determine spectrum attributes.
    nold = len(wold)  # number of old points
    nnew = len(wnew)  # number of new points
    psold = (wold[-1] - wold[0]) / (nold - 1)  # old pixel scale
    psnew = (wnew[-1] - wnew[0]) / (nnew - 1)  # new pixel scale

    # Verify that new wavelength scale is a subset of old wavelength scale.
    wnew_lo = wnew[0] - 0.5 * (wnew[1] - wnew[0])
    wold_lo = wold[0] - 0.5 * (wold[1] - wold[0])
    wnew_hi = wnew[-1] + 0.5 * (wnew[-1] - wnew[-2])
    wold_hi = wold[-1] + 0.5 * (wold[-1] - wold[-2])
    # if (wnew_lo < wold_lo) or (wnew_hi > wold_hi):
    #     raise ValueError("New wavelength scale not subset of old.")

    # Select integration or interpolation depending on change in dispersion.
    if psnew < psold:  # pixel scale decreased
        # Interpolation by cubic spline.
        if trace:
            print("Interpolating onto new wavelength scale.")
        snew = bezier_interp(wold, sold, wnew)
    else:  # pixel scale increased
        #  Integration under cubic spline.
        if trace:
            print("Integrating onto new wavelength scale.")
        xfac = int(psnew / psold + 0.5)  # pixel scale expansion factor
        if trace:
            print("Pixel scale expansion factor: %f", xfac)

        #  Construct another wavelength scale (w) with a pixel scale close to that of
        #    the old wavelength scale (wold), but with the additional constraint that
        #    every xfac pixels in w will exactly fill a pixel in the new wavelength
        #    scale (wnew). Optimized for xfac < nnew.
        dw = 0.5 * (wnew[2:] - wnew[:-2])  # local pixel scale
        dw = np.concatenate([[2 * dw[0] - dw[1]], dw, [2 * dw[-3] - dw[-4]]])  # add trailing endpoint first add leading endpoint last
        w = np.zeros((xfac, nnew))  # initialize W as array
        for i in range(xfac):  # loop thru subpixels
            w[i] = wnew + dw * (2 * i + 1 / (2 * xfac) - 0.5)  # pixel centers in W

        w = w.T  # transpose W before Merging
        w = w.flatten()  # make W into 1-dim vector
        #  Interpolate old spectrum (sold) onto wavelength scale w to make s. Then
        #    sum every xfac pixels in s to make a single pixel in the new spectrum
        #    (snew). Equivalent to integrating under cubic spline through sold.
        s = bezier_interp(wold, sold, w)
        s = s / xfac  # take average in each pixel
        s.shape = nnew, xfac  # initialize sdummy as array
        snew = np.sum(s, axis=1)  # most efficient pixel sum
    return snew  # return resampled spectrum
