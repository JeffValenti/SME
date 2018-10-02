import numpy as np

from scipy.ndimage.filters import convolve
from .bezier import bezier_interp

def apply_broadening(ipres, x_seg, y_seg, type="gauss", sme=None):
    # Using the log-linear wavelength grid requires using the first point
    # for specifying the the width of the instrumental profile
    hwhm = 0.5 * x_seg[0] / ipres if ipres > 0 else 0

    if sme.iptype == "table":
        y_seg = tablebroad(x_seg, y_seg, sme.ip_x, sme.ip_y)
    elif sme.iptype == "gauss":
        y_seg = gaussbroad(x_seg, y_seg, hwhm)
    elif sme.iptype == "sinc":
        y_seg = sincbroad(x_seg, y_seg, hwhm)
    else:
        raise AttributeError("Unknown IP type - " + sme.iptype)

    return y_seg


def tablebroad(w, s, xip, yip):
    """ Convolves a spectrum with an instrumental profile.
    22-May-92 JAV	Switched instrumental profile from multiple gaussians
    		 to gaussian with power-law wings.
    04-Aug-92 JAV	Renamed from ipsmo.pro# changed f/ procedure to function.
    		 Switched f/ 10 to 15 Hamilton pixels in each wing.
    20-Oct-92 JAV	Switched from gpfunc to ipfun (4 to 5 par).
    23-Aug-94 JAV	Switched to explicitly passed IPs.
    """

    dsdh = s

    # Define sizes.
    ns = len(s)  ## points in spectrum
    nip = 2 * int(15 / dsdh) + 1  ## profile points

    # Generate instrumental profile on model pixel scale.
    x = (
        np.arange(nip, dtype=float) - (nip - 1) / 2
    ) * dsdh  # offset in Hamilton pixels
    ip = bezier_interp(xip, yip, x)  # spline onto new scale
    ip = ip[::-1]  # reverse for convolution
    ip = ip / np.sum(ip)  # ensure unit area

    # Pad spectrum ends to minimize impact of Fourier ringing.
    snew = convolve(s, ip, mode="nearest")
    # npad = nip + 2  ## pad pixels on each end
    # snew = np.pad(s, pad_width=npad, mode="edge")  # add lots of end padding

    # # Convolve and trim.
    # snew = np.convolve(snew, ip)  # convolve with gaussian
    # snew = snew[npad : -npad]  # trim to original data/length
    return snew  # return convolved spectrum


def gaussbroad(w, s, hwhm):
    """ #Smooths a spectrum by convolution with a gaussian of specified hwhm.
    # w (input vector) wavelength scale of spectrum to be smoothed
    # s (input vector) spectrum to be smoothed
    # hwhm (input scalar) half width at half maximum of smoothing gaussian.
    #Returns a vector containing the gaussian-smoothed spectrum.
    #Edit History:
    #  -Dec-90 GB,GM Rewrote with fourier convolution algorithm.
    #  -Jul-91 AL	Translated from ANA to IDL.
    #22-Sep-91 JAV	Relaxed constant dispersion check# vectorized, 50% faster.
    #05-Jul-92 JAV	Converted to function, handle nonpositive hwhm.
    """

    # Warn user if hwhm is negative.
    if hwhm < 0:
        print("Warning! Forcing negative smoothing width to zero.")

    # Return input argument if half-width is nonpositive.
    if hwhm <= 0:
        return s  # true: no broadening

    # Calculate (uniform) dispersion.
    nw = len(w)  ## points in spectrum
    dw = (w[-1] - w[0]) / (nw - 1)  # wavelength change per pixel

    # Make smoothing gaussian# extend to 4 sigma.
    # Note: 4.0 / sqrt(2.0*alog(2.0)) = 3.3972872 and sqrt(alog(2.0))=0.83255461
    #  sqrt(alog(2.0)/pi)=0.46971864 (*1.0000632 to correct for >4 sigma wings)
    if hwhm >= 5 * (w[-1] - w[0]):
        return np.full(nw, np.sum(s) / nw)
    nhalf = int(3.3972872 * hwhm / dw)  ## points in half gaussian
    ng = 2 * nhalf + 1  ## points in gaussian (odd!)
    wg = dw * (
        np.arange(ng, dtype=float) - (ng - 1) / 2
    )  # wavelength scale of gaussian
    xg = (0.83255461 / hwhm) * wg  # convenient absisca
    gpro = (0.46974832 * dw / hwhm) * np.exp(-xg * xg)  # unit area gaussian w/ FWHM
    gpro = gpro / np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    sout = convolve(s, gpro, mode="nearest")
    # npad = nhalf + 2  ## pad pixels on each end
    # spad = np.pad(s, npad, mode="edge")

    # # Convolve and trim.
    # sout = np.convolve(spad, gpro)  # convolve with gaussian
    # sout = sout[npad : -npad]  # trim to original data/length
    return sout  # return broadened spectrum.


def sincbroad(w, s, hwhm):
    """
    #Smooths a spectrum by convolution with a sinc function of specified hwhm.
    # w (input vector) wavelength scale of spectrum to be smoothed
    # s (input vector) spectrum to be smoothed
    # hwhm (input scalar) half width at half maximum of smoothing gaussian.
    #Returns a vector containing the gaussian-smoothed spectrum.
    #Edit History:
    #  -Dec-90 GB,GM Rewrote with fourier convolution algorithm.
    #  -Jul-91 AL	Translated from ANA to IDL.
    #22-Sep-91 JAV	Relaxed constant dispersion check# vectorized, 50% faster.
    #05-Jul-92 JAV	Converted to function, handle nonpositive hwhm.
    #14-Nov-93 JAV	Adapted from macbro.pro
    #23-Apr-93 JAV	Verified that convolution kernel has specified hwhm. For IR FTS
    #		 spectra: hwhm=0.0759 Angstroms, max change in profile is 0.4%
    #		 of continuum.
    """

    # Warn user if hwhm is negative.
    if hwhm < 0:
        print("Warning! Forcing negative smoothing width to zero.")

    # Return input argument if half-width is nonpositive.
    if hwhm <= 0:
        return s  # true: no broadening

    # Calculate (uniform) dispersion.
    nw = len(w)  ## points in spectrum
    dw = (w[-1] - w[0]) / (nw - 1)  # wavelength change per pixel

    # Make sinc function out to 20th zero-crossing on either side. Error due to
    # ignoring additional lobes is less than 0.2% of continuum. Reducing extent
    # to 10th zero-crossing doubles maximum error.
    fwhm = 2.0 * hwhm  # full width at half maximum
    rperfw = 0.26525  # radians per fwhm of sinc
    xrange = 20 * np.pi  # 20th zero of sinc (radians)
    wrange = xrange * fwhm * rperfw  # 20th zero of sinc (wavelength)
    nhalf = int(wrange / dw + 0.999)  ## points in half sinc
    nsinc = 2 * nhalf + 1  ## points in sinc (odd!)
    wsinc = (np.arange(nsinc, dtype=float) - nhalf) * dw  # absissca (wavelength)
    xsinc = wsinc / (fwhm * rperfw)  # absissca (radians)
    xsinc[nhalf] = 1.0  # avoid divide by zero
    sinc = np.sin(xsinc) / xsinc  # calculate sinc
    sinc[nhalf] = 1.0  # insert midpoint
    xsinc[nhalf] = 0.0  # fix xsinc
    sinc = sinc / np.sum(sinc)  # normalize sinc

    # Pad spectrum ends to minimize impact of Fourier ringing.
    sout = convolve(s, sinc, mode="nearest")
    # npad = nhalf + 2  ## pad pixels on each end
    # spad = np.pad(s, npad, mode="edge")

    # # Convolve and trim.
    # sout = np.convolve(spad, sinc)  # convolve with gaussian
    # sout = sout[npad:-npad]  # trim to original data/length
    return sout  # return broadened spectrum.
