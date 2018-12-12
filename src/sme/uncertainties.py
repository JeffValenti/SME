import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def uncertainties(pder, resid, unc, freep_name, plot=False):
    binsize = np.median(np.abs(resid))
    nbins = (np.max(resid) - np.min(resid)) / binsize + 1
    nbins = int(nbins)
    h, x = np.histogram(resid, bins=nbins)
    x = x[:-1] + np.diff(x) / 2

    _, a = gaussfit(x, h, nterms="constant")
    sigma = a[2]

    # 6 sigma threshold
    resid_threshold = 6 * sigma

    nfree = len(freep_name)
    freep_unc = np.zeros(nfree)
    freep_med = np.zeros(nfree)
    freep_msig = np.zeros(nfree)
    freep_psig = np.zeros(nfree)

    for ifree in range(nfree):
        gradlim = np.median(np.abs(pder[:, ifree]))
        i = np.where(
            (np.abs(pder[:, ifree]) > gradlim) & (np.abs(resid) < resid_threshold)
        )[0]
        ni = len(i)
        if ni > 21:
            ii = np.argsort(resid[i] / pder[i, ifree])
            # Sort pixels according to the change of the ifree
            # parameter needed to match the observations
            ch_x = resid[i[ii]] / pder[i[ii], ifree]
            # Weights of the individual pixels also sorted
            ch_y = np.abs(pder[i[ii], ifree]) / unc[i[ii]]
            # Cumulative weights
            ch_y = np.cumsum(ch_y)
            # Normalized cumulative weights
            ch_y = ch_y / ch_y[-1]

            # value of distribution at -sigma
            freep_msig[ifree] = np.interp(0.5 - 0.6827 / 2, ch_x, ch_y)
            # value of distribution at +sigma
            freep_psig[ifree] = np.interp(0.5 + 0.6827 / 2, ch_x, ch_y)
            # Mean sigma
            freep_unc[ifree] = (freep_psig[ifree] - freep_msig[ifree]) * 0.5
            # Median change
            freep_med[ifree] = np.median(ch_y)

        print(
            "%s : %.3f %.3f %.3f %.3f"
            % (
                freep_name[ifree],
                freep_med[ifree],
                freep_msig[ifree],
                freep_psig[ifree],
                freep_unc[ifree],
            )
        )

        if plot:
            # TODO plot
            pass

    return freep_med, freep_msig, freep_psig, freep_unc


def gaussfit(x, y, nterms="none"):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma) = a * exp(-z**2/2)
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    z = lambda x, A1, A2: (x - A1) / A2

    if nterms in ["none", 3]:
        gauss = lambda x, A0, A1, A2: A0 * np.exp(-z(x, A1, A2) ** 2 / 2)
        p0 = [max(y), 1, 1]
    elif nterms in ["constant", 4]:
        gauss = lambda x, A0, A1, A2, A3: A0 * np.exp(-z(x, A1, A2) ** 2 / 2) + A3
        p0 = [max(y), 1, 1, 0]
    elif nterms in ["linear", 5]:
        gauss = (
            lambda x, A0, A1, A2, A3, A4: A0 * np.exp(-z(x, A1, A2) ** 2 / 2)
            + A3
            + A4 * x
        )
        p0 = [max(y), 1, 1, 0, 0]

    elif nterms in ["quadratic", 6]:
        gauss = (
            lambda x, A0, A1, A2, A3, A4, A5: A0 * np.exp(-z(x, A1, A2) ** 2 / 2)
            + A3
            + A4 * x
            + A5 * x ** 2
        )
        p0 = [max(y), 1, 1, 0, 0, 0]

    popt, _ = curve_fit(gauss, x, y, p0=p0)
    return gauss(x, *popt), popt
