import numpy as np
import os.path
import glob

from scipy.ndimage.filters import convolve
from .bezier import interpolate as spl_interp


def rtint(mu, inten, deltav, vsini, vrt, osamp=1):
    """
    Produces a flux profile by integrating intensity profiles (sampled
    at various mu angles) over the visible stellar surface.

    Intensity profiles are weighted by the fraction of the projected
    stellar surface they represent, apportioning the area between
    adjacent MU points equally. Additional weights (such as those
    used in a Gauss-Legendre quadrature) can not meaningfully be
    used in this scheme.  About twice as many points are required
    with this scheme to achieve the precision of Gauss-Legendre
    quadrature.
    DELTAV, VSINI, and VRT must all be in the same units (e.g. km/s).
    If specified, OSAMP should be a positive integer.

    Parameters:
    ----------
    mu : array(float) of size (nmu,)
        cosine of the angle between the outward normal and
        the line of sight for each intensity spectrum in INTEN.
    inten : array(float) of size(nmu, npts)
        intensity spectra at specified values of MU.
    deltav : float
        velocity spacing between adjacent spectrum points
        in INTEN (same units as VSINI and VRT).
    vsini : float
        maximum radial velocity, due to solid-body rotation.
    vrt : float
        radial-tangential macroturbulence parameter, i.e.
        np.sqrt(2) times the standard deviation of a Gaussian distribution
        of turbulent velocities. The same distribution function describes
        the radial motions of one component and the tangential motions of
        a second component. Each component covers half the stellar surface.
        See _The Observation and Analysis of Stellar Photospheres_, Gray.
    osamp : int, optional
        internal oversampling factor for convolutions.
        By default convolutions are done using the input points (OSAMP=1),
        but when OSAMP is set to higher integer values, the input spectra
        are first oversampled by cubic spline interpolation.

    Returns:
    -------
    value : array(float) of size (npts,)
        Disk integrated flux profile.

    Author's request:
    ------------
        If you use this algorithm in work that you publish, please cite
        Valenti & Anderson 1996, PASP, currently in preparation.

    History:
    -----------
    Feb-88  GM
        Created ANA version.
    13-Oct-92 JAV
        Adapted from G. Marcy's ANA routi!= of the same name.
    03-Nov-93 JAV
        Switched to annular convolution technique.
    12-Nov-93 JAV
        Fixed bug. Intensity compo!=nts not added when vsini=0.
    14-Jun-94 JAV
        Reformatted for "public" release. Heavily commented.
        Pass deltav instead of 2.998d5/deltav. Added osamp
        keyword. Added rebinning logic at end of routine.
        Changed default osamp from 3 to 1.
    20-Feb-95 JAV
        Added mu as an argument to handle arbitrary mu sampling
        and remove ambiguity in intensity profile ordering.
        Interpret VTURB as np.sqrt(2)*sigma instead of just sigma.
        Replaced call_external with call to spl_{init|interp}.
    03-Apr-95 JAV
        Multiply flux by pi to give observed flux.
    24-Oct-95 JAV
        Force "nmk" padding to be at least 3 pixels.
    18-Dec-95 JAV
        Renamed from dskint() to rtint(). No longer make local
        copy of intensities. Use radial-tangential instead
        of isotropic Gaussian macroturbulence.
    26-Jan-99 JAV
        For NMU=1 and VSINI=0, assume resolved solar surface#
        apply R-T macro, but supress vsini broadening.
    01-Apr-99 GMH
        Use annuli weights, rather than assuming ==ual area.
    07-Mar-12 JAV
        Force vsini and vmac to be scalars.
    """

    # Make local copies of various input variables, which will be altered below.
    # Force vsini and especially vmac to be scalars. Otherwise mu dependence fails.

    if np.size(vsini) > 1:
        vsini = vsini[0]
    if np.size(vrt) > 1:
        vrt = vrt[0]
    vrt = abs(vrt)  # ensure real number

    # Determine oversampling factor.
    os = round(np.clip(osamp, 1, None))  # force integral value > 1

    # Convert input MU to projected radii, R, of annuli for a star of unit radius
    #  (which is just sine, rather than cosine, of the angle between the outward
    #  normal and the line of sight).
    rmu = np.sqrt(1 - mu ** 2)  # use simple trig identity

    # Sort the projected radii and corresponding intensity spectra into ascending
    #  order (i.e. from disk center to the limb), which is equivalent to sorting
    #  MU in descending order.
    isort = np.argsort(rmu)  # sorted indicies
    rmu = rmu[isort]  # reorder projected radii
    nmu = np.size(mu)  # number of radii
    if nmu == 1:
        vsini = 0  # ignore vsini if only 1 mu

    # Calculate projected radii for boundaries of disk integration annuli.  The n+1
    # boundaries are selected such that r(i+1) exactly bisects the area between
    # rmu(i) and rmu(i+1). The in!=rmost boundary, r(0) is set to 0 (disk center)
    # and the outermost boundary, r(nmu) is set to 1 (limb).
    if nmu > 1 or vsini != 0:  # really want disk integration
        r = np.sqrt(0.5 * (rmu[:-1] ** 2 + rmu[1:] ** 2))  # area midpoints between rmu
        r = np.concatenate(([0], r, [1]))

        # Calculate integration weights for each disk integration annulus.  The weight
        # is just given by the relative area of each annulus, normalized such that
        # the sum of all weights is unity.  Weights for limb darkening are included
        # explicitly in the intensity profiles, so they aren't needed here.
        wt = r[1:] ** 2 - r[:-1] ** 2  # weights = relative areas
    else:
        wt = np.array([1.0])  # single mu value, full weight

    # Generate index vectors for input and oversampled points. Note that the
    # oversampled indicies are carefully chosen such that every "os" finely
    # sampled points fit exactly into one input bin. This makes it simple to
    # "integrate" the finely sampled points at the end of the routine.
    npts = inten.shape[1]  # number of points
    xpix = np.arange(npts)  # point indices
    nfine = int(os * npts)  # number of oversampled points
    xfine = (0.5 / os) * (2 * np.arange(nfine) - os + 1)  # oversampled points indices

    # Loop through annuli, constructing and convolving with rotation kernels.
    dummy = 0  # init call_ext() return value
    yfine = np.empty(nfine)  # init oversampled intensities
    flux = np.zeros(nfine)  # init flux vector
    for imu in range(nmu):  # loop thru integration annuli

        #  Use external cubic spline routine (adapted from Numerical Recipes) to make
        #  an oversampled version of the intensity profile for the current annulus.
        ypix = inten[isort[imu]]  # extract intensity profile
        if os == 1:
            # just copy (use) original profile
            yfine = ypix
        else:
            # spline onto fine wavelength scale
            yfine = spl_interp(xpix, ypix, xfine)

        # Construct the convolution kernel which describes the distribution of
        # rotational velocities present in the current annulus. The distribution has
        # been derived analytically for annuli of arbitrary thickness in a rigidly
        # rotating star. The kernel is constructed in two pieces: o!= piece for
        # radial velocities less than the maximum velocity along the inner edge of
        # the annulus, and one piece for velocities greater than this limit.
        if vsini > 0:
            # nontrivial case
            r1 = r[imu]  # inner edge of annulus
            r2 = r[imu + 1]  # outer edge of annulus
            dv = deltav / os  # oversampled velocity spacing
            maxv = vsini * r2  # maximum velocity in annulus
            nrk = 2 * int(maxv / dv) + 3  ## oversampled kernel point
            # velocity scale for kernel
            v = dv * (np.arange(nrk, dtype=float) - ((nrk - 1) / 2))
            rkern = np.zeros(nrk)  # init rotational kernel
            j1 = np.abs(v) < vsini * r1  # low velocity points
            rkern[j1] = np.sqrt((vsini * r2) ** 2 - v[j1] ** 2) - np.sqrt(
                (vsini * r1) ** 2 - v[j1] ** 2
            )  # generate distribution

            j2 = (np.abs(v) >= vsini * r1) & (np.abs(v) <= vsini * r2)
            rkern[j2] = np.sqrt((vsini * r2) ** 2 - v[j2] ** 2)  # generate distribution

            rkern = rkern / np.sum(rkern)  # normalize kernel

            # Convolve the intensity profile with the rotational velocity kernel for this
            # annulus. Pad each end of the profile with as many points as are in the
            # convolution kernel. This reduces Fourier ringing. The convolution may also
            # be do!= with a routi!= called "externally" from IDL, which efficiently
            # shifts and adds.
            if nrk > 3:
                yfine = convolve(yfine, rkern, mode="nearest")

        # Calculate projected sigma for radial and tangential velocity distributions.
        muval = mu[isort[imu]]  # current value of mu
        sigma = os * vrt / np.sqrt(2) / deltav  # standard deviation in points
        sigr = sigma * muval  # reduce by current mu value
        sigt = sigma * np.sqrt(1.0 - muval ** 2)  # reduce by np.sqrt(1-mu**2)

        # Figure out how many points to use in macroturbulence kernel.
        nmk = np.clip(10 * sigma, None, (nfine - 3) / 2)
        # extend kernel to 10 sigma
        nmk = int(np.clip(nmk, 3, None))  # pad with at least 3 pixels

        # Construct radial macroturbulence kernel with a sigma of mu*VRT/np.sqrt(2).
        if sigr > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigr  # expo!=ntial argument
            mrkern = np.exp(
                np.clip(-0.5 * xarg ** 2, -20, None)
            )  # compute the gaussian
            mrkern = mrkern / np.sum(mrkern)  # normalize the profile
        else:
            mrkern = np.zeros(2 * nmk + 1)  # init with 0d0
            mrkern[nmk] = 1.0  # delta function

        # Construct tangential kernel with a sigma of np.sqrt(1-mu**2)*VRT/np.sqrt(2).
        if sigt > 0:
            xarg = (np.arange(2 * nmk + 1) - nmk) / sigt  # expo!=ntial argument
            mtkern = np.exp(
                np.clip(-0.5 * xarg ** 2, -20, None)
            )  # compute the gaussian
            mtkern = mtkern / np.sum(mtkern)  # normalize the profile
        else:
            mtkern = np.zeros(2 * nmk + 1)  # init with 0d0
            mtkern[nmk] = 1.0  # delta function

        # Sum the radial and tangential components, weighted by surface area.
        area_r = 0.5  # assume equal areas
        area_t = 0.5  # ar+at must equal 1
        mkern = area_r * mrkern + area_t * mtkern  # add both components

        # Convolve the total flux profiles, again padding the spectrum on both ends to
        # protect against Fourier ringing.
        yfine = convolve(yfine, mkern, mode="nearest")  # add the padding and convolve

        # Add contribution from current annulus to the running total.
        flux = flux + wt[imu] * yfine  # add profile to running total

    flux = np.reshape(flux, (npts, os))  # convert to an array
    flux = np.pi * np.sum(flux, axis=1) / os  # sum, normalize
    return flux


def rdpop(species, wave, e_low, model, pop_dir=None, eps_wave=4e-6, eps_energy=0.01):
    """
    rdpop reads departure coefficients from one or several files and returns them
    as a 2 x ndepth array referring to the two levels involved in the transition and
    the number of depth points in model atmosphere.

    History:
    --------
    XX-XX-2012
        NP wrote.
    27-03-2013
        NP added optional parameters for pop-file(s) directory replacing
        explicit pop file name.
    28-03-2013 NP
        added checks for energy of the lower level and ionization,
        and optional parameters for the test criteria.
    """

    sp = species.strip().replace(
        " ", "_"
    )  # Trim species name and replace all spaces with underscore
    sss = sp.split()
    ion = int(sss[1])
    sp = sss[0]

    if pop_dir is not None:  # Construct pop-file template
        file = os.path.join(pop_dir, sp + "*_*" + model + "*.pop")
    else:
        file = os.path.join(".", sp + "*_*" + model + "*.pop")

    files = glob.glob(file)
    nfiles = len(files)  # Find all relevant pop-files

    if nfiles == 0:
        raise IOError(
            "No files matching this template: " + file + " were found! Returning ..."
        )

    if nfiles == 1:
        files = [files]

    level = {
        "__name__": "energy_level",
        "id": " ",
        "number": -1,
        "energy": 0.,
        "ion": 0,
    }
    b = 0
    for ifile in range(nfiles):
        # Loop through the pop-files
        un = open(files[ifile])
        nlines = len(un)
        if nlines <= 0:
            un.close()
            continue
        s = ""
        i1 = 0
        i2 = 0
        id = ""
        w = 0
        energy = 0.0
        n_low = -1
        n_upp = -1
        levels = np.full(2000, level).view(np.recarray)
        nlevels = 0
        for i in range(nlines):
            # Read file
            s = un.readline()
            if s[0] == "%":
                continue
            ss = s.strip().replace(" ", "_")
            if "ATOM=" in ss.upper():
                atom = ss[5:]
            if "NDEPTH=" in ss.upper():
                ndepth = int(ss[7:])
            if "LEVEL=" in ss.upper():
                sss = ss.split()
                levels[nlevels].id = sss[1].strip()
                levels[nlevels].number = int(sss[2])
                levels[nlevels].energy = float(sss[3])
                levels[nlevels].ion = int(sss[4])
                nlevels = nlevels + 1
            if "RBB=" in ss.upper():
                levels = levels[0:nlevels]
                i1, i2, w = ss[4:].split()
                # reads, strmid(ss, 4), i1, i2, w
                ii = levels.number == i1
                nii = np.count_nonzero(ii)
                if nii > 0:
                    energy = levels[ii[0]].energy
                    if (
                        abs(wave - w) / wave < eps_wave
                        and abs(e_low - energy) < eps_energy
                        and levels[ii[0]].ion == ion
                    ):
                        n_low = i1
                        n_upp = i2
            if "DEPARTURES=" in ss.upper():
                break

        if n_low < 0 or n_upp < 0:
            un.close()
            continue

        b_low = np.zeros(ndepth)
        b_upp = np.zeros(ndepth)
        bb = np.zeros((ndepth, 2))
        for i in range(1, n_upp + 1):
            level = int(ss[11:])
            bb = un.readline()
            if level == n_low:
                b_low = bb[:, 0]
            if level == n_upp:
                b_upp = bb[:, 0]
            s = un.readline()
            ss = s.strip().replace(" ", "_")

        b = np.transpose(([[[b_low]], [[b_upp]]]))

        b[b == 0] = 1.
    return b
