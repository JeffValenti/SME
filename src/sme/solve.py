"""
Calculates the spectrum, based on a set of stellar parameters
And also determines the best fit parameters
"""

import os.path
import warnings
from itertools import combinations, product

import numpy as np
from scipy.io import readsav
from scipy.constants import speed_of_light
from scipy.optimize import OptimizeWarning, least_squares
from scipy.optimize._numdiff import approx_derivative

from . import broadening, sme_synth
from .interpolate_atmosphere import interp_atmo_grid
from .rtint import rdpop, rtint
from .sme_crvmatch import match_rv_continuum
from .nlte import update_depcoeffs
from .abund import Abund

# from . import sme as SME

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", OptimizeWarning)

clight = speed_of_light * 1e-3  # km/s
elements = Abund._elem


def pass_nlte(sme):
    nlines = len(sme.species)
    ndep = len(sme.atmo.temp)
    b_nlte = np.ones((ndep, nlines, 2))  # initialize the departure coefficient array
    modname = os.path.basename(sme.atm_file[0] + ".krz")
    poppath = os.path.dirname(sme.atm_file[0])
    for iline in range(nlines):
        bnlte = rdpop(
            sme.species[iline],
            sme.atomic[2, iline],
            sme.atomic[3, iline],
            modname,
            pop_dir=poppath,
        )
        if len(bnlte) == 2 * ndep:
            b_nlte[:, iline, :] = bnlte

    error = sme_synth.InputNLTE(b_nlte, 0)
    return error


# @memory.cache
def sme_func_atmo(sme):
    """
    Purpose:
     Return an atmosphere based on specification in an SME structure

    Inputs:
     SME (structure) atmosphere specification

    Outputs:
     ATMO (structure) atmosphere structure
     [.WLSTD] (scalar) wavelength for continuum optical depth scale [A]
     [.TAU] (vector[ndep]) optical depth scale,
     [.RHOX] (vector[ndep]) mass column scale
      .TEMP (vector[ndep]) temperature vs. depth
      .XNE (vector[ndep]) electron number density vs. depth
      .XNA (vector[ndep]) atomic number density vs. depth
      .RHO (vector[ndep]) mass density vs. depth

    History:
     2013-Sep-23 Valenti Extracted and adapted from sme_func.pro
     2013-Dec-13 Valenti Bundle atmosphere variables in ATMO structure
    """

    # Handle atmosphere grid or user routine.
    atmo = sme.atmo
    self = sme_func_atmo

    if hasattr(self, "msdi_save"):
        msdi_save = self.msdi_save
        prev_msdi = self.prev_msdi
    else:
        msdi_save = None
        prev_msdi = None

    if atmo.method == "grid":
        reload = msdi_save is None or atmo.source != prev_msdi[1]
        atmo = interp_atmo_grid(sme.teff, sme.logg, sme.monh, sme.atmo, reload=reload)
        prev_msdi = [atmo.method, atmo.source, atmo.depth, atmo.interp]
        setattr(self, "prev_msdi", prev_msdi)
        setattr(self, "msdi_save", True)
    elif atmo.method == "routine":
        atmo = atmo.source(sme, atmo)
    elif atmo.method == "embedded":
        # atmo structure already extracted in sme_main
        pass
    else:
        raise AttributeError("Source must be 'grid', 'routine', or 'file'")

    sme.atmo = atmo
    return sme


def get_cscale(cscale, flag, il):
    # Extract flag and value that specifies continuum normalization.
    #
    #  VALUE  IMPLICATION
    #  -3     Return residual intensity. Continuum is unity. Ignore sme.cscale
    #  -2     Return physical flux at stellar surface (units? erg/s/cm^2/A?)
    #  -1     Determine one scalar normalization that applies to all segments
    #   0     Determine separate scalar normalization for each spectral segment
    #   1     Determine separate linear normalization for each spectral segment
    #
    # Don't solve for single scalar normalization (-1) if there is no observation
    # CSCALE_FLAG is polynomial degree of continuum scaling, when fitting segments.

    if flag == -3:
        cscale = 1
    elif flag in [-1, -2]:
        cscale = flag
    elif flag == 0:
        cscale = cscale[il, 0]
    elif flag >= 1:
        cscale = cscale[il, :]
    else:
        raise AttributeError("invalid cscale_flag: %i" % flag)

    if flag >= 0:
        ndeg = flag
    else:
        ndeg = 0

    return cscale, ndeg


def get_rv(vrad, flag, il):
    # Extract flag and value that specifies radial velocity.
    #
    #  VALUE  IMPLICATION
    #  -2     Do not solve for radial velocity. Use input value(s).
    #  -1     Determine global radial velocity that applies to all segments
    #   0     Determine a separate radial velocity for each spectral segment
    #
    # Can't solve for radial velocities if there is no observation.
    # Express radial velocities as dimensionless wavelength scale factor.
    # Formula includes special relativity, though correction is negligible.

    if flag == -2:
        return 0, 1
    else:
        vrad = vrad if vrad.ndim == 0 else vrad[il]  # km/s
        vfact = np.sqrt((1 + vrad / clight) / (1 - vrad / clight))
        return vrad, vfact


def get_wavelengthrange(wran, vrad, vsini):
    # 30 km/s == maximum barycentric velocity
    vrad_pad = 30.0 + 0.5 * np.clip(vsini, 0, None)  # km/s
    vbeg = vrad_pad + np.clip(vrad, 0, None)  # km/s
    vend = vrad_pad - np.clip(vrad, None, 0)  # km/s

    wbeg = wran[0] * (1 - vbeg / clight)
    wend = wran[1] * (1 + vend / clight)
    return wbeg, wend


def synthetize_spectrum(wavelength, param, sme, save=True, update=True):
    """
    Create a synthetic spectrum, with a given wavelength

    Parameters
    ----------
    wavelength : array (npoints,)
        wavelength grid
    param : array (nparam,)
        parameter values to use. Only parameters that are varied need to be set. Which parameter is in which position is determined by param_names
    sme : SME_Struct
        input sme structure
    param_names : list
        names of parameters in param
    save : {bool, str}, optional
        wether to save the sme structure. If save is a string it will be used as the filename (default: True)

    Returns
    -------
    spec : array (npoints,)
        synthetic spectrum
    """

    # change parameters
    for name, value in param.items():
        sme[name] = value

    # run spectral synthesis
    sme2 = sme_func(sme)

    # Return values by reference to sme
    if update:
        sme.wave = sme2.wave
        sme.smod = sme2.smod
        sme.vrad = sme2.vrad
        sme.cscale = sme2.cscale

    # Also save intermediary results, because we can
    if isinstance(save, bool):
        if save:
            # use default name
            sme2.save()
    else:
        sme2.save(save)

    mask = (sme2.mob != 0) & (sme2.uob != 0)
    if wavelength.size != sme2.wave[mask].size or not np.allclose(
        wavelength, sme2.wave[mask]
    ):
        # interpolate to required wavelenth grid
        res = np.interp(wavelength, sme2.wave[mask], sme2.smod[mask])
    else:
        res = sme2.smod[mask]

    return res


def linelist_errors(wave, spec, linelist):
    # make linelist errors
    rel_error = linelist.error
    width = sme_synth.GetLineRange(len(linelist))

    sig_syst = np.zeros(wave.size, dtype=float)

    for i, line_range in enumerate(width):
        # find closest wavelength region
        w = (wave >= line_range[0]) & (wave <= line_range[1])
        sig_syst[w] += rel_error[i]

    sig_syst *= np.clip(1 - spec, 0, 1)
    return sig_syst


def determine_continuum(wave, spec, linelist, deg=2):
    width = sme_synth.GetLineRange(len(linelist))
    mask = np.full(wave.size, True)

    for line in width:
        w = (wave >= line[0]) & (wave <= line[1])
        mask[w] = False

    coeff = np.polyfit(wave[mask], spec[mask], deg=deg)

    return coeff


def get_bounds(param_names, atmo_file):
    bounds = {}

    dir = os.path.dirname(__file__)
    atmo_file = os.path.join(dir, "atmospheres", atmo_file)
    atmo_grid = readsav(atmo_file)["atmo_grid"]

    teff = np.unique(atmo_grid.teff)
    teff = np.min(teff), np.max(teff)
    bounds["teff"] = teff

    logg = np.unique(atmo_grid.logg)
    logg = np.min(logg), np.max(logg)
    bounds["logg"] = logg

    monh = np.unique(atmo_grid.monh)
    monh = np.min(monh), np.max(monh)
    bounds["monh"] = monh

    return bounds


def solve(
    sme, param_names=("teff", "logg", "monh"), filename="sme.npy", plot=False, **kwargs
):
    """
    Find the least squares fit parameters to an observed spectrum

    NOTE: intermediary results will be saved in "sme.npy", which is also used to transfer data

    Parameters
    ----------
    sme : SME_Struct
        sme struct containing all input (and output) parameters
    param_names : list, optional
        the names of the parameters to fit (default: ["teff", "logg", "monh"])
    filename : str, optional
        the sme structure will be saved to this file, use None to suppress this behaviour (default: "sme.npy")

    Returns
    -------
    sme : SME_Struct
        same sme structure with fit results in sme.fitresults, and best fit spectrum in sme.smod
    """

    # TODO: get bounds for all parameters. Bounds are given by the precomputed tables
    # TODO: create more efficient jacobian function ?

    param_names = [p.casefold() for p in param_names]
    param_names = [p.capitalize() if p[-5:] == "abund" else p for p in param_names]

    # replace "grav" with equivalent logg fit
    param_names = [p if p != "grav" else "logg" for p in param_names]
    param_names = [p if p != "feh" else "monh" for p in param_names]

    bounds = get_bounds(param_names, sme.atmo.source)
    bounds.update({"vmic": [0, np.inf], "vmac": [0, np.inf]})
    bounds.update({"%s abund" % el: [-10, 10] for el in Abund._elem})

    nparam = len(param_names)

    p0 = [sme[s] for s in param_names]
    bounds = np.array([bounds[s] for s in param_names]).T
    # Get constant data from sme structure
    mask = (sme.mob != 0) & (sme.uob != 0)
    wave = sme.wave[mask]
    spec = sme.sob[mask]
    uncs = sme.uob[mask]

    # Divide the uncertainties by the spectrum, to improve the fit in the continuum
    # Just as in IDL SME
    uncs /= spec

    def residuals(param, param_names, wave, spec, uncs, isJacobian=False):
        """ func = (model - obs) / sigma """
        self = residuals

        param = {n: v for n, v in zip(param_names, param)}
        # print(f"Is Jacobian {isJacobian}")
        synth = synthetize_spectrum(
            wave, param, sme, update=not isJacobian, save=not isJacobian
        )

        # TODO: linelist uncertainties, how large should they be?
        # uncs2 = linelist_errors(wave, spec, sme.linelist)
        uncs_linelist = 0

        resid = (synth - spec) / (uncs + uncs_linelist)
        resid = np.nan_to_num(resid, copy=False)

        if not isJacobian:
            print(param)
            self.resid = resid
            if not hasattr(self, "iteration"):
                self.iteration = 0
            self.iteration += 1
            if plot is not False:
                plot.add(wave, synth, f"Iteration {self.iteration}")

        return resid

    def jacobian(param, *args):
        return approx_derivative(
            residuals,
            param,
            method="3-point",
            f0=residuals.resid,
            bounds=bounds,
            args=args,
            kwargs={"isJacobian": True},
        )

    res = least_squares(
        residuals,
        x0=p0,
        jac=jacobian,  # is 2-point good enough or do we need 3-point ?
        bounds=bounds,
        loss="soft_l1",  # linear or soft_l1 ?
        verbose=2,
        args=(param_names, wave, spec, uncs),
        method="trf",  # method "dogbox" does not fit properly ???
        max_nfev=kwargs.get("maxiter"),
    )

    # The values in the last call are usually from the jacobian, i.e. not with the exactly correct parameters
    for i, name in enumerate(param_names):
        sme[name] = res.x[i]
    # sme = sme_func(sme)

    # SME structure is updated inside synthetize_spectrum to contain the results of the calculation
    # If for some reason that should not work, one can load the intermediary "sme.npy" file
    # sme = SME.SME_Struct.load("sme.npy")

    popt = res.x
    sme.pfree = np.atleast_2d(popt)  # 2d for compatibility
    sme.pname = param_names

    for i, name in enumerate(param_names):
        sme[name] = popt[i]

    # Determine the covariance matrix of the fit
    # Do Moore-Penrose inverse discarding zero singular values.
    # _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
    # threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    # s = s[s > threshold]
    # VT = VT[: s.size]
    # pcov = np.dot(VT.T / s ** 2, VT)

    # Alternative for determining the Covariance, with the exact same result
    fisher = res.jac.T.dot(res.jac)  # hessian == fisher information matrix
    covar = np.linalg.pinv(fisher)
    sig = np.sqrt(covar.diagonal())

    sme.fitresults.clear()
    sme.fitresults.covar = covar
    sme.fitresults.grad = res.grad
    sme.fitresults.pder = res.jac
    sme.fitresults.resid = res.fun
    sme.fitresults.chisq = res.cost * 2 / (sme.sob.size - nparam)

    sme.fitresults.punc = {}
    sme.fitresults.punc2 = {}
    for i in range(nparam):
        # Errors based on covariance matrix
        sme.fitresults.punc[param_names[i]] = sig[i]
        # Errors based on ad-hoc metric
        tmp = np.abs(res.fun) / np.median(np.abs(res.jac[:, i]))
        sme.fitresults.punc2[param_names[i]] = np.median(tmp)

    sme.nlte.flags = sme_synth.GetNLTEflags(sme.linelist)

    if filename is not None:
        sme.save(filename)

    print(res.message)
    for name, value, unc in zip(param_names, popt, sme.fitresults.punc.values()):
        print(f"{name}\t{value:.5f} +- {unc:.5g}")

    return sme


def new_wavelength_grid(wint):
    wmid = 0.5 * (wint[-1] + wint[0])  # midpoint of segment
    wspan = wint[-1] - wint[0]  # width of segment
    jmin = np.argmin(np.diff(wint))
    vstep1 = np.diff(wint)[jmin]
    vstep1 = vstep1 / wint[jmin] * clight  # smallest step
    vstep2 = 0.1 * wspan / (len(wint) - 1) / wmid * clight  # 10% mean dispersion
    vstep3 = 0.05  # 0.05 km/s step
    vstep = max(vstep1, vstep2, vstep3)  # select the largest

    # Generate model wavelength scale X, with uniform wavelength step.
    nx = int(
        np.abs(np.log10(wint[-1] / wint[0])) / np.log10(1 + vstep / clight) + 1
    )  # number of wavelengths
    if nx % 2 == 0:
        nx += 1  # force nx to be odd
    x_seg = np.geomspace(wint[0], wint[-1], num=nx)
    return x_seg, vstep


# @memory.cache
def sme_func(sme, setLineList=True, passAtmosphere=True, passNLTE=True):
    """
    Calculate the synthetic spectrum based on the parameters passed in the SME structure
    The wavelength range of each segment is set in sme.wran
    The specific wavelength grid is given by sme.wave, or is generated on the fly if sme.wave is None

    Will try to fit radial velocity RV and continuum to observed spectrum, depending on vrad_flag and cscale_flag

    Other important fields:
    sme.iptype: instrument broadening type

    Parameters
    ----------
    sme : SME_Struct
        sme structure, with all necessary parameters for the calculation
    setLineList : bool, optional
        wether to pass the linelist to the c library (default: True)
    passAtmosphere : bool, optional
        wether to pass the atmosphere to the c library (default: True)
    passNLTE : bool, optional
        wether to pass NLTE departure coefficients to the c library (default:True)

    Returns
    -------
    sme : SME_Struct
        same sme structure with synthetic spectrum in sme.smod
    """

    # Define constants
    n_segments = sme.nseg
    nmu = len(sme.mu)

    # fix sme input
    if "sob" not in sme:
        sme.vrad_flag = -2
    if "sob" not in sme and sme.cscale_flag >= -1:
        sme.cscale_flag = -3

    # Prepare arrays
    wint = [None for _ in range(n_segments)]
    sint = [None for _ in range(n_segments)]
    cint = [None for _ in range(n_segments)]
    vrad = [None for _ in range(n_segments)]

    cscale = [None for _ in range(n_segments)]
    wave = [None for _ in range(n_segments)]
    smod = [None for _ in range(n_segments)]
    wind = [None for _ in range(n_segments)]

    # Input atmosphere model
    if setLineList:
        sme_synth.SetLibraryPath()
        sme_synth.InputLineList(sme.atomic, sme.species)
    if passAtmosphere:
        sme = sme_func_atmo(sme)
        sme_synth.InputModel(sme.teff, sme.logg, sme.vmic, sme.atmo)
        sme_synth.InputAbund(sme.abund, sme.monh)
        sme_synth.Ionization(0)
        sme_synth.SetVWscale(sme.gam6)
        sme_synth.SetH2broad(sme.h2broad)
    if passNLTE:
        # TODO ???
        if "atmo_pro" in sme:
            pass_nlte(sme)
        update_depcoeffs(sme)

    # Loop over segments
    #   Input Wavelength range and Opacity
    #   Calculate spectral synthesis for each
    #   Interpolate onto geomspaced wavelength grid
    #   Apply instrumental and turbulence broadening
    #   Determine Continuum / Radial Velocity for each segment
    for il in range(n_segments):
        #   Input Wavelength range and Opacity
        vrad_seg, _ = get_rv(sme.vrad, sme.vrad_flag, il)
        wran_seg = sme.wran[il]
        wbeg, wend = get_wavelengthrange(sme.wran[il], vrad_seg, sme.vsini)

        sme_synth.InputWaveRange(wbeg, wend)
        sme_synth.Opacity()

        if sme.wave is None:
            seg_wave = None
        else:
            seg_wind = [0, *(sme.wind + 1)][il : il + 2]
            seg_wave = sme.wave[seg_wind[0] : seg_wind[1]]

        #   Calculate spectral synthesis for each
        nw, wint[il], sint[il], cint[il] = sme_synth.Transf(
            sme.mu,
            sme.accrt,
            sme.accwi,
            keep_lineop=il != 0,
            long_continuum=1,
            wave=seg_wave,
        )

        # Create new geomspaced wavelength grid, to be used for intermediary steps
        x_seg, vstep = new_wavelength_grid(wint[il])

        # Continuum
        cflx_seg = rtint(sme.mu, cint[il], 1, 0, 0)
        yc_seg = np.interp(x_seg, wint[il], cflx_seg)
        # Spectrum
        yi_seg = np.empty((nmu, len(x_seg)))
        for imu in range(nmu):
            yi_seg[imu] = np.interp(x_seg, wint[il], sint[il][imu])

        # Turbulence broadening
        y_seg = rtint(sme.mu, yi_seg, vstep, abs(sme.vsini), abs(sme.vmac))
        # instrument broadening
        if "iptype" in sme:
            ipres = sme.ipres if np.size(sme.ipres) == 1 else sme.ipres[il]
            y_seg = broadening.apply_broadening(
                ipres, x_seg, y_seg, type=sme["iptype"], sme=sme
            )

        # Divide calculated spectrum by continuum
        y_seg /= yc_seg

        if "wave" in sme:  # wavelengths already defined
            # first pixel in current segment
            ibeg, iend = seg_wind
            wind[il] = iend - ibeg

            wave[il] = sme.wave[ibeg:iend]  # wavelengths for current segment
            sob_seg = sme.sob[ibeg:iend]  # observed spectrum
            uob_seg = sme.uob[ibeg:iend]  # associated uncertainties
            mob_seg = sme.mob[ibeg:iend]  # ignore/line/cont mask

        else:  # else must build wavelengths
            itrim = (x_seg > wran_seg[0]) & (x_seg < wran_seg[1])  # trim padding
            wave[il] = np.pad(
                x_seg[itrim],
                1,
                mode="constant",
                constant_value=[wran_seg[0], wran_seg[1]],
            )
            sob_seg = uob_seg = mob_seg = None
            wind[il] = len(wave[il])

        # Determine Continuum / Radial Velocity for each segment
        cscale_seg, ndeg = get_cscale(sme.cscale, sme.cscale_flag, il)

        fix_c = sme.cscale_flag < 0
        fix_rv = "wave" not in sme or sme.vrad_flag < 0

        vrad[il], cscale[il] = match_rv_continuum(
            wave[il],
            sob_seg,
            uob_seg,
            x_seg,
            y_seg,
            ndeg=ndeg,
            mask=mob_seg,
            rvel=vrad_seg,
            cscale=cscale_seg,
            fix_rv=fix_rv,
            fix_c=fix_c,
        )
        smod[il] = np.interp(wave[il], x_seg * (1 + vrad[il] / clight), y_seg)

    # Merge all segments
    sme.smod = smod = np.concatenate(smod)
    # if sme already has a wavelength this should be the same
    sme.wave = wave = np.concatenate(wave)
    sme.wind = wind = np.cumsum(wind) - 1

    sme.vrad = np.array(vrad)
    sme.cscale = np.stack(cscale)

    return sme


def fisher(sme):
    """ Calculate fisher information matrix """
    nparam = len(sme.pname)
    fisher_matrix = np.zeros((nparam, nparam), dtype=np.float64)

    x = sme.wave
    y = sme.sob
    yerr = sme.uob
    parameter_names = [s.decode() for s in sme.pname]
    p0 = sme.pfree[-1, :nparam]

    # step size = machine precision ** (1/number of points)
    # see scipy.optimize._numdiff.approx_derivative
    # step = np.finfo(np.float64).eps ** (1 / 3)
    step = np.abs(sme.pfree[-3, :nparam] - sme.pfree[-1, :nparam])

    second_deriv = lambda f, x, h: (f(x + h) - 2 * f(x) + f(x - h)) / np.sum(h) ** 2

    sme_synth.SetLibraryPath()
    sme_synth.InputLineList(sme.atomic, sme.species)
    # chi squared function, i.e. log likelihood
    # func = 0.5 * sum ((model - obs) / sigma)**2
    func = lambda p: 0.5 * np.sum(
        ((synthetize_spectrum(x, *p, sme=sme, param_names=parameter_names) - y) / yerr)
        ** 2
    )

    # Diagonal elements
    for i in range(nparam):
        h = np.zeros(nparam)
        h[i] = step[i]
        fisher_matrix[i, i] = -second_deriv(func, p0, h)

    # Cross terms, fisher matrix is symmetric, so only calculate one half
    for i, j in combinations(range(nparam), 2):
        h = np.zeros(nparam)
        total = 0
        for k, m in product([-1, 1], repeat=2):
            h[i] = k * step[i]
            h[j] = m * step[j]
            total += func(p0 + h) * k * m

        total /= 4 * np.abs(h[i] * h[j])
        print(i, j, total)
        fisher_matrix[i, j] = -total
        fisher_matrix[j, i] = -total

    np.save("fisher_matrix", fisher_matrix)
    print(fisher_matrix)
    return fisher_matrix
