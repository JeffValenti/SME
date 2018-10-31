"""
Calculates the spectrum, based on a set of stellar parameters
And also determines the best fit parameters
"""

import os.path
import warnings
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy.constants import speed_of_light
from scipy.optimize import OptimizeWarning, least_squares

from . import broadening
from . import sme as SME
from . import sme_synth
from .abund import Abund

from .cwrapper import idl_call_external
from .interpolate_atmosphere import interp_atmo_grid
from .rtint import rdpop, rtint
from .sme_crvmatch import match_rv_continuum

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", OptimizeWarning)

memory = Memory("./__cache__", verbose=0)

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

    error = idl_call_external("InputNLTE", b_nlte)
    if error != b"":
        raise ValueError(
            "InputDepartureCoefficients (call_external): %s" % error.decode()
        )
    return error


@memory.cache
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

    # Static storage
    #   common common_sme_func_atmo, prev_msdi

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
        atmo = interp_atmo_grid(sme.teff, sme.logg, sme.feh, sme.atmo, reload=reload)
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


def get_flags(sme):
    tags = np.array(list(sme.names))
    f_ipro = "IPTYPE" in tags
    f_opro = "sob" in tags
    f_wave = "wave" in tags
    f_h2broad = "h2broad" in tags and sme["h2broad"]
    f_NLTE = False
    f_glob = "glob_free" in tags
    f_gf = "gf_free" in tags
    f_vw = "vw_free" in tags
    f_ab = "ab_free" in tags

    flags = {
        "opro": f_opro,
        "glob": f_glob,
        "wave": f_wave,
        "h2broad": f_h2broad,
        "nlte": f_NLTE,
        "ipro": f_ipro,
        "gf": f_gf,
        "vw": f_vw,
        "ab": f_ab,
    }
    return flags


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
        cscale = cscale[il]
    elif flag == 1:
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


def synthetize_spectrum(wavelength, *param, sme=None, param_names=[], setLineList=True):

    # change parameters
    for name, value in zip(param_names, param):
        sme[name] = value

    # run spectral synthesis
    sme = sme_func(sme, setLineList=setLineList)
    sme.save()

    mask = (sme.mob != 0) & (sme.uob != 0)
    if not np.allclose(wavelength, sme.wave[mask]):
        # interpolate to required wavelenth grid
        res = np.interp(wavelength, sme.wave[mask], sme.smod[mask])
    else:
        res = sme.smod[mask]

    return res

def linelist_errors(sme, linelist):
    # make linelist errors
    mask = (sme.mob != 0) & (sme.uob != 0)
    rel_error = linelist.error
    width = sme_synth.GetLineRange(len(linelist))  # TODO
    sig_syst = np.zeros(np.count_nonzero(mask), dtype=float)
    wave = sme.wave[mask]

    for i, line in enumerate(linelist):
        # find closest wavelength region
        w = (wave >= width[i, 0]) & (wave <= width[i, 1])
        sig_syst[w] += rel_error[i]

    sig_syst *= np.clip(1 - sme.sob[mask], 0, 1)
    return sig_syst

def solve(sme, param_names=["teff", "logg", "feh"]):
    # TODO: get bounds for all parameters. Bounds are given by the precomputed tables
    # TODO: Set up a sparsity scheme for the jacobian (if some parameters are sufficiently independent)
    # TODO: create more efficient jacobian function
    bounds = {"teff": [3500, 7000], "logg": [3, 5], "feh": [-5, 1]}
    bounds.update({"%s abund" % el.casefold(): [-10, 10] for el in Abund._elem})

    nparam = len(param_names)

    p0 = [sme[s] for s in param_names]
    bounds = np.array([bounds[s.casefold()] for s in param_names]).T
    # func = (model - obs) / sigma

    def residuals(param):
        mask = (sme.mob != 0) & (sme.uob != 0) 
        wave = sme.wave[mask]
        spec = sme.sob[mask]
        uncs = sme.uob[mask]
        
        synth = synthetize_spectrum(wave, *param, param_names=param_names, sme=sme)
        uncs2 = linelist_errors(sme, sme.linelist)
        resid = (synth - spec) / (uncs + uncs2)
        return np.nan_to_num(resid, copy=False)

    # Prepare LineList only once
    sme_synth.SetLibraryPath()
    sme_synth.InputLineList(sme.atomic, sme.species)

    res = least_squares(
        residuals, x0=p0, jac="2-point", bounds=bounds, loss="linear", verbose=2
    )

    sme = SME.SME_Struct.load("sme.npy")

    popt = res.x
    sme.pfree = np.atleast_2d(popt)  # 2d for compatibility
    sme.pname = param_names

    for i, name in enumerate(param_names):
        sme[name] = popt[i]

    # Determine the covariance matrix of the fit
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s ** 2, VT)

    # Alternative for determining the Covariance
    # hess = res.jac.T.dot(res.jac) # hessian == fisher information matrix
    # covar = np.linalg.pinv(hess)
    # sig = np.sqrt(covar.diag())

    sme.covar = pcov
    sme.grad = res.grad
    sme.pder = res.jac
    sme.resid = res.fun
    sme.chisq = res.cost * 2 / (sme.sob.size - nparam)

    sme.punc = {name: 0 for name in param_names}
    for i in range(nparam):
        tmp = np.abs(res.fun) / np.median(np.abs(res.jac[:, i]))
        sme.punc[param_names[i]] = tmp

    sme.punc2 = np.sqrt(np.diag(pcov))

    sme.save()

    print(res.message)
    for name, value, unc in zip(param_names, popt, sme.punc.values()):
        print("%s\t%.5f +- %.5f" % (name, value, unc))

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
    #
    nx = int(
        np.log10(wint[-1] / wint[0]) / np.log10(1 + vstep / clight) + 1
    )  # number of wavelengths
    if nx % 2 == 0:
        nx += 1  # force nx to be odd
    x_seg = np.geomspace(wint[0], wint[-1], num=nx)
    return x_seg, vstep


# @memory.cache
def sme_func(sme, setLineList=True, passAtmosphere=True):
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
    jint = [None for _ in range(n_segments)]
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
        # Compile the table of departure coefficients if NLTE flag is set
        if "nlte" in sme and "atmo_pro" in sme:
            pass_nlte(sme)

        sme_synth.InputAbund(sme.abund, sme.feh)
        sme_synth.Ionization(0)
        sme_synth.SetVWscale(sme.gam6)
        sme_synth.SetH2broad(sme.h2broad)

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

        #   Calculate spectral synthesis for each
        nw, wint[il], sint[il], cint[il] = sme_synth.Transf(
            sme.mu, sme.accrt, sme.accwi, keep_lineop=il != 0, long_continuum=1
        )
        jint[il] = jint[il - 1] + nw if il != 0 else nw - 1

        #   Interpolate onto geomspaced wavelength grid
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

        y_seg /= yc_seg

        if "wave" in sme:  # wavelengths already defined
            # first pixel in current segment
            ibeg = 0 if il == 0 else sme.wind[il - 1] + 1
            # last pixel in current segment
            iend = sme.wind[il]
            wind[il] = iend - ibeg
            if il > 0:
                wind[il] += 1
            wave[il] = sme.wave[ibeg : iend + 1]  # wavelengths for current segment

            sob_seg = sme.sob[ibeg : iend + 1]  # observed spectrum
            uob_seg = sme.uob[ibeg : iend + 1]  # associated uncertainties
            mob_seg = sme.mob[ibeg : iend + 1]  # ignore/line/cont mask

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
    sme.wind = wind = np.cumsum(wind)

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


def sme_main(sme, only_func=False):

    flags = get_flags(sme)

    # Decide which global parameters, if any, are free parameters.
    freep = []

    if "glob_free" in sme:
        freep += sme.glob_free

    # Decide which log(gf), if any, are free parameters.
    if flags["gf"]:
        igf = sme.gf_free > 0
        freep += [
            "%s %i LOGGF" % (s, i)
            for s, i in np.zip(sme.species[igf], sme.atomic.T[2, igf])
        ]

    # Decide which van der Waal's constants, if any, are free parameters.
    if flags["vw"]:
        ivw = sme.vw_free > 0
        freep += [
            "%s%i %i LOGVW " % (s, i, j)
            for s, i, j in zip(
                elements[sme.atomic.T[0, ivw] - 1],
                sme.atomic.T[1, ivw],
                sme.atomic.T[2, ivw],
            )
        ]

    # Decide which abundances, if any, are free parameters.
    if flags["ab"]:
        iab = sme.ab_free > 0
        freep += [s + " ABUND" for s in elements[iab]]

    # TODO: sme_nlte_reset

    # Call model evaluator/solver.
    if len(freep) > 0 and not only_func:  # true: call gradient solver
        sme = solve(sme, param_names=freep)  # solve for best parameters
    else:  # else: parameters known
        sme = sme_func(sme)  # just evaluate model once

    sme.save()

    return sme
