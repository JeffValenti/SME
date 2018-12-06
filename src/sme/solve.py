"""
Calculates the spectrum, based on a set of stellar parameters
And also determines the best fit parameters
"""

import logging
import os.path
import warnings

import numpy as np
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d
from scipy.io import readsav
from scipy.optimize import OptimizeWarning, least_squares
from scipy.optimize._numdiff import approx_derivative

from . import broadening, sme_synth
from .abund import Abund
from .atmosphere import interp_atmo_grid, krz_file
from .continuum_and_radial_velocity import match_rv_continuum
from .integrate_flux import integrate_flux
from .nlte import update_depcoeffs

clight = speed_of_light * 1e-3  # km/s
warnings.filterwarnings("ignore", category=OptimizeWarning)


def residuals(
    param,
    names,
    sme,
    spec,
    uncs,
    mask,
    isJacobian=False,
    fname="sme.npy",
    fig=None,
    **_,
):
    """
    Calculates the synthetic spectrum with sme_func and
    returns the residuals between observation and synthetic spectrum

    residual = (obs - synth) / uncs

    Parameters
    ----------
    param : list(float) of size (n,)
        parameter values to use for synthetic spectrum, order is the same as names
    names : list(str) of size (n,)
        names of the parameters to set, as defined by SME_Struct
    sme : SME_Struct
        sme structure holding all relevant information for the synthetic spectrum generation
    spec : array(float) of size (m,)
        observed spectrum
    uncs : array(float) of size (m,)
        uncertainties of the observed spectrum
    mask : array(bool) of size (k,)
        mask to apply to the synthetic spectrum to select the same points as spec
        The size of the synthetic spectrum is given by sme.wave
        then mask must have the same size, with m True values
    isJacobian : bool, optional
        Flag to use when within the calculation of the Jacobian (default: False)
    fname : str, optional
        filename of the intermediary product (default: "sme.npy")
    fig : Figure, optional
        plotting interface, fig.add(x, y, title) will be called each non jacobian iteration

    Returns
    -------
    resid : array(float) of size (m,)
        residuals of the synthetic spectrum
    """

    self = residuals
    if not hasattr(self, "iteration"):
        self.iteration = 0

    update = not isJacobian
    save = not isJacobian
    reuse_wavelength_grid = isJacobian

    # change parameters
    for name, value in zip(names, param):
        sme[name] = value
    # run spectral synthesis
    try:
        sme2 = synthesize_spectrum(sme, reuse_wavelength_grid=reuse_wavelength_grid)
    except ValueError as ve:
        # Something went wrong (left the grid? Don't go there)
        # If returned value is not finite it will be ignored?
        logging.debug(ve)
        return np.inf

    # Return values by reference to sme
    if update:
        sme.wave = sme2.wave
        sme.smod = sme2.smod
        sme.vrad = sme2.vrad
        sme.cscale = sme2.cscale

    # Also save intermediary results, because we can
    if save:
        sme2.save(fname, verbose=False)

    _, synth = sme2.spectrum(syn=True)
    synth = synth.flatten()
    if mask is not None:
        synth = synth[mask]

    # TODO: update based on lineranges
    uncs_linelist = 0

    resid = (synth - spec) / (uncs + uncs_linelist)
    resid = np.nan_to_num(resid, copy=False)

    if not isJacobian:
        # Save result for jacobian
        self.resid = resid
        self.iteration += 1
        logging.debug("%s", {n: v for n, v in zip(names, param)})
        # Plot
        if fig is not None:
            wave = sme2.wave
            try:
                fig.add(wave, synth, f"Iteration {self.iteration}")
            except AttributeError:
                warnings.warn(f"Figure {repr(fig)} doesn't have a 'add' method")
            except Exception as e:
                warnings.warn(f"Error during Plotting: {e.message}")

    return resid


def jacobian(param, *args, bounds=None, **_):
    """
    Approximate the jacobian numerically
    The calculation is the same as "3-point"
    but we can tell residuals that we are within a jacobian
    """
    return approx_derivative(
        residuals,
        param,
        method="3-point",
        # This feels pretty bad, passing the latest synthetic spectrum
        # by reference as a parameter of the residuals function object
        f0=residuals.resid,
        bounds=bounds,
        args=args,
        kwargs={"isJacobian": True},
    )


def linelist_errors(wave, spec, linelist):
    """ make linelist errors, based on the effective wavelength range
    of each line and the uncertainty value of that line """
    rel_error = linelist.error
    width = sme_synth.GetLineRange(len(linelist))

    sig_syst = np.zeros(wave.size, dtype=float)

    for i, line_range in enumerate(width):
        # find closest wavelength region
        w = (wave >= line_range[0]) & (wave <= line_range[1])
        sig_syst[w] += rel_error[i]

    sig_syst *= np.clip(1 - spec, 0, 1)
    return sig_syst


def get_bounds(param_names, atmo_file):
    """
    Create Bounds based on atmosphere grid and general rules

    Note that bounds define by definition a cube in the parameter space,
    but the grid might not be a cube. I.e. Not all combinations of teff, logg, monh are valid
    This method will choose the outerbounds of that space as the boundary, which means that
    we can still run into problems when interpolating the atmospheres

    Parameters
    ----------
    param_names : array(str)
        names of the parameters to vary
    atmo_file : str
        filename of the atmosphere grid

    Raises
    ------
    IOError
        If the atmosphere file can't be read, allowed types are IDL savefiles (.sav), and .krz files

    Returns
    -------
    bounds : dict
        Bounds for the given parameters
    """

    bounds = {}

    # Create bounds based on atmosphere grid
    if "teff" in param_names or "logg" in param_names or "monh" in param_names:
        folder = os.path.dirname(__file__)
        atmo_file = os.path.basename(atmo_file)
        _, ext = os.path.splitext(atmo_file)
        atmo_file = os.path.join(folder, "atmospheres", atmo_file)

        if ext == ".sav":
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
        elif ext == ".krz":
            # krz atmospheres are fixed to one parameter set
            # allow just "small" area around that
            atmo = krz_file(atmo_file)
            bounds["teff"] = atmo.teff - 500, atmo.teff + 500
            bounds["logg"] = atmo.logg - 1, atmo.logg + 1
            bounds["monh"] = atmo.monh - 1, atmo.monh + 1
        else:
            raise IOError(f"File extension {ext} not recognized")

    # Add generic bounds
    bounds.update({"vmic": [0, np.inf], "vmac": [0, np.inf], "vsini": [0, np.inf]})
    bounds.update({"%s abund" % el: [-10, 11] for el in Abund._elem})

    # Select bounds requested
    bounds = np.array([bounds[s] for s in param_names]).T

    return bounds


def get_scale(param_names):
    """
    Returns scales for each parameter so that values are on order ~1

    Parameters
    ----------
    param_names : list(str)
        names of the parameters

    Returns
    -------
    scales : list(float)
        scales of the parameters in the same order as input array
    """

    # The only parameter we want to scale right now is temperature,
    # as it is orders of magnitude larger than all others
    scales = {"teff": 1000}
    scales = [scales[name] if name in scales.keys() else 1 for name in param_names]
    return scales


def solve(
    sme, param_names=("teff", "logg", "monh"), filename="sme.npy", fig=None, **kwargs
):
    """
    Find the least squares fit parameters to an observed spectrum

    NOTE: intermediary results will be saved in filename ("sme.npy")

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

    # Sanitize parameter names
    param_names = [p.casefold() for p in param_names]
    param_names = [p.capitalize() if p[-5:] == "abund" else p for p in param_names]

    param_names = [p if p != "grav" else "logg" for p in param_names]
    param_names = [p if p != "feh" else "monh" for p in param_names]

    # Parameters are unique
    # But keep the order the same
    param_names, index = np.unique(param_names, return_index=True)
    param_names = param_names[np.argsort(index)]

    if "vrad" in param_names:
        param_names.remove("vrad")
        sme.vrad_flag = "each"

    if "cont" in param_names:
        param_names.remove("cont")
        sme.cscale_flag = 1

    nparam = len(param_names)

    # Create appropiate bounds
    bounds = get_bounds(param_names, sme.atmo.source)
    scales = get_scale(param_names)

    # Starting values
    p0 = [sme[s] for s in param_names]

    # Get constant data from sme structure
    spec = sme.spec.flatten()
    uncs = sme.uncs.flatten()
    mask = sme.mask.flatten()

    mask = mask != 0
    spec = spec[mask]
    uncs = uncs[mask]

    # Divide the uncertainties by the spectrum, to improve the fit in the continuum
    # Just as in IDL SME
    uncs /= spec

    logging.info("Fitting Spectrum with Parameters: " + "%s, " * nparam, *param_names)

    # Do the heavy lifting
    res = least_squares(
        residuals,
        x0=p0,
        jac=jacobian,
        bounds=bounds,
        x_scale=scales,
        loss="soft_l1",
        method="trf",
        verbose=2,
        args=(param_names, sme, spec, uncs, mask),
        kwargs={"bounds": bounds, "fig": fig, "fname": filename},
        max_nfev=kwargs.get("maxiter"),
    )

    # SME structure is updated inside synthetize_spectrum to contain the results of the calculation
    # If for some reason that should not work, one can load the intermediary "sme.npy" file
    # sme = SME.SME_Struct.load("sme.npy")
    for i, name in enumerate(param_names):
        sme[name] = res.x[i]

    # Update SME structure
    popt = res.x
    sme.pfree = np.atleast_2d(popt)  # 2d for compatibility
    sme.pname = param_names

    for i, name in enumerate(param_names):
        sme[name] = popt[i]

    # Determine the covariance
    # hessian == fisher information matrix
    fisher = res.jac.T.dot(res.jac)
    covar = np.linalg.pinv(fisher)
    sig = np.sqrt(covar.diagonal())

    # Update fitresults
    sme.fitresults.clear()
    sme.fitresults.covar = covar
    sme.fitresults.grad = res.grad
    sme.fitresults.pder = res.jac
    sme.fitresults.resid = res.fun
    sme.fitresults.chisq = res.cost * 2 / (sme.spec.size - nparam)

    sme.fitresults.punc = {}
    sme.fitresults.punc2 = {}
    for i in range(nparam):
        # Errors based on covariance matrix
        sme.fitresults.punc[param_names[i]] = sig[i]
        # Errors based on ad-hoc metric
        tmp = np.abs(res.fun) / np.clip(np.median(np.abs(res.jac[:, i])), 1e-5, None)
        sme.fitresults.punc2[param_names[i]] = np.median(tmp)

    sme.nlte.flags = sme_synth.GetNLTEflags(len(sme.linelist))

    if filename is not None:
        sme.save(filename)

    logging.debug("Reduced chi square: %.3f", sme.fitresults.chisq)
    for name, value, unc in zip(param_names, popt, sme.fitresults.punc.values()):
        logging.info("%s\t%.5f +- %.5g", name.ljust(10), value, unc)

    return sme


def get_atmosphere(sme):
    """
    Return an atmosphere based on specification in an SME structure

    sme.atmo.method defines mode of action:
        "grid"
            interpolate on atmosphere grid
        "embedded"
            No change
        "routine"
            calls sme.atmo.source(sme, atmo)

    Parameters
    ---------
        sme : SME_Struct
            sme structure with sme.atmo = atmosphere specification

    Returns
    -------
    sme : SME_Struct
        sme structure with updated sme.atmo
    """

    # Handle atmosphere grid or user routine.
    atmo = sme.atmo
    self = get_atmosphere

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
        raise AttributeError("Source must be 'grid', 'routine', or 'embedded'")

    sme.atmo = atmo
    return sme


def get_wavelengthrange(wran, vrad, vsini):
    """
    Determine wavelengthrange that needs to be calculated
    to include all lines within velocity shift vrad + vsini
    """
    # 30 km/s == maximum barycentric velocity
    vrad_pad = 30.0 + 0.5 * np.clip(vsini, 0, None)  # km/s
    vbeg = vrad_pad + np.clip(vrad, 0, None)  # km/s
    vend = vrad_pad - np.clip(vrad, None, 0)  # km/s

    wbeg = wran[0] * (1 - vbeg / clight)
    wend = wran[1] * (1 + vend / clight)
    return wbeg, wend


def new_wavelength_grid(wint):
    """ Generate new wavelength grid within bounds of wint"""
    # Determine step size for a new model wavelength scale, which must be uniform
    # to facilitate convolution with broadening kernels. The uniform step size
    # is the larger of:
    #
    # [1] smallest wavelength step in WINT_SEG, which has variable step size
    # [2] 10% the mean dispersion of WINT_SEG
    # [3] 0.05 km/s, which is 1% the width of solar line profiles

    wbeg, wend = wint[[0, -1]]
    wmid = 0.5 * (wend + wbeg)  # midpoint of segment
    wspan = wend - wbeg  # width of segment
    diff = wint[1:] - wint[:-1]
    jmin = np.argmin(diff)
    vstep1 = diff[jmin] / wint[jmin] * clight  # smallest step
    vstep2 = 0.1 * wspan / (len(wint) - 1) / wmid * clight  # 10% mean dispersion
    vstep3 = 0.05  # 0.05 km/s step
    vstep = max(vstep1, vstep2, vstep3)  # select the largest

    # Generate model wavelength scale X, with uniform wavelength step.
    nx = int(
        np.abs(np.log10(wend / wbeg)) / np.log10(1 + vstep / clight) + 1
    )  # number of wavelengths
    if nx % 2 == 0:
        nx += 1  # force nx to be odd

    # Resolution
    # IDL way
    # resol_out = 1 / ((wend / wbeg) ** (1 / (nx - 1)) - 1)
    # vstep = clight / resol_out
    # x_seg = wbeg * (1 + 1 / resol_out) ** np.arange(nx)

    # Python way (not identical, as IDL endpoint != wend)
    # difference approx 1e-7
    x_seg = np.geomspace(wbeg, wend, num=nx)
    resol_out = 1 / np.diff(np.log(x_seg[:2]))[0]
    vstep = clight / resol_out
    return x_seg, vstep


def synthesize_spectrum(
    sme,
    setLineList=True,
    passAtmosphere=True,
    passNLTE=True,
    reuse_wavelength_grid=False,
):
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
        wether to pass NLTE departure coefficients to the c library (default: True)
    reuse_wavelength_grid : bool, optional
        wether to use sme.wint as the output grid of the function or create a new grid (default: False)

    Returns
    -------
    sme : SME_Struct
        same sme structure with synthetic spectrum in sme.smod
    """

    # Define constants
    n_segments = sme.nseg
    nmu = np.size(sme.mu)

    # fix impossible input
    if "spec" not in sme:
        sme.vrad_flag = "none"
    if "spec" not in sme and sme.cscale_flag >= -1:
        sme.cscale_flag = "none"
    if "wint" not in sme:
        reuse_wavelength_grid = False

    # Prepare arrays
    wran = sme.wran

    wint = [None for _ in range(n_segments)]
    sint = [None for _ in range(n_segments)]
    cint = [None for _ in range(n_segments)]
    vrad = [None for _ in range(n_segments)]

    cscale = [None for _ in range(n_segments)]
    wave = [None for _ in range(n_segments)]
    smod = [None for _ in range(n_segments)]
    wind = [0] + [None for _ in range(n_segments)]

    # If wavelengths are already defined use those as output
    if "wave" in sme:
        wave = sme.wave
        wind = sme.wind

    # Input Model data to C library
    if setLineList:
        sme_synth.SetLibraryPath()
        sme_synth.InputLineList(sme.atomic, sme.species)
    if passAtmosphere:
        sme = get_atmosphere(sme)
        sme_synth.InputModel(sme.teff, sme.logg, sme.vmic, sme.atmo)
        sme_synth.InputAbund(sme.abund)
        sme_synth.Ionization(0)
        sme_synth.SetVWscale(sme.gam6)
        sme_synth.SetH2broad(sme.h2broad)
    if passNLTE:
        update_depcoeffs(sme)

    # Loop over segments
    #   Input Wavelength range and Opacity
    #   Calculate spectral synthesis for each
    #   Interpolate onto geomspaced wavelength grid
    #   Apply instrumental and turbulence broadening
    #   Determine Continuum / Radial Velocity for each segment
    for il in range(n_segments):
        logging.debug("Segment %i", il)
        # Input Wavelength range and Opacity
        vrad_seg = sme.vrad if np.size(sme.vrad) == 1 else sme.vrad[il]
        wbeg, wend = get_wavelengthrange(wran[il], vrad_seg, sme.vsini)

        sme_synth.InputWaveRange(wbeg, wend)
        sme_synth.Opacity()

        # Reuse adaptive wavelength grid in the jacobians
        wint_seg = sme.wint[il] if reuse_wavelength_grid else None
        # Only calculate line opacities in the first segment
        keep_line_opacity = il != 0
        #   Calculate spectral synthesis for each
        _, wint[il], sint[il], cint[il] = sme_synth.Transf(
            sme.mu,
            sme.accrt,  # threshold line opacity / cont opacity
            sme.accwi,
            keep_lineop=keep_line_opacity,
            wave=wint_seg,
        )

        # Create new geomspaced wavelength grid, to be used for intermediary steps
        wgrid, vstep = new_wavelength_grid(wint[il])

        # Continuum
        # rtint = Radiative Transfer Integration
        cont_flux = integrate_flux(sme.mu, cint[il], 1, 0, 0)
        cont_flux = np.interp(wgrid, wint[il], cont_flux)

        # Broaden Spectrum
        y_integrated = np.empty((nmu, len(wgrid)))
        for imu in range(nmu):
            y_integrated[imu] = np.interp(wgrid, wint[il], sint[il][imu])

        # Turbulence broadening
        # Apply macroturbulent and rotational broadening while integrating intensities
        # over the stellar disk to produce flux spectrum Y.
        flux = integrate_flux(sme.mu, y_integrated, vstep, sme.vsini, sme.vmac)
        # instrument broadening
        if "iptype" in sme:
            ipres = sme.ipres if np.size(sme.ipres) == 1 else sme.ipres[il]
            flux = broadening.apply_broadening(
                ipres, wgrid, flux, type=sme.iptype, sme=sme
            )

        # Divide calculated spectrum by continuum
        if sme.cscale_flag != -2:
            flux /= cont_flux

        # Create a wavelength array if it doesn't exist
        if "wave" not in sme:
            # trim padding
            wbeg, wend = wran[il]
            itrim = (wgrid > wbeg) & (wgrid < wend)
            # Force endpoints == wavelength range
            wave[il] = np.concatenate(([wbeg], wgrid[itrim], [wend]))
            wind[il + 1] = wind[il] + len(wave[il])

        # Fit continuum and radial velocity
        cscale[il], vrad[il] = match_rv_continuum(sme, il, wgrid, flux)
        # Interpolate spectrum on ouput wavelength grid
        if vrad[il] is not None:
            rv_factor = np.sqrt((1 + vrad[il] / clight) / (1 - vrad[il] / clight))
            wgrid *= rv_factor

        # TODO: how to resample/interpolate here? Does it matter?
        # smod[il] = resamp(x_seg, y_seg, wave[il])
        smod[il] = interp1d(
            wgrid, flux, kind="cubic", fill_value=0, bounds_error=False
        )(wave[il])

    # Merge all segments
    sme.smod = smod = np.concatenate(smod)
    # if sme already has a wavelength this should be the same
    sme.wave = wave = np.concatenate(wave)
    sme.wind = wind
    sme.wint = wint

    sme.cscale = np.stack(cscale)

    if sme.vrad_flag in [-1, "whole"]:
        # If we want to determine radial velocity on the whole spectrum at once (instead of segments)
        _, vrad = match_rv_continuum(sme, -1, wave, smod)
        rv_factor = np.sqrt((1 + vrad / clight) / (1 - vrad / clight))
        sme.smod = interp1d(
            wave * rv_factor,
            sme.smod,
            kind="cubic",
            fill_value="constant",
            bounds_error=False,
        )(wave)

    sme.vrad = np.asarray(vrad)

    return sme
