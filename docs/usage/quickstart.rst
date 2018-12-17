Quickstart
==========

The first step in each SME project is to create an SME structure
    >>> from PySME.src.sme.sme import SME_Struct

This can be done in done in a few different ways:
    * load an existing SME save file (from Python or IDL)
        >>> sme = SME_Struct.load("sme.inp")
    * load an .ech file spectrum
        >>> sme = SME_Struct.load("obs.ech")
    * assign values manually
        >>> sme = SME_Struct()

Either way one has to make sure that a few essential properties are set in the object, those are:
    * Stellar parameters (teff, logg, monh, abund)
        >>> from PySME.src.sme.abund import Abund
        >>> sme.teff, sme.logg, sme.monh = 5700, 4.4, -0.1
        >>> sme.abund = Abund.solar()
    * Wavelength range(s) in Ångstrom
        >>> sme.wran = [[4500, 4600], [5200, 5400]]
    * LineList (linelist), e.g. from VALD
        >>> from PySME.src.sme.vald import ValdFile
        >>> vald = ValdFile("linelist.lin")
        >>> sme.linelist = vald.linelist
    * Atmosphere (atmo), the file has to be in PySME/src/sme/atmospheres
        >>> sme.atmo.source = "marcs2012p_t2.0.sav"
        >>> sme.atmo.method = "grid"

Furthermore for fitting to an observation an observation is required:
    * Wavelength wave
        >>> sme.wave = Wavelength
    * Spectrum spec
        >>> sme.spec = Spectrum
    * Uncertainties uncs
        >>> sme.uncs = Uncertainties
    * Mask mask
        >>> sme.mask = np.ones(len(Spectrum))
    * Wavelength Segment indices wind
        >>> # index of the first wavelength element in each segment
        >>> # No overlap, no empty region between segments
        >>> # Currently it is the users responsibility to make sure
        >>> # wavelength ranges and segment indices agree
        >>> sme.wind = [0, 4096, 7250]
    * radial velocity and continuum flags
        >>> sme.vrad_flag = "each"
        >>> sme.cscale_flag = "linear"

Optionally the following can be set:
    * NLTE nlte for non local thermal equilibrium calculations
        >>> sme.nlte.set_nlte("Ca")

Once the SME structure is prepared, SME can be run in one of its two modes:
    1. Synthesize a spectrum
        >>> from PySME.src.sme.solve import synthesize_spectrum
        >>> sme = synthesize_spectrum(sme)
    2. Finding the best fit (least squares) solution
        >>> from PySME.src.sme.solve import solve
        >>> fitparameters = ["teff", "logg", "monh", "Mg Abund"]
        >>> sme = solve(sme, fitparameters)

The results will be contained in the output sme structure. These can for example be plotted using the gui module.
    >>> from PySME.src.gui import plot_plotly
    >>> fig = plot_plotly.FinalPlot(sme)
    >>> fig.save(filename="sme.html")

.. raw:: html
    :file: ../_static/sun.html

or saved with
    >>> sme.save("out.npy")
