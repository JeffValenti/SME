from sys import version_info


class ContinuousOpacityFlags(dict):
    """Manage continuous opacity flags needed by the SME external library.

    Subclass of the standard dict class. Initialization populates dict
    with one item per continuous opacity source. Key identifies the opacity
    source (e.g., 'H-'). Value indicates whether the SME external library
    should include the continuous opacity source (True or False).

    Use standard dictionary syntax to get a flag value (e.g., cof['H-'])
    or to set a flag (e.g., cof['H-'] = False). Attempting to set a flag
    raises ValueError if the key is not a valid continuous opacity source
    (e.g., 'H++') or the value is not boolean (e.g., 1).

    Overrides __str__() so that print lists keys (opacity sources) with
    value True followed by keys with value False. The cof.smelib property
    returns flag values as integers (0 or 1) for use with the SME external
    library.

    Example
    -------
    >>> from sme.atmo import ContinuousOpacityFlags
    >>> cof = ContinuousOpacityFlags()
    >>> print(cof.smelib)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> cof['hot'] = False
    >>> print(cof)
    True: H H2+ H- HRay He He+ He- HeRay cool luke e- H2Ray | False: hot
    """

    def __init__(self):
        """Create continuous opacity flags object with default values.
        Key order in _defaults must match order expected by SME library.
        """
        self._defaults = {
            'H': True, 'H2+': True, 'H-': True, 'HRay': True, 'He': True,
            'He+': True, 'He-': True, 'HeRay': True, 'cool': True,
            'luke': True, 'hot': True, 'e-': True, 'H2Ray': True}
        self.defaults()

    def __setitem__(self, key, value):
        """Set value of a continuous opacity flag. Raise ValueError exception
        if continuous opacity key is not valid or if the value is not boolean.
        """
        if key not in self._defaults:
            raise ValueError(
                f'Invalid continuous opacity key: {key}\n' +
                'Valid keys: ' + ' '.join(self._defaults))
        if not isinstance(value, bool):
            raise ValueError(
                f'Invalid continuous opacity flag value: {value}\n' +
                'Valid values: True False')
        super().__setitem__(key, value)

    def __str__(self):
        """Return string that summarizes continuous opacity flag values.
        """
        return(' '.join([
            'True:', *[k for k, v in self.items() if v is True], '|',
            'False:', *[k for k, v in self.items() if v is False]]))

    def defaults(self):
        """Set each continuous opacity flag to its default value.
        """
        for key in self._defaults:
            self[key] = self._defaults[key]

    @property
    def smelib(self):
        """Get opacity flag values as list of integers (0 for False,
        1 for True) in the order expected by the SME external library.
        Python 3.7 or later required because code assumes dict is ordered.
        """
        assert version_info[0:2] >= (3, 7)
        return(list(map(lambda x: 1 if x is True else 0, self.values())))


class SmeAtmo:
    """Manage atmosphere attributes used by the SME external library.

    Parameters
    ----------
    radius : float
        Stellar radius (in cm) base of atmosphere grid. Mandatory for
        spherical geometry. Not allowed for plane-parallel geometry.

    opacity_flags : dictionary
        Text

    Notes
    -----
    Data in this class yield arguments required by the InputModel() external
    function in the SME external library. Those arguments are:

    ====== ================================================================
    arg[0] Number of depths in the atmosphere
    arg[1] Reserved for future use (currently read into TEFF, but not used)
    arg[2] Reserved for future use (currently read into GRAV, but not used)
    arg[3] Wavelength for reference continuous opacities (used if MOTYPE=0)
    arg[4] Type of  model atmosphere ('TAU', 'RHOX', or 'SPH')
    ====== ================================================================
    """
    def __init__(self, modeltype, scale, wavelength=None, radius=None):
        self._modeltypes = ['rhox', 'tau', 'sph']
        self.set_scale(modeltype, scale, wavelength=wavelength, radius=radius)

    def __str__(self):
        """Return string that summarizes atmosphere.
        """
        return(
            f'modeltype: {self.modeltype}, ' +
            f'wavelength: {self.wavelength}, ' +
            f'radius: {self.radius}' +
            f'nlayer: {self.nlayer}')

    @property
    def modeltype(self):
        """Combination of radiative transfer geometry (plane parallel or
        spherical) and depth scale type (mass column, continuum optical
        depth, or height). The SME external library can handle the
        following model types:

        ====== ============== ======================= =======
        Value  Geometry       Depth scale type        Units
        ====== ============== ======================= =======
        'rhox' plane-parallel mass column             g/cm**2
        'tau'  plane-parallel continuum optical depth
        'sph'  spherical      height                  cm
        ====== ============== ======================= =======

        This read only property is set when an atmosphere is loaded.
        """
        return(self._modeltype)

    @property
    def wavelength(self):
        """Wavelength (in Angstrom) of continuum optical depth scale.
        Set for modeltype 'tau', otherwise None. This read only property
        is set when an atmosphere is loaded.
        """
        return(self._wavelength)

    @property
    def radius(self):
        """Radius (in cm) of deepest point in an atmosphere grid. Set
        for modeltype 'sph', otherwise None. This read only property is
        set when an atmosphere is loaded.
        """
        return(self._radius)

    @property
    def scale(self):
        """Physical scale for each layer in the atmosphere. See `modeltype`
        for description of scale types (mass column, continuum optical depth,
        or height). This read only property is set when an atmosphere is
        loaded.
        """
        return(self._scale)

    @property
    def nlayer(self):
        """Number of layers (depths or heights) in the model atmosphere.
        This read only property is set when an atmosphere is loaded.
        """
        return(self._nlayer)

    def set_scale(self, modeltype, scale, wavelength=None, radius=None):
        """Set model type and atmosphere scale. Input modeltype is case
        insensitive, but will be forced to lowercase for subsequent use.
        For modeltype 'tau', specify wavelength of the continuum optical
        depth scale. For modeltype 'sph', specify stellar radius (in cm)
        of deepest point in the atmosphere grid.

        Setting a new scale updates the number oflayers and invalidates
        existing values of temperature, electron number density, total
        number density, and mass density at each layer in the atmosphere.

        Raise ValueError exception if modeltype is not valid.
        Raise AttributeError exception if wavelength and/or radius
        are missing when expected or specifed when not expected.
        """
        if modeltype.lower() not in self._modeltypes:
            raise ValueError(
                f'Invalid modeltype: {modeltype}\n' +
                'Valid modeltypes: ' + ' '.join(self._modeltypes))
        self._modeltype = modeltype.lower()
        self._radius = None
        self._wavelength = None
        if self.modeltype == 'rhox':
            if wavelength is not None or radius is not None:
                raise AttributeError(
                    "For modeltype 'rhox' do not specify wavelength or radius")
        if self.modeltype == 'tau':
            if wavelength is None or radius is not None:
                raise AttributeError(
                    "For modeltype 'tau' specify wavelength but not radius")
            else:
                self._wavelength = float(wavelength)
        if self.modeltype == 'sph':
            if wavelength is not None or radius is None:
                raise AttributeError(
                    "For modeltype 'sph' specify radius but not wavelength")
            else:
                self._radius = float(radius)
        self._scale = scale
        self._nlayer = len(self.scale)
        self._temperature = None
        self._elecnumbdens = None
        self._atomnumbdens = None
        self._massdensity = None


class Atlas9AtmoFile:
    """Contents of an ATLAS9 atmosphere file.
    """
    def __init__(self, filename):
        self._filename = filename
        self.read(filename)

    def __str__(self):
        if 'ON' in self.conv:
            convstr = f'{self.conv.strip().lower()}, L/H={self.mixlen}'
        else:
            convstr = f'{self.conv.strip().lower()}'
        if 'ON' in self.turb:
            turbstr = f'{self.turb.strip().lower()}, param={self.turbparam}'
        else:
            turbstr = f'{self.turb.strip().lower()}'
        return(
            f"file='{self.filename}'\n"
            f"teff={self.teff}, logg={self.logg}, ifop={self.ifop[1::2]}\n"
            f"title='{self.title.strip()}'\n"
            f"convection={convstr}, turbulence={turbstr}")

    @property
    def filename(self):
        return self._filename

    @property
    def teff(self):
        return self._teff

    @property
    def logg(self):
        return self._logg

    @property
    def title(self):
        return self._title

    @property
    def ifop(self):
        return self._ifop

    @property
    def conv(self):
        return self._conv

    @property
    def mixlen(self):
        return self._mixlen

    @property
    def turb(self):
        return self._turb

    @property
    def turbparam(self):
        return self._turbparam

    def read(self, filename):
        """Read data from an ATLAS9 atmosphere file.
        """
        with open(filename, 'r') as file:
            lines = file.read().splitlines()
        self.parse_header(*lines[0:4])

    def parse_header(self, line0, line1, line2, line3):
        """Parse four header lines from an ATLAS9 atmosphere file.
        Print detailed diagnostics if expected text is not found.
        """
        expected = [
            'TEFF ', '  GRAVITY', 'TITLE ', ' OPACITY IFOP',
            ' CONVECTION ', ' TURBULENCE']
        actual = [
            line0[0:5], line0[12:21], line1[0:6], line2[0:13],
            line3[0:12], line3[21:32]]
        if expected != actual:
            fmt = lambda e, m, a: f"  {e:15} {m:2} {a:15}"
            quote = lambda s: "'" + s + "'"
            print(fmt('Expected Text', '', 'Actual Text'))
            print(fmt('---------------', '??', '---------------'))
            for e, a in zip(expected, actual):
                if e == a:
                    print(fmt(quote(e), '==', quote(a)))
                else:
                    print(fmt(quote(e), '!=', quote(a)))
            raise ValueError(
                f'{self._filename} does not have expected text')
        self._teff = float(line0[5:12])
        self._logg = float(line0[21:29])
        self._title = line1[6:]
        self._ifop = line2[13:53]
        self._conv = line3[12:16]
        self._mixlen = float(line3[16:22])
        self._turb = line3[32:36]
        chop = lambda s, w: [float(s[w*i:w*(i+1)]) for i in range(len(s)//w)]
        self._turbparam = chop(line3[36:60], 6)
