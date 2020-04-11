from math import log10
from sys import version_info

from sme.abund import Abund
from sme.util import FileError, filesection


class AtmoFileError(Exception):
    """Raise when attempt to read an atmosphere file fails.
    """


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
        return ' '.join([
            'True:', *[k for k, v in self.items() if v is True], '|',
            'False:', *[k for k, v in self.items() if v is False]])

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
        return list(map(lambda x: 1 if x is True else 0, self.values()))


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
        return (
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
        return self._modeltype

    @property
    def wavelength(self):
        """Wavelength (in Angstrom) of continuum optical depth scale.
        Set for modeltype 'tau', otherwise None. This read only property
        is set when an atmosphere is loaded.
        """
        return self._wavelength

    @property
    def radius(self):
        """Radius (in cm) of deepest point in an atmosphere grid. Set
        for modeltype 'sph', otherwise None. This read only property is
        set when an atmosphere is loaded.
        """
        return self._radius

    @property
    def scale(self):
        """Physical scale for each layer in the atmosphere. See `modeltype`
        for description of scale types (mass column, continuum optical depth,
        or height). This read only property is set when an atmosphere is
        loaded.
        """
        return self._scale

    @property
    def nlayer(self):
        """Number of layers (depths or heights) in the model atmosphere.
        This read only property is set when an atmosphere is loaded.
        """
        return self._nlayer

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


class AtmoFileAtlas9:
    """Contents of an ATLAS9 atmosphere file.

    Parameters
    ----------
    path : path-like object
        Path to an ATLAS9 atmosphere file
    """
    def __init__(self, path):
        self._path = path
        self._read(path)

    def __str__(self):
        p = self.param
        if 'ON' in p['conv']:
            convstr = f"{p['conv'].strip().lower()}, L/H={p['mixlen']}"
        else:
            convstr = f"{p['conv'].strip().lower()}"
        if 'ON' in p['turb']:
            turbstr = f"{p['turb'].strip().lower()}, param={p['turbparam']}"
        else:
            turbstr = f"{p['turb'].strip().lower()}"
        return (
            f"file='{self.path}'\n"
            f"teff={self.teff}, logg={self.logg}, ifop={p['ifop'][1::2]}\n"
            f"title='{p['title'].strip()}'\n"
            f"convection={convstr}, turbulence={turbstr}\n"
            f"ndepth={self.ndepth}, niter={p['niter']}, pradk={p['pradk']}")

    @property
    def path(self):
        """Path to the ATLAS9 atmosphere file used to initialize object.
        """
        return self._path

    @property
    def param(self):
        """Dictionary of parameters read from the ATLAS9 atmosphere file.
        """
        return self._param

    @property
    def teff(self):
        """Effective temperature for the ATLAS9 atmosphere.
        """
        return self.param['teff']

    @property
    def logg(self):
        """Logarithm of surface gravity for the ATLAS9 atmosphere.
        """
        return self.param['logg']

    @property
    def abund(self):
        """Abundance pattern and metallicity for the ATLAS9 atmosphere.
        """
        return self._abund

    @property
    def ndepth(self):
        """Number of layers in the ATLAS9 atmosphere.
        """
        return self._ndepth

    @property
    def atmo(self):
        """Dictionary that contains depth-dependent atmosphere data.

        Keys come from the ATLAS9 atmosphere file. Nominal keys:

        ======== ========= ==============================
        Key      Units     Description
        ======== ========= ==============================
        RHOX     g/cm**2   Mass column density
        T        K         Kinetic temperature
        P        erg/cm**3 Total gas pressure
        XNE      1/cm**3   Electron number fraction
        ABROSS   cm**2/g   Rosseland mean mass extinction
        ACCRAD   cm/s**2   Radiative acceleration
        VTURB    cm/s      Microturbulence velocity
        FLXCNV             Convective flux
        VCONV    cm/s      Velocity of convective cells
        VELSND   cm/s      Local sound speed
        ======== ========= ==============================
        """
        return self._atmo

    def _read(self, path):
        """Read data from an ATLAS9 atmosphere file.
        """
        try:
            with open(path, 'r') as fobj:
                self._param = self._parse_header(
                    filesection(fobj, 'header', nline=4))
                self._abund = self._parse_abund(
                    filesection(fobj, 'abundance', nline=18))
                self._ndepth = self._parse_ndepth(
                    filesection(fobj, 'ndepth', nline=1))
                self._atmo = self._parse_atmo(
                    filesection(fobj, 'atmo', nline=self._ndepth))
                self._param['pradk'] = self._parse_pradk(
                    filesection(fobj, 'pradk', nline=1))
                self._param['niter'] = self._parse_niter(
                    filesection(fobj, 'niter', nline=1))
        except FileError as e:
            raise AtmoFileError(e)

    def _parse_header(self, lines):
        """Parse header from an ATLAS9 atmosphere file.
        """
        try:
            expected = [
                'TEFF ', '  GRAVITY', 'TITLE ', ' OPACITY IFOP',
                ' CONVECTION ', ' TURBULENCE']
            actual = [
                lines[0][:5], lines[0][12:21], lines[1][:6],
                lines[2][:13], lines[3][:12], lines[3][21:32]]
            if expected != actual:
                diag = '\n   Expected_Label    Actual_Label'
                for e, a in zip(expected, actual):
                    operator = '==' if e == a else '!='
                    diag += f'\n  {e!r:>15} {operator} {a!r:15}'
                raise ValueError(
                    f'error parsing header: {self._path}' + diag)
            param = {}
            param['teff'] = float(lines[0][5:12])
            param['logg'] = float(lines[0][21:29])
            param['title'] = lines[1][6:]
            param['ifop'] = lines[2][13:53]
            param['conv'] = lines[3][12:16]
            param['mixlen'] = float(lines[3][16:22])
            param['turb'] = lines[3][32:36]
            param['turbparam'] = [
                float(lines[3][i:i+6]) for i in range(36, 60, 6)]
            return param
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing header: {self._path}')

    def _parse_abund(self, lines):
        """Parse abundances from an ATLAS9 atmosphere file.

        Check 'ABUNDANCE SCALE' label. Parse abundance scale factor.
        Join remaining text into string. Split on 'ABUNDANCE CHANGE' label.
        Split again on white space. Even index words are atomic number.
        Odd index words are abundances relative to total number of nuclei.
        Leave H abundances linear. Convert H3 abundance to log10.
        Leave all other abundances log10.
        """
        try:
            assert lines[0][0:16] == 'ABUNDANCE SCALE '
            monh = log10(float(lines[0][16:25]))
            abstr = lines[0][25:] + ''.join(lines[1:])
            words = ''.join(abstr.split('ABUNDANCE CHANGE')).split()
            assert [int(s) for s in words[0::2]] == list(range(1, 100))
            abund = [float(s) for s in words[1::2]]
            abund[1] = log10(abund[1])
            elements = Abund(0, 'Empty').elements
            values = {el: ab for el, ab in zip(elements, abund)}
            return Abund(monh, values, 'sme')
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing abund: {self._path}')

    def _parse_ndepth(self, lines):
        """Parse header for atmosphere section of an ATLAS9 atmosphere file.
        """
        try:
            assert lines[0][0:10] == 'READ DECK6'
            self._keys = [c.strip() for c in lines[0][13:].split(',')]
            return int(lines[0][10:13])
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing ndepth: {self._path}')

    def _parse_atmo(self, lines):
        """Parse atmosphere section of an ATLAS9 atmosphere file.

        Check that number of values per input line matches number of keys.
        """
        try:
            nkey = len(self._keys)
            layers = []
            for line in lines:
                layer = [float(s) for s in line.split()]
                assert len(layer) == nkey
                layers.append(layer)
            values = map(list, zip(*layers))
            return dict(zip(self._keys, values))
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing atmo: {self._path}')

    def _parse_pradk(self, lines):
        """Parse radiation pressure section of an ATLAS9 atmosphere file.
        """
        try:
            assert lines[0][0:5] == 'PRADK'
            return float(lines[0][5:])
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing pradk: {self._path}')

    def _parse_niter(self, lines):
        """Parse number of iterations section of an ATLAS9 atmosphere file.
        """
        try:
            assert lines[0][25:35] == 'ITERATION '
            return int(lines[0][35:38])
        except (AssertionError, ValueError):
            raise AtmoFileError(f'error parsing niter: {self._path}')
