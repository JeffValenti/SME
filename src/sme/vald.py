from bisect import bisect
from re import findall
from collections import OrderedDict
from itertools import zip_longest
from sme.abund import Abund
from sme.util import change_waveunit, change_energyunit, vacuum_angstroms


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails.
    """


class SmeLine:
    """Basic data required by SME for each atomic or molecular transition.

    Attributes
    ----------
    species : str
        Name of atom (e.g., 'Co') or molecule (e.g., 'CO'), followed by a
        space and the ionization state (e.g., ' 1'). Atom name begins with
        a capital letter (e.g., 'C'). Additional letters (if any) are lower
        case (e.g., 'Co'). Molecule name is a sequence of atom names (e.g.,
        'CO') with multiplicity factors as needed (e.g., 'H2O') Ionization
        state is a number: '1' for neutral, '2' for singly ionized, etc.
        *Examples:* ``'C 4'``, ``'Co 2'``, ``'CO 1'``, or ``'H2O 1``'.
    wlcent : float or str that yields a float
        Wavelength (in Angstroms) of the transition.
        *Examples:* ``6564.61`` or ``'6564.6100'``.
    excit : float or str that yields a float
        Energy (in eV) of the lower state of the transition.
        *Examples:* ``10.1988`` or ``'10.1988'``.
    loggf : float or str that yields a float
        Logarithm of the product of the statistical weight (g) of the lower
        state of the transition times the oscillator strength (f) of the
        transition. *Examples:* ``0.71`` or ``'0.710'``.
    gamrad : float or str that yields a float
        Logarithm of the radiative damping parameter for the transition.
        *Examples:* ``8.766`` or ``'8.766'``.
    gamqst : float or str that yields a float
        For species other than atomic hydrogen, logarithm of the quadratic
        Stark broading parameter for the transition. For atomic hydrogen,
        principal quantum number (n) of the lower state. A value of zero
        requests use of an approximate Stark broadening formula from
        Cowley (1971).
        *Examples:* ``-6.14``, ``'-6.14'``, ``2``, ``'2.000'``, ``0``,
        or ``'0.000'``.
    gamvw : float or str that yields a float
        For species other than atomic hydrogen and value less than zero,
        logarithm of the van der Waals collisional broading parameter for
        the transition at 10000 K (e.g., '-7.510'). For species other than
        atomic hydrogen and value greater than 20, broadening cross section
        (sigma in atomic units) at 10 km/s  and velocity exponent (alpha)
        packed into a single parameter int(sigma)+alpha (e.g., '230.192').
        See Barklem for more info.
        For atomic hydrogen, principal quantum number (n) of the upper state.
        A value of zero requests use of the Unsold approximation (1955).
        *Examples:* ``-7.510``, ``'-7.510'``, ``230.192``, ``'230.192'``,
        ``3``, ``'3'``, ``0``, or ``'0.000'``.
    gamvw : float or str that yields a float
        Van der Waals damping parameter (line FWHM per perturber in units
        of rad/s cm**3) at 10000 K for collisions with neutral species.
        For species other than hydrogen:

        * A value less than zero is the base 10 logarithm of the damping
          parameter. *Examples:* ``-7.510`` or ``'-7.510'``.
        * A value of zero indicates that no damping parameter is specified.
          In such cases, SME uses a modified Unsold (1955) approximation to
          calculate the damping parameter. *Examples:* ``0`` or ``'0.000'``.
        * A value greater than 20 is the broadening cross section (sigma in
          atomic units) at 10 km/s and the velocity exponent (alpha), packed
          into a single parameter, int(sigma)+alpha. See Barklem (1998) for
          more information. *Examples:* ``230.192`` or ``'230.192'``.

        For atomic hydrogen:

        * Principal quantum number (n) of the upper state.
          *Examples:* ``3`` or ``'3'``.

    Notes
    -----
    These parameter conventions are used by the Vienna Atomic Line Database
    (VALD) extract stellar service, except that VALD users may select other
    units for `wlcent` and `excit`. Other conventions (e.g., ionization state
    expressed by a roman numeral) are not valid in SME.

    References
    ----------
    | VALD3 output formats for Extract Stellar request
    |   http://www.astro.uu.se/valdwiki/select_output
    | Barklem et al. (1998)
    |   https://ui.adsabs.harvard.edu/#abs/2000A&AS..142..467B
    |   https://www.astro.uu.se/%7Ebarklem/howto.html
    | Cowley (1971)
    |   https://ui.adsabs.harvard.edu/#abs/1971Obs....91..139C

    Examples
    --------
    >>> line = SmeLine('H 1', '6564.61', '10.20', '0.71', '8.766', '2', '3')
    >>> line = SmeLine('Co 1', 6565.21, 2.042, 3.93, 7.70, -6.14, 270.243)
    """
    def __init__(self, species, wlcent, excit, loggf, gamrad, gamqst, gamvw):
        self.species = str(species)
        self.wlcent = float(wlcent)
        self.excit = float(excit)
        self.loggf = float(loggf)
        self.gamrad = float(gamrad)
        self.gamqst = float(gamqst)
        self.gamvw = float(gamvw)

    def __str__(self):
        """Return line data as a string in VALD extract sellar short format."

        Example
        -------
        >>> print(line)
        'Co 1',       6565.2100,   2.0420,  3.930, 7.700,-6.140, 270.243

        """
        quote_species_quote_comma = "'" + self.species + "',"
        return ' '.join(
            f"{quote_species_quote_comma:13s}"
            f"{self.wlcent:10.4f},"
            f"{self.excit:9.4f},"
            f"{self.loggf:7.3f},"
            f"{self.gamrad:6.3f},"
            f"{self.gamqst:6.3f},"
            f"{self.gamvw:8.3f}".split())

    def __repr__(self):
        """Return python string representation of this object.

        Examples
        --------
        >>> line = SmeLine('Co 1', 6565.21, 2.042, 3.93, 7.70, -6.14, 270.243)
        >>> repr(line)
        "SmeLine('Co 1', 6565.2100, 2.0420, 3.930, 7.700,-6.140, 270.243)"

        """
        return f'{self.__class__.__name__}({self.__str__()})'

    def __eq__(self, other):
        """Test whether two SmeLine objects have identical lines parameters.

        Parameters
        ----------
        other : SmeLine object
            Another spectral line to compare with this spectral line.

        Examples
        --------
        >>> line1 = SmeLine('Co 1', 6565.21, 2.042, 3.93, 7.70, -6.14, 270.243)
        >>> line2 = SmeLine('Co 1', 6565.21, 2.042, 3.93, 7.70, -6.14, 270.243)
        >>> line1 == line2, line1.__eq__(line2), line1 is line2
        (True, True, False)

        """
        if self.__class__ is other.__class__:
            return self.species == other.species and \
                self.wlcent == other.wlcent and \
                self.excit == other.excit and \
                self.loggf == other.loggf and \
                self.gamrad == other.gamrad and \
                self.gamqst == other.gamqst and \
                self.gamvw == other.gamvw
        else:
            return False


class ValdShortLine:
    """Data for one atomic or molecular line from a short-format VALD file.
    """
    def __init__(self, line, wlmedium=None, wlunits=None, exunits=None):
        data, shortref = line.strip().split(", '")
        words = [w.strip() for w in data.split(',')]
        assert len(words) >= 8
        spec_ion = words[0]
        if spec_ion[0] == "'" and spec_ion[-1] == "'":
            spec_ion = spec_ion[1:-1]
        self.species = spec_ion
        self.wlcent = float(words[1])
        self.excit = float(words[2])
        self.vmicro = float(words[3])
        self.loggf = float(words[4])
        self.gamrad = float(words[5])
        self.gamqst = float(words[6])
        self.gamvw = float(words[7])
        self.lande_mean = float(words[8])
        self.depth = float(words[9])
        self.ref = ValdShortRef(shortref[:-1])

    def __str__(self):
        """Return line data as they would appear in a VALD line data file.
        """
        quote_species_quote_comma = "'" + self.species + "',"
        return \
            f"{quote_species_quote_comma:13s}" \
            f"{self.wlcent:10.4f}," \
            f"{self.excit:9.4f}," \
            f"{self.vmicro:4.1f}," \
            f"{self.loggf:7.3f}," \
            f"{self.gamrad:6.3f}," \
            f"{self.gamqst:6.3f}," \
            f"{self.gamvw:8.3f}," \
            f"{self.lande_mean:7.3f}," \
            f"{self.depth:6.3f}, " \
            f"{self.ref}"

    def __repr__(self):
        """Return python string representation of this object.
        """
        return f'{self.__class__.__name__}({self.__str__()!r})'


class ValdShortRef:
    """Reference information for one transition in a short format VALD file.
    """
    def __init__(self, shortref):
        try:
            words = shortref.split()
            assert len(words) == 15
        except AssertionError:
            raise FileError(f'expected 15 words in VALD line ref: {lineref}')
        id = [int(w) for w in words[:13:2]]
        ref = words[1::2]
        self.species = words[14]
        self.wlgf_labels = ref[0][0:3] == 'wl:' and ref[2][0:3] == 'gf:'
        if self.wlgf_labels:
            ref[0] = ref[0][3:]
            ref[2] = ref[2][3:]
        if ref.count(ref[0]) == len(ref):
            self.id = id[0]
            self.ref = ref[0]
        else:
            self.id = id
            self.ref = ref

    def __str__(self):
        """Return references as they would appear in a short format VALD file.
        """
        if type(self.ref) is str:
            id = [self.id] * 7
            ref = [self.ref] * 7
        else:
            id = self.id.copy()
            ref = self.ref.copy()
        if self.wlgf_labels:
            ref[0] = 'wl:' + ref[0]
            ref[2] = 'gf:' + ref[2]
        idref = []
        for i, r in zip(id, ref):
            idref.append(f'{i:3d} {r}')
        out = ' '.join(idref) + f' {self.species:14s}'
        return "' " + out + "'"

    @property
    def wlcent(self):
        return self.ref if type(self.ref) is str else self.ref[0]

    @property
    def excit(self):
        return self.ref if type(self.ref) is str else self.ref[1]

    @property
    def loggf(self):
        return self.ref if type(self.ref) is str else self.ref[2]

    @property
    def gamrad(self):
        return self.ref if type(self.ref) is str else self.ref[3]

    @property
    def gamqst(self):
        return self.ref if type(self.ref) is str else self.ref[4]

    @property
    def gamvw(self):
        return self.ref if type(self.ref) is str else self.ref[5]

    @property
    def lande_mean(self):
        return self.ref if type(self.ref) is str else self.ref[6]


class ValdLongLine:
    """Data for one atomic or molecular line from a long-format VALD file.
    """
    def __init__(self, chunk):
        line = chunk[0]
        words = [w.strip() for w in line.split(',')]
        assert len(words) >= 8
        spec_ion = words[0]
        if spec_ion[0] == "'" and spec_ion[-1] == "'":
            spec_ion = spec_ion[1:-1]
        self.species = spec_ion
        self.wlcent = float(words[1])
        self.excit = float(words[2])
        self.loggf = float(words[4])
        self.gamrad = float(words[5])
        self.gamqst = float(words[6])
        self.gamvw = float(words[7])
        self.lande_mean = float(words[8])
        self.depth = float(words[9])


class LineList:
    """Line data for a list of atomic and molecular lines.
    """
    _valid_line_types = (SmeLine, ValdShortLine, ValdLongLine)

    def __init__(self):
        self._lines = []

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, key):
        return self._lines[key]

    def __setitem__(self, key, line):
        if type(line) in self._valid_line_types:
            self._lines[key] = line
        else:
            self._raise_invalid_line_type(line)

    def _raise_invalid_line_type(self, line):
        raise TypeError(
            f'line in LineList has invalid type: {type(line).__name__}\n'
            f'  Valid line types: ' + \
            ' '.join([type.__name__ for type in self._valid_line_types]))

    def __str__(self):
        out = []
        for line in self._lines:
            out.append(line.__str__())
        return '\n'.join(out)

    def _raise_invalid_line_type(self, line):
        raise TypeError(
            f'line in LineList has invalid type: {type(line).__name__}\n'
            f'  Valid line types: ' + \
            ' '.join([type.__name__ for type in self._valid_line_types]))

    @property
    def species(self):
        return [line.species for line in self._lines]

    @property
    def wlcent(self):
        return [line.wlcent for line in self._lines]

    @property
    def excit(self):
        return [line.excit for line in self._lines]

    @property
    def loggf(self):
        return [line.loggf for line in self._lines]

    @property
    def gamrad(self):
        return [line.gamrad for line in self._lines]

    @property
    def gamqst(self):
        return [line.gamqst for line in self._lines]

    @property
    def gamvw(self):
        return [line.gamvw for line in self._lines]

    def append(self, line):
        if type(line) in self._valid_line_types:
            self._lines.append(line)
        else:
            self._raise_invalid_line_type(line)

class ValdFile:
    """Contents of a VALD3 line data file.
    """
    def __init__(self, filename, standard=True):
        self._filename = filename
        self.read(filename)
        if standard:
            self.standardize()

    @property
    def filename(self):
        return self._filename

    @property
    def wlrange(self):
        return (self._wavelo, self._wavehi)

    @property
    def nlines(self):
        return self._nlines

    @property
    def nprocessed(self):
        return self._nprocessed

    @property
    def wlmedium(self):
        return self._wlmedium

    @property
    def wlunits(self):
        return self._wlunits

    @property
    def exunits(self):
        return self._exunits

    @property
    def linelist(self):
        return self._linelist

    @property
    def valdatmo(self):
        return self._valdatmo

    @property
    def abund(self):
        return self._abund

    @property
    def isotopes(self):
        return self._isotopes

    @property
    def references(self):
        return self._references

    @property
    def version(self):
        return self._version

    def read(self, filename):
        """Read line data file from the VALD extract stellar service.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
        ibeg, iend = 0, 3
        self.parse_header(lines[0:3])
        chunksize = 1 if self._format == 'short' else 4
        ibeg, iend = iend, iend+chunksize*self.nlines
        self._linelist = self.parse_linedata(lines[ibeg:iend])
        self._valdatmo = self.parse_valdatmo(lines[iend])
        ibeg, iend = iend+1, iend+19
        self._abund = self.parse_abund(lines[ibeg:iend])
        self._isotopes = self.parse_isotopes(lines[iend])
        self._references = self.parse_references(lines[iend+1:])

    def parse_header(self, lines):
        """Parse header lines from a VALD line data file.
        Presence of wavelength medium ('_air' or '_vac') implies VALD3.
        """
        words = [w.strip() for w in lines[0].split(',')]
        if len(words) < 5 or words[5] != 'Wavelength region':
            raise FileError(f'{self._filename} is not a VALD line data file')
        self._wavelo = float(words[0])
        self._wavehi = float(words[1])
        self._nlines = int(words[2])
        self._nprocessed = int(words[3])
        self._vmicro = float(words[4])
        self._wlmedium = lines[2].partition('WL_')[2].partition('(')[0]
        self._version = 2 if self._wlmedium == '' else 3
        self._format = 'long' if 'J lo' in lines[2] else 'short'
        self._wlunits, self._exunits = findall(r'\(([^)]+)', lines[2])[0:2]

    def parse_linedata(self, lines):
        """Parse line data from a VALD line data file.
        """
        linelist = LineList()
        if self._format == 'short':
            for line in lines:
                linelist.append(ValdShortLine(
                    line, self.wlmedium, self.wlunits, self.exunits))
        elif self._format == 'long':
            for chunk in zip_longest(*[iter(lines)]*4):
                linelist.append(ValdLongLine(chunk))
        else:
            raise FileError(f'{self._filename} has unknown format')
        return linelist

    def parse_valdatmo(self, line):
        """Parse VALD model atmosphere line from a VALD line data file.
        """
        lstr = line.strip()
        if lstr[0] != "'" or lstr[-2:] != "',":
            raise FileError(f'error parsing model atmosphere: {lstr}')
        return lstr[1:-2]

    def parse_abund(self, lines):
        """Parse VALD abundance lines from a VALD line data file.
        """
        abstr = ''.join([''.join(line.split()) for line in lines])
        words = [w[1:-1] for w in abstr.split(',')]
        if len(words) != 100 or words[99] != 'END':
            raise FileError(f'error parsing abundances: {abstr}')
        pattern = [w.split(':') for w in words[:-1]]
        pattern = OrderedDict([(el, float(ab)) for el, ab in pattern])
        monh = 0.0
        return Abund(monh, pattern, type='sme')

    def parse_isotopes(self, line):
        """Infer whether isotopic scaling was applied in VALD line data file.
        """
        return 'oscillator strengths were scaled by' in line

    def parse_references(self, lines):
        references = {}
        for line in lines:
            id, period, ref = line.partition('.')
            if 'References:' not in line:
                references[id.strip()] = ref.strip()
        return references

    def standardize(self):
        '''Convert linelist wavelengths to vacuum Angstroms and energies to eV.

        If input wavelength units are inverse centimeters, reverse the
        wavelength range and reverse the order of lines in the linelist.

        If any input wavelengths are bluer than 2000 Angstroms, search
        for the first air wavelength, which can be slightly below 2000
        Angstroms (see docstring for `util._first_air_wavelength`).
        '''
        # Reverse order of lines if units are cm^-1
        nline = len(self.linelist)
        if self.wlunits.lower() in ['cm^-1', '1/cm']:
            self._wavelo, self._wavehi = self._wavehi, self._wavelo
            for i in range(nline // 2):
                self._linelist[i], self._linelist[nline-i-1] = \
                    self.linelist[nline-i-1], self.linelist[i]

        # Convert wavelengths of line centers to Angstroms
        wlcent = self.linelist.wlcent
        if self.wlunits.upper() != 'A':
            wlcent = change_waveunit(wlcent, self.wlunits, 'A')

        # Convert air wavelengths to vacuum wavelengths
        if self.wlmedium.lower() == 'air':
            iair = self._first_air_wavelength(wlcent)
            if iair < len(wlcent):
                wlcent[iair:] = vacuum_angstroms(wlcent[iair:], 'A', 'air')

        # Update wavelengths in line list, if not already vacuum Angstroms
        if self.wlunits != 'A' or self.wlmedium != 'vac':
            self._wavelo, self._wavehi = vacuum_angstroms(
                [self._wavelo, self._wavehi], self.wlunits, self.wlmedium)
            for line, w in zip(self.linelist, wlcent):
                line.wlcent = w
            self._wlmedium = 'vac'
            self._wlunits = 'A'

        # Convert energies to eV and update line list, if not already eV
        excit = self.linelist.excit
        if self.exunits.lower() != 'eV':
            excit = change_energyunit(excit, self.exunits, 'eV')
            for line, e in zip(self.linelist, excit):
                line.excit = e
            self._exunit = 'eV'

    def _first_air_wavelength(self, wave):
        '''Return index of first air wavelength in input list.

        By convention, air wavelengths are actually vacuum wavelengths for
        vacuum wavelengths below 2000 Angstroms. However, using the VALD3
        :any:`vaccum_to_air` formula, a vacuum wavelength of 2000 Angstroms
        corresponds to an air wavelength of 1999.3520267833612 Angstroms.
        Thus, 'air' wavelengths between 1999.3520 and 2000 Angstroms can
        either be vacuum wavelengths in that range or vacuum wavelengths
        in the range 2000 to 2000.6480857571032 Angstroms that have been
        converted to air. Context may help distinguish these two cases.
        A VALD3 line list in air may have vacuum wavelengths that increase
        up to the 2000 Angstrom threshold and then jump backwards once to
        shorter wavelengths between 1999.352 and 2000 Angstroms. In this
        case, the backwards jump indicates the transition from vacuum
        wavelengths to air wavelengths in the line list. If there are lines
        in the 1999.352 to 2000 Angstrom interval, but no jump backwards
        to shorter wavelengths, then it is not possible to infer from the
        wavelengths alone whether the medium for these lines is air or
        vacuum. This routine assumes such lines are vacuum wavelengths,
        but that assumption could be wrong. Request vacuum wavelengths
        from VALD to avoid this issue.

        If all wavelengths are greater than 2000 Angstroms, return
        iair = 0, so that wave[iair:] returns all elements in wave.
        Example: wave = [2001, 2002] returns 0

        If all wavelengths are less than or equal to 1999.352 Angstroms,
        return iair = len(wave), so that wave[iair:] is an empty list.
        Example: wave = [1998, 1999] returns 2

        If there is a backwards jump between 1999.352 and 2000 Angstroms,
        return iair corresponding to the element immediately after the
        backwards jump, so that wave[iair:] returns the non-decreasing
        sequence of wavelengths after the backwards jump.
        Example: wave = [1999.6, 1999.5] returns 1

        Otherwise return iair corresponding to the first wavelength
        greater than 2000 Angstroms, so that wave[iair:] returns all
        wavelengths greater than 2000 Angstroms or an empty list if there
        are not elements greater than 2000 Angstroms.
        Example: wave = [1999.5, 1999.6] returns 2
        Example: wave = [1999, 2000, 2001] returns 2
        '''
        nwave = len(wave)
        wlo, whi = 1999.352026783362, 2000
        ilo, ihi = [bisect(wave, w) for w in (wlo, whi)]

        # All wavelengths > 2000 Angstroms. All are air.
        if ihi == 0:
            return(0)

        # All wavelengths <= 1999.352 Angstroms. All are vacuum.
        if ilo == nwave:
            return(nwave)

        # Air begins at backwards jump between 1999.352 and 2000 Angstroms.
        if ihi - ilo > 1:
            jump = [i for i in range(ilo+1, ihi) if wave[i] < wave[i-1]]
            assert len(jump) < 2
            if jump:
                return(jump[0])

        # Air begin at first wavelength > 2000 Angstroms.
        return(ihi)
