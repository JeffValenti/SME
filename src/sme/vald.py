from re import findall
from collections import OrderedDict
from itertools import zip_longest
from sme.abund import Abund


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails.
    """


class SmeLine:
    """Data required by SME for each atomic or molecular line.
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
        """Return comma separated list of line data.
        """
        return "{:13s}{:10.4f},{:9.4f},{:7.3f},{:6.3f},{:7.3f},{:8.3f}". \
            format(
                "'" + self.species + "',",
                self.wlcent,
                self.excit,
                self.loggf,
                self.gamrad,
                self.gamqst,
                self.gamvw
                )

    def __repr__(self):
        return "SmeLine({})".format(self.__str__())


class ValdShortLine:
    """Data for one atomic or molecular line from a short-format VALD file.
    """
    def __init__(self, line):
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
        return '{:13s}{:10.4f},{:9.4f},{:4.1f},{:7.3f},{:6.3f},{:6.3f},' \
            '{:8.3f},{:7.3f},{:6.3f}, {}'. \
            format(
                "'" + self.species + "',",
                self.wlcent,
                self.excit,
                self.vmicro,
                self.loggf,
                self.gamrad,
                self.gamqst,
                self.gamvw,
                self.lande_mean,
                self.depth,
                self.ref
                )

    def __repr__(self):
        return 'ValdShortLine("{}")'.format(self.__str__())


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
            id = self.id
            ref = self.ref
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
    def __init__(self):
        self._lines = []

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, key):
        return self._lines[key]

    def __str__(self):
        out = []
        for line in self._lines:
            out.append(line.__str__())
        return '\n'.join(out)

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

    @property
    def lande_mean(self):
        return [line.lande_mean for line in self._lines]

    @property
    def depth(self):
        return [line.depth for line in self._lines]

    def add(self, line):
        self._lines.append(line)


class ValdFile:
    """Contents of a VALD3 line data file.
    """
    def __init__(self, filename):
        self._filename = filename
        self.read(filename)

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
                linelist.add(ValdShortLine(line))
        elif self._format == 'long':
            for chunk in zip_longest(*[iter(lines)]*4):
                linelist.add(ValdLongLine(chunk))
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
