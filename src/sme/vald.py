from collections import OrderedDict
from .abund import Abund


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails.
    """


class Line:
    """Atomic data for a single spectral line.
    """
    def __init__(self, species, wlcent, excit, gflog, gamrad, gamqst, gamvw):
        self.species = str(species)
        self.wlcent = float(wlcent)
        self.excit = float(excit)
        self.gflog = float(gflog)
        self.gamrad = float(gamrad)
        self.gamqst = float(gamqst)
        self.gamvw = float(gamvw)

    def __str__(self):
        return "'{}',{:10.4f},{:7.4f},{:7.3f},{:6.3f},{:7.3f},{:8.3f}". \
            format(
                self.species,
                self.wlcent,
                self.excit,
                self.gflog,
                self.gamrad,
                self.gamqst,
                self.gamvw
                )


class LineList:
    """Atomic data for a list of spectral lines.
    """
    def __init__(self):
        self._lines = []

    def __len__(self):
        return len(self._lines)

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
    def gflog(self):
        return [line.gflog for line in self._lines]

    @property
    def gamrad(self):
        return [line.gamrad for line in self._lines]

    @property
    def gamqst(self):
        return [line.gamqst for line in self._lines]

    @property
    def gamvw(self):
        return [line.gamvw for line in self._lines]

    def add(self, species, wlcent, excit, gflog, gamrad, gamqst, gamvw):
        line = Line(species, wlcent, excit, gflog, gamrad, gamqst, gamvw)
        self._lines.append(line)


class ValdFile:
    """Atomic data for a list of spectral lines.
    """
    def __init__(self, filename):
        self._filename = filename
        self.read(filename)

    @property
    def filename(self):
        return self._filename

    @property
    def n(self):
        return self._nlines

    @property
    def linelist(self):
        return self._linelist

    @property
    def valdatmo(self):
        return self._valdatmo

    @property
    def abund(self):
        return self._abund

    def read(self, filename):
        """Read line data file from the VALD extract stellar service.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
        self.parse_header(lines[0])
        self._linelist = self.parse_linedata(lines[3:3+self.n])
        self._valdatmo = self.parse_valdatmo(lines[3+self.n])
        self._abund = self.parse_abund(lines[4+self.n:22+self.n])

    def parse_header(self, line):
        """Parse header line from a VALD line data file.
        """
        words = [w.strip() for w in line.split(',')]
        if len(words) < 5 or words[5] != 'Wavelength region':
            raise FileError(f'{self._filename} is not a VALD line data file')
        self._wavelo = float(words[0])
        self._wavehi = float(words[1])
        self._nlines = int(words[2])
        self._nlines_proc = int(words[3])
        self._vmicro = float(words[4])

    def parse_linedata(self, lines):
        """Parse line data from a VALD line data file.
        """
        linelist = LineList()
        for line in lines:
            words = [w.strip() for w in line.split(',')]
            assert len(words) >= 8
            spec_ion = words[0]
            if spec_ion[0] == "'" and spec_ion[-1] == "'":
                spec_ion = spec_ion[1:-1]
            linelist.add(spec_ion, *words[1:3], *words[4:8])
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
