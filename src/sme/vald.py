from collections import OrderedDict

import numpy as np
import numpy.lib.recfunctions

from .abund import Abund


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails.
    """


class Line:
    """Atomic data for a single spectral line.
    """

    def __init__(
        self,
        species,
        wlcent,
        excit,
        gflog,
        gamrad,
        gamqst,
        gamvw,
        lulande=None,
        extra=None,
    ):
        self.species = str(species)
        self.wlcent = float(wlcent)
        self.excit = float(excit)
        self.gflog = float(gflog)
        self.gamrad = float(gamrad)
        self.gamqst = float(gamqst)
        self.gamvw = float(gamvw)
        self.lulande = lulande
        self.extra = extra

        if self.lulande is not None:
            self.lulande = [float(l) for l in self.lulande]
        if self.extra is not None:
            self.extra = [float(e) for e in self.extra]

    def __str__(self):
        return "'{}',{:10.4f},{:7.4f},{:7.3f},{:6.3f},{:7.3f},{:8.3f}".format(
            self.species,
            self.wlcent,
            self.excit,
            self.gflog,
            self.gamrad,
            self.gamqst,
            self.gamvw,
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
        return "\n".join(out)

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

    def add(
        self,
        species,
        wlcent,
        excit,
        gflog,
        gamrad,
        gamqst,
        gamvw,
        lulande=None,
        extra=None,
    ):
        line = Line(
            species,
            wlcent,
            excit,
            gflog,
            gamrad,
            gamqst,
            gamvw,
            lulande=lulande,
            extra=extra,
        )
        self._lines.append(line)

    @property
    def atomic(self):
        # Data (ouput array) array of important atomic data
        # data(0,*) atomic number (H=1, He=2, Li=3, etc.)
        # data(1,*) ionization stage (neutral=0, singly=1, etc.)
        # data(2,*) central wavelength of transition (Angstroms)
        # data(3,*) excitation potential of lower state (eV)
        # data(4,*) log(gf)
        # data(5,*) radiative damping constant
        # data(6,*) quadratic Stark damping constant
        # data(7,*) van der Waal's damping constant
        # dtype = c_double
        atomic = np.zeros((len(self), 8), dtype=float)
        atomic[:, 0] = [Abund._elem_dict[s] for s in self.species]  # Atomic number
        atomic[:, 1] = [int(s[-1]) for s in self.species]  # Ionization
        atomic[:, 2] = self.wlcent  # Central Wavelength
        atomic[:, 3] = self.excit  # Excitation
        atomic[:, 4] = self.gflog  # Oscillator strength
        atomic[:, 5] = self.gamrad  # radiative damping
        atomic[:, 6] = self.gamqst  # Stark
        atomic[:, 7] = self.gamvw  # van der Waals
        return atomic


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
        with open(filename, "r") as file:
            lines = file.readlines()
        self.parse_header(lines[0])

        # TODO how to recognise extended format
        # and what to do about it
        if lines[4][:2] == "' ":
            fmt = "long"
        else:
            fmt = "short"

        if fmt == "long":
            linedata = lines[3 : 3 + self.n * 4]
            atmodata = lines[3 + self.n * 4]
            abunddata = lines[4 + self.n * 4 : 22 + self.n * 4]
        elif fmt == "short":
            linedata = lines[3 : 3 + self.n]
            atmodata = lines[3 + self.n]
            abunddata = lines[4 + self.n : 22 + self.n]

        self._linelist = self.parse_linedata(linedata, fmt=fmt)
        self._valdatmo = self.parse_valdatmo(atmodata)
        self._abund = self.parse_abund(abunddata)

    def parse_header(self, line):
        """Parse header line from a VALD line data file.
        """
        words = [w.strip() for w in line.split(",")]
        if len(words) < 5 or words[5] != "Wavelength region":
            raise FileError(f"{self._filename} is not a VALD line data file")
        self._wavelo = float(words[0])
        self._wavehi = float(words[1])
        self._nlines = int(words[2])
        self._nlines_proc = int(words[3])
        self._vmicro = float(words[4])

    def parse_line_error(self, error_flags, values):
        nist = {
            "AAA": 0.003,
            "AA": 0.01,
            "A+": 0.02,
            "A": 0.03,
            "B+": 0.07,
            "B": 0.1,
            "C+": 0.18,
            "C": 0.25,
            "C-": 0.3,
            "D+": 0.4,
            "D": 0.5,
            "D-": 0.6,
            "E": 0.7,
        }
        error = np.ones(len(error_flags), dtype=float)
        for i, (flag, value) in enumerate(zip(error_flags, values)):
            if flag[0] in [" ", "_", "P"]:
                # undefined or predicted
                error[i] = 0.5
            elif flag[0] == "E":
                # absolute error in dex
                # TODO absolute?
                error[i] = 10 ** float(flag[1:])
            elif flag[0] == "C":
                # Cancellation Factor, i.e. relative error
                error[i] = abs(float(flag[1:]))
            elif flag[0] == "N":
                # NIST quality class
                flag = flag[1:5].strip()
                error[i] = nist[flag]
        return error

    def parse_linedata(self, lines, fmt="short"):
        """Parse line data from a VALD line data file.
        """
        linelist = LineList()

        if fmt == "short":
            dtype = np.dtype(
                [
                    ("species", "<U6"),
                    ("wlcent", float),
                    ("excit", float),
                    ("vmic", float),
                    ("gflog", float),
                    ("gamrad", float),
                    ("gamqst", float),
                    ("gamvw", float),
                    ("lande", float),
                    ("depth", float),
                    ("ref", "<U100"),
                ]
            )
        elif fmt == "long":
            dtype = np.dtype(
                [
                    ("species", "<U5"),
                    ("wlcent", float),
                    ("gflog", float),
                    ("excit", float),
                    ("j_lo", float),
                    ("e_upp", float),
                    ("j_up", float),
                    ("lande_lower", float),
                    ("lande_upper", float),
                    ("lande", float),
                    ("gamrad", float),
                    ("gamqst", float),
                    ("gamvw", float),
                    ("depth", float),
                    # ("term_lower", "<U255"),
                    # ("term_upper", "<U255"),
                    # ("reference", "<U255"),
                    # ("ionization", int),
                    # ("error", float),
                ]
            )
            term_lower = lines[1::4]
            term_upper = lines[2::4]
            comment = lines[3::4]
            lines = lines[::4]

        # TODO assign correct dtype once and then fill values
        linelist = np.genfromtxt(
            lines, dtype=dtype, delimiter=",", converters={0: lambda s: s[1:-1]}
        )
        linelist = linelist.view(np.recarray)

        if fmt == "long":
            # Convert from cm^-1 to eV
            linelist["excit"] /= 8065.544
            linelist["e_upp"] /= 8065.544

            # add extra fields
            ionization = np.array([int(s[-1]) for s in linelist["species"]])
            error = np.array([s[1:11].strip() for s in comment])
            error = self.parse_line_error(error, linelist["depth"])

            linelist = np.lib.recfunctions.append_fields(
                linelist,
                ("term_lower", "term_upper", "reference", "ionization", "error"),
                (term_lower, term_upper, comment, ionization, error),
                usemask=False,
                asrecarray=True,
            )

            # additional data arrays for sme
            linelist.lulande = linelist[["lande_lower", "lande_upper"]]
            linelist.extra = linelist[["j_lo", "e_upp", "j_up"]]

        return linelist

    def parse_valdatmo(self, line):
        """Parse VALD model atmosphere line from a VALD line data file.
        """
        lstr = line.strip()
        if lstr[0] != "'" or lstr[-2:] != "',":
            raise FileError(f"error parsing model atmosphere: {lstr}")
        return lstr[1:-2]

    def parse_abund(self, lines):
        """Parse VALD abundance lines from a VALD line data file.
        """
        abstr = "".join(["".join(line.split()) for line in lines])
        words = [w[1:-1] for w in abstr.split(",")]
        if len(words) != 100 or words[99] != "END":
            raise FileError(f"error parsing abundances: {abstr}")
        pattern = [w.split(":") for w in words[:-1]]
        pattern = OrderedDict([(el, float(ab)) for el, ab in pattern])
        monh = 0.0
        return Abund(monh, pattern, type="sme")
