import numpy as np

from .abund import Abund


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails.
    """


class LineList:
    """Atomic data for a list of spectral lines.
    """

    @staticmethod
    def from_IDL_SME(**kwargs):
        """ extract LineList from IDL SME structure keywords """
        species = kwargs.pop("species").astype("U")
        atomic = kwargs.pop("atomic")
        lande = kwargs.pop("lande")
        depth = kwargs.pop("depth")
        lineref = kwargs.pop("lineref").astype("U")
        short_line_format = kwargs.pop("short_line_format")
        if short_line_format == 2:
            line_extra = kwargs.pop("line_extra")
            line_lulande = kwargs.pop("line_lulande")
            line_term_low = kwargs.pop("line_term_low").astype("U")
            line_term_upp = kwargs.pop("line_term_upp").astype("U")

        arrays = [species, *[atomic[:, i] for i in range(8)], lande, depth, lineref]
        names = [
            "species",
            "atom_number",
            "ionization",
            "wlcent",
            "excit",
            "gflog",
            "gamrad",
            "gamqst",
            "gamvw",
            "lande",
            "depth",
            "ref",
        ]

        if short_line_format == 1:
            lineformat = "short"
            term_lower = term_upper = error = comment = None
        if short_line_format == 2:
            lineformat = "long"
            arrays = arrays + [
                line_lulande[:, 0],
                line_lulande[:, 1],
                line_extra[:, 0],
                line_extra[:, 1],
                line_extra[:, 2],
            ]
            names = names + ["lande_lower", "lande_upper", "j_lo", "e_upp", "j_up"]
            term_lower = line_term_low
            term_upper = line_term_upp
            comment = lineref
            error = [s[0:11].strip() for s in lineref]
            error = ValdFile.parse_line_error(error, depth)

        linedata = np.rec.fromarrays(arrays, names=names)
        ionization = linedata["ionization"]
        atom_number = linedata["atom_number"]

        return (
            linedata,
            lineformat,
            ionization,
            atom_number,
            term_lower,
            term_upper,
            comment,
            error,
        )

    @staticmethod
    def fromList(linedata):
        """ extract data from """
        names = [
            "species",
            "wlcent",
            "excit",
            "vmic",
            "gflog",
            "gamrad",
            "gamqst",
            "gamvw",
            "lande",
            "depth",
            "ref",
        ]
        linedata = np.rec.fromrecords(linedata, names=names)
        return linedata

    def __init__(self, linedata, lineformat="short", **kwargs):
        if linedata is None:
            # everything is in the kwargs (usually by loading from old SME file)
            ld, lf, ion, an, tl, tu, com, err = LineList.from_IDL_SME(**kwargs)
        else:
            if not isinstance(linedata, np.recarray):
                linedata = LineList.fromList(linedata)
            ld = linedata
            lf = lineformat
            if "atom_number" in kwargs.keys():
                an = kwargs["atom_number"]
            else:
                an = np.ones(linedata.shape, dtype=int)

            if "ionization" in kwargs.keys():
                ion = kwargs["ionization"]
            else:
                ion = np.array([int(s[-1]) for s in linedata["species"]])

            tu = kwargs.get("term_upper")
            tl = kwargs.get("term_lower")
            com = kwargs.get("comment")
            err = kwargs.get("error")

        self._lines = ld
        self.lineformat = lf
        self.ionization = np.asarray(ion)
        self.atom_number = np.asarray(an)
        self.term_lower = np.asarray(tl)
        self.term_upper = np.asarray(tu)
        self.comment = np.asarray(com)
        self.error = np.asarray(err)

    def __len__(self):
        return len(self._lines)

    def __str__(self):
        return str(self._lines)

    def __iter__(self):
        return self._lines.__iter__()

    def __next__(self):
        return self._lines.__next__()

    @property
    def species(self):
        return self._lines["species"]

    @property
    def wlcent(self):
        return self._lines["wlcent"]

    @property
    def excit(self):
        return self._lines["excit"]

    @property
    def gflog(self):
        return self._lines["gflog"]

    @property
    def gamrad(self):
        return self._lines["gamrad"]

    @property
    def gamqst(self):
        return self._lines["gamqst"]

    @property
    def gamvw(self):
        return self._lines["gamvw"]

    @property
    def lulande(self):
        if self.lineformat == "short":
            raise AttributeError("LuLande is only available in the long line format")

            # additional data arrays for sme
        tmp = self._lines[["lande_lower", "lande_upper"]]
        tmp = np.array(tmp.tolist())
        return tmp

    @property
    def extra(self):
        if self.lineformat == "short":
            raise AttributeError("Extra is only available in the long line format")
        tmp = self._lines[["j_lo", "e_upp", "j_up"]]
        tmp = np.array(tmp.tolist())
        return tmp

    @property
    def atomic(self):
        names = ["wlcent", "excit", "gflog", "gamrad", "gamqst", "gamvw"]
        # Select fields
        tmp = [self.atom_number, self.ionization]
        tmp += [self._lines[n] for n in names]
        # Convert to 2D array
        tmp = np.array(tmp)
        return tmp.T


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

    @staticmethod
    def parse_line_error(error_flags, values):
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
            conversion_factor = 8065.544
            linelist["excit"] /= conversion_factor
            linelist["e_upp"] /= conversion_factor

            # extract error data
            error = np.array([s[1:11].strip() for s in comment])
            error = self.parse_line_error(error, linelist["depth"])

            linelist = LineList(
                linelist,
                lineformat=fmt,
                term_lower=term_lower,
                term_upper=term_upper,
                reference=comment,
                error=error,
            )
        else:
            linelist = LineList(linelist)

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
            raise FileError(f"Error parsing abundances: {abstr}")
        pattern = [w.split(":") for w in words[:-1]]
        pattern = {el: float(ab) for el, ab in pattern}
        monh = 0
        return Abund(monh, pattern, type="sme")
