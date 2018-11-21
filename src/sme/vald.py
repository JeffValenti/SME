import numpy as np
import numpy.lib.recfunctions

from io import StringIO
import pandas as pd

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
        lineref = kwargs.pop("reference").astype("U")
        short_line_format = kwargs.pop("short_line_format")
        if short_line_format == 2:
            line_extra = kwargs.pop("line_extra")
            line_lulande = kwargs.pop("line_lulande")
            line_term_low = kwargs.pop("line_term_low").astype("U")
            line_term_upp = kwargs.pop("line_term_upp").astype("U")

        data = {
            "species": species,
            "atom_number": atomic[:, 0],
            "ionization": atomic[:, 1],
            "wlcent": atomic[:, 2],
            "excit": atomic[:, 3],
            "gflog": atomic[:, 4],
            "gamrad": atomic[:, 5],
            "gamqst": atomic[:, 6],
            "gamvw": atomic[:, 7],
            "lande": lande,
            "depth": depth,
            "reference": lineref,
        }

        if short_line_format == 1:
            lineformat = "short"
        if short_line_format == 2:
            lineformat = "long"
            error = [s[0:11].strip() for s in lineref]
            error = ValdFile.parse_line_error(error, depth)
            data["error"] = error
            data["lande_lower"] = line_lulande[:, 0]
            data["lande_upper"] = line_lulande[:, 1]
            data["j_lo"] = line_extra[:, 0]
            data["e_upp"] = line_extra[:, 1]
            data["j_up"] = line_extra[:, 2]
            data["term_lower"] = [t[10:].strip() for t in line_term_low]
            data["term_upper"] = [t[10:].strip() for t in line_term_upp]

        linedata = pd.DataFrame.from_dict(data)

        return (linedata, lineformat)

    def __init__(self, linedata, lineformat="short", **kwargs):
        if linedata is None:
            # everything is in the kwargs (usually by loading from old SME file)
            linedata, lineformat = LineList.from_IDL_SME(**kwargs)
        else:
            if "atom_number" in kwargs.keys():
                linedata["atom_number"] = kwargs["atom_number"]
            elif "atom_number" not in linedata.columns:
                linedata["atom_number"] = np.ones(len(linedata), dtype=float)

            if "ionization" in kwargs.keys():
                linedata["ionization"] = kwargs["ionization"]
            elif "ionization" not in linedata.columns:
                linedata["ionization"] = np.array(
                    [int(s[-1]) for s in linedata["species"]], dtype=float
                )

            if "term_upper" in kwargs.keys():
                linedata["term_upper"] = kwargs["term_upper"]
            if "term_lower" in kwargs.keys():
                linedata["term_lower"] = kwargs["term_lower"]
            if "reference" in kwargs.keys():
                linedata["reference"] = kwargs["reference"]
            if "error" in kwargs.keys():
                linedata["error"] = kwargs["error"]

        self.lineformat = lineformat
        self._lines = linedata  # should have all the fields (20)

    def __len__(self):
        return len(self._lines)

    def __str__(self):
        return str(self._lines)

    def __iter__(self):
        return self._lines.itertuples(index=False)

    # def __next__(self):
    # return self._lines.__next__()

    def __getitem__(self, index):
        if isinstance(index, (list, str)):
            return self._lines[index].values
        else:
            # just pass on a subsection of the linelist data, but keep it a linelist object
            return LineList(self._lines.iloc[index], self.lineformat)

    def __getattribute__(self, name):
        if name[0] != "_" and name not in dir(self):
            return self._lines[name].values
        return super().__getattribute__(name)

    @property
    def species(self):
        return self._lines["species"].values.astype("U")

    @property
    def lulande(self):
        if self.lineformat == "short":
            raise AttributeError("LuLande is only available in the long line format")

            # additional data arrays for sme
        names = ["lande_lower", "lande_upper"]
        return self._lines[names].values

    @property
    def extra(self):
        if self.lineformat == "short":
            raise AttributeError("Extra is only available in the long line format")
        names = ["j_lo", "e_upp", "j_up"]
        return self._lines[names].values

    @property
    def atomic(self):
        names = [
            "ionization",
            "atom_number",
            "wlcent",
            "excit",
            "gflog",
            "gamrad",
            "gamqst",
            "gamvw",
        ]
        # Select fields
        return self._lines[names].values

    def sort(self, field, ascending=True):
        self._lines = self._lines.sort_values(by=field, ascending=ascending)
        return self._lines


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
        fmt = "long" if lines[4][:2] == "' " else "short"

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
                "reference",
            ]

        elif fmt == "long":
            names = [
                "species",
                "wlcent",
                "gflog",
                "excit",
                "j_lo",
                "e_upp",
                "j_up",
                "lande_lower",
                "lande_upper",
                "lande",
                "gamrad",
                "gamqst",
                "gamvw",
                "depth",
            ]
            term_lower = lines[1::4]
            term_upper = lines[2::4]
            comment = lines[3::4]
            lines = lines[::4]

        data = StringIO("".join(lines))
        linelist = pd.read_table(
            data,
            sep=",",
            names=names,
            header=None,
            quotechar="'",
            skipinitialspace=True,
            usecols=range(len(names)),
        )

        if fmt == "long":
            # Convert from cm^-1 to eV
            conversion_factor = 8065.544
            linelist["excit"] /= conversion_factor
            linelist["e_upp"] /= conversion_factor

            comment = [c.replace("'", "").strip() for c in comment]
            linelist["reference"] = comment

            # Parse energy level terms
            term_lower = [t.replace("'", "").split(maxsplit=1)[-1] for t in term_lower]
            term_upper = [t.replace("'", "").split(maxsplit=1)[-1] for t in term_upper]
            linelist["term_lower"] = term_lower
            linelist["term_upper"] = term_upper

            # extract error data
            error = np.array([s[:10].strip() for s in comment])
            error = self.parse_line_error(error, linelist["depth"])
            linelist["error"] = error

        linelist = LineList(linelist, lineformat=fmt)

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
