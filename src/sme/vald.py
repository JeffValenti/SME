""" VALD data handling module """
import logging
from io import StringIO

import numpy as np
import pandas as pd

from .abund import Abund
from .linelist import FileError, LineList


class ValdError(FileError):
    """ Vald Data File Error """


class ValdFile:
    """Atomic data for a list of spectral lines.
    """

    def __init__(self, filename):
        self._filename = filename
        self.read(filename)

    @property
    def filename(self):
        """ Source filename """
        return self._filename

    @property
    def n(self):
        """ number of spectral lines """
        return self._nlines

    @property
    def linelist(self):
        """ LineList data """
        return self._linelist

    @property
    def valdatmo(self):
        """ Atmopshere used by Vald """
        return self._valdatmo

    @property
    def abund(self):
        """ Elemental abundances used by Vald """
        return self._abund

    def read(self, filename):
        """Read line data file from the VALD extract stellar service.
        """
        logging.info("Loading VALD file %s", filename)

        try:
            with open(filename, "r") as file:
                lines = file.readlines()
        except Exception as ex:
            raise ValdError(str(ex))

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
            raise ValdError(f"{self._filename} is not a VALD line data file")
        self._wavelo = float(words[0])
        self._wavehi = float(words[1])
        self._nlines = int(words[2])
        self._nlines_proc = int(words[3])
        self._vmicro = float(words[4])

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
            term_lower = [t.replace("'", "").split(maxsplit=1) for t in term_lower]
            term_upper = [t.replace("'", "").split(maxsplit=1) for t in term_upper]
            term_lower = [t[-1][:-1] if len(t) != 0 else "" for t in term_lower]
            term_upper = [t[-1][:-1] if len(t) != 0 else "" for t in term_upper]

            linelist["term_lower"] = term_lower
            linelist["term_upper"] = term_upper

            # extract error data
            error = np.array([s[:10].strip() for s in comment])
            error = LineList.parse_line_error(error, linelist["depth"])
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
