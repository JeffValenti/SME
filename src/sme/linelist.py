"""
Handles abstract LineList data
Implementation of a specific source (e.g. Vald) should be in its own file

Uses a pandas dataframe under the hood to handle the data
"""
import numpy as np
import pandas as pd


class FileError(Exception):
    """Raise when attempt to read a VALD line data file fails"""


class LineList:
    """Atomic data for a list of spectral lines
    """

    @staticmethod
    def parse_line_error(error_flags, values):
        """ Transform Line Error flags into relative error values """
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
        for i, (flag, _) in enumerate(zip(error_flags, values)):
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

    @staticmethod
    def from_IDL_SME(**kwargs):
        """ extract LineList from IDL SME structure keywords """
        species = kwargs.pop("species").astype("U")
        atomic = np.asarray(kwargs.pop("atomic"), dtype="<f8")
        lande = np.asarray(kwargs.pop("lande"), dtype="<f8")
        depth = np.asarray(kwargs.pop("depth"), dtype="<f8")
        lineref = kwargs.pop("reference").astype("U")
        short_line_format = kwargs.pop("short_line_format")
        if short_line_format == 2:
            line_extra = np.asarray(kwargs.pop("line_extra"), dtype="<f8")
            line_lulande = np.asarray(kwargs.pop("line_lulande"), dtype="<f8")
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
            error = LineList.parse_line_error(error, depth)
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

    def __getitem__(self, index):
        if isinstance(index, (list, str)):
            return self._lines[index].values
        else:
            if isinstance(index, int):
                index = slice(index, index + 1)
            # just pass on a subsection of the linelist data, but keep it a linelist object
            return LineList(self._lines.iloc[index], self.lineformat)

    def __getattribute__(self, name):
        if name[0] != "_" and name not in dir(self):
            return self._lines[name].values
        return super().__getattribute__(name)

    @property
    def species(self):
        """ Species name of each line """
        return self._lines["species"].values

    @property
    def lulande(self):
        """ Lower and Upper Lande factors """
        if self.lineformat == "short":
            raise AttributeError(
                "Lower and Upper Lande Factors are only available in the long line format"
            )

            # additional data arrays for sme
        names = ["lande_lower", "lande_upper"]
        return self._lines[names].values

    @property
    def extra(self):
        """ additional line level information for NLTE calculation """
        if self.lineformat == "short":
            raise AttributeError("Extra is only available in the long line format")
        names = ["j_lo", "e_upp", "j_up"]
        return self._lines[names].values

    @property
    def atomic(self):
        """ Data array passed to C library, should only be used for this purpose """
        names = [
            "atom_number",
            "ionization",
            "wlcent",
            "excit",
            "gflog",
            "gamrad",
            "gamqst",
            "gamvw",
        ]
        # Select fields
        return self._lines[names].values

    def sort(self, field="wlcent", ascending=True):
        """ sort the linelist 'in place' """
        self._lines = self._lines.sort_values(by=field, ascending=ascending)
        return self._lines
