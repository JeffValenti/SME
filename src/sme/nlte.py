"""
NLTE module of SME
reads and interpolates departure coefficients from library files
"""

import itertools
import logging
import os.path
import warnings

import numpy as np
from scipy import interpolate

from .abund import Abund


class DirectAccessFile:
    """
    This function reads a single record from binary file that has the following
    structure:
    Version string              - 64 byte long
    # of directory bocks        - short int
    directory block length      - short int
    # of used directory blocks  - short int
    1st directory block
    key      - string of up to 256 characters padded with ' '
    datatype - 32-bit int 23-element array returned by SIZE
    pointer  - 64-bit int pointer to the beginning of the record

    2nd directory block
    ...
    last directory block
    1st data record
    ...
    last data record
    """

    def __init__(self, filename):
        key, pointer, dtype, shape, version = DirectAccessFile.read_header(filename)
        self.file = filename
        self.version = version
        self.key = key
        self.shape = shape
        self.pointer = pointer
        self.dtype = dtype

    def __getitem__(self, key):
        # Access data via brackets
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found")
        return value

    def get(self, key, alt=None):
        """ get field from file """
        idx = np.where(self.key == key)[0]

        if idx.size == 0:
            return alt
        else:
            idx = idx[0]

        return np.memmap(
            self.file,
            mode="r",
            offset=self.pointer[idx],
            dtype=self.dtype[idx],
            shape=self.shape[idx][::-1],
        )

    @staticmethod
    def idl_typecode(i):
        """
        relevant IDL typecodes and corresponding Numpy Codes
        Most specify a byte size, but not all
        """
        typecode = {
            0: "V",
            1: "B",
            2: "i2",
            3: "i4",
            4: "f4",
            5: "f8",
            6: "c4",
            7: "S",
            8: "O",
            9: "c8",
            10: "i8",
            11: "O",
            12: "u2",
            13: "u4",
            14: "i8",
            15: "u8",
        }
        return typecode[i]

    @staticmethod
    def read_header(fname):
        """ parse Header data """
        with open(fname, "rb") as file:
            header_dtype = np.dtype(
                [
                    ("version", "S64"),
                    ("nblocks", "<i2"),
                    ("dir_length", "<i2"),
                    ("ndir", "<i2"),
                ]
            )
            dir_dtype = np.dtype(
                [("key", "S256"), ("size", "<i4", 23), ("pointer", "<i8")]
            )

            header = np.fromfile(file, header_dtype, count=1)
            version = header["version"][0].decode().strip()
            ndir = header["ndir"][0]

            directory = np.fromfile(file, dir_dtype, count=ndir)

        # Decode bytes to strings
        key = directory["key"]
        key = np.char.strip(np.char.decode(key))
        # Get relevant info from size parameter
        # ndim, n1, n2, ..., typecode, size
        dtype = np.array(
            [DirectAccessFile.idl_typecode(d[1 + d[0]]) for d in directory["size"]],
            dtype="U5",
        )
        shape = np.array([tuple(d[1 : d[0] + 1]) for d in directory["size"]])
        # Pointer to data arrays
        pointer = directory["pointer"]

        # Bytes (which represent strings) get special treatment to get the dimensions right
        # And to properly convert them to strings
        # Also Null terminator is important to remember
        idx = dtype == "B"
        dtype[idx] = [f"S{s[0]}" for s in shape[idx]]
        shape[idx] = [s[1:] for s in shape[idx]]

        return key, pointer, dtype, shape, version


class Grid:
    """NLTE Grid class that handles all NLTE data reading and interpolation    
    """

    def __init__(self, sme, elem):
        #:str: Element of the NLTE grid
        self.elem = elem
        #:LineList: Whole LineList that was passed to the C library
        self.linelist = sme.linelist
        #:array(str): Elemental Species Names for the linelist
        self.species = sme.species

        localdir = os.path.dirname(__file__)
        #:str: complete filename of the NLTE grid data file
        self.fname = os.path.join(localdir, "nlte_grids", sme.nlte.grids[elem])
        depth_name = str.lower(sme.atmo.interp)
        #:array(float): depth points of the atmosphere that was passed to the C library (in log10 scale)
        self.target_depth = sme.atmo[depth_name]
        self.target_depth = np.log10(self.target_depth)

        #:DirectAccessFile: The NLTE data file
        self.directory = DirectAccessFile(self.fname)
        self._teff = self.directory["teff"]
        self._grav = self.directory["grav"]
        self._feh = self.directory["feh"]
        self._xfe = self.directory["abund"]
        self._keys = self.directory["models"].astype("U")
        self._depth = self.directory[depth_name]

        self._grid = None
        self._points = None

        #:list(int): number of points in the grid to cache for each parameter, order: abund, teff, logg, monh
        self.subgrid_size = sme.nlte.subgrid_size
        #:float: Solar Abundance of the element
        self.solar = Abund.solar()[self.elem]

        #:dict: upper and lower parameters covered by the grid
        self.limits = {}
        #:array: NLTE data array
        self.bgrid = None
        #:array: Depth points of the NLTE data
        self.depth = None

        #:array: Indices of the lines in the NLTE data
        self.linerefs = None
        #:array: Indices of the lines in the LineList
        self.lineindices = None

    def get(self, abund, teff, logg, monh):
        rabund = abund - self.solar

        if len(self.limits) == 0 or not (
            (self.limits["xfe"][0] <= rabund <= self.limits["xfe"][-1])
            and (self.limits["teff"][0] <= teff <= self.limits["teff"][-1])
            and (self.limits["grav"][0] <= logg <= self.limits["grav"][-1])
            and (self.limits["feh"][0] <= monh <= self.limits["feh"][-1])
        ):
            _ = self.read_grid(rabund, teff, logg, monh)

        return self.interpolate(rabund, teff, logg, monh)

    def read_grid(self, rabund, teff, logg, monh):
        """ Read the NLTE coefficients from the nlte_grid files for the given element
        The class will cache subgrid_size points around the target values as well

        Parameters
        ----------
        rabund : float
            relative (to solar) abundance of the element
        teff : float
            temperature in Kelvin
        logg : float
            surface gravity in log(cgs)
        monh : float
            Metallicity in H=12

        Returns
        -------
        nlte_grid : dict
            collection of nlte coefficient data (memmapped)
        linerefs : array (nlines,)
            linelevel descriptions (Energy level terms)
        lineindices: array (nlines,)
            indices of the used lines in the linelist
        """
        # find the n nearest parameter values in the grid (n == subgrid_size)
        x = np.argsort(np.abs(rabund - self._xfe))[: self.subgrid_size[0]]
        t = np.argsort(np.abs(teff - self._teff))[: self.subgrid_size[1]]
        g = np.argsort(np.abs(logg - self._grav))[: self.subgrid_size[2]]
        f = np.argsort(np.abs(monh - self._feh))[: self.subgrid_size[3]]

        x = x[np.argsort(self._xfe[x])]
        t = t[np.argsort(self._teff[t])]
        g = g[np.argsort(self._grav[g])]
        f = f[np.argsort(self._feh[f])]

        # Read the models with those parameters, and store depth and level
        # Create storage array
        ndepths, nlevel = self.directory[self._keys[0, 0, 0, 0]].shape
        nabund = len(x)
        nteff = len(t)
        ngrav = len(g)
        nfeh = len(f)

        self.bgrid = np.zeros((ndepths, nlevel, nabund, nteff, ngrav, nfeh))

        for i, j, k, l in np.ndindex(nabund, nteff, ngrav, nfeh):
            model = self._keys[f[l], g[k], t[j], x[i]]
            if model != "":
                self.bgrid[:, :, i, j, k, l] = self.directory[model]
            else:
                warnings.warn(
                    f"Missing Model for element {self.elem}: T={self._teff[t[j]]}, logg={self._grav[g[k]]}, feh={self._feh[f[l]]}, abund={self._xfe[x[i]]:.2f}"
                )
        mask = np.zeros(self._depth.shape[:-1], bool)
        for i, j, k in itertools.product(f, g, t):
            mask[i, j, k] = True
        self.depth = self._depth[mask, :]
        self.depth.shape = nfeh, ngrav, nteff, -1

        # Read more data from the table (conf, term, spec, J)
        conf = self.directory["conf"].astype("U")
        term = self.directory["term"].astype("U")
        species = self.directory["spec"].astype("U")
        rotnum = self.directory["J"]  # rotational number of the atomic state

        # call sme_nlte_select_levels
        self.bgrid, self.linerefs, self.lineindices = self.select_levels(
            conf, term, species, rotnum
        )

        self._points = (self._xfe[x], self._teff[t], self._grav[g], self._feh[f])
        self.limits = {
            "teff": self._points[1][[0, -1]],
            "grav": self._points[2][[0, -1]],
            "feh": self._points[3][[0, -1]],
            "xfe": self._points[0][[0, -1]],
        }

        # Interpolate the depth scale to the target depth, this is unstructured data
        # i.e. each combination of parameters has a different depth scale (given in depth)
        ndepths, _, *nparam = self.bgrid.shape
        ntarget = len(self.target_depth)

        self._grid = np.empty((*nparam, ntarget, ndepths), float)
        for l, x, t, g, f in np.ndindex(ndepths, *nparam):
            xp = self.depth[f, g, t, :]
            yp = self.bgrid[l, :, x, t, g, f]

            xp = np.log10(xp)
            self._grid[x, t, g, f, :, l] = interpolate.interp1d(
                xp, yp, bounds_error=False, fill_value=(yp[0], yp[-1]), kind="cubic"
            )(self.target_depth)

        return self.bgrid, self.linerefs, self.lineindices

    def select_levels(self, conf, term, species, rotnum):
        """
        Match our NLTE terms to transitions in the vald3-format linelist.

        Level descriptions in the vald3 long format look like this:
        'LS                                                           2p6.3s                   2S'
        'LS                                                             2p6.3p                2P*'
        These are stored in line3_term_low and line3_term_upp.
        The array line3_extra has dimensions [3 x nline3s]. It stores J_low, E_up, J_up
        The sme.atomic array stores:
        0) atomic number, 1) ionization state, 2) wavelength (in A),
        3) excitation energy of lower level (in eV), 4) log(gf), 5) radiative,
        6) Stark, 7) and van der Waals damping parameters

        Parameters
        ----------
        conf : array (nl,)
            electronic configuration (for identification), e.g., 2p6.5s
        term : array (nl,)
            term designations (for identification), e.g., a5F
        species : array (nl,)
            Element and ion for each atomic level (for identification), e.g. Na 1.
        rotnum : array (nl,)
            rotational number J of atomic state (for identification).

        Returns
        -------
        bgrid : array (nd, nlines, nx, nt, ng, nf,)
            grid of departure coefficients, reduced to the lines used
        level_labels : array (nl,)
            string descriptions of each atomic level, usually
            "[species]_[conf]_[term]_[2*J+1]", according to definitions above
        linelevels : array (2, nlines,)
            Cross references for the lower and upper level in each transition,
            to their indices in the list of atomic levels.
            Missing levels use indices of -1.
        lineindices : array (nl,)
            Indices of the used lines in the linelist
        """

        self.lineindices = np.asarray(self.species, "U")
        self.lineindices = np.char.startswith(self.lineindices, self.elem)
        if not np.any(self.lineindices):
            warnings.warn(f"No NLTE transitions for {self.elem} found")
            return None, None, None, None

        sme_species = self.species[self.lineindices]

        # Extract data from linelist
        parts_low = self.linelist["term_lower"][self.lineindices]
        parts_upp = self.linelist["term_upper"][self.lineindices]
        # Remove quotation marks (if any are there)
        # parts_low = [s.replace("'", "") for s in parts_low]
        # parts_upp = [s.replace("'", "") for s in parts_upp]
        # Get only the relevant part
        parts_low = np.array([s.rsplit(" ", 1) for s in parts_low])
        parts_upp = np.array([s.rsplit(" ", 1) for s in parts_upp])

        # Transform into term symbol J (2*S+1) ?
        extra = self.linelist.extra[self.lineindices]
        extra = extra[:, [0, 2]] * 2 + 1
        extra = np.rint(extra).astype("i8")

        # Transform into term symbol J (2*S+1) ?
        rotnum = np.rint(2 * rotnum + 1).astype(int)

        # Create record arrays for each set of labels
        dtype = [
            ("species", sme_species.dtype),
            ("configuration", parts_upp.dtype),
            ("term", parts_upp.dtype),
            ("J", extra.dtype),
        ]
        level_labels = np.rec.fromarrays((species, conf, term, rotnum), dtype=dtype)
        line_label_low = np.rec.fromarrays(
            (sme_species, parts_low[:, 0], parts_low[:, 1], extra[:, 0]), dtype=dtype
        )
        line_label_upp = np.rec.fromarrays(
            (sme_species, parts_upp[:, 0], parts_upp[:, 1], extra[:, 1]), dtype=dtype
        )

        # Prepare arrays
        nlines = parts_low.shape[0]
        self.linerefs = np.full((nlines, 2), -1)
        iused = np.zeros(len(species), dtype=bool)

        # Loop through the NLTE levels
        # and match line levels
        for i, level in enumerate(level_labels):
            idx_l = line_label_low == level
            self.linerefs[idx_l, 0] = i
            iused[i] = iused[i] or np.any(idx_l)

            idx_u = line_label_upp == level
            self.linerefs[idx_u, 1] = i
            iused[i] = iused[i] or np.any(idx_u)

        # Reduce the stored data to only relevant energy levels
        # Remap the previous indices into a collapsed sequence
        # level_labels = level_labels[iused]
        self.bgrid = self.bgrid[iused, ...]
        self.lineindices = np.where(self.lineindices)[0]

        # Remap the linelevel references
        for j, i in enumerate(np.where(iused)[0]):
            self.linerefs[self.linerefs == i] = j

        # bgrid, level_labels, linelevels, lineindices
        return self.bgrid, self.linerefs, self.lineindices

    def interpolate(self, rabund, teff, logg, monh):
        """
        interpolate nlte coefficients on the model grid

        Parameters
        ----------
        rabund : float
            relative (to solar) abundance of the element
        teff : float
            temperature in Kelvin
        logg : float
            surface gravity in log(cgs)
        monh : float
            Metallicity in H=12

        Returns
        -------
        subgrid : array (ndepth, nlines)
            interpolated grid values
        """

        assert self._grid is not None
        assert self._points is not None

        # Interpolate on the grid
        # self._points and self._grid are interpolated when reading the data in read_grid
        target = (rabund, teff, logg, monh)
        subgrid = interpolate.interpn(
            self._points, self._grid, target, bounds_error=False, fill_value=None
        )

        return subgrid[0]


def nlte(sme, dll, elem):
    """ Read and interpolate the NLTE grid for the current element and parameters """
    if sme.nlte.grids[elem] is None:
        raise ValueError(f"Element {elem} has not been prepared for NLTE")

    if elem in dll._nlte_grids.keys():
        grid = dll._nlte_grids[elem]
    else:
        grid = Grid(sme, elem)
        dll._nlte_grids[elem] = grid

    subgrid = grid.get(sme.abund[elem], sme.teff, sme.logg, sme.monh)

    return subgrid, grid.linerefs, grid.lineindices


# TODO should this be in sme_synth instead ?
def update_nlte_coefficients(sme, dll):
    """ pass departure coefficients to C library """

    # Only print "Running in NLTE" message on the first run each time
    self = update_nlte_coefficients
    if not hasattr(self, "first"):
        setattr(self, "first", True)

    if (
        not "nlte" in sme
        or "elements" not in sme.nlte
        or "grids" not in sme.nlte
        or np.all(sme.nlte.grids == "")
        or np.size(sme.nlte.elements) == 0
    ):
        # No NLTE to do
        if self.first:
            self.first = False
            logging.info("Running in LTE")
        return sme
    if sme.linelist.lineformat == "short":
        if self.first:
            self.first = False
            warnings.warn(
                "NLTE line formation was requested, but VALD3 long-format linedata\n"
                "are required in order to relate line terms to NLTE level corrections!\n"
                "Line formation will proceed under LTE."
            )
        return sme

    # Reset the departure coefficient every time, just to be sure
    # It would be more efficient to just Update the values, but this doesn't take long
    dll.ResetNLTE()

    elements = sme.nlte.elements

    if self.first:
        self.first = False
        logging.info("Running in NLTE: %s", ", ".join(elements))

    # Call each element to update and return its set of departure coefficients
    for elem in elements:
        # Call function to retrieve interpolated NLTE departure coefficients
        bmat, linerefs, lineindices = nlte(sme, dll, elem)

        if bmat is None or len(linerefs) < 2:
            # no data were returned. Don't bother?
            pass
        else:
            # Put corrections into the nlte_b matrix, don't cache the data
            for lr, li in zip(linerefs, lineindices):
                # loop through the list of relevant _lines_, substitute both their levels into the main b matrix
                # Make sure both levels have corrections available
                if lr[0] != -1 and lr[1] != -1:
                    dll.InputNLTE(bmat[:, lr].T, li)

    # flags = sme_synth.GetNLTEflags(sme.linelist)

    return sme


if __name__ == "__main__":
    in_file = "sme.npy"

    localdir = os.path.dirname(__file__)
    fname = os.path.join(localdir, "nlte_grids/marcs2012_Na.grd")

    sme = np.load(in_file)
    sme = np.atleast_1d(sme)[0]

    grid = nlte(sme, "Na")
