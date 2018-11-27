"""
NLTE module of SME
reads and interpolates departure coefficients from library files
"""

import itertools
import os.path

import numpy as np
from scipy import interpolate

from . import sme_synth
from .abund import Abund


class DirectAccessFile:
    """
    This function reads a single record from binary file that has the following
    structure:
    Version string              - 64 byte long
    # of directory bocks        - short int
    directory block length      - short int
    # of used directory blocks  - short int
    1st directory block:
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


def read_grid(sme, elem):
    """ Read the NLTE coefficients from the nlte_grid files for the given element

    Parameters
    ----------
    sme : SME_Struct
        sme structure with parameters and nlte grid locations
    elem : str
        current element name

    Returns
    -------
    nlte_grid : dict
        collection of nlte coefficient data (memmapped)
    linerefs : array (nlines,)
        linelevel descriptions (Energy level terms)
    lineindices: array (nlines,)
        indices of the used lines in the linelist
    """

    # array[4] = abund, teff, logg, feh
    # subgridsize > 2, doesn't really matter,
    # as we are just going to linearly interpolate between the closest pointsanyway
    subgrid_size = sme.nlte.nlte_subgrid_size
    subgrid_size[:] = 2
    solar = Abund(0, "asplund2009")
    relative_abundance = sme.abund[elem] - solar[elem]
    sme_values = relative_abundance, sme.teff, sme.logg, sme.monh

    # Get NLTE filename
    # TODO: external setting parameter for the location of the grids
    localdir = os.path.dirname(__file__)
    fname = os.path.join(localdir, "nlte_grids", sme.nlte.nlte_grids[elem])

    # Get data from file
    directory = DirectAccessFile(fname)
    teff = directory["teff"]
    grav = directory["grav"]
    feh = directory["feh"]
    xfe = directory["abund"]
    keys = directory["models"].astype("U")
    depth = directory[str.lower(sme.atmo.interp)]

    # Determine parameter limits
    # x_limits = np.min(xfe), np.max(xfe)
    # t_limits = np.min(teff), np.max(teff)
    # g_limits = np.min(grav), np.max(grav)
    # f_limits = np.min(feh), np.max(feh)

    # find the n nearest parameter values in the grid (n == subgrid_size)
    x = np.argsort(np.abs(sme_values[0] - xfe))[: subgrid_size[0]]
    t = np.argsort(np.abs(sme_values[1] - teff))[: subgrid_size[1]]
    g = np.argsort(np.abs(sme_values[2] - grav))[: subgrid_size[2]]
    f = np.argsort(np.abs(sme_values[3] - feh))[: subgrid_size[3]]

    x = x[np.argsort(xfe[x])]
    t = t[np.argsort(teff[t])]
    g = g[np.argsort(grav[g])]
    f = f[np.argsort(feh[f])]

    # Read the models with those parameters, and store depth and level
    # Create storage array
    ndepths, nlevel = directory[keys[0, 0, 0, 0]].shape
    nabund = len(x)
    nteff = len(t)
    ngrav = len(g)
    nfeh = len(f)

    bgrid = np.zeros((ndepths, nlevel, nabund, nteff, ngrav, nfeh))

    for i, j, k, l in np.ndindex(nabund, nteff, ngrav, nfeh):
        model = keys[f[l], g[k], t[j], x[i]]
        if model != "":
            bgrid[:, :, i, j, k, l] = directory[model]
        else:
            print(
                f"Missing Model for element {elem}: T={teff[t[j]]}, logg={grav[g[k]]}, feh={feh[f[l]]}, abund={xfe[x[i]]:.2f}"
            )
    mask = np.zeros(depth.shape[:-1], bool)
    for i, j, k in itertools.product(f, g, t):
        mask[i, j, k] = True
    depth = depth[mask, :]
    depth.shape = nfeh, ngrav, nteff, -1

    # Read more data from the table (conf, term, spec, J)
    conf = directory["conf"].astype("U")
    term = directory["term"].astype("U")
    species = directory["spec"].astype("U")
    rotnum = directory["J"]  # rotational number of the atomic state

    # call sme_nlte_select_levels
    bgrid, levels, linerefs, lineindices = select_levels(
        sme, elem, bgrid, conf, term, species, rotnum
    )

    # store results in nlte_grid as a dictionary
    nlte_grid = {
        # Parameter Limits
        # "Tlims": t_limits,  # min/max Teff
        # "glims": g_limits,  # min/max logg
        # "flims": f_limits,  # min/max metallicity
        # "xlims": x_limits,  # min/max elemental abundance
        # Parameters of the subgrid
        "teff": teff[t],
        "grav": grav[g],
        "feh": feh[f],
        "xfe": xfe[x],  # elemental abundance
        "depth": depth,  # corresponding depth points for each grid model
        "bgrid": bgrid,  # subgrid departure coefficients
        "levels": levels,  # level term designations, including J and configuration
    }

    return nlte_grid, linerefs, lineindices


def select_levels(sme, elem, bgrid, conf, term, species, rotnum):
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

    Parameters:
    ----------
    sme : SME_Struct
        SME input structure. Communicates the linelist.
    elem: str
        atomic element for which departure coefficients are given
    bgrid : array (nd, nl, nx, nt, ng, nf,)
        grid of departure coefficients
    conf : array (nl,)
        electronic configuration (for identification), e.g., 2p6.5s
    term : array (nl,)
        term designations (for identification), e.g., a5F
    species : array (nl,)
        Element and ion for each atomic level (for identification), e.g. Na 1.
    rotnum : array (nl,)
        rotational number J of atomic state (for identification).

    Returns:
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

    lineindices = np.char.startswith(sme.species, elem)
    if not np.any(lineindices):
        print(f"No NLTE transitions for {elem} found")
        return None, None, None, None

    sme_species = sme.species[lineindices]

    # Extract data from linelist
    parts_low = sme.linelist["term_lower"][lineindices]
    parts_upp = sme.linelist["term_upper"][lineindices]
    # Remove quotation marks (if any are there)
    # parts_low = [s.replace("'", "") for s in parts_low]
    # parts_upp = [s.replace("'", "") for s in parts_upp]
    # Get only the relevant part
    parts_low = np.array([s.split() for s in parts_low])
    parts_upp = np.array([s.split() for s in parts_upp])

    # Transform into term symbol J (2*S+1) ?
    extra = sme.linelist.extra[lineindices]
    extra = extra[:, [0, 2]] * 2 + 1
    extra = np.rint(extra).astype("i8")

    # Transform into term symbol J (2*S+1) ?
    rotnum = np.rint(2 * rotnum + 1).astype(int)

    # Create record arrays for each set of labels
    dtype = [
        ("species", sme.species.dtype),
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
    linelevels = np.full((nlines, 2), -1)
    iused = np.zeros(len(species), dtype=bool)

    # Loop through the NLTE levels
    # and match line levels
    for i, level in enumerate(level_labels):
        idx_l = line_label_low == level
        linelevels[idx_l, 0] = i
        iused[i] = iused[i] or np.any(idx_l)

        idx_u = line_label_upp == level
        linelevels[idx_u, 1] = i
        iused[i] = iused[i] or np.any(idx_u)

    # Reduce the stored data to only relevant energy levels
    # Remap the previous indices into a collapsed sequence
    level_labels = level_labels[iused]
    bgrid = bgrid[iused, ...]
    lineindices = np.where(lineindices)[0]

    # Remap the linelevel references
    for j, i in enumerate(np.where(iused)[0]):
        linelevels[linelevels == i] = j

    return bgrid, level_labels, linelevels, lineindices


def interpolate_grid(sme, elem, nlte_grid):
    """
    interpolate nlte coefficients on the model grid

    Parameters
    ----------
    sme : SME_Struct
        sme structure with parameters and atmosphere
    elem : str
        Name of the NLTE element element
    nlte_grid : dict
        NLTE parameter grid (usually from read_grid)

    Returns
    -------
    subgrid : array (ndepth, nlines)
        interpolated grid values
    """
    # Get parameters from sme structure
    teff = sme.teff
    grav = sme.logg
    feh = sme.monh
    solar = Abund(0, "asplund2009")
    abund = sme.abund[elem] - solar[elem]

    # find target depth
    target_depth = sme.atmo[sme.atmo.interp]
    target_depth = np.log10(target_depth)

    bgrid = nlte_grid["bgrid"]
    depth = nlte_grid["depth"]

    # Interpolation in 2 steps:
    # first interpolate the depth scale to the target depth, this is unstructured data
    # i.e. each combination of parameters has a different depth scale (given in depth)
    # Second interpolate on the grid of parameters, to get the value we want

    nparam = bgrid.shape[2:]  # dimensions of the parameters
    ndepth = len(target_depth)  # number of target depth layers
    nlines = bgrid.shape[0]  # number of nlte line transitions

    # have grid parameters first, and values second
    grid = np.empty((*nparam, ndepth, nlines), float)
    for l, x, t, g, f in np.ndindex(nlines, *nparam):
        xp = depth[f, g, t, :]
        yp = bgrid[l, :, x, t, g, f]

        xp = np.log10(xp)
        cubic = interpolate.interp1d(
            xp, yp, bounds_error=False, fill_value=(yp[0], yp[-1]), kind="cubic"
        )(target_depth)
        grid[x, t, g, f, :, l] = cubic

    # Interpolate on the grid
    points = (nlte_grid["xfe"], nlte_grid["teff"], nlte_grid["grav"], nlte_grid["feh"])
    target = (abund, teff, grav, feh)
    subgrid = interpolate.interpn(
        points, grid, target, bounds_error=False, fill_value=None
    )

    return subgrid[0]


def nlte(sme, elem):
    if sme.nlte.nlte_grids[elem] is None:
        raise ValueError(f"Element {elem} has not been prepared for NLTE")

    nlte_grid, linerefs, lineindices = read_grid(sme, elem)
    if nlte_grid["bgrid"] is None:
        return None, None, None

    subgrid = interpolate_grid(sme, elem, nlte_grid)
    return subgrid, linerefs, lineindices


# TODO should this be in sme_synth instead ?
def update_depcoeffs(sme):
    """ pass departure coefficients to C library """
    # # Common block keeps track of the currently stored subgrid,
    # #  i.e. that which surrounds a previously used grid points,
    # #  as well as the current matrix of departure coefficients.

    ## Reset departure coefficients from any previous call, to ensure LTE as default:

    if not "nlte" in sme:
        return sme  # no NLTE is requested
    if (
        "nlte_pro" not in sme.nlte
        or "nlte_elem_flags" not in sme.nlte
        or "nlte_grids" not in sme.nlte
        or np.all(sme.nlte.nlte_grids == "")
    ):
        # Silent fail to do LTE only.
        if not hasattr(update_depcoeffs, "first"):
            setattr(update_depcoeffs, "first", False)
            print("Running in LTE")
        return sme  # no NLTE routine available
    if sme.linelist.lineformat == "short":
        if not hasattr(update_depcoeffs, "first"):
            setattr(update_depcoeffs, "first", False)
            print("---")
            print("NLTE line formation was requested, but VALD3 long-format linedata ")
            print(
                "are required in order to relate line terms to NLTE level corrections!"
            )
            print("Line formation will proceed under LTE.")
        return sme  # no NLTE line data available

    # Reset the departure coefficient every time, just to be sure
    # It would be more efficient to just Update the values, but this doesn't take long
    sme_synth.ResetNLTE()

    # Only print "Running in NLTE" message on the first run each time
    if not hasattr(update_depcoeffs, "first"):
        setattr(update_depcoeffs, "first", False)
        print("Running in NLTE")

    # TODO store results for later reuse

    elements = sme.nlte.nlte_elem_flags
    elements = [elem for elem, i in zip(Abund._elem, elements) if i == 1]

    # Call each element to update and return its set of departure coefficients
    for elem in elements:
        # Call function to retrieve interpolated NLTE departure coefficients
        bmat, linerefs, lineindices = nlte(sme, elem)

        if bmat is None or len(linerefs) < 2:
            # no data were returned. Don't bother?
            pass
        else:
            # Put corrections into the nlte_b matrix, don't cache the data
            for lr, li in zip(linerefs, lineindices):
                # loop through the list of relevant _lines_, substitute both their levels into the main b matrix
                # Make sure both levels have corrections available
                if lr[0] != -1 and lr[1] != -1:
                    sme_synth.InputNLTE(bmat[:, lr].T, li)

    # flags = sme_synth.GetNLTEflags(sme.linelist)

    return sme


if __name__ == "__main__":
    in_file = "sme.npy"

    localdir = os.path.dirname(__file__)
    fname = os.path.join(localdir, "nlte_grids/marcs2012_Na.grd")

    sme = np.load(in_file)
    sme = np.atleast_1d(sme)[0]

    grid = nlte(sme, "Na")
