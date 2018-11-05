
import os.path
import itertools
import numpy as np

from scipy.interpolate import griddata, interpn

from SME.src.sme.abund import Abund


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
        typecode = {0: "V", 1: "B", 2: "i2", 3: "i4", 4: "f4", 5: "f8"}
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
            dir_dtype = np.dtype("S256, (23)<i4, <i8")

            header = np.fromfile(file, header_dtype, count=1)
            version = header["version"][0].decode().strip()
            ndir = header["ndir"][0]

            directory = np.fromfile(file, dir_dtype, count=ndir)

        # Decode bytes to strings
        key = directory["f0"]
        key = np.char.strip(np.char.decode(key))
        # Get relevant info from size parameter
        # ndim, n1, n2, ..., typecode, size
        dtype = np.array(
            [DirectAccessFile.idl_typecode(d[1 + d[0]]) for d in directory["f1"]],
            dtype="U5",
        )
        shape = np.array([tuple(d[1 : d[0] + 1]) for d in directory["f1"]])
        # Pointer to data arrays
        pointer = directory["f2"]

        # Bytes (which represent strings) get special treatment to get the dimensions right
        # And to properly convert them to strings
        # Also Null terminator is important to remember
        idx = dtype == "B"
        dtype[idx] = [f"S{s[0]}" for s in shape[idx]]
        shape[idx] = [s[1:] for s in shape[idx]]

        return key, pointer, dtype, shape, version


def read_grid(sme, elem):
    # array[4] = abund, teff, logg, feh
    subgrid_size = sme.nlte.nlte_subgrid_size
    sme_values = sme.abund[elem], sme.teff, sme.logg, sme.feh

    # Get NLTE filename
    localdir = os.path.dirname(__file__)
    if isinstance(sme.nlte.nlte_grids, np.ndarray):
        elem_num = Abund._elem_dict[elem]
    else:
        elem_num = elem
    fname = os.path.join(localdir, "nlte_grids", sme.nlte.nlte_grids[elem_num])

    # Get data from file
    directory = DirectAccessFile(fname)
    teff = directory["teff"]
    t_limits = np.min(teff), np.max(teff)
    grav = directory["grav"]
    g_limits = np.min(grav), np.max(grav)
    feh = directory["feh"]
    f_limits = np.min(feh), np.max(feh)
    xfe = directory["abund"]
    x_limits = np.min(xfe), np.max(xfe)
    keys = directory["models"].astype("U")
    depth = directory[str.lower(sme.atmo.interp)]

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

    for i, j, k, l in itertools.product(
        range(nabund), range(nteff), range(ngrav), range(nfeh)
    ):
        model = keys[f[l], g[k], t[j], x[i]]
        if model != "":
            # print(
            #     f"T={teff[t[j]]}, logg={grav[g[k]]}, feh={feh[f[l]]}, abund={xfe[x[i]]:.2f}"
            # )
            # print(model)
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

    # store results in nlte_grid as a dictionary ?
    nlte_grid = {
        # Parameter Limits
        "Tlims": t_limits,  # min/max Teff
        "glims": g_limits,  # min/max logg
        "flims": f_limits,  # min/max metallicity
        "xlims": x_limits,  # min/max elemental abundance
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
    ;		'LS                                                           2p6.3s                   2S'
    ;		'LS                                                             2p6.3p                2P*'
    ; These are stored in line3_term_low and line3_term_upp.
    ; The array line3_extra has dimensions [3 x nline3s]. It stores
    ;	J_low, E_up, J_up
    ; The sme.atomic array stores:
    ; 	0) atomic number, 1) ionization state, 2) wavelength (in A), 
    ; 	3) excitation energy of lower level (in eV), 4) log(gf), 5) radiative, 
    ;	6) Stark, 7) and van der Waals damping parameters

    ; Input:
    ;	sme (struct): SME input structure. Communicates the linelist.
    ;	elem: index of atomic element for which departure coefficients are given, 
    ;	  i.e., atomic number minus 1.
    ; (Terms describing the provided departure coefficients - one per atomic level):
    ;	bgrid (vector[nd, nl, nx, nt, ng, nf]): grid of departure coefficients
    ;	nl: Number of atomic levels
    ; (Terms describing the atomic levels, for matching lines in the SME linelist):
    ;	conf (vector[nl]): electronic configuration (for identification), e.g., 2p6.5s
    ;	term (vector[nl]): term designations (for identification), e.g., a5F
    ;	species (vector[nl]): Element and ion for each atomic level (for identification), e.g. Na 1.
    ;	J (vector[nl]): rotational number of atomic state (for identification).
    ;
    ; Output:
    ;	level_labels (vector[nl]): string descriptions of each atomic level, usually
    ;	  "[species]_[conf]_[term]_[2*J+1]", according to definitions above
    ;	linelevels (vector[2,nlines]): Cross references for the lower and upper level
    ;	  in each transition, to their indices in the list of atomic levels.
    ;	  Missing levels use indices of -1.
    """

    # TODO there must be a better way of doing this
    # just compare the values instead of using string ?

    # TODO check line_extra format, should it really be that small?

    # ion = np.array([s[-1] for s in species], dtype=int)
    lineindices = np.char.startswith(sme.species, elem)
    if not np.any(lineindices):
        print(f"No NLTE transitions for {elem} found")
        return None, None, None

    parts_low = sme.line_term_low[lineindices]
    parts_upp = sme.line_term_upp[lineindices]

    parts_low = np.array([s.split()[1:] for s in parts_low])
    parts_upp = np.array([s.split()[1:] for s in parts_upp])

    level_labels = np.array(
        [
            f"{s}_{c}_{t}_{round(2*j+1):.0f}"
            for s, c, t, j in zip(species, conf, term, rotnum)
        ]
    )

    line_label_low = [
        f"{s}_{c}_{t}_{round(2*j+1):.0f}"
        for s, c, t, j, in zip(
            sme.species[lineindices],
            parts_low[:, 0],
            parts_low[:, 1],
            sme.line_extra[lineindices, 0],
        )
    ]

    line_label_upp = [
        f"{s}_{c}_{t}_{round(2*j+1):.0f}"
        for s, c, t, j, in zip(
            sme.species[lineindices],
            parts_upp[:, 0],
            parts_upp[:, 1],
            sme.line_extra[lineindices, 2],
        )
    ]

    iblevels = np.argsort(level_labels)
    illevels = np.argsort(line_label_low)
    iulevels = np.argsort(line_label_upp)

    nlines = parts_low.shape[0]
    linelevels = np.full((nlines, 2), -1)
    levels_used = np.zeros(len(species))

    # Loop through the NLTE levels:
    l = u = 0
    nl, nu = len(line_label_low), len(line_label_upp)
    for i, level in enumerate(level_labels):
        # Skip through the line levels up to current nlte level
        while l < nl and line_label_low[illevels[l]] < level_labels[iblevels[i]]:
            l += 1
        while u < nu and line_label_upp[iulevels[u]] < level_labels[iblevels[i]]:
            u += 1

        # Now match all corresponding nlte-line levels to current nlte level
        while l < nl and line_label_low[illevels[l]] == level_labels[iblevels[i]]:
            linelevels[illevels[l], 0] = iblevels[i]
            levels_used[iblevels[i]] += 1
            l += 1
        while u < nu and line_label_upp[iulevels[u]] == level_labels[iblevels[i]]:
            linelevels[iulevels[u], 1] = iblevels[i]
            levels_used[iblevels[i]] += 1
            u += 1

    iused = levels_used > 0
    # Remap the previous indices into a collapsed sequence:
    iremap = np.cumsum(iused) - 1  # increments by one wherever levels_used is true.
    level_labels = level_labels[iused]

    # Remap the linelevel references:
    imatched0 = linelevels[:, 0] >= 0
    imatched1 = linelevels[:, 1] >= 0
    linelevels[imatched0, 0] = iremap[linelevels[imatched0, 0]]
    linelevels[imatched1, 1] = iremap[linelevels[imatched1, 1]]

    # Reduce the stored data to only relevant energy levels
    bgrid = bgrid[iused, ...]

    return bgrid, level_labels, linelevels, np.where(lineindices)[0]


def get_gridindices(points, param, var, name=""):
    if param <= np.min(points):
        return [np.argmin(points)], 0
    if param >= np.max(points):
        return [np.argmax(points)], 0

    below = np.where(points <= param)[0][-1]
    above = np.where(points > param)[0][0]

    i = np.array([below, above])
    x = points[i]
    frac = (param - x[0]) / (x[1] - x[0])
    return i, frac


def interpolate_grid_param(grid, frac, dim):
    if len(grid[0, 0, :]) == 1:
        # Only one grid point
        return np.reshape(grid[:, :, 0, ...], dim)

    # Interpolate linearly
    tmp = grid[:, :, 0] * (1 - frac) + grid[:, :, 1] * frac
    return np.reshape(tmp, dim)


def interpolate_grid(sme, elem, nlte_grid):
    # Get parameters from sme structure
    teff = sme.teff
    grav = sme.logg
    feh = sme.feh
    abund = sme.abund[elem]

    # Retrieve grid data
    bgrid = nlte_grid["bgrid"]
    blevels = nlte_grid["levels"]
    depth = nlte_grid["depth"]

    nl = len(blevels)

    # find target depth
    target_depth = sme.atmo[sme.atmo.interp]

    # TODO use some scipy method for interpolation

    # Linear interpolation, between nearest grid points
    nbd = bgrid.shape[0]
    nd = len(target_depth)

    xi, xfrac = get_gridindices(nlte_grid["xfe"], abund, 0, name="abund")
    ti, tfrac = get_gridindices(nlte_grid["teff"], teff, 1, name="teff")
    gi, gfrac = get_gridindices(nlte_grid["grav"], grav, 2, name="grav")
    fi, ffrac = get_gridindices(nlte_grid["feh"], feh, 3, name="feh")

    nabund = len(xi)
    nteff = len(ti)
    ngrav = len(gi)
    nfeh = len(fi)

    subgrid = np.full((nd, nl, nabund, nteff, ngrav, nfeh), -1.)

    for x, t, g, f, l in itertools.product(
        range(nabund), range(nteff), range(ngrav), range(nfeh), range(nl)
    ):
        #  bgrid[:, :, i, j, k, l] = keys[f[l], g[k], t[j], x[i]]
        xp = depth[fi[f], gi[g], ti[t], :]
        yp = bgrid[l, :, xi[x], ti[t], gi[g], fi[f]]
        subgrid[:, l, x, t, g, f] = np.interp(np.log10(target_depth), np.log10(xp), yp)

    subgrid = interpolate_grid_param(subgrid, xfrac, [nd, nl, nteff, ngrav, nfeh])
    subgrid = interpolate_grid_param(subgrid, tfrac, [nd, nl, ngrav, nfeh])
    subgrid = interpolate_grid_param(subgrid, gfrac, [nd, nl, nfeh])
    subgrid = interpolate_grid_param(subgrid, ffrac, [nd, nl])

    return subgrid


def nlte(sme, elem):
    if isinstance(sme.nlte.nlte_grids, np.ndarray):
        elem_num = Abund._elem_dict[elem]
    else:
        elem_num = elem

    if sme.nlte.nlte_grids[elem_num] is None:
        raise ValueError(f"Element {elem} has not been prepared for NLTE")

    if sme.line_extra is None:
        raise ValueError(
            "VALD3 long-format linedata is required to relate line terms to NLTE level corrections"
        )

    nlte_grid, linerefs, lineindices = read_grid(sme, elem)
    subgrid = interpolate_grid(sme, elem, nlte_grid)
    return subgrid, linerefs, lineindices


if __name__ == "__main__":
    in_file = "sme.npy"

    localdir = os.path.dirname(__file__)
    fname = os.path.join(localdir, "nlte_grids/marcs2012_Na.grd")

    sme = np.load(in_file)
    sme = np.atleast_1d(sme)[0]
    # sme.nlte = lambda: None
    # sme.nlte.nlte_grids = {"Na": "marcs2012_Na.grd"}
    # sme.nlte.nlte_subgrid_size = [2, 2, 2, 2]
    # sme.teff = 5000
    # sme.logg = 2.2
    # sme.feh = 0.3
    # sme.abund = {"Na": 0.1}

    grid = nlte(sme, "Na")
