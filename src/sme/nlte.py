import os.path
import numpy as np

idl_typecode = {0: "V", 1: "b", 2: "i2", 3: "i4", 4: "f4", 5: "f8"}


def read_direct_access_file(fname):
    # This function reads a single record from binary file that has the following
    # structure:
    # Version string              - 64 byte long
    # # of directory bocks        - short int
    # directory block length      - short int
    # # of used directory blocks  - short int
    # 1st directory block:
    #                      key    - string of up to 256 characters padded with ' '
    #                    datatype - 32-bit int 23-element array returned by SIZE
    #                    pointer  - 64-bit int pointer to the beginning of the
    #                               record
    # 2nd directory block
    # ...
    # last directory block
    # 1st data record
    # ...
    # last data record
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

        key = [directory["f0"][i].decode().strip() for i in range(ndir)]
        # ndim, n1, n2, ..., typecode, size
        dtype = [idl_typecode[d[1 + d[0]]] for d in directory["f1"]]
        shape = [tuple(d[1 : d[0] + 1]) for d in directory["f1"]]
        pointer = directory["f2"]

    # lazy load arrays, with pointers p
    tmp = {
        k: np.memmap(fname, mode="r", dtype=d, offset=p, shape=s)
        for k, d, p, s in zip(key, dtype, pointer, shape)
    }

    return tmp


def read_grid(fname):
    directory = read_direct_access_file(fname)
    teff = directory["teff"]
    bounds_teff = np.min(teff), np.max(teff)

if __name__ == "__main__":
    localdir = os.path.dirname(__file__)
    fname = os.path.join(localdir, "nlte_grids/marcs2012_Na.grd")
    read_grid(fname)
