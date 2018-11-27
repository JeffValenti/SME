import os.path
import numpy as np
import ctypes as ct

import sys
import platform


class IDL_String(ct.Structure):
    _fields_ = [("slen", ct.c_int), ("stype", ct.c_ushort), ("s", ct.c_char_p)]


def get_lib_name():
    """ Get the name of the sme C library """
    system = sys.platform
    arch = platform.machine()
    bits = platform.architecture()[0][:-3]

    return f"sme_synth.so.{system}.{arch}.{bits}"


def get_typenames(arg):
    if isinstance(arg, (str, np.str)) or (
        isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.str)
    ):
        return "unicode"
    if isinstance(arg, (float, np.floating)) or (
        isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.floating)
    ):
        return "double"
    if isinstance(arg, (int, np.integer)) or (
        isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.integer)
    ):
        return "int"

    raise ValueError(f"argument datatype not understood")


def get_dtype(type):
    if type in ["i", "int", int]:
        return ct.c_int
    elif type in ["s", "short"]:
        return ct.c_short
    elif type in ["l", "long"]:
        return ct.c_long
    elif type in ["f", "float"]:
        return ct.c_float
    elif type in ["d", "double", float]:
        return ct.c_double
    elif type in ["u", "unicode", "str", str]:
        return IDL_String
    else:
        raise ValueError(f"Data type {type} not understood")


def idl_call_external(funcname, *args, restype="str", type=None):
    """Call an external C library (here the SME lib) function that uses the IDL type interface
    i.e. restype func(int n, void *args[]), where n is the number of arguments, and args is a list of pointers to the arguments

    Input arrays will be converted to the required datatype for the C function (see type keyword),
    and any changes to input arrays will be written back if possible.
    Input arrays that are already in the correct datatype will not be copied (and the values can therefore change in the C call)

    Note that all strings are converted into IDL_String objects, even those that are in arrays

    Parameters
    ----------
    funcname : str
        Name of the function to call in the library
    restype : str, optional
        expected type of the return value (default: "str")
    type : str, list(str), optional
        type of the input parameters, will default to int/double for all integer/floating point values.
        Accepted values are ('short', 'int', 'long', 'float', 'double', 'unicode') or their respective first letters.
        This means one can use a string as shorthand, e.g. "iidds"

    Returns
    -------
    value : restype
        return value of the function call
    """

    # Load library if that wasn't done yet
    if not hasattr(idl_call_external, "lib"):
        localdir = os.path.dirname(__file__)
        # libfile = "./dll/idl_test.so"
        libfile = get_lib_name()
        libfile = os.path.join(localdir, "dll", libfile)
        idl_call_external.lib = ct.CDLL(libfile)

    # prepare input arguments
    args = list(args)
    staying_alive = [a for a in args]
    original = [a for a in args]

    # datatype is determined by passed type keyword
    # defaults to 'int' for all integer type values and 'double' for all floating point values
    if type is None:
        type = [get_typenames(a) for a in args]
    elif type in ["short", "int", "long", "float", "double"]:
        type = [type for i in range(len(args))]

    # Parse arguments into c values
    # keep Python elements in staying alive, so they are not discarded by the garbage collection
    for i in range(len(args)):
        # Single values
        if isinstance(args[i], (int, float, np.number)):
            dtype = get_dtype(type[i])
            staying_alive[i] = np.array(args[i]).astype(dtype, copy=False).ctypes
            args[i] = staying_alive[i].data
        elif isinstance(args[i], str):
            staying_alive[i] = IDL_String(
                slen=len(args[i]), stype=1, s=args[i].encode()
            )
            args[i] = ct.addressof(staying_alive[i])
        # Arrays
        elif isinstance(args[i], np.ndarray):
            if np.issubdtype(args[i].dtype, np.number) or np.issubdtype(
                args[i].dtype, np.bool_
            ):
                dtype = get_dtype(type[i])
                args[i] = np.require(
                    args[i], dtype=dtype, requirements=["C", "A", "W", "O"]
                )
                staying_alive[i] = args[i].ctypes
                args[i] = staying_alive[i].data
            elif np.issubdtype(args[i].dtype, np.str):
                args[i] = args[i].astype("S")
                staying_alive.append(args[i])
                length = [len(a) for a in args[i]]
                args[i] = [
                    IDL_String(slen=l, stype=1, s=s) for s, l in zip(args[i], length)
                ]
                staying_alive.append(args[i])

                strarr = (IDL_String * len(args[i]))()
                for j in range(len(args[i])):
                    strarr[j] = args[i][j]

                staying_alive[i] = strarr
                args[i] = ct.addressof(strarr)

    # Load function and define parameters
    func = getattr(idl_call_external.lib, funcname)
    func.argtypes = (ct.c_int, ct.POINTER(ct.c_void_p))
    if restype in ["str", str]:
        func.restype = ct.c_char_p
    else:
        func.restype = get_dtype(restype)

    # Convert input parameters to list of void pointers
    a = np.array(args, dtype=ct.c_void_p)
    a = np.ascontiguousarray(a)

    argc = len(args)
    argv = a.ctypes.data_as(ct.POINTER(ct.c_void_p))

    # C function call
    res = func(argc, argv)

    # Try to copy back data to the original array memory (if necessary)
    for i in range(len(original)):
        if isinstance(original[i], np.ndarray):
            if np.issubdtype(original[i].dtype, np.number) or np.issubdtype(
                original[i].dtype, np.bool_
            ):
                # do nothing if its the same array
                if original[i] is staying_alive[i]._arr:
                    continue
                arr = staying_alive[i]._arr
            elif np.issubdtype(original[i].dtype, np.str):
                # For string arrays recover the strings from the IDL_String structure
                arr = [s.s.decode() for s in staying_alive[i]]
            else:
                # Shouldn't happen
                continue

            # If nothing was changed then all is good
            if not np.all(original[i] == arr):
                try:
                    original[i][:] = arr
                except ValueError as ve:
                    print(
                        f"WARNING: Array values changed, but could not be written back to the original array\n{str(ve)}"
                    )

    return res


if __name__ == "__main__":
    print(idl_call_external("IDLtest"))
