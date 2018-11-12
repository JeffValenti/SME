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


def idl_call_external(funcname, *args, restype="str", inttype="int"):
    # Load library if that wasn't done yet
    if not hasattr(idl_call_external, "lib"):
        localdir = os.path.dirname(__file__)
        # libfile = "./dll/idl_test.so"
        libfile = get_lib_name()
        libfile = os.path.join(localdir, "dll", libfile)
        idl_call_external.lib = ct.CDLL(libfile)

    # Parse arguments into c values
    # keep Python elements in staying alive, so they are not discarded by the garbage collection

    # funcname = "IDLtest"
    # from awlib.sme import sme as SME
    # file = "/home/ansgar/Documents/IDL/SME/wasp21_20d.out"
    # sme = SME.read(file)
    # linelist = sme.species
    # args = [len(linelist), linelist]

    args = list(args)
    staying_alive = [a for a in args]
    if isinstance(inttype, str):
        inttype = [inttype for i in range(len(args))]

    for i in range(len(args)):
        if isinstance(args[i], int):
            if inttype[i] == "int":
                staying_alive[i] = np.array(args[i]).astype(ct.c_int, copy=False).ctypes
            elif inttype[i] == "short":
                staying_alive[i] = (
                    np.array(args[i]).astype(ct.c_short, copy=False).ctypes
                )
            else:
                raise ValueError("Integer type must be 'int' or 'short'")
            args[i] = staying_alive[i].data
        elif isinstance(args[i], np.integer):
            # if integers are passed as numpy integers, assume the type was chosen for a reason
            if np.issubdtype(args[i], np.int64):
                staying_alive[i] = (
                    np.array(args[i]).astype(ct.c_long, copy=False).ctypes
                )
            if np.issubdtype(args[i], np.int32):
                staying_alive[i] = np.array(args[i]).astype(ct.c_int, copy=False).ctypes
            elif np.issubdtype(args[i], np.int16):
                staying_alive[i] = (
                    np.array(args[i]).astype(ct.c_short, copy=False).ctypes
                )
            else:
                staying_alive[i] = np.array(args[i]).ctypes
            args[i] = staying_alive[i].data
        if isinstance(args[i], float) or isinstance(args[i], np.floating):
            staying_alive[i] = np.array(args[i]).astype(ct.c_double, copy=False).ctypes
            args[i] = staying_alive[i].data
        elif isinstance(args[i], str):
            staying_alive[i] = IDL_String(
                slen=len(args[i]), stype=1, s=args[i].encode()
            )
            args[i] = ct.addressof(staying_alive[i])
        elif isinstance(args[i], np.ndarray):
            if np.issubdtype(args[i].dtype, np.integer):
                if inttype[i] == "int":
                    args[i] = np.require(
                        args[i], dtype=ct.c_int, requirements=["C", "A", "W", "O"]
                    )
                elif inttype[i] == "short":
                    args[i] = np.require(
                        args[i], dtype=ct.c_short, requirements=["C", "A", "W", "O"]
                    )
                staying_alive[i] = args[i].ctypes
                args[i] = staying_alive[i].data
            elif np.issubdtype(args[i].dtype, np.floating):
                args[i] = np.require(
                    args[i], dtype=ct.c_double, requirements=["C", "A", "W", "O"]
                )
                staying_alive[i] = args[i].ctypes
                args[i] = staying_alive[i].data
            elif np.issubdtype(args[i].dtype, np.dtype(str).type):
                args[i] = args[i].astype("S")
                staying_alive.append(args[i])
                length = len(args[i][0])
                args[i] = [IDL_String(slen=length, stype=1, s=s) for s in args[i]]
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
    elif restype in ["int", int]:
        func.restype = ct.c_int
    else:
        func.restype = restype

    a = np.array(args, dtype=ct.c_void_p)
    a = np.ascontiguousarray(a)

    argc = len(args)
    argv = a.ctypes.data_as(ct.POINTER(ct.c_void_p))

    res = func(argc, argv)

    return res


if __name__ == "__main__":
    print(idl_call_external("IDLtest"))
