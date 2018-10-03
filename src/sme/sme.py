"""Simple Python interface for SME

SME (Spectrsocopy Made Easy) is a idl package for spectroscopy.
This module provides a simple interface for some useful interactions with SME
"""


import os
import os.path
import subprocess
import tempfile

import numpy as np
from scipy.io import readsav

# np.set_printoptions(threshold=np.nan)


class SME_Structure:
    def __init__(self, sme):
        self.__fields__ = {}
        if isinstance(sme, str):
            self.filename = sme
            sme = readsav(sme)["sme"]
        else:
            self.filename = None

        self.__temps__ = []
        if sme is not None:
            for field in sme.dtype.names:
                field = field.lower()
                value = sme[field][0]
                if isinstance(value, np.recarray):
                    value = SME_Structure(value)
                elif isinstance(value, bytes):
                    value = value.decode()
                elif isinstance(value, np.ndarray) and value.dtype == object:
                    value = value.astype(str)

                if isinstance(value, np.ndarray):
                    value = np.require(value, requirements=["W", "O"])

                self.__fields__[field] = value

    def __repr__(self):
        fields = [
            "{key!s}:{value!r}".format(key=key, value=self.__fix__(value))
            for key, value in self.__dict__.items()
        ]
        fields = ", ".join(fields)
        fields = "{%s}" % fields
        return fields

    def __getitem__(self, index):
        return self.__fields__[index.lower()]

    def __setitem__(self, index, value):
        setattr(self, index.lower(), value)
        # self.__fields__[index.lower()] = value

    def __getattr__(self, index):
        if index[0] != "_":
            return self.__fields__[index]
        else:
            return super().__getattribute__(index)

    def __setattr__(self, name, value):
        if name[0] != "_":
            self.__fields__[name] = value
        super().__setattr__(name, value)

    def __contains__(self, item):
        return item.lower() in self.names

    def __fix__(self, value):
        if isinstance(value, np.ndarray):
            value = repr(value)
            i = value.rfind("dtype") - 2
            value = value[6:i].replace("\n", "").replace(" ", "")
        return value

    def __get_typecode__(self, dtype):
        """ Get the IDL typecode for a given dtype """
        if dtype.name[:5] == "bytes":
            return "1"
        if dtype.name == "int16":
            return "2"
        if dtype.name == "int32":
            return "3"
        if dtype.name == "float32":
            return "4"
        if dtype.name == "float64":
            return "5"
        if dtype.name[:3] == "str":
            return dtype.name[3:]

    def __write__(self, file):
        """ 
        Write SME structure into and idl format 
        data arrays are stored in seperate temp files, and only the filename is passed to idl
        """
        iterator = list(self.__dict__.items())
        sep = ""

        for key, value in iterator:
            if key in ["__temps__", "filename"]:
                continue
            if isinstance(value, SME_Structure):
                file.write(",{key!s}:{{".format(key=key))
                value.__write__(file)
                self.__temps__ += value.__temps__
                file.write("}$\n")
            else:
                if isinstance(value, np.ndarray):
                    if value.dtype.name[:3] == "str":
                        value = value.astype(bytes)
                        shape = (value.dtype.itemsize, len(value))
                    else:
                        shape = value.shape[::-1]
                    with tempfile.NamedTemporaryFile(
                        "w+", suffix=".dat", delete=False
                    ) as temp:
                        value.tofile(temp)
                        value = [
                            temp.name,
                            str(list(shape)),
                            self.__get_typecode__(value.dtype),
                        ]
                        self.__temps__.append(temp.name)

                field = "{sep}{key!s}:{value!r} $\n".format(
                    key=key, value=value, sep=sep
                )
                sep = ","
                file.write(field)

    def __clean_temp__(self):
        """ Clean temporary files, created for saving """
        for temp in self.__temps__:
            if os.path.exists(temp):
                os.remove(temp)

        self.__temps__ = []

    def get(self, key, alt=None):
        if key in self.names:
            return self[key]
        else:
            self[key] = alt
            return self[key]

    @property
    def names(self):
        """ Names of the existing SME fields """
        return self.__fields__.keys()

    @property
    def dtype(self):
        # this emulates a numpy.recarray, as read from a IDL save file
        dtype = lambda: None
        dtype.names = [s.upper() for s in self.names]
        return dtype

    def save_py(self, file="sme.npy"):
        np.save(file, self)

    @staticmethod
    def load_py(file="sme.npy"):
        s = np.load(file)
        s = np.atleast_1d(s)[0]
        return s

    def save(self, fname=None):
        """ Save the SME structure to disk """
        if fname is None and self.filename is not None:
            fname = self.filename
        elif fname is None:
            raise AttributeError(
                "fname needs to be set or the structure needs to be assigned a filename"
            )

        with tempfile.NamedTemporaryFile("w+", suffix=".pro") as temp:
            tempname = temp.name
            temp.write("sme = {")
            self.__write__(temp)
            temp.write("} \n")
            temp.write(
                """tags = tag_names(sme)
new_sme = {}

for i =0, n_elements(tags)-1 do begin
    arr = sme.(i)
    s = size(arr)
    if (s[0] eq 1) and (s[1] eq 3) then begin
        void = execute('shape = ' + arr[1])
        type = fix(arr[2])
        arr = read_binary(arr[0], data_dims=shape, data_type=type, endian='big')
        if type eq 1 then begin
            ;string
            arr = string(arr)
        endif
    endif
    if (s[0] eq 1) and (s[1] ne 3) then begin
        ;struct
        tags2 = tag_names(sme.(i))
        new2 = {}
        tmp = sme.(i)

        for j = 0, n_elements(tags2)-1 do begin
            arr2 = tmp.(j)
            s = size(arr2)
            if (s[0] eq 1) and (s[1] eq 3) then begin
                void = execute('shape = ' + arr2[1])
                type = fix(arr2[2])
                arr2 = read_binary(arr2[0], data_dims=shape, data_type=type, endian='big')
                if type eq 1 then begin
                    ;string
                    arr2 = string(arr2)
                endif
            endif
            new2 = create_struct(temporary(new2), tags2[j], arr2)
        endfor
        arr = new2
    endif
    new_sme = create_struct(temporary(new_sme), tags[i], arr)
endfor

sme = new_sme\n"""
            )
            temp.write('save, sme, filename="{fn}"\n'.format(fn=fname))
            temp.write("end\n")
            temp.flush()

            # with open(os.devnull, 'w') as devnull:
            subprocess.run(["idl", "-e", ".r %s" % tempname])
            self.__clean_temp__()

    def check(self):
        """Check a SME input file for validity

        Calls the sme_struct_valid procedure in idl

        Parameters
        ----------
        filename : str
        input file
        Returns
        -------
        bool
        True if file is valid, False otherwise
        """
        with tempfile.NamedTemporaryFile() as file:
            self.save(file.name)
            with open(os.devnull, "w") as devnull:
                process = subprocess.run(
                    [
                        "idl",
                        "-e",
                        'restore, "%s" & sme_struct_valid, sme, valid, /user, extra=extra & print, valid'
                        % file.name,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=devnull,
                )
            result = process.stdout.decode().split("\n")[-2].strip()
            result = bool(result)
            return result

    def load_spectrum(self, syn=False):
        """load the wavelength and spectrum, with wavelength sets seperated into seperate arrays

        syn : bool, optional
            wether to load the synthetic spectrum instead (the default is False, which means the observed spectrum is used)

        Returns
        -------
        wave, spec : 1d-array(1d-array)
            As the size of each wavelength set is not equal in general, numpy can't usually create a 2d array from the results
        """

        # wavelength grid
        wave = self.wave
        # wavelength indices of the various sections
        # +1 because of different indexing between idl and python
        section_index = self.wind + 1

        if syn:
            # synthetic spectrum
            obs_flux = self.smod
        else:
            # observed spectrum
            obs_flux = self.sob

        w, s = [], []
        for i, j in zip(section_index[:-1], section_index[1:]):
            w += [wave[i:j]]
            s += [obs_flux[i:j]]

        return np.array(w), np.array(s)

    def mask_set_bad(self, threshold=0.05):
        """Find bad pixels, by comparing with synthetic spectrum and set the mask appropiately

        threshold : float, optional
            difference above which to set the current pixel as bad (the default is 0.05)
        """

        # Spectrum observed
        spec = self.sob
        # Spectrum synthetic
        synt = self.smod
        # Mask
        # cont = 2, line = 1, bad = 0
        CONT, LINE, BAD = 2, 1, 0

        self.mob[np.abs(spec - synt) > threshold] = BAD
        # sme.save(join(dir, 'wasp29_7_tmp.out'))





def read(filename):
    """read a SME input or output file

    its just a IDL save file, so no special tricks required

    Parameters
    ----------
    filename : str
        in/output filename
    Returns
    -------
    sme : sme
        SME structure
    """
    return SME_Structure(filename)


def change_parameter(filename, parameter, value):
    """Change a single parameter in an existing sme structure

    This uses an IDL call, as there is no direct way in Python to create an IDL save structure

    Parameters
    ----------
    filename : str
        sme structure filename
    parameter : str
        name of the parameter to change
    value : {str, float, int}
        new value of the parameter
    """

    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [
                "idl",
                "-e",
                'restore, "{fn}" & sme.{par} = {val} & save, sme, filename="{fn}"'.format(
                    fn=filename, par=parameter, val=value
                ),
            ],
            stderr=devnull,
            stdout=devnull,
        )


def launch(filename):
    """launch a SME process

    the SME process loads the given filename as input
    the input is NOT checked before processing

    Parameters
    ----------
    filename : str
        input filename
    Returns
    -------
    filename : str
        output filename
    """

    with open(os.devnull, "w") as devnull:
        _ = subprocess.run(
            [
                "idl",
                "-e",
                'restore, "%s" & sme_main, sme & save, sme, file="%s"'
                % (filename, filename),
            ],
            check=True,
            stderr=devnull,
        )
    filename = os.path.splitext(filename)[0] + ".out"
    return filename


def check_input(filename):
    """Check a SME input file for validity

    Calls the sme_struct_valid procedure in idl

    Parameters
    ----------
    filename : str
        input file
    Returns
    -------
    bool
        True if file is valid, False otherwise
    """

    with open(os.devnull, "w") as devnull:
        process = subprocess.run(
            [
                "idl",
                "-e",
                'restore, "%s" & sme_struct_valid, sme, valid, /user, extra=extra & print, valid'
                % filename,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=devnull,
        )
    result = process.stdout.decode().split("\n")[-2].strip()
    result = bool(result)
    return result
