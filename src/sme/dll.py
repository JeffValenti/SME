from ctypes import (
        CDLL, POINTER, Structure, pointer, byref,
        c_short, c_ushort, c_int, c_double, c_char_p)
from pathlib import Path
from platform import system, machine, architecture


class LibSme:
    """Access SME external library code.
    """
    def __init__(self, file=None):
        if file:
            self._file = file
        else:
            self._file = self.default_libfile()
        self.lib = CDLL(str(self._file))
        self._wfirst = None
        self._wlast = None
        self._vwscale = None
        self._h2broad = None
        self._linelist = None

    @property
    def file(self):
        """Absolute path to SME library file for the current platform.
        """
        return self._file

    @property
    def wfirst(self):
        return self._wfirst

    @property
    def wlast(self):
        return self._wlast

    @property
    def vwscale(self):
        return self._vwscale

    @property
    def h2broad(self):
        return self._h2broad

    @property
    def linelist(self):
        return self._linelist

    def default_libfile(self):
        """Return default absolute path to SME library file.
        """
        dir = Path(__file__).parent.joinpath('dll')
        file = '.'.join([
            'sme_synth.so',
            system().lower(),
            machine(),
            architecture()[0][0:2]
            ])
        return dir.joinpath(file)

    def SMELibraryVersion(self):
        """Return version number reported by SME library code.
        """
        self.lib.SMELibraryVersion.restype = c_char_p
        return self.lib.SMELibraryVersion().decode('utf-8')

    def InputWaveRange(self, wfirst, wlast):
        """Pass wavelength range for spectrum synthesis to library code.
        """
        libfunc = self.lib.InputWaveRange

        class Args(Structure):
            _fields_ = [
                ('wfirst', POINTER(c_double)),
                ('wlast', POINTER(c_double))]

            def __init__(self, wfirst, wlast):
                self.wfirst = pointer(c_double(wfirst))
                self.wlast = pointer(c_double(wlast))

        argv = Args(wfirst, wlast)
        argc = len(argv._fields_)
        libfunc.argtypes = [c_int, *[POINTER(f[1]) for f in argv._fields_]]
        libfunc.restype = c_char_p
        error = libfunc(
            argc,
            byref(argv.wfirst),
            byref(argv.wlast)
            ).decode('utf-8')
        if error != '':
            raise ValueError(error)
        self._wfirst = wfirst
        self._wlast = wlast

    def SetVWscale(self, vwscale):
        """Pass van der Waals broadening enhancement factor to library code.
        """
        libfunc = self.lib.SetVWscale
        argv = pointer(c_double(vwscale))
        libfunc.argtypes = [c_int, POINTER(POINTER(c_double))]
        libfunc.restype = c_char_p
        error = libfunc(1, byref(argv)).decode('utf-8')
        if error != '':
            raise ValueError(error)
        self._vwscale = vwscale

    def SetH2broad(self):
        """Enable collisional broadening by molecular hydrogen in library code.
        """
        self.lib.SetH2broad.restype = c_char_p
        assert self.lib.SetH2broad().decode('utf-8') == ''
        self._h2broad = True

    def ClearH2broad(self):
        """Disable collisional broadening by molecular hydrogen in library code.
        """
        self.lib.ClearH2broad.restype = c_char_p
        assert self.lib.ClearH2broad().decode('utf-8') == ''
        self._h2broad = False

    def InputLineList(self, linelist):
        """Pass atomic and molecular line data to library code.
        """
        libfunc = self.lib.InputLineList
        nlines = len(linelist)
        m = 8

        class Args(Structure):
            _fields_ = [
                ('nlines', POINTER(c_int)),
                ('species', POINTER(_IdlString * nlines)),
                ('atomic', POINTER(c_double * nlines * m))
                ]

            def __init__(self, linelist):
                self._nlines = c_int(nlines)
                self._species = _IdlStringArray(linelist.species)
                self._atomic = Array2D(c_double, nlines, m)
                self._atomic.data[2][:] = linelist.wlcent
                self._atomic.data[3][:] = linelist.excit
                self._atomic.data[4][:] = linelist.loggf
                self._atomic.data[5][:] = linelist.gamrad
                self._atomic.data[6][:] = linelist.gamqst
                self._atomic.data[7][:] = linelist.gamvw
                self.nlines = pointer(self._nlines)
                self.species = pointer(self._species.data)
                self.atomic = pointer(self._atomic.data)

        argv = Args(linelist)
        argc = len(argv._fields_)
        libfunc.argtypes = [c_int, *[POINTER(f[1]) for f in argv._fields_]]
        libfunc.restype = c_char_p
        error = libfunc(
            argc,
            byref(argv.nlines),
            byref(argv.species),
            byref(argv.atomic)
            ).decode('utf-8')
        if error != '':
            raise ValueError(error)
        self._linelist = linelist

    def OutputLineList(self):
        """Returns WLCENT, GF, EXCIT, log10(GAMRAD), log10(GAMQST), GAMVW.
        Note that InputLineList elements are 2:WLCENT, 3:log(GF), 4:EXCIT,
        whereas  OutputLineList elements are 0:WLCENT, 1:EXCIT,   2:GF.
        """
        libfunc = self.lib.OutputLineList
        assert self._linelist is not None
        nlines = len(self._linelist)
        m = 6

        class Args(Structure):
            _fields_ = [
                ('nlines', POINTER(c_int)),
                ('atomic', POINTER(c_double * m * nlines))
                ]

            def __init__(self):
                self._nlines = c_int(nlines)
                self._atomic = Array2D(c_double, m, nlines)
                self.nlines = pointer(self._nlines)
                self.atomic = pointer(self._atomic.data)

        argv = Args()
        argc = len(argv._fields_)
        libfunc.argtypes = [c_int, *[POINTER(f[1]) for f in argv._fields_]]
        libfunc.restype = c_char_p
        error = libfunc(
            argc,
            byref(argv.nlines),
            byref(argv.atomic)
            ).decode('utf-8')
        if error != '':
            raise ValueError(error)
        return argv._atomic.data

    def UpdateLineList(self, newlinedata, index):
        """Pass new line data to library code for lines specified by index.
        """
        libfunc = self.lib.UpdateLineList
        nlines = len(newlinedata)
        if len(index) != nlines:
            raise ValueError(f'mismatch: {nlines} lines, {len(index)} indexes')
        m = 8

        class Args(Structure):
            _fields_ = [
                ('nlines', POINTER(c_short)),
                ('species', POINTER(_IdlString * nlines)),
                ('atomic', POINTER(c_double * nlines * m)),
                ('index', POINTER(c_short * nlines))
                ]

            def __init__(self, newlinedata, index):
                self._nlines = c_short(nlines)
                self._species = _IdlStringArray(newlinedata.species)
                self._atomic = Array2D(c_double, nlines, m)
                self._atomic.data[2][:] = newlinedata.wlcent
                self._atomic.data[3][:] = newlinedata.excit
                self._atomic.data[4][:] = newlinedata.loggf
                self._atomic.data[5][:] = newlinedata.gamrad
                self._atomic.data[6][:] = newlinedata.gamqst
                self._atomic.data[7][:] = newlinedata.gamvw
                self._index = Array1D(c_short, nlines)
                self._index.data[:] = index
                self.nlines = pointer(self._nlines)
                self.species = pointer(self._species.data)
                self.atomic = pointer(self._atomic.data)
                self.index = pointer(self._index.data)

        argv = Args(newlinedata, index)
        argc = len(argv._fields_)
        libfunc.argtypes = [c_int, *[POINTER(f[1]) for f in argv._fields_]]
        libfunc.restype = c_char_p
        error = libfunc(
            argc,
            byref(argv.nlines),
            byref(argv.species),
            byref(argv.atomic),
            byref(argv.index)
            ).decode('utf-8')
        if error != '':
            raise RuntimeError(error)
        for i, line in enumerate(newlinedata):
            self._linelist[index[i]] = line


class Array1D:
    class Type:
        def __init__(self, ctype, n):
            self.type = ctype * n

    def __init__(self, ctype, n):
        self.n = n
        datatype = self.Type(ctype, self.n).type
        self.type = POINTER(datatype)
        self.data = datatype()
        self.pointer = pointer(self.data)


class Array2D:
    class Type:
        """
        Memory order has n varying rapidly, m varying slowly.
        Use slow index first, rapid index second, e.g., obj[jm][in].
        """
        def __init__(self, ctype, n, m):
            self.type = ctype * n * m

    def __init__(self, ctype, n, m):
        self.n = n
        self.m = m
        datatype = self.Type(ctype, self.n, self.m).type
        self.type = POINTER(datatype)
        self.data = datatype()
        self.pointer = pointer(self.data)


class _IdlString(Structure):
    def __init__(self, bytes):
        self.len = c_int(len(bytes))
        self.type = c_ushort(0)
        self.bytes = c_char_p(bytes)

    _fields_ = [
        ('len', c_int),
        ('type', c_ushort),
        ('bytes', c_char_p)
        ]

class _IdlStringArray:
    class Type:
        def __init__(self, n):
            self.type = _IdlString * n

    def __init__(self, strlist):
        self.n = len(strlist)
        datatype = self.Type(self.n).type
        self.data = datatype()
        self.data[:] = [_IdlString(s.encode('utf-8')) for s in strlist]
        self.type = POINTER(datatype)
        self.pointer = pointer(self.data)
