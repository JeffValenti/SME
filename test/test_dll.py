from math import log10
from pathlib import Path
from pytest import raises
from sme.dll import LibSme, _IdlString, _IdlStringArray
from sme.vald import LineList, ValdShortLine


def test_load_and_get_version():
    """Load SME library. Get version number of SME library.
    """
    libsme = LibSme()
    version = libsme.SMELibraryVersion()

def test_basic_sequence():
    """Test basic sequence of SME library calls.
    """
    libsme = LibSme()
    libsme.InputWaveRange(6550, 6560.5)
    libsme.InputLineList(_make_linelist([0, 1]))

def test_settings_and_properties():
    """Test SME library settings and corresponding class properties.
    """

    # Path to SME library file (file).
    libsme = LibSme()
    assert Path(libsme.file).is_file()
    libsme = LibSme(libsme.file)

    # Enhancement factor for van der Waals broadening (vwscale).
    assert libsme.vwscale is None
    vwscale = 2.5
    libsme.SetVWscale(vwscale)
    assert libsme.vwscale == vwscale

    # Collisional broadening by H2 (h2broad).
    assert libsme.h2broad is None
    libsme.SetH2broad()
    assert libsme.h2broad is True
    libsme.ClearH2broad()
    assert libsme.h2broad is False

    # Wavelength range (wfirst, wlast).
    assert libsme.wfirst == None
    assert libsme.wlast == None
    wfirst, wlast = 6550, 6560.5
    libsme.InputWaveRange(wfirst, wlast)
    assert libsme.wfirst == wfirst
    assert libsme.wlast == wlast

def test_linelist():
    """Test line data methods and code paths.
    """
    libsme = LibSme()
    libsme.InputWaveRange(6550, 6560.5)

    # Pass line list to SME library. Read line list back. Check equivalence.
    linelist = _make_linelist([0, 1])
    libsme.InputLineList(linelist)
    outlist = libsme.OutputLineList()
    _assert_outputlinelist_matches_input(outlist, libsme.linelist)

    # Update one line. Read line list. Check update
    index = [1]
    delta_loggf = 0.1
    newline = _make_linelist(index, delta_loggf=delta_loggf)
    assert abs(libsme.linelist[index[0]].loggf - newline[0].loggf) > 1e-8
    libsme.UpdateLineList(newline, index)
    assert abs(libsme.linelist[index[0]].loggf - newline[0].loggf) < 1e-8
    outlist = libsme.OutputLineList()
    _assert_outputlinelist_matches_input(outlist, libsme.linelist)

    # Create new data for multiple lines.
    index = [1, 0]
    newlinedata = LineList()
    for i in index:
        vsl = libsme.linelist[i]
        vsl.loggf += 0.01
        newlinedata.append(vsl)

    # Update line data
    libsme.UpdateLineList(newlinedata, index)
    outlist = libsme.OutputLineList()
    _assert_outputlinelist_matches_input(outlist, libsme.linelist)

def test_library_exceptions():
    """Handle exceptions raised by SME Library.
    """
    libsme = LibSme()

    # Wavelength range is not in ascending wavelength order.
    with raises(ValueError, match='Wrong wavelength range'):
        libsme.InputWaveRange(9000, 3000)

    # Line list is not in ascending wavelength order.
    out_of_order_list = _make_linelist([1, 0])
    wlcent = out_of_order_list.wlcent
    assert wlcent[0] > wlcent[1]
    with raises(ValueError, match='not sorted in wavelength ascending order'):
        libsme.InputLineList(out_of_order_list)

    # Attempt to replace line data for a different line.
    linelist = _make_linelist([0, 1])
    libsme.InputLineList(linelist)
    with raises(RuntimeError, match=r'Attempt to replace .* another line'):
        libsme.UpdateLineList(linelist, [1, 0])

def test_python_exceptions():
    """Hanlde exceptions raised by python interface to SME Library.
    """
    libsme = LibSme()

    # van der Waals broadening parameter is not a real number.
    with raises(TypeError, match='must be real number'):
        libsme.SetVWscale('not a real number')

    # Attempt to replace line data for a different line.
    linelist = _make_linelist([0, 1])
    libsme.InputLineList(linelist)
    with raises(ValueError, match=r'mismatch: .* lines, .* indexes'):
        libsme.UpdateLineList(linelist, [0])

    # 
    linelist = _make_linelist([0, 1])
    libsme.InputLineList(linelist)
    with raises(ValueError, match=r'mismatch: .* lines, .* indexes'):
        libsme.UpdateLineList(linelist, [0])

def _make_linelist(index, delta_loggf=None):
    """Make a line list for use in tests.
    """
    vald_short_line_strings = [
        "'Ti 1',       6554.2230,   1.4432, 1.0, -1.150, 7.870,-6.070,"
        " 284.261,  1.070, 0.606, '   9 wl:LGWSC   9 LGWSC   9 gf:LGWSC"
        "   7 K10   7 K10   7 K10  10 BPM Ti            '",
        "'MgH 1',      6556.8086,   0.9240, 1.0, -0.867, 7.060, 0.000,"
        "   0.000, 99.000, 0.021, '  12 KMGH  12 KMGH  12 KMGH"
        "  12 KMGH  12 KMGH  12 KMGH  12 KMGH (24)MgH       '"
        ]
    linelist = LineList()
    for i in index:
        vsl = ValdShortLine(vald_short_line_strings[i])
        if delta_loggf:
            vsl.loggf += delta_loggf
        linelist.append(vsl)
    return(linelist)

def _assert_outputlinelist_matches_input(outlist, inlist):
    """Check that line data returned by libsme.UpdateLineList()
    is equivalent  to line data passed to libsme.InputLineList().
    """
    for outline, inline in zip(outlist, inlist):
        assert outline[0] == inline.wlcent
        assert abs(log10(outline[1]) - inline.loggf) < 1e-8
        assert outline[2] == inline.excit
        assert abs(outline[3] - inline.gamrad) < 1e-8
        assert abs(outline[4] - inline.gamqst) < 1e-8
        assert abs(outline[5] - inline.gamvw) < 1e-8
