from math import log10
from pathlib import Path
from pytest import raises
from sme.dll import LibSme
from sme.vald import LineList, ValdShortLine


def test_dll():
    """Load SME library. Pass parameters. Read results.
    Some later steps require results from earlier steps.
    """

# Load SME library.
    libsme = LibSme()
    assert Path(libsme.file).is_file()
    libsme2 = LibSme(libsme.file)

# Read version number.
    version = libsme.SMELibraryVersion()

# Fail to pass wavelength range.
    with raises(ValueError, match='Wrong wavelength range'):
        libsme.InputWaveRange(9000, 3000)

# Pass wavelength range.
    wfirst, wlast = 6550, 6560.5
    libsme.InputWaveRange(wfirst, wlast)
    assert libsme.wfirst == wfirst
    assert libsme.wlast == wlast

# Fail to pass enhancement factor for van der Waals broadening.
    with raises(TypeError, match='must be real number'):
        libsme.SetVWscale('type')

# Pass enhancement factor for van der Waals broadening.
    vw_scale = 2.5
    libsme.SetVWscale(vw_scale)
    assert libsme.vw_scale == vw_scale
    raises(TypeError, libsme.SetVWscale, 'string')

# Enable collisional broadening by H2.
    libsme.SetH2broad()
    assert libsme.H2broad is True

# Disable collisional broadening by H2.
    libsme.ClearH2broad()
    assert libsme.H2broad is False

# Create line list
    vald_short_line_strings = [
        "'Ti 1',       6554.2230,   1.4432, 1.0, -1.150, 7.870,-6.070,"
        " 284.261,  1.070, 0.606, '   9 wl:LGWSC   9 LGWSC   9 gf:LGWSC"
        "   7 K10   7 K10   7 K10  10 BPM Ti            '",
        "'MgH 1',      6556.8086,   0.9240, 1.0, -0.867, 7.060, 0.000,"
        "   0.000, 99.000, 0.021, '  12 KMGH  12 KMGH  12 KMGH"
        "  12 KMGH  12 KMGH  12 KMGH  12 KMGH (24)MgH       '"
        ]
    linelist = LineList()
    for vslstr in vald_short_line_strings:
        linelist.append(ValdShortLine(vslstr))

# Fail to pass line data.
    out_of_order_list = LineList()
    out_of_order_list.append(linelist[1])
    out_of_order_list.append(linelist[0])
    with raises(ValueError, match='not sorted in wavelength ascending order'):
        libsme.InputLineList(out_of_order_list)

# Pass line data.
    libsme.InputLineList(linelist)

# Read line data and check that output matches input.
    outlist = libsme.OutputLineList()
    assert_outputlinelist_matches_input(outlist, libsme.linelist)

# Create new data for one line.
    index = [1]
    newlinedata = LineList()
    for i in index:
        vsl = libsme.linelist[i]
        vsl.wlcent += 0.01
        newlinedata.append(vsl)

# Fail to update data for one line.
    with raises(ValueError, match=r'mismatch: .* lines, .* indexes'):
        libsme.UpdateLineList(newlinedata, [0, 1])
    with raises(RuntimeError, match=r'Attempt to replace .* another line'):
        libsme.UpdateLineList(newlinedata, [0])
 
# Update data for one line.
    libsme.UpdateLineList(newlinedata, index)
    outlist = libsme.OutputLineList()
    assert_outputlinelist_matches_input(outlist, libsme.linelist)

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
    assert_outputlinelist_matches_input(outlist, libsme.linelist)

def assert_outputlinelist_matches_input(outlist, inlist):
    """Check that line data returned by libsme.UpdateLineList()
    is equivalent  to line data passed to libsme.InputLineList().
    """
    for outline, inline in zip(outlist, inlist):
        assert outline[0] == inline.wlcent
        assert log10(outline[1]) - inline.loggf < 1e-8
        assert outline[2] == inline.excit
        assert outline[3] - inline.gamrad < 1e-8
        assert outline[4] - inline.gamqst < 1e-8
        assert outline[5] - inline.gamvw < 1e-8
