from pathlib import Path
from sme.vald import SmeLine, LineList, ValdFile, ValdShortLine


species = 'Fe 1'
wlcent = 5502.9931
excit = 0.9582
loggf = -3.047
gamrad = 7.19
gamqst = -6.22
gamvw = 239.249
linedata = [species, wlcent, excit, loggf, gamrad, gamqst, gamvw]

vald_short_line_strings = [
    "'Ti 1',       6554.2230,   1.4432, 1.0, -1.150, 7.870,-6.070,"
    " 284.261,  1.070, 0.606, '   9 wl:LGWSC   9 LGWSC   9 gf:LGWSC"
    "   7 K10   7 K10   7 K10  10 BPM Ti            '",
    "'Ti 1',       6554.2230,   1.4432, 1.0, -1.150, 7.870,-6.070,"
    " 284.261,  1.070, 0.606, '   9 LGWSC   9 LGWSC   9 LGWSC"
    "   7 K10   7 K10   7 K10  10 BPM Ti            '",
    "'MgH 1',      6556.8086,   0.9240, 1.0, -0.867, 7.060, 0.000,"
    "   0.000, 99.000, 0.021, '  12 wl:KMGH  12 KMGH  12 gf:KMGH"
    "  12 KMGH  12 KMGH  12 KMGH  12 KMGH (24)MgH       '",
    "'MgH 1',      6556.8086,   0.9240, 1.0, -0.867, 7.060, 0.000,"
    "   0.000, 99.000, 0.021, '  12 KMGH  12 KMGH  12 KMGH"
    "  12 KMGH  12 KMGH  12 KMGH  12 KMGH (24)MgH       '",
    ]


def test_valdshortline():
    for vslstr in vald_short_line_strings:
        vsl = ValdShortLine(vslstr)
        assert vsl.__str__() == vslstr
        assert vsl.__repr__() == type(vsl).__name__ + f'({vslstr!r})'


def test_smeline():
    """Test that property values equal line data passed to __init__().
    """
    line = SmeLine(*linedata)
    assert isinstance(line, SmeLine)
    assert line.species == species
    assert line.wlcent == wlcent
    assert line.excit == excit
    assert line.loggf == loggf
    assert line.gamrad == gamrad
    assert line.gamqst == gamqst
    assert line.gamvw == gamvw
    line2 = eval(repr(line))
    assert line == line2
    line2.excit += 0.1
    assert not line == line2
    assert not line == None


def test_linelist_add_and_len():
    """Test that len() returns the number of lines (including 0) in list.
    """
    linelist = LineList()
    assert isinstance(linelist, LineList)
    assert len(linelist) == 0
    for iline in range(3):
        assert len(linelist) == iline
        linelist.add(SmeLine(*linedata))


def test_linelist_properties():
    """Test that properties are lists with one item per line.
    Test that property value equal line data passed to add().
    """
    linelist = LineList()
    linelist.add(SmeLine(*linedata))
    proplist = [
            linelist.species,
            linelist.wlcent,
            linelist.excit,
            linelist.loggf,
            linelist.gamrad,
            linelist.gamqst,
            linelist.gamvw]
    for iprop, prop in enumerate(proplist):
        assert isinstance(prop, list)
        assert len(prop) == 1
        assert prop[0] == linedata[iprop]


def test_valdfile():
    """Test class to read a VALD line data file.
    """
    testdir = Path(__file__).parent
    vf = ValdFile(testdir / 'testcase1.lin')
