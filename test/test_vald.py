from pathlib import Path
from sme.vald import SmeLine, LineList, ValdFile, ValdShortLine


# Strings containing line data from a short-format VALD extract stellar file.
# Include cases with one and with multiple distinct references in a string.
# Include cases with and without 'wl:' and 'gf:' labels.
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


def test_smeline():
    """Test code paths and cases in vald.SmeLine().
    """
    line = SmeLine('H 1', '6564.61', '10.20', '0.71', '8.766', '2', '3')
    for vslstr in vald_short_line_strings:
        # Pass various VALD short line (vsl) strings to __init__().
        vsl = ValdShortLine(vslstr)
        line = SmeLine(vsl.species, vsl.wlcent, vsl.excit, vsl.loggf,
                       vsl.gamrad, vsl.gamqst, vsl.gamvw)
        assert isinstance(line, SmeLine)

        # __init__() argument order maps to properties as expected.
        assert line.species == vsl.species
        assert line.wlcent == vsl.wlcent
        assert line.excit == vsl.excit
        assert line.loggf == vsl.loggf
        assert line.gamrad == vsl.gamrad
        assert line.gamqst == vsl.gamqst
        assert line.gamvw == vsl.gamvw

        # eval(repr()) yields equal result according to __eq__().
        line2 = eval(repr(line))
        assert line == line2

        # __eq__() yields False when value of a property differs.
        line2.excit += 0.1
        assert not line == line2

        # __eq__() yields False when type of other object is not SmeLine.
        assert not line == ''


def test_valdshortline():
    """Test code paths and cases in vald.ValdShortLine().
    """
    for vslstr in vald_short_line_strings:
        vsl = ValdShortLine(vslstr)
        assert isinstance(vsl, ValdShortLine)
        assert vsl.__str__() == vslstr
        assert vsl.__repr__() == type(vsl).__name__ + f'({vslstr!r})'
        data, shortref = vslstr.strip().split(", '")


def test_linelist():
    """Test code paths and cases in vald.LineList().
    """
    # __init__() creates a LineList with 0 lines.
    linelist = LineList()
    assert isinstance(linelist, LineList)
    assert len(linelist) == 0
    inputs = []
    for iline, vslstr in enumerate(vald_short_line_strings):
        assert len(linelist) == iline
        vsl = ValdShortLine(vslstr)
        inputs.append(vsl)

        # Append SmeLine and ValdShortLine objects (add ValdLongLine).
        # __getitem__() returns the object just appended.
        if iline % 2 == 0:
            linelist.append(vsl)
            assert linelist[iline] == vsl
        else:
            smeline = SmeLine(vsl.species, vsl.wlcent, vsl.excit, vsl.loggf,
                              vsl.gamrad, vsl.gamqst, vsl.gamvw)
            linelist.append(smeline)
            assert linelist[iline] == smeline

    # __len__() returns number of appended lines.
    assert len(linelist) == len(vald_short_line_strings)

    # Properties return lists of values equal to the input values.
    assert isinstance(linelist.species, list)
    assert len(linelist.species) == len(vald_short_line_strings)
    assert linelist.species == [line.species for line in inputs]
    assert linelist.wlcent == [line.wlcent for line in inputs]
    assert linelist.excit == [line.excit for line in inputs]
    assert linelist.loggf == [line.loggf for line in inputs]
    assert linelist.gamrad == [line.gamrad for line in inputs]
    assert linelist.gamqst == [line.gamqst for line in inputs]
    assert linelist.gamvw == [line.gamvw for line in inputs]


def test_valdfile():
    """Test code paths and cases in vald.ValdFile().
    """
    maxdiff = lambda xlist, ylist: max(abs(x-y) for x, y in zip(xlist, ylist))
    datadir = Path(__file__).parent / 'data'
    jobnum_pairs = [[45169, 45170], [45174, 45175]]
    for jobnum1, jobnum2 in jobnum_pairs:
        valdfile1 = datadir / f'vald.{jobnum1:06}'
        vf1 = ValdFile(valdfile1)
        valdfile2 = datadir / f'vald.{jobnum2:06}'
        vf2 = ValdFile(valdfile2)
        assert vf1.nlines == vf2.nlines
        assert maxdiff(vf1.wlrange, vf2.wlrange) < 1e-2
        assert maxdiff(vf1.linelist.wlcent, vf2.linelist.wlcent) < 1e-4

    # Test _first_air_wavelength() method:
    # All wavelengths > 2000 Angstroms. All are air.
    assert(vf1._first_air_wavelength([2001]) == 0)
    assert(vf1._first_air_wavelength([2001, 2002]) == 0)

    # All wavelengths <= 1999.352 Angstroms. All are vacuum.
    assert(vf1._first_air_wavelength([1999]) == 1)
    assert(vf1._first_air_wavelength([1998, 1999]) == 2)

    # Air begins at backwards jump between 1999.352 and 2000 Angstroms.
    assert(vf1._first_air_wavelength([1999.6, 1999.5]) == 1)
    assert(vf1._first_air_wavelength([1999.6, 1999.5, 2000]) == 1)
    assert(vf1._first_air_wavelength([1999.6, 1999.5, 2001]) == 1)
    assert(vf1._first_air_wavelength([1999, 1999.6, 1999.5]) == 2)
    assert(vf1._first_air_wavelength([1999, 1999.6, 1999.5, 2000]) == 2)
    assert(vf1._first_air_wavelength([1999, 1999.6, 1999.5, 2001]) == 2)

    # Air begins at first wavelength > 2000 Angstroms.
    assert(vf1._first_air_wavelength([1999.5, 1999.6]) == 2)
    assert(vf1._first_air_wavelength([1999.5, 1999.6, 2000]) == 3)
    assert(vf1._first_air_wavelength([1999.5, 1999.6, 2001]) == 2)
    assert(vf1._first_air_wavelength([1999, 1999.5, 1999.6]) == 3)
    assert(vf1._first_air_wavelength([1999, 1999.5, 1999.6, 2000]) == 4)
    assert(vf1._first_air_wavelength([1999, 1999.5, 1999.6, 2001]) == 3)
    assert(vf1._first_air_wavelength([1999, 2001]) == 1)
    assert(vf1._first_air_wavelength([1999, 2000, 2001]) == 2)
