import pytest
from sme.vald import Line, LineList

species = 'Fe 1'
wlcent = 5502.9931
excit = 0.9582
gflog = -3.047
gamrad = 7.19
gamqst = -6.22
gamvw = 239.249
linedata = [species, wlcent, excit, gflog, gamrad, gamqst, gamvw]

def test_line_init():
    """Test that property values equal line data passed to __init__().
    """
    line = Line(*linedata)
    assert isinstance(line, Line)
    assert line.species == species
    assert line.wlcent == wlcent
    assert line.excit == excit
    assert line.gflog == gflog
    assert line.gamrad == gamrad
    assert line.gamqst == gamqst
    assert line.gamvw == gamvw

def test_linelist_add_and_len():
    """Test that len() returns the number of lines (including 0) in list.
    """
    linelist = LineList()
    assert isinstance(linelist, LineList)
    assert len(linelist) == 0
    for iline in range(3):
        assert len(linelist) == iline
        linelist.add(*linedata)

def test_linelist_properties():
    """Test that properties are lists with one item per line.
    Test that property value equal line data passed to add().
    """
    linelist = LineList()
    linelist.add(*linedata)
    proplist = [
            linelist.species,
            linelist.wlcent,
            linelist.excit,
            linelist.gflog,
            linelist.gamrad,
            linelist.gamqst,
            linelist.gamvw]
    for iprop, prop in enumerate(proplist):
        assert isinstance(prop, list)
        assert len(prop) == 1
        assert prop[0] == linedata[iprop]
