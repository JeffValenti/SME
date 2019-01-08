from pathlib import Path
from sme.src.sme.linelist import LineList
from sme.src.sme.vald import ValdFile

import numpy as np

species = "Fe 1"
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
    line = LineList(linedata)
    assert isinstance(line, LineList)
    assert line.species[0] == species
    assert line.wlcent[0] == wlcent
    assert line.excit[0] == excit
    assert line.gflog[0] == gflog
    assert line.gamrad[0] == gamrad
    assert line.gamqst[0] == gamqst
    assert line.gamvw[0] == gamvw


def test_linelist_add_and_len():
    """Test that len() returns the number of lines (including 0) in list.
    """
    linelist = LineList()
    assert isinstance(linelist, LineList)
    assert len(linelist) == 0
    for iline in range(3):
        print(len(linelist._lines))
        print(linelist._lines)
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
        linelist.gamvw,
    ]
    for iprop, prop in enumerate(proplist):
        print(linelist._lines)
        assert isinstance(prop, np.ndarray)
        assert len(prop) == 1
        assert prop[0] == linedata[iprop]


def test_valdfile():
    """Test class to read a VALD line data file.
    """
    testdir = Path(__file__).parent
    vf = ValdFile(testdir / "testcase1.lin")

    linelist = vf.linelist

    assert len(linelist) == 44
    assert linelist.lineformat == "short"
    assert linelist[0].species == "V 1"
