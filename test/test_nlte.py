# TODO implement NLTE tests

import pytest

from os.path import dirname

import numpy as np

from sme import synthesize_spectrum
from sme import SME_Struct
from sme.src.sme.sme import Iliffe_vector
from sme.src.sme.vald import ValdFile
from sme.src.sme.sme_synth import SME_DLL
from sme.src.sme.nlte import nlte
from sme.src.sme.solve import get_atmosphere

cwd = dirname(__file__)


def make_minimum_structure():
    sme = SME_Struct()
    sme.teff = 5000
    sme.logg = 4.4
    sme.vmic = 1
    sme.vmac = 1
    sme.vsini = 1
    sme.set_abund(0, "asplund2009", "")
    sme.linelist = ValdFile(f"{cwd}/testcase3.lin").linelist
    sme.atmo.source = "marcs2012p_t2.0.sav"
    sme.atmo.method = "grid"

    sme.wran = [[6436, 6444]]

    return sme


def test_activate_nlte():
    sme = make_minimum_structure()

    # Make sure nothing is set yet
    assert len(sme.nlte.elements) == 0

    # Add an element
    sme.nlte.set_nlte("Ca")
    assert len(sme.nlte.elements) == 1
    assert "Ca" in sme.nlte.elements

    # Add it again, shouldn't change anything
    sme.nlte.set_nlte("Ca")
    assert len(sme.nlte.elements) == 1
    assert "Ca" in sme.nlte.elements

    # Try to remove something else
    sme.nlte.remove_nlte("Na")
    assert len(sme.nlte.elements) == 1
    assert "Ca" in sme.nlte.elements

    # Remove the original element
    sme.nlte.remove_nlte("Ca")
    assert len(sme.nlte.elements) == 0

    # Add a element with a custom grid
    sme.nlte.set_nlte("Na", "test_grid.grd")
    assert len(sme.nlte.elements) == 1
    assert "Na" in sme.nlte.elements
    assert sme.nlte.grids["Na"] == "test_grid.grd"

    # Update custom grid
    sme.nlte.set_nlte("Na", "test_grid2.grd")
    assert len(sme.nlte.elements) == 1
    assert sme.nlte.grids["Na"] == "test_grid2.grd"

    # Add element without default grid
    with pytest.raises(ValueError):
        sme.nlte.set_nlte("U")

    # with a grid it should work
    sme.nlte.set_nlte("U", "test_grid.grd")
    assert sme.nlte.grids["U"] == "test_grid.grd"


def test_run_with_nlte():
    # NOTE sme structure must have long format for NLTE
    sme = make_minimum_structure()
    sme.nlte.set_nlte("Ca")

    sme2 = synthesize_spectrum(sme)

    assert isinstance(sme2.nlte.flags, np.ndarray)
    assert np.issubdtype(sme2.nlte.flags.dtype, np.bool)
    assert len(sme2.nlte.flags) == len(sme2.linelist)
    assert np.any(sme2.nlte.flags)


def test_dll():
    sme = make_minimum_structure()
    elem = "Ca"
    sme.nlte.set_nlte(elem)

    libsme = SME_DLL()
    libsme.ResetNLTE()

    sme = get_atmosphere(sme)
    libsme.InputLineList(sme.linelist)
    libsme.InputModel(sme.teff, sme.logg, sme.vmic, sme.atmo)

    # This is essentially what update_depcoefs does, just for one element
    counter = 0
    bmat, linerefs, lineindices = nlte(sme, elem)
    for lr, li in zip(linerefs, lineindices):
        if lr[0] != -1 and lr[1] != -1:
            counter += 1
            libsme.InputNLTE(bmat[:, lr].T, li)

    flags = libsme.GetNLTEflags()
    assert np.any(flags)
    assert np.count_nonzero(flags) == counter
    assert len(flags) == len(sme.linelist)

    idx = np.where(flags)[0][0]
    coeffs = libsme.GetNLTE(idx)
    assert coeffs is not None

    # If we reset NLTE no flags should be set
    libsme.ResetNLTE()
    flags = libsme.GetNLTEflags()
    assert not np.any(flags)
    assert len(flags) == len(sme.linelist)

    with pytest.raises(TypeError):
        libsme.InputNLTE(None, 0)

    with pytest.raises(TypeError):
        libsme.InputNLTE(bmat[:, [0, 1]].T, 0.1)

    with pytest.raises(ValueError):
        libsme.InputNLTE([0, 1], 10)

    with pytest.raises(ValueError):
        libsme.InputNLTE(bmat[:, [0, 1]].T, -10)

