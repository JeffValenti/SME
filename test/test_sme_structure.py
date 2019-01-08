import pytest

import numpy as np
from os.path import dirname
from os import remove
from sme.src.sme.sme import SME_Struct

cwd = dirname(__file__)
filename = f"{cwd}/__test.npy"


def test_empty_structure():
    """ Test that all properties behave well when nothing is set """
    empty = SME_Struct()

    assert empty.version == "5.1"
    assert empty.teff is None
    assert empty.logg is None
    assert empty.vmic is None
    assert empty.vmac is None
    assert empty.vsini is None

    assert empty.nseg is None
    assert empty.wave is None
    assert empty.spec is None
    assert empty.uncs is None
    assert empty.synth is None
    assert empty.mask is None
    assert empty.mask_good is None
    assert empty.mask_bad is None
    assert empty.mask_line is None
    assert empty.mask_continuum is None

    assert empty.cscale == [[1]]
    assert empty.vrad is None
    assert empty.cscale_flag == "none"
    assert empty.vrad_flag == "none"
    assert empty.cscale_degree == 0

    assert empty.mu == [1]
    assert empty.nmu == 1

    assert empty.md5 is not None

    assert empty.linelist is None
    assert empty.species is None
    assert empty.atomic is None

    assert empty.monh is None
    assert np.isnan(empty["Fe Abund"])
    assert np.isnan(empty.abund["H"])
    assert np.isnan(empty.abund()["Mg"])

    assert empty.idlver is not None
    assert empty.idlver.arch is None

    assert len(empty.fitparameters) == 0
    assert empty.fitresults is not None
    assert empty.fitresults.covar is None

    assert empty.atmo is not None
    assert empty.atmo.depth is None

    assert empty.nlte is not None
    assert empty.nlte.elements == []


def test_save_and_load_structure():
    sme = SME_Struct()
    assert sme.teff is None

    sme.teff = 5000
    sme.save(filename)
    sme = SME_Struct.load(filename)
    remove(filename)
    assert sme.teff == 5000

    data = np.linspace(1000, 2000, 100)
    sme.wave = data
    sme.spec = data
    sme.save(filename)
    sme = SME_Struct.load(filename)
    remove(filename)
    assert np.all(sme.wave[0] == data)
    assert np.all(sme.spec[0] == data)
    assert sme.nseg == 1


def test_load_idl_savefile():
    filename = f"{cwd}/testcase1.inp"
    sme = SME_Struct.load(filename)

    assert sme.teff == 5770
    assert sme.wave is not None

    assert sme.nseg == 1
    assert sme.cscale_flag == "linear"
    assert sme.vrad_flag == "each"


def test_cscale_degree():
    sme = SME_Struct()
    sme.cscale = 1

    flags = ["none", "fix", "constant", "linear", "quadratic"]
    degrees = [0, 0, 0, 1, 2]

    for f, d in zip(flags, degrees):
        sme.cscale_flag = f
        assert sme.cscale_degree == d
        assert sme.cscale.shape[0] == 1
        assert sme.cscale.shape[1] == d + 1


def test_idlver():
    sme = SME_Struct()
    sme.idlver.update()
    # assert sme.idlver.arch == "x86_64"


def test_fitresults():
    sme = SME_Struct()
    sme.fitresults.chisq = 100
    sme.fitresults.clear()
    assert sme.fitresults.chisq is None
