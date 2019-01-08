# TODO implement synthesis tests
import pytest
from os.path import dirname

import numpy as np

from sme import synthesize_spectrum
from sme import SME_Struct
from sme.src.sme.sme import Iliffe_vector
from sme.src.sme.vald import ValdFile

cwd = dirname(__file__)


def make_minimum_structure():
    sme = SME_Struct()
    sme.teff = 5000
    sme.logg = 4.4
    sme.vmic = 1
    sme.vmac = 1
    sme.vsini = 1
    sme.set_abund(0, "asplund2009", "")
    sme.linelist = ValdFile(f"{cwd}/testcase1.lin").linelist
    sme.atmo.source = "marcs2012p_t2.0.sav"
    sme.atmo.method = "grid"

    sme.wran = [[6550, 6560], [6560, 6574]]
    return sme


def test_synthesis_simple():
    sme = make_minimum_structure()
    sme2 = synthesize_spectrum(sme)

    # Check if a result is there it has the expected data type
    assert sme2.synth is not None
    assert isinstance(sme2.synth, Iliffe_vector)
    assert isinstance(sme2.smod, np.ndarray)
    assert np.all(sme2.smod != 0)

    assert sme.wave is not None
    assert isinstance(sme2.wave, Iliffe_vector)
    assert isinstance(sme2.wob, np.ndarray)
    assert np.all(sme2.wob != 0)

    assert sme.spec is None


def test_synthesis_segment():
    sme = make_minimum_structure()
    # Out of range
    with pytest.raises(IndexError):
        synthesize_spectrum(sme, segments=[3])

    with pytest.raises(IndexError):
        synthesize_spectrum(sme, segments=[-1])

    sme2 = synthesize_spectrum(sme, segments=[0])
    assert len(sme2.synth[0]) != 0
    assert len(sme2.synth[1]) == 0

    assert len(sme2.wave[0]) != 0
    assert len(sme2.wave[1]) == 0

    assert sme2.wave.shape[0] == 2
    assert sme2.wave.shape[1][1] == 0

    orig = np.copy(sme2.synth[0])
    sme2 = synthesize_spectrum(sme2, segments=[1])

    assert sme2.wave.shape[1][0] != 0
    assert sme2.wave.shape[1][1] != 0

    assert np.all(sme2.synth[0] == orig)
