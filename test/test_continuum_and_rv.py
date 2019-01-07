# TODO implement continuum and radial velocity tests

from os.path import dirname

import pytest
import numpy as np

from scipy.constants import speed_of_light

from SME.src.sme.sme import SME_Struct
from SME.src.sme.continuum_and_radial_velocity import determine_rv_and_cont

c_light = speed_of_light * 1e-3

# TODO get better test case for this
cwd = dirname(__file__)
sme = SME_Struct.load(cwd + "/testcase1.npy")

rv = 10
x_syn = sme.wave[0] * (1 - rv / c_light)
y_syn = sme.synth[0]


def test_match_both():
    # Nothing should change
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"

    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert rvel == 0
    assert cscale == [1]

    sme.vrad_flag = "each"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert cscale == [1]

    sme.cscale_flag = "constant"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [1], atol=1e-2)

    sme.cscale_flag = "linear"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 1], atol=1e-2)

    sme.cscale_flag = "quadratic"
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 0, 1], atol=1e-2)


def test_nomask():
    sme.cscale_flag = "linear"
    sme.vrad_flag = "each"

    mask = np.copy(sme.mask[0])

    sme.mask[0] = 1

    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert np.allclose(rvel, rv, atol=1)
    assert np.allclose(cscale, [0, 1], atol=1e-2)

    sme.mask[0] = 0
    rvel, cscale = determine_rv_and_cont(sme, 0, x_syn, y_syn)

    assert rvel == 0
    assert cscale == [1]

    sme.mask[0] = mask
