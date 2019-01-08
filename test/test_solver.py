import pytest
from os.path import dirname

import numpy as np

from sme import solve, SME_Struct

cwd = dirname(__file__)
filename = f"{cwd}/testcase1.inp"


def test_simple():
    sme = SME_Struct.load(filename)
    sme2 = solve(sme, ["teff"])

    assert sme2.synth is not None
    assert sme2.fitresults is not None
    assert sme2.fitresults.covar is not None
    assert isinstance(sme2.fitresults.covar, np.ndarray)
    assert np.all(sme2.fitresults.covar != 0)

    assert isinstance(sme2.fitresults.punc, dict)
    assert len(sme2.fitresults.punc) == 1
    assert len(sme2.fitresults.punc.keys()) == 1
    assert list(sme2.fitresults.punc.keys())[0] == "teff"
    assert list(sme2.fitresults.punc.values())[0] != 0

    assert np.array_equal(sme2.fitresults.covar.shape, [1, 1])
    assert sme2.fitresults.covar.ndim == 2

    assert sme2.fitresults.chisq != 0
