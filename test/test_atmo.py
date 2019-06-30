import pytest
from sme.atmo import OpacityFlags


def test_opacityflags():
    """Code ceverage tests for OpacityFlags() class and methods.
    """
    defaults = OpacityFlags('defaults')
    assert OpacityFlags() == defaults
    assert OpacityFlags(defaults) == defaults
    assert all(defaults) == True
    with pytest.raises(AssertionError):
        flags = OpacityFlags({'Xe++': True})
