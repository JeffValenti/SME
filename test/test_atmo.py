import pytest
from sme.atmo import ContinuousOpacityFlags, SmeAtmo


def test_continuousopacityflags():
    """Code coverage tests for ContinuousOpacityFlags() class and methods.
    """
    cof = ContinuousOpacityFlags()
    assert cof.smelib[2] == 1
    cof['H-'] = False
    assert cof.smelib[2] == 0
    with pytest.raises(ValueError):
        cof['H++'] = True
    with pytest.raises(ValueError):
        cof['H-'] = 1
    assert 'H-' in cof.__str__()

def test_smeatmo():
    """Code coverage tests for SmeAtmo() class and methods.
    Demonstrate that modeltype is case insensitive ('rhox', 'RHOX').
    Test that wavelength and radius are only specified when appropriate.
    """
    scale = [1, 2, 3]
    atmo = SmeAtmo('RhoX', scale)
    assert 'rhox' in atmo.__str__()
    atmo = SmeAtmo('tau', scale, wavelength=5000)
    assert '5000' in atmo.__str__()
    atmo = SmeAtmo('SPH', scale, radius='7e10')
    assert 'None' in atmo.__str__()
    with pytest.raises(ValueError, match='Invalid modeltype'):
        atmo = SmeAtmo('_', scale)
    with pytest.raises(AttributeError, match='do not specify'):
        atmo = SmeAtmo('rhox', scale, radius=7e10)
    with pytest.raises(AttributeError, match='do not specify'):
        atmo = SmeAtmo('rhox', scale, wavelength=5000)
    with pytest.raises(AttributeError, match='specify wavelength'):
        atmo = SmeAtmo('tau', scale)
    with pytest.raises(AttributeError, match='but not radius'):
        atmo = SmeAtmo('tau', scale, radius=7e10)
    with pytest.raises(AttributeError, match='specify radius'):
        atmo = SmeAtmo('sph', scale)
    with pytest.raises(AttributeError, match='but not wavelength'):
        atmo = SmeAtmo('sph', scale, wavelength=5000)
