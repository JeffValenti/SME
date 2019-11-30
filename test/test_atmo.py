from pathlib import Path
from pytest import raises

from sme.atmo import (
    ContinuousOpacityFlags, SmeAtmo, AtmoFileAtlas9, AtmoFileError)


def test_continuousopacityflags():
    """Code coverage tests for ContinuousOpacityFlags() class and methods.
    """
    cof = ContinuousOpacityFlags()
    assert cof.smelib[2] == 1
    cof['H-'] = False
    assert cof.smelib[2] == 0
    with raises(ValueError):
        cof['H++'] = True
    with raises(ValueError):
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
    with raises(ValueError, match='Invalid modeltype'):
        atmo = SmeAtmo('_', scale)
    with raises(AttributeError, match='do not specify'):
        atmo = SmeAtmo('rhox', scale, radius=7e10)
    with raises(AttributeError, match='do not specify'):
        atmo = SmeAtmo('rhox', scale, wavelength=5000)
    with raises(AttributeError, match='specify wavelength'):
        atmo = SmeAtmo('tau', scale)
    with raises(AttributeError, match='but not radius'):
        atmo = SmeAtmo('tau', scale, radius=7e10)
    with raises(AttributeError, match='specify radius'):
        atmo = SmeAtmo('sph', scale)
    with raises(AttributeError, match='but not wavelength'):
        atmo = SmeAtmo('sph', scale, wavelength=5000)

def test_atmofileatlas9():
    """Code coverage tests for Atlas9AtmoFile() class and methods.
    """
    datadir = Path(__file__).parent / 'atmo'
    a9file = AtmoFileAtlas9(datadir / 'ap00t5750g45k2odfnew.dat')
    assert 'turbulence' in str(a9file)

def test_atmofileatlas9_exceptions():
    """Handle exceptions raised by AtmoFileAtlas9().
    """
    datadir = Path(__file__).parent / 'atmo'
    af = AtmoFileAtlas9(datadir / 'complete_file.atlas9')
    with raises(AtmoFileError, match='incomplete header'):
        AtmoFileAtlas9(datadir / 'incomplete_header.atlas9')
    with raises(AtmoFileError, match='error parsing header'):
        AtmoFileAtlas9(datadir / 'bad_header_labels.atlas9')

