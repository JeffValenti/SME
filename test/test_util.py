from pytest import raises
from sme.util import (
    change_waveunit, air_to_vacuum, vacuum_to_air, vacuum_angstroms)


def test_change_waveunit():
    """Test code paths and cases in util.change_waveunit().
    Test conversion from/to Angstroms to/from all units listed in `cases`.
    Test units specified in uppercase and lowercase.
    Test scalar wavelength and list of wavelengths.
    """
    cases = {
        'a': 5000,
        'nm': 500,
        'um': 0.5,
        'micron': 0.5,
        'cm-1': 20000,
        '1/cm': 20000}
    for unit in cases:
        wave = cases[unit]
        assert change_waveunit(5000, 'A', unit) == wave
        assert change_waveunit(5000, 'A', unit.upper()) == wave
        assert change_waveunit([5000], 'A', unit) == [wave]
        assert change_waveunit(wave, unit, 'A') == 5000
        assert change_waveunit(wave, unit.upper(), 'A') == 5000
        assert change_waveunit([wave, wave], unit, 'A') == [5000, 5000]
    with raises(ValueError, match='invalid waveunit specified'):
        change_waveunit(5000, 'A', 'erg')
    with raises(ValueError, match='invalid waveunit specified'):
        change_waveunit(5000, 'erg', 'A')

def test_air_to_vacuum():
    """Test code paths and cases in util.air_to_vacuum().
    Test conversion for input wavelengths in Angstroms and nm.
    Test wavelength above and below 2000 Angstroms.
    Test scalar wavelength and list of wavelengths.
    Test that air_to_vacuum(vacuum_to_air( )) is an identity operator.
    Allow discrepancies smaller than 1e-8 Angstroms.
    """
    wvac_a = [1999, 5000, 5010]
    wair_a = [1999, 4998.605522013399, 5008.602864587058]
    wvac_nm = [100, 200, 1000, 2000, 5000, 10000, 20000]
    for wv, wa in zip(wvac_a, wair_a):
        assert abs(air_to_vacuum(wa, 'A') - wv) < 1e-8
    wvac_nm = air_to_vacuum([w / 10 for w in wair_a], 'nm')
    assert all([abs(10 * wnm - wv) < 1e-8 for wnm, wv in zip(wvac_nm, wvac_a)])
    for wnm in wvac_nm:
        assert abs(air_to_vacuum(vacuum_to_air(wnm, 'nm'), 'nm') - wnm) < 1e-8

def test_vacuum_to_air():
    """Test code paths and cases in util.vacuum_to_air().
    Test conversion for input wavelengths in Angstroms and nm.
    Test wavelength above and below 2000 Angstroms.
    Test scalar wavelength and list of wavelengths.
    Allow discrepancies smaller than 1e-8 Angstroms.
    """
    wvac_a = [1999, 5000, 5010]
    wair_a = [1999, 4998.605522013399, 5008.602864587058]
    for wv, wa in zip(wvac_a, wair_a):
        assert abs(vacuum_to_air(wv, 'A') - wa) < 1e-8
    wair_nm = vacuum_to_air([w / 10 for w in wvac_a], 'nm')
    assert all([abs(10 * wnm - wa) < 1e-8 for wnm, wa in zip(wair_nm, wair_a)])

def test_vacuum_angstroms():
    """Test code paths and cases in util.vacuum_angstroms().
    """
    win  = [5000, 20000, 500]
    uin  = ['A' , 'cm-1', 'nm']
    for w, u in zip(win, uin):
        assert vacuum_angstroms(w, u, 'vac') == 5000
        assert vacuum_angstroms(w, u, 'air') == 5001.39484863807
    with raises(ValueError, match='invalid waveunit specified'):
        vacuum_angstroms(5000, 'erg', 'vac')
    with raises(ValueError, match='invalid medium specified'):
        vacuum_angstroms(5000, 'A', 'water')

