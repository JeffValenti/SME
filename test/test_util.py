from pytest import raises
from sme.util import change_waveunit


def test_change_waveunit():
    """Test code paths and cases in util.change_waveunit().
    """
    cases = {
        'a': 5000, 'nm': 500, 'um': 0.5, 'micron': 0.5,
        'cm-1': 20000, '1/cm': 20000}
    for unit in cases:
        wave = cases[unit]
        assert change_waveunit(5000, 'A', unit) == wave
        assert change_waveunit(5000, 'A', unit.upper()) == wave
        assert change_waveunit(wave, unit, 'A') == 5000
        assert change_waveunit(wave, unit.upper(), 'A') == 5000
    with raises(ValueError, match='Invalid waveunit specified'):
        change_waveunit(5000, 'A', 'erg')
    with raises(ValueError, match='Invalid waveunit specified'):
        change_waveunit(5000, 'erg', 'A')
