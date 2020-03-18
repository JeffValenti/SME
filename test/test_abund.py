import pytest
from sme.abund import Abund


pattern_names = ['Asplund2009', 'Grevesse2007', 'Empty']
types = ['H=12', 'n/nH', 'n/nTot', 'SME']


def test_init_with_too_few_args():
    """Test that __init__ raise an error if too few arguments are passed.
    """
    # Passing zero arguments to __init__() raises an error.
    with pytest.raises(TypeError):
        Abund()

    # Passing one argument to __init__() raises an error.
    with pytest.raises(TypeError):
        Abund(0)


def test_init_using_pattern_names():
    """Test handling of abundance pattern name passed to __init__().
    """
    # Each abundance pattern name yields an Abund object.
    for pattern_name in pattern_names:
        assert isinstance(Abund(0, pattern_name), Abund)

    # The 'Empty' abundance pattern has a value of None for all elements.
    abund = Abund(0, 'Empty')
    assert all([value is None for value in abund.pattern.values()])

    # An invalid abundance pattern name raises an error.
    with pytest.raises(ValueError):
        Abund(0, 'INVALID')


def test_call_returns_abund_in_odict():
    """Test return value, which is an ordered dictionary with element
    abbreviations as the keys and abundances as the values.
    """
    abund = Abund(0, pattern_names[0])
    assert isinstance(abund(), dict)
    assert tuple(abund().keys()) == abund.elements


def test_getitem_returns_abund_values():
    """Test getitem method, which return computed abundance values for
    the specified element or list of elements.
    """
    abund = Abund(0, pattern_names[0])
    assert abund['H'] == 12


def test_monh_property_set_and_get():
    """Test setting and getting monh property. Set converts input to float.
    """
    # Input str convertable to float yields a float with the specified value.
    abund = Abund('-6e-1', pattern_names[0])
    assert isinstance(abund.monh, float)
    assert abund.monh == -0.6

    # Input int yields a float with the specified value.
    abund.monh = -2
    assert isinstance(abund.monh, float)
    assert abund.monh == -2.0

    # Input float yields a float with the specified value.
    abund.monh = 0.3
    assert isinstance(abund.monh, float)
    assert abund.monh == 0.3

    # Input str that cannot be converted to float raises an error.
    with pytest.raises(ValueError):
        abund = Abund('ABC', pattern_names[0])

    # Input that is not a string or a number raises an error.
    with pytest.raises(TypeError):
        abund.monh = []


def test_pattern_property_set_and_get():
    """Test setting and getting pattern property. Set is not allowed.
    """
    # Raise error is user tries to set pattern
    abund = Abund(0, 'Empty')
    with pytest.raises(AttributeError):
        abund.pattern = 0.0


def test_update_pattern():
    """Test behavior of update_pattern(), which modifies values in _pattern
    for the specified element(s).
    """
    # Update for one element yields float with the specified value.
    abund = Abund(0, 'Empty')
    assert not abund.pattern['Fe']
    abund.update_pattern({'Fe': '3.14'})
    assert isinstance(abund.pattern['Fe'], float)
    assert abund.pattern['Fe'] == 3.14

    # Update for two elements yields floats with the specified values.
    abund.update_pattern({'C': 8.4, 'F': 5})
    assert isinstance(abund.pattern['C'], float)
    assert isinstance(abund.pattern['F'], float)
    assert abund.pattern['C'] == 8.4
    assert abund.pattern['F'] == 5.0


def test_totype_fromtype():
    """Test behavior of totype() and fromtype(), which are static methods
    that return a copy of the input abundance pattern transformed to or
    from the specified abudnance pattern type.
    """
    # Round trip tests that compare copy=fromtype(totype()) with original.
    orig = Abund(0, pattern_names[0]).pattern
    for type in types:
        copy = Abund.fromtype(Abund.totype(orig, type), type)
        # Same elements in the same order for full dictionary.
        assert copy.keys() == orig.keys()
        # Same elements have abundance defined (!= None).
        o = dict((k, v) for k, v in orig.items() if v)
        c = dict((k, v) for k, v in copy.items() if v)
        assert c.keys() == o.keys()
        # Logarithmic abundances differ by less than 1e-10.
        assert all([abs(o[k]-c[k]) < 1e-10 for k in o])
        # Lowercase type yields same result as mixed case type.
        type_lc = type.lower()
        assert copy == Abund.fromtype(Abund.totype(orig, type_lc), type_lc)

    # Invalid abundance pattern type raises error.
    with pytest.raises(ValueError):
        copy = Abund.totype(orig, 'INVALID')
    with pytest.raises(ValueError):
        copy = Abund.fromtype(orig, 'INVALID')
