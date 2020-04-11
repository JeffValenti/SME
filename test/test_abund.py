from pytest import raises
from sme.abund import AbundError, Abund, AbundPattern, to_H12, from_H12


pattern_names = ['Asplund2009', 'Grevesse2007', 'Empty']
norms = ['H=12', 'n/nH', 'n/nTot', 'SME']


def test_from_h12_to_h12():
    """Code coverage tests for AbundPattern() class and methods.

    Test that to_H12(from_H12()) round trip is identity operation.
    Test == and != operators with AbundPattern as left and right argument.
    """
    pattern = AbundPattern(pattern_names[0])
    for norm in norms:
        altnorm = from_H12(pattern, norm)
        if norm != 'H=12':
            assert altnorm != pattern
            assert pattern != altnorm
            assert not (altnorm == pattern)
            assert not (pattern == altnorm)
        stdnorm = to_H12(altnorm, norm)
        assert pattern == stdnorm
        assert stdnorm == pattern
        assert not (pattern != stdnorm)
        assert not (stdnorm != pattern)
    nohydrogen = dict(pattern)
    nohydrogen.pop('H')
    for function in [from_H12, to_H12]:
        with raises(AbundError, match='norm must be a string'):
            function(pattern, None)
        with raises(AbundError, match='unknown abundance normalization'):
            function(pattern, 'unknown')
        with raises(AbundError, match='must define abundance of H'):
            function(nohydrogen, 'H=12')


def test_abund():
    """Code coverage tests for Abund() class and methods.
    """
    abund = Abund(0, pattern_names[0])
    for name in pattern_names:
        a1 = Abund(0, name)
        a2 = Abund(0, name.lower())
        a3 = Abund(-0.1, Abund(0.1, name).abund, 'H=12')
        a1.__repr__()
        a2.__str__()
        abund.compare(a3)
        a3.compare(abund)
        assert len(a1) == len(a1.elements) == 99
        assert a1 is not a2
        assert a1 == a2
        assert tuple(a1.keys()) == a1.elements
        assert list(a1.values()) == [a1[k] for k, v in a1.items()]
        if a2['Fe'] is not None:
            for norm in norms:
                a1 == to_H12(a1.normalized(norm), norm)
                a1 == to_H12(a1.normalized(norm, prune=True), norm)
            a2.pattern['Fe'] = a1.pattern['Fe'] + 0.999e-4
            assert a1['Fe'] != a2['Fe']
            assert a1 == a2
            assert not a1 != a2
            a2.pattern['Fe'] = a1.pattern['Fe'] + 1.001e-4
            assert a1 != a2
            a2.pattern['Fe'] = None
            assert a1 != a2
            assert a1 != 'wrong object type'
            assert all(
                [abs(a1[k] - a3[k]) < 1e-8 for k in a1.elements if a1[k]])
            assert a1 != a3
    with raises(AbundError, match='set monh and pattern separately'):
        abund['C'] = 8.2
    with raises(ValueError, match='could not convert string'):
        abund.monh = 'text'
    with raises(AbundError, match='must be an AbundPattern object'):
        abund.pattern = {'H': 12, 'He': 11, 'Li': 1}
    with raises(AbundError, match='unknown element key'):
        abund['Water']
    with raises(AbundError, match='unknown element key'):
        abund.pattern._pattern['Water'] = 5
        abund.abund()


def test_abundpattern():
    """Code coverage tests for AbundPattern() class and methods.

    Test that abundance difference less than 1e-4 is treated as equal.
    Test that abundance difference greater than 1e-4 is treated as unequal.
    """
    for name in pattern_names:
        ap1 = AbundPattern(name)
        ap2 = AbundPattern(name.lower())
        ap1.__repr__()
        ap1.__str__()
        assert len(ap1) == len(ap1.elements) == 99
        assert ap1 is not ap2
        assert ap1 == ap2
        if ap2['Fe'] is not None:
            for norm in norms:
                ap1 == to_H12(ap1.normalized(norm), norm)
                ap1 == to_H12(ap1.normalized(norm, prune=True), norm)
            ap2['Fe'] = ap1['Fe'] + 0.999e-4
            assert ap1['Fe'] != ap2['Fe']
            assert ap1 == ap2
            assert not ap1 != ap2
            ap2['Fe'] = ap1['Fe'] + 1.001e-4
            assert ap1 != ap2
            ap2['Fe'] = None
            assert ap1 != ap2
            assert ap1 != 'wrong object type'
    name = pattern_names[0]
    pattern = AbundPattern(name)
    with raises(AbundError, match='normalization not allowed'):
        AbundPattern(name, 'H=12')
    with raises(AbundError, match='normalization required'):
        AbundPattern(pattern)
    with raises(AbundError, match='unknown abundance pattern name'):
        AbundPattern('unknown name')
    with raises(AbundError, match='unknown element key'):
        pattern['Water']
    with raises(AbundError, match='unknown element key'):
        pattern['Water'] = 7.0
    with raises(AbundError, match='cannot convert'):
        pattern['C'] = 'value that cannot be converted to float'
    with raises(AbundError, match='H abundance must be 12'):
        pattern['H'] = None
    with raises(AbundError, match='H abundance must be 12'):
        pattern['H'] = 11.0
