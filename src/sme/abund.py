from collections.abc import Sequence
from math import log10


def from_H12(input_pattern, output_norm):
    """Copy input abundance pattern and convert from H=12 normalization.

    Parameters
    ----------
    input_pattern : dict-like object, including Abund or AbundPattern object
        abundance pattern normalized so that H=12
    output_norm : str
        normalization type: 'H=12', 'n/nH', 'n/nTot', and 'sme'

    Returns
    -------
    dict-like object
        abundance pattern normalized as specified in output_norm
    """
    pattern = {k: v for k, v in input_pattern.items()}
    try:
        norm = output_norm.lower()
    except AttributeError:
        raise AbundError('output_norm must be a string')
    elem = [el for el, ab in pattern.items() if ab]
    if elem[0] != 'H':
        raise AbundError('pattern must define abundance of H')
    if norm == 'h=12':
        return pattern
    newpatt = [pattern[el] for el in elem]
    if norm == 'sme':
        newpatt = [10**(ab-12) for ab in newpatt]
        absum = sum(newpatt)
        newpatt = [ab/absum for ab in newpatt]
        newpatt = [newpatt[0]] + [log10(ab) for ab in newpatt[1:]]
    elif norm == 'n/ntot':
        newpatt = [10**(ab-12) for ab in newpatt]
        absum = sum(newpatt)
        newpatt = [ab/absum for ab in newpatt]
    elif norm == 'n/nh':
        newpatt = [10**(ab-12) for ab in newpatt]
    else:
        raise AbundError(
            f"unknown abundance normalization: {norm}\n" +
            "Known normalizations: 'H=12', 'n/nH', 'n/nTot', 'sme'")
    for iel, el in enumerate(elem):
        pattern[el] = newpatt[iel]
    return pattern


def to_H12(input_pattern, input_norm):
    """Copy input abundance pattern and convert to H=12 normalization.

    Parameters
    ----------
    input_pattern : dict-like object, including Abund or AbundPattern object
        abundance pattern normalized as specified in input_norm
    input_norm : str
        normalization type: 'H=12', 'n/nH', 'n/nTot', and 'sme'

    Returns
    -------
    dict-like object
        abundance pattern normalized such that hydrogen abundance is 12

    Raises
    ------
    AbundError
        Raised if `input_norm` is not a known normalization type
    """
    pattern = {k: v for k, v in input_pattern.items()}
    try:
        norm = input_norm.lower()
    except AttributeError:
        raise AbundError('input_norm must be a string')
    elem = [el for el, ab in pattern.items() if ab]
    if elem[0] != 'H':
        raise AbundError('pattern must define abundance of H')
    if norm == 'h=12':
        return pattern
    newpatt = [pattern[el] for el in elem]
    if norm == 'sme':
        newpatt = [newpatt[0]] + [10**ab for ab in newpatt[1:]]
        newpatt = [ab/newpatt[0] for ab in newpatt]
        newpatt = [12+log10(ab) for ab in newpatt]
    elif norm == 'n/ntot':
        newpatt = [ab/newpatt[0] for ab in newpatt]
        newpatt = [12+log10(ab) for ab in newpatt]
    elif norm == 'n/nh':
        newpatt = [12+log10(ab) for ab in newpatt]
    else:
        raise AbundError(
            f"unknown abundance normalization: '{norm}'\n" +
            "Known normalizations: 'H=12', 'n/nH', 'n/nTot', 'sme'")
    for iel, el in enumerate(elem):
        pattern[el] = newpatt[iel]
    return pattern


class AbundError(Exception):
    """Raise when abundance specification is invalid.
    """


class Abund(Sequence):
    """Manage elemental abundances via metallicity and an abundance pattern.

    Attributes
    ----------
    monh : float
        Metallicity, [M/H], which is the logarithmic offset that will be
        added to logarithmic abundances specified in pattern, for all
        elements except hydrogen.
    input_norm : {'H=12', 'n/nTot', 'n/nH', 'sme'}
        Valid abundance pattern normalizations are:

        * 'H=12' - Abundance values are log10 of the fraction of nuclei of
          each element in any form, relative to the number of hydrogen
          nuclei in any form plus an offset of 12. For the Sun, the
          abundance values of H, He, and Li are approximately 12, 10.9,
          and 1.05.
        * 'n/nH' - Abundance values are the fraction of nuclei of each element
          in any form, relative to the number of hydrogen nuclei in any
          form. For the Sun, the abundance values of H, He, and Li are
          approximately 1, 0.085, and 1.12e-11.
        * 'n/nTot' - Abundance values are the fraction of nuclei of each
          element in any form, relative to the total for all elements in any
          form. For the Sun, the abundance values of H, He, and Li are
          approximately 0.92, 0.078, and 1.03e-11.
        * 'sme' - For hydrogen, the abundance value is the fraction of all
          nuclei that are hydrogen. For the other elements, the
          abundance values are log10 of the fraction of nuclei of each
          element in any form, relative to the total for all elements
          in any form. For the Sun, the abundance values of H, He, and
          Li are approximately 0.92, -1.11, and -11.0.

    Returns
    -------
    Abund object, which a Sequence subclass
        Abundances calculated by applying metallicity to abundance pattern
    """
    def __init__(self, monh, name_or_pattern, input_norm=None):
        self.monh = monh
        self.pattern = AbundPattern(name_or_pattern, input_norm)

    def __len__(self):
        return len(self.abund)

    def __eq__(self, other):
        """Test whether specified abundances equal abundances in this object.

        Parameters
        ----------
        other : sme.abund.Abund object
            Other abundances that will be compared to abundances in this object

        Returns
        -------
        boolean
            True if abundances are defined for the same elements and
            are equal to within 0.0001 dex in the H=12 representation.
        """
        if isinstance(other, Abund):
            return self.monh == other.monh and self.pattern == other.pattern
        return False

    def __getitem__(self, elem, output_norm='H=12'):
        try:
            return self.abund[elem]
        except KeyError:
            raise AbundError(f"unknown element key: '{elem}'")

    def __setitem__(self, elem, abund):
        raise AbundError(
            'set monh and pattern separately, not the combination')

    def __repr__(self):
        return f"Abund({self.monh}, {str(dict(self.pattern))}, 'H=12')"

    def __str__(self, norm='H=12'):
        out = 'Abundances obtained by applying [M/H]=' \
            f'{self.monh:.3f} to the abundance pattern.\n'
        abund = self.abund
        keys = list(abund.keys())
        values = list(self.values())
        for i in range(9):
            for j in range(11):
                out += f'  {keys[11*i+j]:<5s}'
            out += '\n'
            for j in range(11):
                if values[11*i+j]:
                    out += f'{values[11*i+j]:7.3f}'
                else:
                    out += '  None '
            if i < 8:
                out += '\n'
        return out

    def keys(self):
        return self.abund.keys()

    def values(self):
        return self.abund.values()

    def items(self):
        return self.abund.items()

    @property
    def abund(self):
        """Get abundances, calculated by combining metallicity and pattern.

        Returns
        -------
        dict - elemental abundances with the H=12 normalization
        """
        abund = {}
        for key, value in self.pattern.items():
            if key not in self.elements:
                raise AbundError(f'unknown element key: {key}')
            if value is None:
                abund[key] = value
            else:
                try:
                    if key in ('H', 'He'):
                        abund[key] = float(value)
                    else:
                        abund[key] = float(value) + self.monh
                except ValueError:
                    raise AbundError(
                        'cannot convert abundance value to a float')
        return abund

    @property
    def elements(self):
        """Return the standard abbreviation for each element.
        """
        return self.pattern.elements

    @property
    def monh(self):
        """Metallicity, [M/H], that will be applied to the abundance pattern.
        """
        return self._monh

    @monh.setter
    def monh(self, monh):
        """Set metallicity that will be applied to the abundance pattern.

        Metallicity, [M/H], is a logarithmic offset that will be added to
        the abundance pattern for all elements except hydrogen and helium.

        Parameters
        ----------
        monh : float or valid argument for the float() function
            metallicity that will be applied to the abundance pattern
        """
        self._monh = float(monh)

    @property
    def pattern(self):
        """Elemental abundance pattern with H=12 normalization convention.
        """
        return self._pattern

    @pattern.setter
    def pattern(self, pattern):
        """Set abundance pattern. Value must be AbundPattern object.
        """
        if not isinstance(pattern, AbundPattern):
            raise AbundError('pattern must be an AbundPattern object')
        self._pattern = pattern

    def normalized(self, output_norm='H=12', prune=False):
        """Return abundances with the requested normalization.

        Parameters
        ----------
        output_norm : str
            normalization type: 'H=12', 'n/nH', 'n/nTot', and 'sme'
        prune : boolean
            if True, remove items where the abundance value is None
        """
        abund = from_H12(self.abund, output_norm)
        if prune:
            return {k: v for k, v in abund.items() if v is not None}
        else:
            return abund

    def compare(self, refabund):
        """Return string comparing two sets of elemental abundances.

        Parameters
        ----------
        refabund : Abund object
            reference abundances that will be compared to current abundances
        """
        abund = self.abund
        keys = list(abund.keys())
        a = list(self.values())
        r = list(refabund.values())
        out = f'A: Abundances: [M/H]={self.monh:.3f}' \
            ' applied to the abundance pattern.\n' \
            f'R: Reference abundances: [M/H]={refabund.monh:.3f}' \
            ' applied to reference abundance pattern.\n' \
            'D: Differences: Abundances minus Reference abundances (A-R).\n'
        for i in range(9):
            out += '  '
            for j in range(11):
                out += f'  {keys[11*i+j]:<5s}'
            out += '\nA:'
            for j in range(11):
                if a[11*i+j]:
                    out += f'{a[11*i+j]:7.3f}'
                else:
                    out += '  None '
            out += '\nR:'
            for j in range(11):
                if r[11*i+j]:
                    out += f'{r[11*i+j]:7.3f}'
                else:
                    out += '  None '
            out += '\nD:'
            for j in range(11):
                if a[11*i+j] and r[11*i+j]:
                    if abs(a[11*i+j] - r[11*i+j]) > 1e-3:
                        out += f'{a[11*i+j] - r[11*i+j]:7.3f}'
                    else:
                        out += '       '
                elif a[11*i+j] is None and r[11*i+j] is None:
                    out += '       '
                else:
                    out += '  ^^^^ '
            if i < 8:
                out += '\n'
        return out


class AbundPattern(Sequence):
    """A pattern of abundances that is usually scaled by a metallicity.

    Values contain a pattern of abundances on the H=12 scale.
    Pattern is usually scaled by metallicity to obtain abundances.
    Subclass of the standard dict class.
    Keys are standard element abbreviations (e.g., 'Fe').
    Values are floats (e.g., 12.000 for hydrogen).

    Example
    -------
    >>> from sme.abund import AbundPattern
    >>> g = AbundPattern('Grevesse2007')
    """

    _elem = (
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es'
        )

    # Asplund, Grevesse, Sauval, Scott (2009, ARAA, 47, 481)
    _asplund2009 = (
        12.00, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,
        6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,
        3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56,
        3.04, 3.65, 2.30, 3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58,
        1.46, 1.88, None, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,
        1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58, 0.72, 1.42,
        None, 0.96, 0.52, 1.07, 0.30, 1.10, 0.48, 0.92, 0.10, 0.84,
        0.10, 0.85, -0.12, 0.85, 0.26, 1.40, 1.38, 1.62, 0.92, 1.17,
        0.90, 1.75, 0.65, None, None, None, None, None, None, 0.02,
        None, -0.54, None, None, None, None, None, None, None
        )

    # Grevesse, Asplund, Sauval (2007, Space Science Review, 130, 105)
    _grevesse2007 = (
        12.00, 10.93, 1.05, 1.38, 2.70, 8.39, 7.78, 8.66, 4.56, 7.84,
        6.17, 7.53, 6.37, 7.51, 5.36, 7.14, 5.50, 6.18, 5.08, 6.31,
        3.17, 4.90, 4.00, 5.64, 5.39, 7.45, 4.92, 6.23, 4.21, 4.60,
        2.88, 3.58, 2.29, 3.33, 2.56, 3.25, 2.60, 2.92, 2.21, 2.58,
        1.42, 1.92, None, 1.84, 1.12, 1.66, 0.94, 1.77, 1.60, 2.00,
        1.00, 2.19, 1.51, 2.24, 1.07, 2.17, 1.13, 1.70, 0.58, 1.45,
        None, 1.00, 0.52, 1.11, 0.28, 1.14, 0.51, 0.93, 0.00, 1.08,
        0.06, 0.88, -0.17, 1.11, 0.23, 1.25, 1.38, 1.64, 1.01, 1.13,
        0.90, 2.00, 0.65, None, None, None, None, None, None, 0.06,
        None, -0.52, None, None, None, None, None, None, None
        )

    def __init__(self, name_or_pattern, input_norm=None):
        """Create an abundance pattern object with the specified pattern.

        Parameters
        ----------
        name_or_pattern : str or dict-like object
            abundance pattern name or abudance pattern values
        input_norm : None or str
            None if name specified or normalization type if pattern specified

        Returns
        -------
        AbundPattern object

        If `name_or_pattern` has lower() method, assume it is a pattern name.
        Otherwise, assume `name_or_value` contains abundance pattern values.
        Require `input_norm` when pattern is specified. Otherwise, forbid.
        """
        self._pattern = {}
        if hasattr(name_or_pattern, 'lower'):
            if input_norm is None:
                self.named_pattern(name_or_pattern)
            else:
                raise AbundError(
                    'normalization not allowed with named abundance pattern')
        else:
            if input_norm is None:
                raise AbundError(
                    'normalization required with custom abundance pattern')
            else:
                self.custom_pattern(name_or_pattern, input_norm)

    def __len__(self):
        return len(self._pattern)

    def __eq__(self, other):
        """Test whether self and other abundance patterns are equivalent.

        Parameters
        ----------
        other : dict-like object, including Abund or AbundPattern object
            Other abundance pattern to compare to self.

        Returns
        -------
        boolean
            True if abundances are defined for the same elements and
            are equal to within 0.0001 dex in the H=12 representation.
        """
        if not hasattr(other, 'items'):
            return False
        keys = [k for k, v in self.items() if v is not None]
        if [k for k, v in other.items() if v is not None] != keys:
            return False
        return all([abs(self[k] - other[k]) < 1e-4 for k in keys])

    def __getitem__(self, elem):
        try:
            return self._pattern[elem]
        except KeyError:
            raise AbundError(f"unknown element key: '{elem}'")

    def __setitem__(self, elem, value):
        """Set value of abundance pattern for the specified element.

        Update`self[elem]` item in `self`, which is a dict-like object.
        Check that `elem` is a known element abbreviation.
        Check that if `elem` is 'H', then `float(value)` is 12.0000.
        Check that `value` is `None` or `float(value)` is a float.

        Parameters
        ----------
        elem : str
            Element key, e.g., 'H', 'He', 'Li', 'Be', 'B', 'C', ...
        value : object that can be coverted to float by float()
            Abundance pattern value in the H=12 scale.

        Exceptions
        ----------
        Raises AbundError abundance pattern specification is invalid.
        """
        if elem not in self._elem:
            raise AbundError(f"unknown element key: '{elem}'")
        if value is None:
            if elem == 'H':
                raise AbundError('H abundance must be 12 on H=12 scale')
            self._pattern[elem] = value
        else:
            try:
                float_value = float(value)
            except ValueError:
                raise AbundError('cannot convert abundance value to a float')
            if elem == 'H' and abs(float_value - 12) > 1e-5:
                raise AbundError('H abundance must be 12 on H=12 scale')
            self._pattern[elem] = float_value

    def __repr__(self):
        return f"AbundPattern({str(self._pattern)}, 'H=12')"

    def __str__(self):
        """Print alternating rows of element names and abundances.
        """
        out = ''
        keys = list(self.keys())
        values = list(self.values())
        for i in range(9):
            for j in range(11):
                out += f'  {keys[11*i+j]:<5s}'
            out += '\n'
            for j in range(11):
                if values[11*i+j]:
                    out += f'{values[11*i+j]:7.3f}'
                else:
                    out += '  None '
            if i < 8:
                out += '\n'
        return out

    def keys(self):
        return self._pattern.keys()

    def values(self):
        """Return list of abundances in pattern.
        """
        return self._pattern.values()

    def items(self):
        """Return pattern as list of (element, abundance) tuples.
        """
        return self._pattern.items()

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @property
    def elements(self):
        return self._elem

    def normalized(self, output_norm='H=12', prune=False):
        """Return abundances with the requested normalization.

        Parameters
        ----------
        output_norm : str
            normalization type: 'H=12', 'n/nH', 'n/nTot', and 'sme'
        prune : boolean
            if True, remove items where the abundance value is None
        """
        abund = from_H12(self, output_norm)
        if prune:
            return {k: v for k, v in abund.items() if v is not None}
        else:
            return abund

    def named_pattern(self, name):
        """Set abundance pattern to standard values for the specified name.

        Known pattern names: 'Asplund2009', 'Grevesse2007', 'Empty'.
        The 'Empty' pattern creates items for 88 elements from 'H' to 'Es'.

        Parameters
        ----------
        name : str
            identifier for a known abundance pattern, case insensivite

        Raises
        ------
        AbundError
            Raised if `name` is not a known pattern name.
        """
        if name.lower() == 'asplund2009':
            self.update(dict(zip(self._elem, self._asplund2009)))
        elif name.lower() == 'grevesse2007':
            self.update(dict(zip(self._elem, self._grevesse2007)))
        elif name.lower() == 'empty':
            values = [12] + [None]*(len(self._elem) - 1)
            self.update(dict(zip(self._elem, values)))
        else:
            raise AbundError(
                f"unknown abundance pattern name: '{name}'\n"
                f"Known pattern names: 'Asplund2009', 'Grevesse2007', 'Empty'")

    def custom_pattern(self, pattern, input_norm):
        """Set custom abundance pattern from input pattern and normalization.

        Valid input normalizations: 'H=12', 'n/nH', 'n/nTot', 'sme'
        Internally, pattern is stored with H=12 normalization.
        Reset all abundances: 12 for 'H', None for elements 'He' through 'Es'.
        Update individual abundances based on items in pattern.

        Parameters
        ----------
        pattern : dict-like object with keys that are element abbreviations
            abundance pattern with normalization specified in `input_norm`
        input_norm : str
            abundance normalization type, case insensitive
        """
        self.named_pattern('empty')
        self.update(to_H12(pattern, input_norm))
