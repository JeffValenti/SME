import math


class Abund:
    """Manage elemental abundances via metallicity and an abundance pattern.

    Attributes
    ----------
    monh : float
        Metallicity, [M/H], which is the logarithmic offset that will be
        added to logarithmic abundances specified in pattern, for all
        elements except hydrogen.
    type : {'H=12', 'n/nTot', 'n/nH', 'sme'}
        Valid abundance pattern types are:

        * 'H=12' - Abundance values are log10 of the fraction of nuclei of
          each element in any form relative to the number of hydrogen
          in any form plus an offset of 12. For the Sun, the nuclei
          abundance values of H, He, and Li are approximately 12,
          10.9, and 1.05.
        * 'n/nTot' - Abundance values are log10 of the fraction of nuclei
          of each element in any form relative to the total for all
          elements in any form. For the Sun, the abundance values of
          H, He, and Li are approximately 0.92, 0.078, and 1.03e-11.
        * 'n/nH' - Abundance values are log10 of the fraction of nuclei
          of each element in any form relative to the number of
          hydrogen nuclei in any form. For the Sun, the abundance
          values of H, He, and Li are approximately 1, 0.085, and
          1.12e-11.
        * 'sme' - For hydrogen, the abundance value is the fraction of all
          nuclei that are hydrogen, including all ionization states
          and treating molecules as constituent atoms. For the other
          elements, the abundance values are log10 of the fraction of
          nuclei of each element in any form relative to the total for
          all elements in any form. For the Sun, the abundance values
          of H, He, and Li are approximately 0.92, -1.11, and -11.0.

    Returns
    -------
    dict
        Keys are element names. Values are abundances computed by adding
        [M/H] to abundance pattern.
    """
    def __init__(self, monh, pattern, type=None):
        self.monh = monh
        if isinstance(pattern, str):
            self.set_pattern_by_name(pattern)
        else:
            self.set_pattern_by_value(pattern, type)

    def __call__(self, type='H=12'):
        """Return abundances for all elements.
        Apply current [M/H] value to the current abundance pattern.
        Transform the resulting abundances to the requested abundance type.
        """
        abund = dict(
            (el, ab+self._monh if ab is not None else ab)
            for el, ab in self._pattern.items()
            )
        for el in ['H', 'He']:
            abund[el] = self._pattern[el]
        return self.totype(abund, type)

    def __getitem__(self, elems):
        abund = self.__call__()
        if isinstance(elems, str):
            try:
                return abund[elems]
            except KeyError:
                msg = "got element abbreviation '{}', should be one of "
                raise KeyError(msg.format(elems) + ', '.join(abund.keys()))
        else:
            abunds = []
            try:
                for elem in elems:
                    abunds.append(abund[elem])
            except TypeError:
                raise TypeError(
                    "got item descriptor '{}'".format(elem) +
                    ", should an element abbreviation or a list of" +
                    " element abbreviations"
                    )
            except KeyError:
                msg = "got element abbreviation '{}', should be one of "
                raise KeyError(msg.format(elem) + ', '.join(abund.keys()))
            return abunds

    def __setitem__(self, elem, abund):
        raise TypeError(
            "can't set abundance directly; " +
            "instead set monh and pattern separately"
            )

    def __str__(self):
        a = list(self.get_pattern('H=12').values())
        for i in range(2, len(a)-1):
            if a[i]:
                a[i] += self._monh
        out = 'Abundances obtained by applying [M/H]=' \
            f'{self.monh:.3f} to the abundance pattern.\n'
        for i in range(9):
            for j in range(11):
                out += f'  {self._elem[11*i+j]:<5s}'
            out += '\n'
            for j in range(11):
                if a[11*i+j]:
                    out += f'{a[11*i+j]:7.3f}'
                else:
                    out += '  None '
            if i < 8:
                out += '\n'
        return out

    @staticmethod
    def fromtype(pattern, fromtype):
        """Return a copy of the input abundance pattern, transformed from
        the input type to the 'H=12' type. Valid abundance pattern types
        are 'sme', 'n/nTot', 'n/nH', and 'H=12'.
        """
        if fromtype is None:
            raise ValueError(f'invalid abundance pattern type: {fromtype}')
        patt = pattern.copy()
        type = fromtype.lower()
        if type == 'h=12':
            return patt
        elem = [el for el, ab in patt.items() if ab]
        if elem[0] != 'H':
            raise ValueError('pattern must define abundance of H')
        abund = [patt[el] for el in elem]
        if type == 'sme':
            abund = [abund[0]] + [10**ab for ab in abund[1:]]
            abund = [ab/abund[0] for ab in abund]
            abund = [12+math.log10(ab) for ab in abund]
        elif type == 'n/ntot':
            abund = [ab/abund[0] for ab in abund]
            abund = [12+math.log10(ab) for ab in abund]
        elif type == 'n/nh':
            abund = [12+math.log10(ab) for ab in abund]
        else:
            raise ValueError(
                f"invalid abundance type: {type}\n" +
                "Allowed abundance types: 'H=12', 'n/nH', 'n/nTot', 'sme'")
        for iel, el in enumerate(elem):
            patt[el] = abund[iel]
        return patt

    @staticmethod
    def totype(pattern, totype):
        """Return a copy of the input abundance pattern, transformed from
        the 'H=12' type to the output type. Valid abundance pattern types
        are 'sme', 'n/nTot', 'n/nH', and 'H=12'.
        """
        patt = pattern.copy()
        type = totype.lower()
        if type == 'h=12':
            return patt
        elem = [el for el, ab in patt.items() if ab]
        abund = [patt[el] for el in elem]
        if elem[0] != 'H':
            raise ValueError('pattern must define abundance of H')
        if type == 'sme':
            abund = [10**(ab-12) for ab in abund]
            absum = sum(abund)
            abund = [ab/absum for ab in abund]
            abund = [abund[0]] + [math.log10(ab) for ab in abund[1:]]
        elif type == 'n/ntot':
            abund = [10**(ab-12) for ab in abund]
            absum = sum(abund)
            abund = [ab/absum for ab in abund]
        elif type == 'n/nh':
            abund = [10**(ab-12) for ab in abund]
        else:
            raise ValueError(
                "got abundance type '{}',".format(type) +
                " should be 'H=12', 'n/nH', 'n/nTot', or 'sme'"
                )
        for iel, el in enumerate(elem):
            patt[el] = abund[iel]
        return patt

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

    """Asplund, Grevesse, Sauval, Scott (2009,  Annual Review of Astronomy
    and Astrophysics, 47, 481)
    """
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

    """Grevesse, Asplund, Sauval (2007, Space Science Review, 130, 105)
    """
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

    @property
    def elements(self):
        """Return the standard abbreviation for each element.
        Use property so user will not redefine elements.
        """
        return self._elem

    @property
    def monh(self):
        return self._monh

    @monh.setter
    def monh(self, monh):
        """Set [M/H] metallicity, which is the logarithmic offset added to
        the abundance pattern for all elements except hydrogen and helium.
        """
        self._monh = float(monh)

    @property
    def pattern(self):
        return self._pattern

    def set_pattern_by_name(self, pattern_name):
        if pattern_name.lower() == 'asplund2009':
            self._pattern = dict(zip(self._elem, self._asplund2009))
        elif pattern_name.lower() == 'grevesse2007':
            self._pattern = dict(zip(self._elem, self._grevesse2007))
        elif pattern_name.lower() == 'empty':
            self._pattern = self.empty_pattern()
        else:
            raise ValueError(
                "got abundance pattern name '{}',".format(pattern_name) +
                " should be 'Asplund2009', 'Grevesse2007', 'empty'."
                )

    def set_pattern_by_value(self, pattern, type):
        self._pattern = self.fromtype(pattern, type)

    def update_pattern(self, updates):
        for key in updates:
            if key in self._pattern.keys():
                self._pattern[key] = float(updates[key])
            else:
                raise KeyError(
                    "got element abbreviation '{}'".format(key) +
                    ", should be one of " +
                    ", ".join(self._pattern.keys())
                    )

    def get_pattern(self, type='sme'):
        """Transform the specified abundance pattern from type used
        internally by SME to the requested type. Valid abundance pattern
        types are:

        Parameters
        ----------
        type : {'sme', 'n/nTot', 'n/nH', 'H=12'}

        * 'sme' - For hydrogen, the abundance value is the fraction of
          all nuclei that are hydrogen, including all ionization states
          and treating molecules as constituent atoms. For the other
          elements, the abundance values are log10 of the fraction of
          nuclei of each element in any form relative to the total for
          all elements in any form. For the Sun, the abundance values
          of H, He, and Li are approximately 0.92, -1.11, and -11.0.
        * 'n/nTot' - Abundance values are log10 of the fraction of nuclei
          of each element in any form relative to the total for all
          elements in any form. For the Sun, the abundance values of
          H, He, and Li are approximately 0.92, 0.078, and 1.03e-11.
        * 'n/nH' - Abundance values are log10 of the fraction of nuclei
          of each element in any form relative to the number of
          hydrogen nuclei in any form. For the Sun, the abundance
          values of H, He, and Li are approximately 1, 0.085, and
          1.12e-11.
        * 'H=12' - Abundance values are log10 of the fraction of nuclei of
          each element in any form relative to the number of hydrogen
          in any form plus an offset of 12. For the Sun, the nuclei
          abundance values of H, He, and Li are approximately 12,
          10.9, and 1.05.
        """
        return self.totype(self._pattern, type)

    def empty_pattern(self):
        """Return an abundance pattern with value None for all elements.
        """
        return dict.fromkeys(self._elem)

    def compare(self, refabund):
        a = list(self.get_pattern('H=12').values())
        for i in range(2, len(a)-1):
            if a[i]:
                a[i] += self._monh
        r = list(refabund.get_pattern('H=12').values())
        for i in range(2, len(r)-1):
            if r[i]:
                r[i] += refabund._monh
        out = f'A: Abundances: [M/H]={self.monh:.3f}' \
            ' applied to the abundance pattern.\n' \
            f'R: Reference abundances: [M/H]={refabund.monh:.3f}' \
            ' applied to reference abundance pattern.\n' \
            'D: Differences: Abundances minus Reference abundances (A-R).\n'
        for i in range(9):
            out += '  '
            for j in range(11):
                out += f'  {self._elem[11*i+j]:<5s}'
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
        print(out)

    apply_monh = lambda monh, abund: abund + monh if abund else abund


class AbundPattern(dict):
    """A pattern of abundances that must be scaled by a metallicity.

    Subclass of the standard dict class. Initialization populates dict
    with one item per element. Key identifies the element (e.g., 'Fe').
    Values contain a pattern of abundances on the H=12 scale, which must
    be scaled by metallicity to obtain abundances.

    Use standard dictionary syntax to get a pattern value (e.g., ap['Fe'])
    or to set a pattern value (e.g., ap['Fe'] = 7.5). Attempting to set
    a pattern value raises ValueError if the key is not a valid element
    (e.g., 'CN') or the value is not None and cannot by converted into
    a float.

    Overrides __str__() so that print lists keys (elements) and pattern
    values in tabular format.

    Example
    -------
    >>> from sme.abund import AbundPattern
    >>> ap = AbundPattern('Asplund2009')
    >>> print(ap)
    >>> ap['Fe'] = 7.5
    >>> print(ap['Fe'])
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

    """Asplund, Grevesse, Sauval, Scott (2009,  Annual Review of Astronomy
    and Astrophysics, 47, 481)
    """
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

    """Grevesse, Asplund, Sauval (2007, Space Science Review, 130, 105)
    """
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

    def __init__(self, pattern):
        """Create an abundance pattern object with the specified pattern.
        """
        if isinstance(pattern, str):
            self.set_pattern_by_name(pattern)
        else:
            self.set_pattern_by_value(pattern, type)

    def __setitem__(self, key, value):
        """Set pattern abundance for the specfied element. Raise ValueError
        exception if key is not a valid element.
        """
        if key not in self._elem:
            raise ValueError(f'Invalid element key: {key}')
        if value is None:
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, float(value))

    def update(self, keyval):
        """Override update() method in the list superclass. Local method is
        purposely less powerful because we don't want to change the ordered
        list of keys. Call local __setitem__() to check that key is valid.
        """
        for key, value in keyval.items():
            self.__setitem__(key, value)

    def set_pattern_by_name(self, pattern_name):
        if pattern_name.lower() == 'asplund2009':
            super().update(zip(self._elem, self._asplund2009))
        elif pattern_name.lower() == 'grevesse2007':
            super().update(zip(self._elem, self._grevesse2007))
        elif pattern_name.lower() == 'empty':
            super().update(zip(self._elem, [None]*len(self._elem)))
        else:
            raise ValueError(
                f'Invalid abundance pattern name {pattern_name}\n'
                f'Valid names: Asplund2009, Grevesse2007, empty')

