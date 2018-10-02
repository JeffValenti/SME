import math
from collections import OrderedDict


class Abund:
    """Elemental abundance data and methods.
    Valid abundance pattern types are:
        'sme' - For hydrogen, the abundance value is the fraction of all
            nuclei that are hydrogen, including all ionization states
            and treating molecules as constituent atoms. For the other
            elements, the abundance values are log10 of the fraction of
            nuclei of each element in any form relative to the total for
            all elements in any form. For the Sun, the abundance values
            of H, He, and Li are approximately 0.92, -1.11, and -11.0.
        'n/nTot' - Abundance values are log10 of the fraction of nuclei
            of each element in any form relative to the total for all
            elements in any form. For the Sun, the abundance values of
            H, He, and Li are approximately 0.92, 0.078, and 1.03e-11.
        'n/nH' - Abundance values are log10 of the fraction of nuclei
            of each element in any form relative to the number of
            hydrogen nuclei in any form. For the Sun, the abundance
            values of H, He, and Li are approximately 1, 0.085, and
            1.12e-11.
        'H=12' - Abundance values are log10 of the fraction of nuclei of
            each element in any form relative to the number of hydrogen
            in any form plus an offset of 12. For the Sun, the nuclei
            abundance values of H, He, and Li are approximately 12,
            10.9, and 1.05.
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
        abund = OrderedDict(
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
        a = a[0:2] + [ab+self._monh for ab in a[2:]]
        out = ' [M/H]={:.3f} applied to abundance pattern. ' \
            'Values below are abundances.\n'.format(self._monh)
        for i in range(9):
            for j in range(11):
                out += '  {:<5s}'.format(self._elem[11*i+j])
            out += '\n'
            for j in range(11):
                out += '{:7.3f}'.format(a[11*i+j])
            if i < 8:
                out += '\n'
        return out

    @staticmethod
    def fromtype(pattern, fromtype):
        """Return a copy of the input abundance pattern, transformed from
        the input type to the 'H=12' type. Valid abundance pattern types
        are 'sme', 'n/nTot', 'n/nH', and 'H=12'.
        """
        patt = pattern.copy()
        type = fromtype.lower()
        if type == 'h=12':
            return patt

        if type == "sme":
            elem = Abund._elem
            abund = patt
            patt = {el:ab for el, ab in zip(elem, abund) if ab}
        else:
            elem = [el for el, ab in patt.items() if ab]
            abund = [patt[el] for el in elem]

        if elem[0] != 'H':
            raise ValueError('pattern must define abundance of H')
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
                "got abundance type '{}',".format(type) +
                " should be 'H=12', 'n/nH', 'n/nTot', or 'sme'"
                )
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
    def elem(self):
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
            self._pattern = OrderedDict(zip(self._elem, self._asplund2009))
        elif pattern_name.lower() == 'grevesse2007':
            self._pattern = OrderedDict(zip(self._elem, self._grevesse2007))
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
            'sme' - For hydrogen, the abundance value is the fraction of all
                nuclei that are hydrogen, including all ionization states
                and treating molecules as constituent atoms. For the other
                elements, the abundance values are log10 of the fraction of
                nuclei of each element in any form relative to the total for
                all elements in any form. For the Sun, the abundance values
                of H, He, and Li are approximately 0.92, -1.11, and -11.0.
            'n/nTot' - Abundance values are log10 of the fraction of nuclei
                of each element in any form relative to the total for all
                elements in any form. For the Sun, the abundance values of
                H, He, and Li are approximately 0.92, 0.078, and 1.03e-11.
            'n/nH' - Abundance values are log10 of the fraction of nuclei
                of each element in any form relative to the number of
                hydrogen nuclei in any form. For the Sun, the abundance
                values of H, He, and Li are approximately 1, 0.085, and
                1.12e-11.
            'H=12' - Abundance values are log10 of the fraction of nuclei of
                each element in any form relative to the number of hydrogen
                in any form plus an offset of 12. For the Sun, the nuclei
                abundance values of H, He, and Li are approximately 12,
                10.9, and 1.05.
        """
        return self.totype(self._pattern, type)

    def empty_pattern(self):
        """Return an abundance pattern with value None for all elements.
        """
        return OrderedDict.fromkeys(self._elem)

    def print(self):
        if self._pattern is None:
            if self._monh is None:
                print('[M/H] is not set. Abundance pattern is not set.')
            else:
                print(
                    '[M/H]={:.3f}. Abundance pattern is not set'.
                    format(self._monh)
                    )
        else:
            if self._monh is None:
                print(
                    '[M/H] is not set. Values below are the abundance pattern.'
                    )
                a = self.get_pattern('H=12')
            else:
                print(
                    '[M/H]={:.3f} applied to abundance pattern. '
                    'Values below are abundances.'.format(self._monh)
                    )
                a = self.get_pattern('H=12')
                a = a[0:2] + [ab+self._monh for ab in a[2:]]
            for i in range(9):
                estr = ''
                astr = ''
                for j in range(11):
                    estr += '  {:<5s}'.format(self._elem[11*i+j])
                    astr += '{:7.3f}'.format(a[11*i+j])
                print(estr)
                print(astr)
