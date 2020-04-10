.. toctree::
   :maxdepth: 2

################
User Information
################

.. warning::
   This project is very far from completion, so use the IDL version.

**********
Abundances
**********

.. |dict| replace:: :class:`dict`

.. |AbundError| replace:: :class:`AbundError`
.. |Abund| replace:: :class:`Abund`
.. |AbundPattern| replace:: :class:`AbundPattern`
.. |from_H12| replace:: :func:`from_H12`
.. |to_H12| replace:: :func:`to_H12`

.. |None| replace:: :code:`None`
.. |H_12| replace:: :code:`'H=12'`
.. |n_nH| replace:: :code:`'n/nH'`
.. |n_nTot| replace:: :code:`'n/nTot'`
.. |smenorm| replace:: :code:`'sme'`

.. _dict: https://docs.python.org/3/library/stdtypes.html#dict
.. _float: https://docs.python.org/3/library/functions.html?highlight=float#float
.. _Sequence: https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence

SME factors elemental abundances into an *abundance pattern*, which is the
relative abundance of each element, and a *metallicity*, which is the (base 10)
logarithmic scale factor applied to all elements except hydrogen and helium.

Create an abundance object
==========================

.. _SSRv, 130, 105: https://ui.adsabs.harvard.edu/abs/2007SSRv..130..105G
.. _ARA&A, 47, 481: https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A

Create an abundance object by specifying a metallicity and an abundance
pattern. Specify the abundance pattern by name or as a sequence of custom
values. When specifying custom values, you must also indicate how the input
abundances are normalized.

SME knows the following named abundance patterns:

.. _table-named-patterns:

.. table:: Named abundance patterns

    ====================== =========
    Pattern name           Reference
    ====================== =========
    :code:`'Grevesse2007'` Grevesse, Asplund, & Sauval (2007, `SSRv, 130, 105`_)
    :code:`'Asplund2009'`  Asplund, Grevesse, Sauval, & Scott (2009, `ARA&A, 47, 481`_)
    :code:`'Empty'`        Abundance is 12 for hydrogen, |None| for other elements
    ====================== =========

Here is an example that sets elemental abundances equal to the solar abundance
pattern in Asplund et al. (2009), scaled by a metallicity of -0.25:

.. code-block:: python

    >>> from sme.abund import Abund
    >>> abund = Abund(-0.25, 'Asplund2009')
    >>> abund
    Abund(-0.25, {'H': 12.0, 'He': 10.93, 'Li': 1.05, 'Be': 1.38, 'B': 2.7, 'C': 8.43, 'N': 7.83, 'O': 8.69, 'F': 4.56, 'Ne': 7.93, 'Na': 6.24, 'Mg': 7.6, 'Al': 6.45, 'Si': 7.51, 'P': 5.41, 'S': 7.12, 'Cl': 5.5, 'Ar': 6.4, 'K': 5.03, 'Ca': 6.34, 'Sc': 3.15, 'Ti': 4.95, 'V': 3.93, 'Cr': 5.64, 'Mn': 5.43, 'Fe': 7.5, 'Co': 4.99, 'Ni': 6.22, 'Cu': 4.19, 'Zn': 4.56, 'Ga': 3.04, 'Ge': 3.65, 'As': 2.3, 'Se': 3.34, 'Br': 2.54, 'Kr': 3.25, 'Rb': 2.52, 'Sr': 2.87, 'Y': 2.21, 'Zr': 2.58, 'Nb': 1.46, 'Mo': 1.88, 'Tc': None, 'Ru': 1.75, 'Rh': 0.91, 'Pd': 1.57, 'Ag': 0.94, 'Cd': 1.71, 'In': 0.8, 'Sn': 2.04, 'Sb': 1.01, 'Te': 2.18, 'I': 1.55, 'Xe': 2.24, 'Cs': 1.08, 'Ba': 2.18, 'La': 1.1, 'Ce': 1.58, 'Pr': 0.72, 'Nd': 1.42, 'Pm': None, 'Sm': 0.96, 'Eu': 0.52, 'Gd': 1.07, 'Tb': 0.3, 'Dy': 1.1, 'Ho': 0.48, 'Er': 0.92, 'Tm': 0.1, 'Yb': 0.84, 'Lu': 0.1, 'Hf': 0.85, 'Ta': -0.12, 'W': 0.85, 'Re': 0.26, 'Os': 1.4, 'Ir': 1.38, 'Pt': 1.62, 'Au': 0.92, 'Hg': 1.17, 'Tl': 0.9, 'Pb': 1.75, 'Bi': 0.65, 'Po': None, 'At': None, 'Rn': None, 'Fr': None, 'Ra': None, 'Ac': None, 'Th': 0.02, 'Pa': None, 'U': -0.54, 'Np': None, 'Pu': None, 'Am': None, 'Cm': None, 'Bk': None, 'Cf': None, 'Es': None}, 'H=12')
    >>> print(abund)
    Abundances obtained by applying [M/H]=-0.250 to the abundance pattern.
     H      He     Li     Be     B      C      N      O      F      Ne     Na   
    12.000 10.930  0.800  1.130  2.450  8.180  7.580  8.440  4.310  7.680  5.990
     Mg     Al     Si     P      S      Cl     Ar     K      Ca     Sc     Ti   
     7.350  6.200  7.260  5.160  6.870  5.250  6.150  4.780  6.090  2.900  4.700
     V      Cr     Mn     Fe     Co     Ni     Cu     Zn     Ga     Ge     As   
     3.680  5.390  5.180  7.250  4.740  5.970  3.940  4.310  2.790  3.400  2.050
     Se     Br     Kr     Rb     Sr     Y      Zr     Nb     Mo     Tc     Ru   
     3.090  2.290  3.000  2.270  2.620  1.960  2.330  1.210  1.630  None   1.500
     Rh     Pd     Ag     Cd     In     Sn     Sb     Te     I      Xe     Cs   
     0.660  1.320  0.690  1.460  0.550  1.790  0.760  1.930  1.300  1.990  0.830
     Ba     La     Ce     Pr     Nd     Pm     Sm     Eu     Gd     Tb     Dy   
     1.930  0.850  1.330  0.470  1.170  None   0.710  0.270  0.820  0.050  0.850
     Ho     Er     Tm     Yb     Lu     Hf     Ta     W      Re     Os     Ir   
     0.230  0.670 -0.150  0.590 -0.150  0.600 -0.370  0.600  0.010  1.150  1.130
     Pt     Au     Hg     Tl     Pb     Bi     Po     At     Rn     Fr     Ra   
     1.370  0.670  0.920  0.650  1.500  0.400  None   None   None   None   None 
     Ac     Th     Pa     U      Np     Pu     Am     Cm     Bk     Cf     Es   
     None  -0.230  None  -0.790  None   None   None   None   None   None   None 

Here is an example that specifies the abundance pattern as a sequence of
custom values, normalized using the |n_nH| convention:

.. code-block:: python

    >>> abund = Abund(0.1, {'H': 1.0, 'He': 0.085, 'C': 0.00025, 'N': 6e-05, 'O': 0.00046, 'Mg': 3.4e-05, 'Al': 2.3e-06, 'Si': 3.2e-05, 'S': 1.4e-05, 'Ca': 2e-06, 'Ti': 7.9e-08, 'Cr': 4.4e-07, 'Fe': 2.8e-05, 'Ni': 1.7e-06}, 'n/nH')
    >>> print(abund)
    Abundances obtained by applying [M/H]=-0.250 to the abundance pattern.
     H      He     Li     Be     B      C      N      O      F      Ne     Na   
    12.000 10.929  None   None   None   8.148  7.528  8.413  None   None   None 
     Mg     Al     Si     P      S      Cl     Ar     K      Ca     Sc     Ti   
     7.281  6.112  7.255  None   6.896  None   None   None   6.051  None   4.648
     V      Cr     Mn     Fe     Co     Ni     Cu     Zn     Ga     Ge     As   
     None   5.393  None   7.197  None   5.980  None   None   None   None   None 
     Se     Br     Kr     Rb     Sr     Y      Zr     Nb     Mo     Tc     Ru   
     None   None   None   None   None   None   None   None   None   None   None 
     Rh     Pd     Ag     Cd     In     Sn     Sb     Te     I      Xe     Cs   
     None   None   None   None   None   None   None   None   None   None   None 
     Ba     La     Ce     Pr     Nd     Pm     Sm     Eu     Gd     Tb     Dy   
     None   None   None   None   None   None   None   None   None   None   None 
     Ho     Er     Tm     Yb     Lu     Hf     Ta     W      Re     Os     Ir   
     None   None   None   None   None   None   None   None   None   None   None 
     Pt     Au     Hg     Tl     Pb     Bi     Po     At     Rn     Fr     Ra   
     None   None   None   None   None   None   None   None   None   None   None 
     Ac     Th     Pa     U      Np     Pu     Am     Cm     Bk     Cf     Es   
     None   None   None   None   None   None   None   None   None   None   None 

The abundance pattern is set to |None| for all unspecified elements.
Abundances should be specified for all major constituents of the atmosphere.
Otherwise, renormalization will be inaccurate, as illustrated by comparing
abundances in the two examples above.


Update metallicity and abundance pattern
========================================

Abundance objects instantiated from the |Abund| class have a metallicity
property (`monh`) and an abundance pattern property (`pattern`). Both can
be updated. The value for `monh` must be a number or a string that `float`_
can convert to type float. The value for `pattern` must be an object of
type |AbundPattern|.

.. code-block:: python

    >>> print(abund.monh)
    -0.25
    >>> abund.monh = 0.15
    >>> abund.monh -= 0.15
    >>> abund.monh
    0.0
    >>> abund.monh = '-0.25'
    >>> abund.monh
    -0.25
    >>> abund.monh = 'text'
    ValueError: could not convert string to float: 'text'

.. code-block:: python

    >>> type(abund.pattern)
    <class 'sme.abund.AbundPattern'>
    >>> abund.pattern
    AbundPattern({'H': 12.0, 'He': 10.93, 'Li': 1.05, i...}, 'H=12')
    >>> abund.pattern['He'] = 11.02
    >>> print(abund.pattern)
     H      He     Li     Be     B      C      N      O      F      Ne     Na   
    12.000 11.020  1.050  1.380  2.700  8.430  7.830  8.690  4.560  7.930  6.240
     Mg     Al     Si     P      S      Cl     Ar     K      Ca     Sc     Ti   
     7.600  6.450  7.510  5.410  7.120  5.500  6.400  5.030  6.340  3.150  4.950
     V      Cr     Mn     Fe     Co     Ni     Cu     Zn     Ga     Ge     As   
     3.930  5.640  5.430  7.500  4.990  6.220  4.190  4.560  3.040  3.650  2.300
     Se     Br     Kr     Rb     Sr     Y      Zr     Nb     Mo     Tc     Ru   
     3.340  2.540  3.250  2.520  2.870  2.210  2.580  1.460  1.880  None   1.750
     Rh     Pd     Ag     Cd     In     Sn     Sb     Te     I      Xe     Cs   
     0.910  1.570  0.940  1.710  0.800  2.040  1.010  2.180  1.550  2.240  1.080
     Ba     La     Ce     Pr     Nd     Pm     Sm     Eu     Gd     Tb     Dy   
     2.180  1.100  1.580  0.720  1.420  None   0.960  0.520  1.070  0.300  1.100
     Ho     Er     Tm     Yb     Lu     Hf     Ta     W      Re     Os     Ir   
     0.480  0.920  0.100  0.840  0.100  0.850 -0.120  0.850  0.260  1.400  1.380
     Pt     Au     Hg     Tl     Pb     Bi     Po     At     Rn     Fr     Ra   
     1.620  0.920  1.170  0.900  1.750  0.650  None   None   None   None   None 
     Ac     Th     Pa     U      Np     Pu     Am     Cm     Bk     Cf     Es   
     None   0.020  None  -0.540  None   None   None   None   None   None   None 
    >>> abund.pattern.update({'H': 12, 'He': 11.1, 'Li': 1.1})
    >>> abund.pattern
    AbundPattern({'H': 12.0, 'He': 11.1, 'Li': 1.1, 'Be': 1.38, ...}, 'H=12')
    >>> abund.pattern.update(Be=1.4)
    >>> abund.pattern
    AbundPattern({'H': 12.0, 'He': 11.1, 'Li': 1.1, 'Be': 1.4, ....}, 'H=12')
    >>> abund.pattern = {'H': 12, 'He': 11, 'Li': 1}
    sme.abund.AbundError: pattern must be an AbundPattern object



Trying to set abundances directly raises an |AbundError| because SME does
not know whether to adjust metallicity or the underlying abundance pattern.

.. code-block:: python

    >>> abund['Fe'], abund.pattern['Fe'], abund.monh
    (7.25, 7.5, -0.25)
    >>> abund['Fe'] = 7.1
    sme.abund.AbundError: set monh and pattern separately, not the combination

Instead update metallicity or abundance pattern to yield the desired abundance.

.. code-block:: python

    >>> abund.pattern['Fe'] -= 0.15
    >>> abund['Fe'], abund.pattern['Fe'], abund.monh
    (7.1, 7.35, -0.25)

    >>> abund.pattern['Fe'] = 7.5
    >>> abund.monh = -0.4
    >>> abund['Fe'], abund.pattern['Fe'], abund.monh
    (7.1, 7.5, -0.4)

Abundance normalization conventions
===================================

The astronomical community uses multiple conventions to normalize elemental
abundances and abundance patterns. |Abund| and |AbundPattern| can be
initialized and can output (see `normalized`) abundances with these four
normalization conventions:

.. _table-normalization-descriptions:

.. table:: Abundance normalization conventions

    ============= ===========
    Normalization Description
    ============= ===========
    |H_12|        Abundance values are log10 of the fraction of nuclei of each
                  element in any form, relative to the number of hydrogen
                  nuclei in any form plus an offset of 12. For the Sun, the
                  abundance values of H, He, and Li are approximately 12,
                  10.9, and 1.05.
    |n_nH|        Abundance values are the fraction of nuclei of each element
                  in any form, relative to the number of hydrogen nuclei in any
                  form. For the Sun, the abundance values of H, He, and Li are
                  approximately 1, 0.085, and 1.12e-11.
    |n_nTot|      Abundance values are the fraction of nuclei of each element
                  in any form, relative to the total for all elements in any
                  form. For the Sun, the abundance values of H, He, and Li are
                  approximately 0.92, 0.078, and 1.03e-11.
    |smenorm|     For hydrogen, the abundance value is the fraction of all
                  nuclei that are hydrogen. For the other elements, the
                  abundance values are log10 of the fraction of nuclei of each
                  element in any form, relative to the total for all elements
                  in any form. For the Sun, the abundance values of H, He,
                  and Li are approximately 0.92, -1.11, and -11.0.
    ============= ===========

In the descriptions above, "nuclei in any form" includes all ionization states
and treats molecules as constituent atoms. Thus, a bare proton counts as one
hydrogen nucleus and an H\ :sub:`2` molecule counts as two hydrogen nuclei.

This SME python package always stores abundances and abundance patterns using
the |H_12| normalization. The SME shared library use the |smenorm|
normalization convention.

Use the `normalized` method to get abundances or an abundance pattern
with a different normalization convention. The result is a dictionary.

.. code-block:: python

    >>> abund = Abund(-0.5, 'Asplund2009')
    >>> abund.pattern.normalized('H=12')
    {'H': 12.0, 'He': 10.93, 'Li': 1.05, ...}
    >>> abund.normalized('H=12')
    {'H': 12.0, 'He': 10.93, 'Li': 0.55, ...}
    >>> abund.normalized('n/nH')
    {'H': 1.0, 'He': 0.08511380382023759, 'Li': 3.5481338923357604e-12, ...}
    >>> abund.normalized('n/nTot')
    {'H': 0.921282772024301, 'He': 0.07841388112104103, 'Li': 3.2688346278444622e-12, ...}
    >>> abund.normalized('sme')
    {'H': 0.921282772024301, 'He': -1.1056070500624255, 'Li': -11.485607050062425, ...}

Use the optional `input_norm` parameter to initialize |Abund| or |AbundPattern|
from input abundances that do not have the default |H_12| normalization.

.. code-block:: python

    >>> grevesse = Abund(0, 'Grevesse2007')
    >>> grevesse
    Abund(0.0, {'H': 12.0, 'He': 10.93, 'Li': 1.05, ...}, 'H=12')
    >>> dict_altnorm = grevesse.normalized('n/nH')
    >>> dict_altnorm
    {'H': 1.0, 'He': 0.08511380382023759, 'Li': 1.1220184543019653e-11, ...}
    >>> grevesse_from_dict = Abund(0, dict_altnorm, input_norm='n/nH')
    >>> grevesse_from_dict
    Abund(0.0, {'H': 12.0, 'He': 10.93, 'Li': 1.0500000000000007, ...}, 'H=12')

The final |Abund| object (:code:`grevesse_from_dict`) uses |H_12|
normalization, even though it was initialized with abundances having |n_nH|
normalization.  The round trip renormalization from |H_12| to |n_nH| back
to |H_12| yielded a small numerical difference in the abundances.

Renormalize abundances
======================

The stand alone function |to_H12| converts input abundances from the
specified input normalization to |H_12| output normalization. The inverse
function |from_H12| converts input abundances from |H_12| input normalization
to the specified output normalization.

Input abundances must be provided in a |dict|-like object that provides an
`items` method. |Abund| and |AbundPattern| objects have an `items` method.
Normalization conventions are described in a
:ref:`table <table-normalization-descriptions>` above. Here is an example:

.. code-block:: python

    >>> from sme.abund import from_H12, to_H12, Abund
    >>> abund = Abund(-0.5, 'Grevesse2007')
    >>> abund.normalized('H=12')
    {'H': 12.0, 'He': 10.93, 'Li': 0.55, ...}
    >>> from_H12(abund.normalized('H=12'), 'n/nH')
    {'H': 1.0, 'He': 0.08511380382023759, 'Li': 3.5481338923357604e-12, ...}
    >>> to_H12(from_H12(abund.normalized('H=12'), 'n/nH'), 'n/nH')
    {'H': 12.0, 'He': 10.93, 'Li': 0.5500000000000007, ...}

Compare two abundance objects
=============================

|Abund| and |AbundPattern| support the :code:`==` and :code:`!=` comparison
operators.

.. code-block:: python

    >>> grevesse = Abund(0, 'Grevesse2007')
    >>> asplund = Abund(0, 'Asplund2009')
    >>> grevesse == asplund
    False
    >>> grevesse != asplund
    True

.. code-block:: python

    >>> asplund_copy = Abund(0, 'Asplund2009')
    >>> asplund_copy is asplund
    False
    >>> asplund_copy == asplund
    True

For |Abund| objects, the comparison operators separately compare both
`monh` and `pattern`.

.. code-block:: python

    >>> asplund_rich = Abund(0.5, 'Asplund2009')
    >>> asplund_rich.pattern == asplund.pattern
    True
    >>> asplund_rich == asplund
    False

Astrophysically negligible differences are ignored in comparisons.
Two |AbundPattern| objects are considered equal if abundances are specified
for the same set of elements and the absolute value of the difference in
two abundances for each of those elements is less than 10\ :sup:`-4` with
the H=12 normalization convention.

.. code-block:: python

    >>> asplund_copy.pattern['Fe'] += 1e-5
    >>> asplund_copy.pattern['Fe'], asplund.pattern['Fe']
    (7.50001, 7.5)
    >>> asplund_copy.pattern['Fe'] == asplund.pattern['Fe']
    False
    >>> asplund_copy.pattern == asplund.pattern
    True
    >>> asplund_copy.pattern['Fe'] = asplund['Fe'] + 1e-3
    >>> asplund_copy.pattern == asplund.pattern
    False

Use abundance object in software
================================

|Abund| and |AbundPattern| are subclasses of the `Sequence`_ abstract base
class, with additional methods (`items`, `keys`, `values`) that mimic some
methods available in the more capable `dict`_ class. |AbundPattern| also
supports the `update` method. Keys are standard element abbreviations (e.g.,
:code:`'Mg'`) ordered by increasing atomic number. Keys are immutable. Values
are floating point numbers or |None| if an abundance is not specified.

.. code-block:: python

    >>> from sme.abund import AbundPattern
    >>> p = AbundPattern('Grevesse2007')
    >>> p['CN'] = 5.2
    sme.abund.AbundError: unknown element key: 'CN'
    >>> p['Mg'] = 'text'
    sme.abund.AbundError: cannot convert abundance value to a float
    >>> p['Mg'] = '7.6'
    >>> p['Mg']
    7.6
    >>> p.update(Mg=7.5)
    >>> p['Mg']
    7.5
    >>> p.keys()
    dict_keys(['H', 'He', 'Li', 'Be', ..., 'Bk', 'Cf', 'Es'])
    >>> p.values()
    dict_values([12.0, 10.93, 1.05, 1.38, ..., None, None, None])
    >>> {k: v for k, v in p.items()}
    {'H': 12.0, 'He': 10.93, 'Li': 1.05, 'Be': 1.38, ..., 'Bk': None, 'Cf': None, 'Es': None}

**************************
Dynamically Linked Library
**************************

***************************
Vienna Atomic Line Database
***************************
