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

SME factors elemental abundances into an *abundance pattern*, which is the relative
abundance of each element, and a *metallicity*, which is the (base 10) logarithmic
scale factor applied to all elements except hydrogen. Specify both factors to define
the abundance of each element, e.g.:

.. code-block:: python

    >>> from sme.abund import Abund
    >>> a = Abund(-0.25, 'Asplund2009')
    >>> print(a)
     [M/H]=-0.250 applied to abundance pattern. Values below are abundances.
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

Specifying an Abundance Pattern
===============================

SME provides two abundance patterns by case-insensitive name.
Use :code:`'Asplund2009'` for the solar abundance pattern from Asplund, Grevesse,
Sauval, Scott (2009, Annual Review of Astronomy and Astrophysics, 47, 481),
as in the example above.

.. code-block:: python

    >>> a.pattern
    {'H' : 12.0, 'He': 10.93, 'Li': 1.05 , 'Be': 1.38, 'B' : 2.7 ,
     'C' : 8.43, 'N' : 7.83 , 'O' : 8.69 , 'F' : 4.56, 'Ne': 7.93,
     'Na': 6.24, 'Mg': 7.6  , 'Al': 6.45 , 'Si': 7.51, 'P' : 5.41,
     'S' : 7.12, 'Cl': 5.5  , 'Ar': 6.4  , 'K' : 5.03, 'Ca': 6.34,
     'Sc': 3.15, 'Ti': 4.95 , 'V' : 3.93 , 'Cr': 5.64, 'Mn': 5.43,
     'Fe': 7.5 , 'Co': 4.99 , 'Ni': 6.22 , 'Cu': 4.19, 'Zn': 4.56,
     'Ga': 3.04, 'Ge': 3.65 , 'As': 2.3  , 'Se': 3.34, 'Br': 2.54,
     'Kr': 3.25, 'Rb': 2.52 , 'Sr': 2.87 , 'Y' : 2.21, 'Zr': 2.58,
     'Nb': 1.46, 'Mo': 1.88 , 'Tc': None , 'Ru': 1.75, 'Rh': 0.91,
     'Pd': 1.57, 'Ag': 0.94 , 'Cd': 1.71 , 'In': 0.8 , 'Sn': 2.04,
     'Sb': 1.01, 'Te': 2.18 , 'I' : 1.55 , 'Xe': 2.24, 'Cs': 1.08,
     'Ba': 2.18, 'La': 1.1  , 'Ce': 1.58 , 'Pr': 0.72, 'Nd': 1.42,
     'Pm': None, 'Sm': 0.96 , 'Eu': 0.52 , 'Gd': 1.07, 'Tb': 0.3 ,
     'Dy': 1.1 , 'Ho': 0.48 , 'Er': 0.92 , 'Tm': 0.1 , 'Yb': 0.84,
     'Lu': 0.1 , 'Hf': 0.85 , 'Ta': -0.12, 'W' : 0.85, 'Re': 0.26,
     'Os': 1.4 , 'Ir': 1.38 , 'Pt': 1.62 , 'Au': 0.92, 'Hg': 1.17,
     'Tl': 0.9 , 'Pb': 1.75 , 'Bi': 0.65 , 'Po': None, 'At': None,
     'Rn': None, 'Fr': None , 'Ra': None , 'Ac': None, 'Th': 0.02,
     'Pa': None, 'U' : -0.54, 'Np': None , 'Pu': None, 'Am': None,
     'Cm': None, 'Bk': None , 'Cf': None , 'Es': None}

Use :code:`'Grevesse2007'` for the solar abundance pattern from Grevesse,
Asplund, Sauval (2007, Space Science Review, 130, 105):

.. code-block:: python

    >>> a = Abund(0, 'Grevesse2007'))
    >>> a.pattern
    {'H' : 12.0, 'He': 10.93, 'Li': 1.05 , 'Be': 1.38, 'B' : 2.7 ,
     'C' : 8.39, 'N' : 7.78 , 'O' : 8.66 , 'F' : 4.56, 'Ne': 7.84,
     'Na': 6.17, 'Mg': 7.53 , 'Al': 6.37 , 'Si': 7.51, 'P' : 5.36,
     'S' : 7.14, 'Cl': 5.5  , 'Ar': 6.18 , 'K' : 5.08, 'Ca': 6.31,
     'Sc': 3.17, 'Ti': 4.9  , 'V' : 4.0  , 'Cr': 5.64, 'Mn': 5.39,
     'Fe': 7.45, 'Co': 4.92 , 'Ni': 6.23 , 'Cu': 4.21, 'Zn': 4.6 ,
     'Ga': 2.88, 'Ge': 3.58 , 'As': 2.29 , 'Se': 3.33, 'Br': 2.56,
     'Kr': 3.25, 'Rb': 2.6  , 'Sr': 2.92 , 'Y' : 2.21, 'Zr': 2.58,
     'Nb': 1.42, 'Mo': 1.92 , 'Tc': None , 'Ru': 1.84, 'Rh': 1.12,
     'Pd': 1.66, 'Ag': 0.94 , 'Cd': 1.77 , 'In': 1.6 , 'Sn': 2.0 ,
     'Sb': 1.0 , 'Te': 2.19 , 'I' : 1.51 , 'Xe': 2.24, 'Cs': 1.07,
     'Ba': 2.17, 'La': 1.13 , 'Ce': 1.7  , 'Pr': 0.58, 'Nd': 1.45,
     'Pm': None, 'Sm': 1.0  , 'Eu': 0.52 , 'Gd': 1.11, 'Tb': 0.28,
     'Dy': 1.14, 'Ho': 0.51 , 'Er': 0.93 , 'Tm': 0.0 , 'Yb': 1.08,
     'Lu': 0.06, 'Hf': 0.88 , 'Ta': -0.17, 'W' : 1.11, 'Re': 0.23,
     'Os': 1.25, 'Ir': 1.38 , 'Pt': 1.64 , 'Au': 1.01, 'Hg': 1.13,
     'Tl': 0.9 , 'Pb': 2.0  , 'Bi': 0.65 , 'Po': None, 'At': None,
     'Rn': None, 'Fr': None , 'Ra': None , 'Ac': None, 'Th': 0.06,
     'Pa': None, 'U' : -0.52, 'Np': None , 'Pu': None, 'Am': None,
     'Cm': None, 'Bk': None , 'Cf': None , 'Es': None}


**************************
Dynamically Linked Library
**************************

***************************
Vienna Atomic Line Database
***************************
