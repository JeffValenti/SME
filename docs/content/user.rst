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
Sauval, Scott (2009,  Annual Review of Astronomy and Astrophysics, 47, 481):

.. code-block:: python

    >>> print(Abund(0, 'Asplund2009'))
    Abundances obtained by applying [M/H]=0.000 to the abundance pattern.
      H      He     Li     Be     B      C      N      O      F      Ne     Na   
     12.000 10.930  1.050  1.380  2.700  8.430  7.830  8.690  4.560  7.930  6.240
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

Use :code:`'Grevesse2007'` for the solar abundance pattern from Grevesse,
Asplund, Sauval (2007, Space Science Review, 130, 105):

.. code-block:: python

    >>> print(Abund(0, 'Grevesse2007'))
    Abundances obtained by applying [M/H]=0.000 to the abundance pattern.
      H      He     Li     Be     B      C      N      O      F      Ne     Na   
     12.000 10.930  1.050  1.380  2.700  8.390  7.780  8.660  4.560  7.840  6.170
      Mg     Al     Si     P      S      Cl     Ar     K      Ca     Sc     Ti   
      7.530  6.370  7.510  5.360  7.140  5.500  6.180  5.080  6.310  3.170  4.900
      V      Cr     Mn     Fe     Co     Ni     Cu     Zn     Ga     Ge     As   
      4.000  5.640  5.390  7.450  4.920  6.230  4.210  4.600  2.880  3.580  2.290
      Se     Br     Kr     Rb     Sr     Y      Zr     Nb     Mo     Tc     Ru   
      3.330  2.560  3.250  2.600  2.920  2.210  2.580  1.420  1.920  None   1.840
      Rh     Pd     Ag     Cd     In     Sn     Sb     Te     I      Xe     Cs   
      1.120  1.660  0.940  1.770  1.600  2.000  1.000  2.190  1.510  2.240  1.070
      Ba     La     Ce     Pr     Nd     Pm     Sm     Eu     Gd     Tb     Dy   
      2.170  1.130  1.700  0.580  1.450  None   1.000  0.520  1.110  0.280  1.140
      Ho     Er     Tm     Yb     Lu     Hf     Ta     W      Re     Os     Ir   
      0.510  0.930  None   1.080  0.060  0.880 -0.170  1.110  0.230  1.250  1.380
      Pt     Au     Hg     Tl     Pb     Bi     Po     At     Rn     Fr     Ra   
      1.640  1.010  1.130  0.900  2.000  0.650  None   None   None   None   None 
      Ac     Th     Pa     U      Np     Pu     Am     Cm     Bk     Cf     Es   
      None   0.060  None  -0.520  None   None   None   None   None   None   None 



**************************
Dynamically Linked Library
**************************

***************************
Vienna Atomic Line Database
***************************
