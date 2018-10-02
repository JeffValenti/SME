
import numpy as np

def solar_abund():
    #Return solar abundances used in SME.
    #
    #Input:
    # None
    #
    #Output:
    # abund (array(99)) abundances of 99 elements
    # elem (string array(99)) names of 99 elements
    #
    #History:
    # 2002-Mar-26 Valenti  Initial coding. Abundances from sme.pro
    # 2002-May-29 Valenti  Add optional elem return argument
    # 2004-Feb-22 Valenti  Add optional atwe return argument
    # 2009-07-01 Heiter  Change abundances to those by Grevesse et al. 2007,
    #                    the same as adopted in the MARCS-2008 model atmosphere grid

    #Logarithmic abundances relative to hydrogen with hydrogen set to 12.
    #Reference: Grevesse, Asplund & Sauval 2007, Space Sciences Review 130, 105
    #Unknown value: -8.00
    # fmt: off
    #       "H",  "He",  "Li",  "Be",   "B",   "C",   "N",   "O",   "F",  "Ne",
    s1=np.array([12.00, 10.93,  1.05,  1.38,  2.70,  8.39,  7.78,  8.66,  4.56,  7.84, 
    #      "Na",  "Mg",  "Al",  "Si",   "P",   "S",  "Cl",  "Ar",   "K",  "Ca",
        6.17,  7.53,  6.37,  7.51,  5.36,  7.14,  5.50,  6.18,  5.08,  6.31, 
    #      "Sc",  "Ti",   "V",  "Cr",  "Mn",  "Fe",  "Co",  "Ni",  "Cu",  "Zn",
        3.17,  4.90,  4.00,  5.64,  5.39,  7.45,  4.92,  6.23,  4.21,  4.60, 
    #      "Ga",  "Ge",  "As",  "Se",  "Br",  "Kr",  "Rb",  "Sr",   "Y",  "Zr",
        2.88,  3.58,  2.29,  3.33,  2.56,  3.25,  2.60,  2.92,  2.21,  2.58, 
    #      "Nb",  "Mo",  "Tc",  "Ru",  "Rh",  "Pd",  "Ag",  "Cd",  "In",  "Sn",
        1.42,  1.92, -8.00,  1.84,  1.12,  1.66,  0.94,  1.77,  1.60,  2.00,
    #      "Sb",  "Te",   "I",  "Xe",  "Cs",  "Ba",  "La",  "Ce",  "Pr",  "Nd",
        1.00,  2.19,  1.51,  2.24,  1.07,  2.17,  1.13,  1.70,  0.58,  1.45, 
    #      "Pm",  "Sm",  "Eu",  "Gd",  "Tb",  "Dy",  "Ho",  "Er",  "Tm",  "Yb",
        -8.00,  1.00,  0.52,  1.11,  0.28,  1.14,  0.51,  0.93,  0.00,  1.08, 
    #      "Lu",  "Hf",  "Ta",   "W",  "Re",  "Os",  "Ir",  "Pt",  "Au",  "Hg",
        0.06,  0.88, -0.17,  1.11,  0.23,  1.25,  1.38,  1.64,  1.01,  1.13, 
    #      "Tl",  "Pb",  "Bi",  "Po",  "At",  "Rn",  "Fr",  "Ra",  "Ac",  "Th",
        0.90,  2.00,  0.65, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00,  0.06, 
    #      "Pa",   "U",  "Np",  "Pu",  "Am",  "Cm",  "Bk",  "Cs",  "Es", "TiO"
        -8.00, -0.52, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00, -8.00])
    # fmt: on
    
    log_eonh = s1 - 12.0

    #Convert to abundances relative to total number of nuclei.
    eonh = 10 ** log_eonh
    renorm = np.sum(eonh)
    eontot = eonh / renorm

    #SME expects Hydrogen abundance to NOT be logarithmic.
    log_eontot = float(np.log10(eontot))
    abund = log_eontot
    abund[0] = eonh[0] / renorm

    #Force single precision.
    abund = float(abund)

    #Build array of element names.
    # fmt: off
    elem=np.array([ 'H', 'He', 'Li', 'Be',  'B',  'C',  'N',  'O',  'F', 'Ne' 
        ,'Na', 'Mg', 'Al', 'Si',  'P',  'S', 'Cl', 'Ar',  'K', 'Ca' 
        ,'Sc', 'Ti',  'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn' 
        ,'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',  'Y', 'Zr' 
        ,'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn' 
        ,'Sb', 'Te',  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd' 
        ,'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb' 
        ,'Lu', 'Hf', 'Ta',  'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg' 
        ,'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th' 
        ,'Pa',  'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cs', 'Es' ])
    # fmt: on

    #Build array of atomic weights.
    #From Loss, R. D. 2003, Pure Appl. Chem., 75, 1107, "Atomic Weights
    # of the Elements 2001". The isotopic composition of the Sun and stars
    # may differ from the terrestrial values listed here!
    #For radionuclides that appear only in Table 3, we have adopted the
    # atomic mass of the longest lived isotope, rounding to at most two
    # digits after the decimal point. These elements are: 43_Tc_98, 
    # 61_Pm_145, 85_At_210, 86_Rn_222, 87_Fr_223, 88_Ra_226, 89_Ac_227,
    # 93_Np_237, 94_Pu_244, 95_Am_243, 96_Cm_247, 97_Bk_247, 98_Cf_251,
    # 99_Es_252
    #
    
    # fmt: off
    atwe=[   1.00794 ,   4.002602 ,   6.941   ,   9.012182 ,  10.811    
        ,  12.0107  ,  14.0067   ,  15.9994  ,  18.9984032,  20.1797   
        ,  22.989770,  24.3050   ,  26.981538,  28.0855   ,  30.973761 
        ,  32.065   ,  35.453    ,  39.948   ,  39.0983   ,  40.078    
        ,  44.955910,  47.867    ,  50.9415  ,  51.9961   ,  54.938049 
        ,  55.845   ,  58.933200 ,  58.6934  ,  63.546    ,  65.409    
        ,  69.723   ,  72.64     ,  74.92160 ,  78.96     ,  79.904    
        ,  83.798   ,  85.4678   ,  87.62    ,  88.90585  ,  91.224    
        ,  92.90638 ,  95.94     ,  97.91    ,  95.94     , 101.07     
        , 102.90550 , 106.42     , 107.8682  , 112.411    , 114.818    
        , 118.710   , 121.760    , 127.60    , 126.90447  , 131.293    
        , 132.90545 , 137.327    , 138.9055  , 140.116    , 140.90765  
        , 144.24    , 144.91     , 150.36    , 151.964    , 157.25     
        , 158.92534 , 162.500    , 164.93032 , 167.259    , 168.93421  
        , 173.04    , 174.967    , 178.49    , 180.9479   , 183.84     
        , 186.207   , 190.23     , 192.217   , 195.078    , 196.966    
        , 200.59    , 204.3833   , 207.2     , 208.98038  , 209.99     
        , 222.02    , 223.02     , 226.03    , 227.03     , 232.0381   
        , 231.03588 , 238.02891  , 237.05    , 244.06     , 243.06     
        , 247.07    , 247.07     , 251.08    , 252.08     ]
    # fmt: on
    
    return atmo, elem, atwe