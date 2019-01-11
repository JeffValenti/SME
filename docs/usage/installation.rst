Installation
=============

How to install SME:

1. Download source code from git
2. Download data files as part of IDL SME from http://www.stsci.edu/~valenti/sme.html
    - in ./src/sme create new folders and move data inside:
        - dll:
            - sme_synth.so.*
            - Fe1_Bautista..., etc
            - i.e. everything from SME/lib
        - atmospheres
            - everything from SME/atmospheres
        - nlte_grids
            - *.grd from SME/NLTE

Optional: create a virtual environment

3. Install dependencies:
    - numpy
    - scipy
    - pandas
    - astropy
    - Ipython
    - jupyter
    - matplotlib
    - plotly
    - pytest

4. pip install --editable .