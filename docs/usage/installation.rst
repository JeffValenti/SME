Installation
=============

How to install SME:

1. Download source code from git
    - e.g. git clone https://github.com/AWehrhahn/SME.git
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

3: create a virtual environment, and activate it.
    - cd to SME directory
    - conda env create -f environment.yml
    - source activate pysme

4. Install SME
    - pip install --editable .

5. Run SME
    - python main.py
