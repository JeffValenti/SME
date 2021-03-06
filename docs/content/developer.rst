.. toctree::
   :maxdepth: 2

#####################
Developer Information
#####################

***********
Environment
***********

SME is not pure python. SME calls C++ and fortran dynamic libraries,
which may call other dynamic libraries. Hence, SME should be run in
an environment that manages C++ and fortran versions as well as
python packages. For this reason, we recommend running SME in a
:command:`conda` environment.

Create and maintain one :command:`conda` environment for SME. Use
that *sme* environment in every terminal window where you develop
or run SME. Use other :command:`conda` environments for projects.

Use environment
===============

Activate the *sme* environment in each terminal window before working
with SME in that window. Activiating the *sme* environment adds a
``(sme)`` prefix to your normal shell prompt. Create the environment
before using it.

.. code-block:: bash

    $ conda activate sme
    (sme) $

To stop using the *sme* environment in a terminal window, close the
window or explicitly deactivate the current environment.

.. code-block:: bash

    (sme) $ conda deactivate
    $

Create environment
==================

To create an environment for SME development:

1. Use :command:`bash` shell because it is fully compatible with
:command:`conda`.

.. _Install Miniconda or Anaconda:
   https://conda.io/docs/user-guide/install/

2. `Install Miniconda or Anaconda`_. Miniconda includes a minimal set
of python packages. Anaconda incudes many additional python packages
that are not needed by SME. In either case, users will install
additional packages as needed.

Installing conda typically prepends the conda :file:`bin` directory
in the definition of :envvar:`PATH` in :file:`~/.bash_profile`. Open
and begin working in a new terminal window, so that the new
:envvar:`PATH` definition takes effect.

3. `Clone <https://help.github.com/articles/cloning-a-repository/>`_
the `SME repository`_ on GitHub to a local :file:`SME` directory.

.. _SME repository:
   https://github.com/JeffValenti/SME

.. code-block:: bash

    $ git clone https://github.com/JeffValenti/SME.git
    $ cd SME
    SME$

.. _conda-env-create-step:

4. `Create a conda environment`_ named *sme* using the
:file:`environment.yml` file distributed with SME. The name of the
new environment (*sme*) is specified in :file:`environment.yml`.


.. _Create a conda environment:
   https://conda.io/docs/user-guide/tasks/manage-environments.html
   #creating-an-environment-from-an-environment-yml-file

.. code-block:: bash

    SME$ conda env create -f environment.yml

5. Install SME from the local repository. Activate the *sme* environment,
so that SME is installed in the *sme* environment, not the *base*
environment. Execute :command:`pip install` from the top-level directory
of the local SME repository (typically :file:`sme`). Use the `--editable`
option to keep the installed version synchronized with changes to the
local repository.

.. code-block:: bash

    SME$ conda activate sme
    (sme) SME$ pip install --editable .


Update environment
==================

Periodically update packages in the *sme* environment to obtain the
enviroment a new user or Travis CI would create.

.. code-block:: bash

    $ conda activate sme
    (sme) $ conda update

Recreate environment
====================

Recreate the *sme* environment, if the :file:`environment.yml` file
changes or to restore the *sme* environment to its canonical state.
First remove the existing *sme* environment.

.. code-block:: bash

    (sme) $ conda deactivate
    $ conda env remove --name sme

Then recreate the *sme* environment, starting with the
:ref:`conda env create step <conda-env-create-step>`.


Supplement environment
======================

If needed, use :command:`conda install` to add additional packages
while in the *sme* environment. Conda may have to install different
versions of required packages to reconcile dependencies in the newly
installed additional packages, but SME should still work.

.. code-block:: bash

    $ conda activate sme
    (sme) $ conda install <package>

If an additional package is not available via a conda channel, use
:command:`pip install` while in the *sme* environment. Package
management is not as robust when using :command:`pip` to install in
a :command:`conda` environment.

.. code-block:: bash

    $ conda activate sme
    (sme) $ pip install <package>

Additional packages will be available in the local *sme* environment,
but not in Travis CI or other user environments. To add packages for
all SME users, update the :file:`environment.yml` file and recreate
the local *sme* environment.

****************
Software Testing
****************

SME uses the `pytest framework <https://docs.pytest.org/latest>`_
to demonstrate that SME behaves as intended, despite inevitable
changes to SME code, python modules, and the environment.

Unit Tests
==========

Unit test code is located in the directory :file:`SME/test` in files
named :file:`test_<module>.py`. Each such file contains functions
named :code:`test_<purpose>`. Each such function contains python
:code:`assert` statements that test whether a particular code path
behaves as expected.

Run unit tests locally before committing code. Use the `-v` option
to see results for each individual test function. Use the `--cov=sme`
option to see what fraction of statements in each SME module are
covered by unit tests. Unit tests should cover 100% of statements.

.. code-block:: bash

    $ cd SME
    SME$ conda activate sme
    (sme) SME$ pytest -v --cov=sme --cov-report html

To see which specific statements are not covered, use a web browser
to view the file :file:`htmlcov/index.html` and then click each
module name. On a mac, use `open` to view the file.

.. code-block:: bash

    (sme) SME$ open htmlcov/index.html


Continuous Integration
======================

.. _Travis CI:
   https://travis-ci.org/JeffValenti/SME/builds

The `SME repository`_ on GitHub is linked to the `Travis CI`_
continuous integration service. Updating SME on GitHub causes
Travis CI to follow instructions in :file:`SME/.travis.yml`.
These instructions build SME in one or more environments (e.g.,
linux, osx) and run :command:`pytest`. When the tests complete,
GitHub indicates the result with a green check mark (pass) or
a red 'X' (fail). Resolve failures before merging a pull request
or making a release.


Test Data
=========

Test data for :func:`test_vald` were generated using the VALD3
`extract stellar` feature and the following parameter values.

=====================  ========  ========  ========  ========
VALD job date          12/30/18  12/30/18  12/30/18  12/30/18
VALD job number        045169    045170    045174    045175
Starting wavelength    6562      656.0188  49958.83  1999
Ending wavelength      6567      656.5186  50025.01  2001
Detection threshold    0.001     0.001     0.001     0.001
Microturbulence        0         0         0         0
Effective temperature  5750      5750      3500      3500
Log surface gravity    4.5       4.5       5.0       5.0
Extraction format      long      short     long      short
Energy level units     eV        cm-1      cm-1      eV
Medium                 vacuum    air       vacuum    air
Wavelength units       Angstrom  nm        cm-1      Angstrom
Van der Waals syntax   extended  extended  extended  extended
Isotopic scaling       on        on        off       off
=====================  ========  ========  ========  ========

*************
Documentation
*************

.. _Sphinx:
   https://www.sphinx-doc.org/en/master/contents.html

.. _reStructuredText:
   http://docutils.sourceforge.net/rst.html

SME uses the `Sphinx`_ documentation builder to translate
`reStructuredText`_ source files into various output formats,
including HTML and PDF. SME documentation source is located in
the directory :file:`SME/docs`.  The master document is
:file:`SME/docs/index.rst`. Additional source files are in the
subdirectory :file:`SME/docs/content`. SME has documentation
for users and for developers.

Online
======

.. _readthedocs:
   https://sme.readthedocs.io

The `SME repository`_ on GitHub is linked to the `readthedocs`_
online documentation service. Updating SME on GitHub causes
readthedocs to rebuild and publish the latest SME documentation
from source.

Local
=====

Use :command:`make` to build a local copy of the documentation in the
directory :file:`SME/docs/_build`. Use the argument `html` to build
HTML documentation and `latexpdf` to build PDF documentation.

.. code-block:: bash

    $ cd SME/docs
    SME$ conda activate sme
    (sme) SME$ make html
    (sme) SME$ make latexpdf

Use a web browser to view the file :file:`_build/html/index.html`.
Use a PDF browser to view the file :file:`_build/latex/SME.pdf`.
On a mac, use `open` to view either type of output format.

.. code-block:: bash

    (sme) SME$ open _build/html/index.html
    (sme) SME$ open _build/latex/SME.pdf

