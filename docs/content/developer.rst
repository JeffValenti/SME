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

    $ source activate sme
    (sme) $

To stop using the *sme* environment in a terminal window, close the
window or explicitly deactivate the current environment.

.. code-block:: bash

    (sme) $ source deactivate
    $

Create environment
==================

To create an environment for SME development:

1. Use :command:`bash` shell because it is fully compatible with
:command:`conda`. Do not use :command:`tcsh` shell because it has
a built-in :command:`source` command that masks a :command:`conda`
executable with the same name.

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
the `SME github repository`_ to a local :file:`sme` directory.

.. _SME github repository:
   https://github.com/JeffValenti/SME

.. code-block:: bash

    $ mkdir sme
    $ cd sme
    $ git clone https://github.com/JeffValenti/SME.git

.. _conda-env-create-step:

4. `Create a conda environment`_ named *sme* using the
:file:`environment.yml` file distributed with SME. The name of the
new environment (*sme*) is specified in :file:`environment.yml`.


.. _Create a conda environment:
   https://conda.io/docs/user-guide/tasks/manage-environments.html
   #creating-an-environment-from-an-environment-yml-file

.. code-block:: bash

    $ cd sme
    $ conda env create -f environment.yml

5. Install SME from the local repository. Activate the *sme* environment,
so that SME is installed in the *sme* environment, not the *base*
environment. Execute :command:`pip install` from the top-level directory
of the local SME repository (typically :file:`sme`). Use the `--editable`
option to keep the installed version synchronized with changes to the
local repository.

.. code-block:: bash

    $ source activate sme
    (sme) $ pip install --editable .


Update environment
==================

Periodically update packages in the *sme* environment to obtain the
enviroment a new user or Travis CI would create.

.. code-block:: bash

    $ source activate sme
    (sme) $ conda update

Recreate environment
====================

Recreate the *sme* environment, if the :file:`environment.yml` file
changes or to restore the *sme* environment to its canonical state.
First remove the existing *sme* environment.

.. code-block:: bash

    (sme) $ source deactivate
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

    $ source activate sme
    (sme) $ conda install <package>

If an additional package is not available via a conda channel, use
:command:`pip install` while in the *sme* environment. Package
management is not as robust when using :command:`pip` to install in
a :command:`conda` environment.

.. code-block:: bash

    $ source activate sme
    (sme) $ pip install <package>

Additional packages will be available in the local *sme* environment,
but not in Travis CI or other user environments. To add packages for
all SME users, update the :file:`environment.yml` file and recreate
the local *sme* environment.

**********
Test Files
**********

VALD3 generated data files for :func:`test_vald` using parameters
in the following table.

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
