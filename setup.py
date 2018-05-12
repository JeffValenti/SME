#!/usr/bin/env python

from distutils.core import setup

setup(
        name='sme',
        version='0.0',
        description='Spectroscopy Made Easy',
        author='Jeff A. Valenti',
        author_email='valenti@stsci.edu',
        package_dir={'': 'src'},
        packages=['sme']
        include_package_data=True
        )
