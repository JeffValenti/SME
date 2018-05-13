#!/usr/bin/env python

from distutils.core import setup

setup(
        name='sme',
        version='0.0',
        description='Spectroscopy Made Easy',
        author='Jeff A. Valenti',
        author_email='valenti@stsci.edu',
        packages=['sme'],
        package_dir={'sme': 'src/sme'},
        package_data={'sme': ['dll/sme_synth.so.*']}
        )
