#!/usr/bin/env python

from setuptools import setup

setup(
        name='sme',
        version='0.0',
        description='Spectroscopy Made Easy',
        author='Jeff A. Valenti',
        author_email='valenti@stsci.edu',
        packages=['sme'],
        package_dir={'': 'src'},
        package_data={'sme': ['dll/sme_synth.so.*', 'dll/intel64_lin/*']}
        )
