#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="sme",
    version="0.0",
    description="Spectroscopy Made Easy",
    author="Jeff A. Valenti",
    author_email="valenti@stsci.edu",
    packages=find_packages(),
    #     package_dir={"": "src"},
    #     package_data={"sme": ["dll/sme_synth.so.*", "dll/intel64_lin/*"]},
)
