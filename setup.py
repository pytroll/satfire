#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Panu Lahtinen / FMI

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi>

"""Setup for satfire
"""
from setuptools import setup
import imp

version = imp.load_source('satfire.version', 'satfire/version.py')

setup(name="satfire",
      version=version.__version__,
      description='Forest fire detection based on satellite imager from ' +
      'AVHRR/3 and VIIRS instruments',
      author='Panu Lahtinen',
      author_email='panu.lahtinen@fmi.fi',
      url="https://github.com/pytroll/satfire",
      packages=['satfire',
                'satfire.tests'
                ],
      data_files=[],
      zip_safe=False,
      install_requires=['pyyaml', 'satpy', 'trollflow', 'trollflow-sat'],
      tests_require=['mock', 'pyyaml', 'satpy', 'trollflow', 'trollflow-sat',
                     'posttroll'],
      test_suite='satfire.tests.suite',
      )
