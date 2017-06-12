#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Panu Lahtinen / FMI

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi>

"""Setup for fffsat
"""
from setuptools import setup
import imp

version = imp.load_source('fffsat.version', 'fffsat/version.py')

setup(name="fffsat",
      version=version.__version__,
      description='Forest fire detection based on satellite imager from ' +
      'AVHRR/3 and VIIRS instruments',
      author='Panu Lahtinen',
      author_email='panu.lahtinen@fmi.fi',
      url="https://github.com/fmidev/fmi-forest-fire-satellite",
      packages=['fffsat',
                'fffsat.tests'
                ],
      data_files=[],
      zip_safe=False,
      install_requires=['pyyaml', 'satpy', 'trollflow', 'trollflow-sat'],
      tests_require=['mock', 'pyyaml', 'satpy', 'trollflow', 'trollflow-sat',
                     'posttroll'],
      test_suite='fffsat.tests.suite',
      )
