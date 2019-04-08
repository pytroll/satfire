#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
"""The tests package."""

import unittest
# import doctest
from satfire.tests import (test_utils,
                           test_forest_fire)


def suite():
    """The global test suite.
    """
    mysuite = unittest.TestSuite()
    # Test the documentation strings
    # mysuite.addTests(doctest.DocTestSuite(image))
    # Use the unittests also
    mysuite.addTests(test_utils.suite())
    mysuite.addTests(test_forest_fire.suite())

    return mysuite
