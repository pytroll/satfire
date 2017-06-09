#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>


"""Unit testing for foo
"""

import sys
import os.path
from collections import OrderedDict

from fffsat import utils

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class TestUtils(unittest.TestCase):

    yaml_config = """config:
    item_1: 1
    item_2: 2
    """

    def test_ordered_load(self):
        fid = StringIO(self.yaml_config)
        res = utils.ordered_load(fid)
        fid.close()
        self.assertTrue(list(res.keys())[0] == "config")
        keys = list(res["config"].keys())
        self.assertTrue(keys[0] == "item_1")
        self.assertTrue(res["config"][keys[0]] == 1)
        self.assertTrue(keys[1] == "item_2")
        self.assertTrue(res["config"][keys[1]] == 2)

    def test_read_config(self):
        config = utils.read_config(os.path.join(os.path.dirname(__file__),
                                                "test_data", "config.yaml"))
        self.assertTrue(len(config) > 0)
        keys = config.keys()
        self.assertTrue(isinstance(config, OrderedDict))
        self.assertEqual(keys[0], 'item_1')
        self.assertTrue(isinstance(config['item_1'], str))
        self.assertTrue(isinstance(config['item_2'], list))
        self.assertTrue(isinstance(config['item_3'], OrderedDict))
        self.assertTrue(isinstance(config['item_4'], int))


def suite():
    """The suite for test_utils
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestUtils))

    return mysuite

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
