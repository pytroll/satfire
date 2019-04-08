#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>


"""Unit testing for utils
"""

import sys
import os.path
from collections import OrderedDict

import numpy as np

from satfire import utils

from posttroll.message import Message

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
        keys = list(config.keys())
        self.assertTrue(isinstance(config, OrderedDict))
        self.assertEqual(keys[0], 'item_1')
        self.assertTrue(isinstance(config['item_1'], str))
        self.assertTrue(isinstance(config['item_2'], list))
        self.assertTrue(isinstance(config['item_3'], OrderedDict))
        self.assertTrue(isinstance(config['item_4'], int))

    def test_get_filenames_from_msg(self):
        config = {"cma_message_tag": "pps",
                  "sat_message_tag": "hrpt"}
        cma_fname = "/tmp/foo.nc"
        sat_fname = "/tmp/bar.l1b"

        # Both files present
        data = {"collection":
                {"pps":
                 {"dataset":
                  [{"uri": cma_fname}]},
                 "hrpt":
                 {"dataset":
                  [{"uri": sat_fname}]}}}
        msg = Message("/topic", "collection", data)
        sat, cma = utils.get_filenames_from_msg(msg, config)
        self.assertEqual(sat, sat_fname)
        self.assertEqual(cma, cma_fname)

        # Only satellite file
        data = {"collection":
                {"hrpt":
                 {"dataset":
                  [{"uri": sat_fname}]}}}
        msg = Message("/topic", "collection", data)
        sat, cma = utils.get_filenames_from_msg(msg, config)
        self.assertEqual(sat, sat_fname)
        self.assertIsNone(cma)

        # Only cloud mask file
        data = {"collection":
                {"pps":
                 {"dataset":
                  [{"uri": cma_fname}]}}}
        msg = Message("/topic", "collection", data)
        sat, cma = utils.get_filenames_from_msg(msg, config)
        self.assertEqual(cma, cma_fname)
        self.assertIsNone(sat)

        # No files
        data = {"collection": {}}
        msg = Message("/topic", "dataset", data)
        sat, cma = utils.get_filenames_from_msg(msg, config)
        self.assertIsNone(cma)
        self.assertIsNone(sat)

    def test_get_idxs_around_location(self):
        side = 5
        # Note that the centre pixel is always masked out
        y_cor = np.array([0, 1, 2, 3, 4,
                          0, 1, 2, 3, 4,
                          0, 1,    3, 4,
                          0, 1, 2, 3, 4,
                          0, 1, 2, 3, 4])
        x_cor = np.array([0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1,
                          2, 2,    2, 2,
                          3, 3, 3, 3, 3,
                          4, 4, 4, 4, 4])
        y_res, x_res = utils.get_idxs_around_location(2, 2, side,
                                                      remove_neighbours=False)
        self.assertTrue(y_res.size == 24)
        self.assertTrue(x_res.size == 24)
        self.assertTrue((y_cor == y_res).all())
        self.assertTrue((x_cor == x_res).all())

        side = 5
        y_cor = np.array([0, 1, 2, 3, 4,
                          0, 4,
                          0, 4,
                          0, 4,
                          0, 1, 2, 3, 4])
        x_cor = np.array([0, 0, 0, 0, 0,
                          1, 1,
                          2, 2,
                          3, 3,
                          4, 4, 4, 4, 4])
        y_res, x_res = utils.get_idxs_around_location(2, 2, side,
                                                      remove_neighbours=True)
        self.assertTrue(y_res.size == side * side - 9)
        self.assertTrue(x_res.size == side * side - 9)
        self.assertTrue((y_cor == y_res).all())
        self.assertTrue((x_cor == x_res).all())

    def test_calc_footprint_size(self):
        sat_zens = np.array([0, 68.5])
        ifov = 1.4e-3
        sat_alt = 830.
        max_swath_width = 1446.58

        along, across = utils.calc_footprint_size(sat_zens, ifov, sat_alt,
                                                  max_swath_width)
        self.assertAlmostEqual(along[0], 1.16, 2)
        self.assertAlmostEqual(along[1], 2.46, 2)
        self.assertAlmostEqual(across[0], 1.16, 2)
        self.assertAlmostEqual(across[1], 6.70, 2)

    def test_haversine(self):
        lon1, lat1 = 25., 60.
        lon2, lat2 = 21.3, 68.3
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=True)
        self.assertAlmostEqual(dists[0], 939.8, 1)
        self.assertAlmostEqual(bearings[0], 350.66, 2)

        lon1, lat1 = 0, 0
        lon2, lat2 = 0, 90
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=True)
        self.assertAlmostEqual(dists[0], 10007.9, 1)
        self.assertAlmostEqual(bearings[0], 0.0, 1)

        lon1, lat1 = 0, 0
        lon2, lat2 = 90, 0
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=True)
        self.assertAlmostEqual(dists[0], 10007.9, 1)
        self.assertAlmostEqual(bearings[0], 90.0, 1)

        lon1, lat1 = 0, 0
        lon2, lat2 = -90, 0
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=True)
        self.assertAlmostEqual(dists[0], 10007.9, 1)
        self.assertAlmostEqual(bearings[0], 270.0, 1)

        lon1, lat1 = 0, 0
        lon2, lat2 = 0, -90
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=True)
        self.assertAlmostEqual(dists[0], 10007.9, 1)
        self.assertAlmostEqual(bearings[0], 180.0, 1)

        lon1, lat1 = 0, 0
        lon2, lat2 = 0, -90
        dists, bearings = utils.haversine(lon1, lat1, lon2, lat2,
                                          calc_bearings=False)
        self.assertAlmostEqual(dists[0], 10007.9, 1)
        self.assertIsNone(bearings)

    def test_ensure_numpy(self):
        res = utils.ensure_numpy(1, dtype=None)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue(res.dtype == np.int64)
        self.assertEqual(res[0], 1)

        res = utils.ensure_numpy(1, dtype=np.float32)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue(res.dtype == np.float32)
        self.assertEqual(res[0], 1.0)

        res = utils.ensure_numpy([1], dtype=np.float32)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue(res.dtype == np.float32)
        self.assertEqual(res[0], 1.0)

        res = utils.ensure_numpy(np.array([1]), dtype=np.float32)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue(res.dtype == np.float32)
        self.assertEqual(res[0], 1.0)

        res = utils.ensure_numpy(np.array(1), dtype=np.float32)
        self.assertTrue(isinstance(res, np.ndarray))
        self.assertTrue(res.dtype == np.float32)
        self.assertEqual(res[0], 1.0)


def suite():
    """The suite for test_utils
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestUtils))

    return mysuite

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
