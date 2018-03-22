#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>


"""Unit testing for ForestFire class
"""

import sys
import os.path

import numpy as np

from fffsat.forest_fire import ForestFire
from fffsat import forest_fire
from fffsat import utils

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestForestFire(unittest.TestCase):

    config = utils.read_config(os.path.join(os.path.dirname(__file__),
                                            "test_data", "config.yaml"))
    data_fname = os.path.join(os.path.dirname(__file__),
                              "test_data", "test_sat_data.npz")
    fff = ForestFire(config)

    def test_init(self):
        self.assertEqual(self.fff.config, self.config)
        self.assertIsNone(self.fff.data)
        self.assertIsNone(self.fff.mask)
        self.assertIsNone(self.fff.cloud_mask)
        self.assertEqual(len(self.fff.fires), 0)
        self.assertTrue(isinstance(self.fff.fires, dict))

    def test_clean(self):
        self.fff.data = 'a'
        self.fff.mask = 'b'
        self.fff.cloud_mask = 'c'
        self.fires = 'd'
        self.fff.clean()
        self.assertIsNone(self.fff.data)
        self.assertIsNone(self.fff.mask)
        self.assertIsNone(self.fff.cloud_mask)
        self.assertEqual(len(self.fff.fires), 0)
        self.assertTrue(isinstance(self.fff.fires, dict))

    def test_create_cloud_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.create_cloud_mask()
        # Certain clear-sky pixels
        self.assertFalse(res[19, 28])
        self.assertFalse(res[15, 35])
        self.assertFalse(res[47, 49])
        # Certain cloud pixels
        self.assertTrue(res[37, 20])
        self.assertTrue(res[35, 28])
        self.assertTrue(res[34, 47])
        # Clean data
        self.fff.clean()

    def test_get_nwc_mask(self):
        # Mask is available
        # self.fff.data = read_sat_data(self.data_fname, self.config)
        self.fff.nwc_mask = 3
        res = self.fff.get_nwc_mask()
        self.assertEqual(res, 3)
        self.fff.clean()
        # Create from swath data
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.get_cloud_mask()
        # Certain clear-sky ground pixels
        self.assertFalse(res[19, 28])
        self.assertFalse(res[15, 35])
        self.assertFalse(res[47, 49])
        # Certain cloud pixels
        self.assertTrue(res[37, 20])
        self.assertTrue(res[35, 28])
        self.assertTrue(res[34, 47])
        # Clean data
        self.fff.clean()

    def test_create_water_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.create_water_mask()
        # Certain water pixels
        self.assertTrue(res[3, 2])
        self.assertTrue(res[19, 17])
        self.assertTrue(res[22, 35])
        # Certain clear-sky ground pixels
        self.assertFalse(res[19, 28])
        self.assertFalse(res[15, 35])
        self.assertFalse(res[47, 49])

        # Clean data
        self.fff.clean()

    def test_create_fcv_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.create_fcv_mask()
        # Certain water pixels
        self.assertTrue(res[3, 2])
        self.assertTrue(res[19, 17])
        self.assertTrue(res[22, 35])

    def test_create_swath_edge_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.create_swath_edge_mask()
        # Original data contains no swath edges
        self.assertFalse(np.any(res))
        # Add artificial border area
        self.fff.data[self.config["sat_za_name"]][:, 0] = \
            self.config["swath_edge_mask"]["threshold"] + 1.
        res = self.fff.create_swath_edge_mask()
        self.assertTrue(np.all(res[:, 0]))
        self.fff.clean()

    def test_create_swath_end_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        # Add initial mask
        self.fff.mask = self.fff.data['mask'].copy()
        # Get the mask length from config
        length = self.config["swath_end_mask"]["threshold"]
        res = self.fff.create_swath_end_mask()
        self.assertTrue(np.all(res[0, :]))
        self.assertTrue(np.all(res[-1, :]))
        self.assertTrue(np.sum(res[:, 0]) == (2 * length))
        self.fff.clean()

    def test_create_swath_masks(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        # Add initial mask
        self.fff.mask = self.fff.data['mask'].copy()
        # Add artificial border area
        self.fff.data[self.config["sat_za_name"]][:, 0] = \
            self.config["swath_edge_mask"]["threshold"] + 1.
        # Get the mask length from config
        length = self.config["swath_end_mask"]["threshold"]
        res = self.fff.create_swath_masks()
        self.assertTrue(np.all(res[:, 0]))
        self.assertTrue(np.all(res[0, :]))
        self.assertTrue(np.all(res[-1, :]))
        self.assertTrue(np.sum(res[:, 10]) == (2 * length))
        self.fff.clean()

    def test_create_sun_glint_mask(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        res = self.fff.create_sun_glint_mask()
        # Original data contains no sun glint areas
        self.assertFalse(np.any(res))
        # Add one pixel where Sun and satellite are in the same line
        # -> sun glint
        self.fff.data[self.config["sat_za_name"]][0, 0] = 0.0
        self.fff.data[self.config["sol_za_name"]][0, 0] = 0.0
        self.fff.data[self.config["rel_az_name"]][0, 0] = 0.0
        res = self.fff.create_sun_glint_mask()
        self.assertTrue(res[0, 0])
        # One pixel where Sun and satellite are on the same azimuth line at
        # same elevation -> sun glint
        self.fff.data[self.config["sat_za_name"]][0, 0] = 5.0
        self.fff.data[self.config["sol_za_name"]][0, 0] = 5.0
        self.fff.data[self.config["rel_az_name"]][0, 0] = 180.0
        res = self.fff.create_sun_glint_mask()
        self.assertTrue(res[0, 0])
        # One pixel where Sun and satellite are on the same azimuth line, but
        # elevation difference is between the two thresholds and 0.8
        # reflectance is BELOW threshold -> no sun glint
        self.fff.data[self.config["sat_za_name"]][0, 0] = 0.0
        self.fff.data[self.config["sol_za_name"]][0, 0] = \
            self.config["sun_glint_mask"]["angle_threshold_1"] + 1.0
        self.fff.data[self.config["rel_az_name"]][0, 0] = 180.0
        res = self.fff.create_sun_glint_mask()
        self.assertFalse(res[0, 0])
        # One pixel where Sun and satellite are on the same azimuth line, but
        # elevation difference is between the two thresholds and 0.8
        # reflectance is ABOVE threshold -> sun glint
        self.fff.data[self.config["sat_za_name"]][0, 0] = 0.0
        self.fff.data[self.config["sol_za_name"]][0, 0] = \
            self.config["sun_glint_mask"]["angle_threshold_1"] + 1.0
        self.fff.data[self.config["rel_az_name"]][0, 0] = 180.0
        self.fff.data[self.config["nir_chan_name"]][0, 0] = \
            self.config["sun_glint_mask"]["nir_refl_threshold"] + 1.0
        res = self.fff.create_sun_glint_mask()
        self.assertTrue(res[0, 0])
        # Sun angle is larger than second threshold -> no glint
        self.fff.data[self.config["sol_za_name"]][0, 0] = \
            self.config["sun_glint_mask"]["angle_threshold_2"] + 1.0
        res = self.fff.create_sun_glint_mask()
        self.assertFalse(res[0, 0])
        self.fff.clean()

    def test_get_background(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        self.fff.mask = self.fff.data['mask'].copy()
        # Original data, no masked pixels
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 16)
        self.assertEqual(ir1.size, 16)
        self.assertEqual(quality, forest_fire.QUALITY_HIGH)
        # Set one pixel masked inside 6x6 box
        self.fff.mask[11, 11] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 16)
        self.assertEqual(ir1.size, 16)
        self.assertEqual(quality, forest_fire.QUALITY_HIGH)
        # Set one pixel masked inside 5x5 box
        self.fff.mask[10, 10] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 15)
        self.assertEqual(ir1.size, 15)
        self.assertEqual(quality, forest_fire.QUALITY_MEDIUM)
        # Set another pixel masked inside 5x5 box
        self.fff.mask[9, 10] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 14)
        self.assertEqual(ir1.size, 14)
        self.assertEqual(quality, forest_fire.QUALITY_MEDIUM)
        # Set one pixel masked inside 3x3 box
        self.fff.mask[9, 9] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 14)  # neighbours are removed, so size stays
        self.assertEqual(ir1.size, 14)  # the same
        self.assertEqual(quality, forest_fire.QUALITY_LOW)
        # Test that enough background pixels (number and fraction) are found
        # Mask 5x5 area
        self.fff.mask[:, :] = False
        self.fff.mask[6:11, 6:11] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertEqual(mir.size, 24)
        self.assertEqual(ir1.size, 24)
        self.assertEqual(quality, forest_fire.QUALITY_LOW)
        # Mask everything
        self.fff.mask[:, :] = True
        mir, ir1, quality = self.fff.get_background(8, 8)
        self.assertIsNone(mir)
        self.assertIsNone(ir1)
        # Quality is "low", but it'll be changed to "not fire" in
        # qualify_fires() because there were no valid data
        self.assertEqual(quality, forest_fire.QUALITY_LOW)
        self.fff.clean()

    def test_qualify_fires(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        self.fff.mask = self.fff.data['mask'].copy()
        # Water should not be on fire
        res = self.fff.qualify_fires(8, 8, is_day=True)
        self.assertEqual(res, forest_fire.QUALITY_NOT_FIRE)
        # Set water on fire
        self.fff.data[self.config["mir_chan_name"]][8, 8] = 340.
        res = self.fff.qualify_fires(8, 8, is_day=True)
        self.assertEqual(res, forest_fire.QUALITY_HIGH)
        # Fire on the ground
        self.fff.data[self.config["mir_chan_name"]][15, 33] = 340.
        res = self.fff.qualify_fires(15, 33, is_day=True)
        self.assertEqual(res, forest_fire.QUALITY_HIGH)
        # Mask everything
        self.fff.mask[:, :] = True
        res = self.fff.qualify_fires(15, 33, is_day=True)
        self.assertEqual(res, forest_fire.QUALITY_UNKNOWN)
        # TODO: add night-time tests
        self.fff.clean()

    def test_mask_data(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        self.fff.mask = self.fff.data['mask'].copy()
        self.fff.mask_data()
        # Few masked pixels
        self.assertTrue(np.all(self.fff.mask[0:20, 0:15]))  # water
        self.assertTrue(np.all(self.fff.mask[33:40, 22:30]))  # cloud
        # Unmasked pixels
        self.assertFalse(np.any(self.fff.mask[18:22, 26:32]))
        self.assertFalse(np.any(self.fff.mask[45:49, 3:6]))
        self.assertFalse(np.any(self.fff.mask[13:18, 30:37]))
        self.fff.clean()

    def test_find_hotspots(self):
        self.fff.data = read_sat_data(self.data_fname, self.config)
        self.fff.mask = self.fff.data['mask'].copy()
        self.assertEqual(len(self.fff.fires), 0)
        self.fff.find_hotspots()
        # No fires on this scene
        self.assertEqual(len(self.fff.fires), 0)
        # Add a fire pixel
        self.fff.data[self.config["mir_chan_name"]][15, 33] = 340.
        self.fff.find_hotspots()
        self.assertEqual(len(self.fff.fires), 1)
        for key in self.fff.fires:
            self.assertEqual(
                self.fff.fires[key]['quality'],
                forest_fire.QUALITY_NAMES[forest_fire.QUALITY_HIGH])
            self.assertEqual(
                self.fff.fires[key]['probability'],
                forest_fire.QUALITY_NAMES[forest_fire.QUALITY_HIGH])
        self.fff.clean()

    def test_get_confidence(self):
        # Low probability
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_LOW,
                                                    forest_fire.QUALITY_LOW),
                         forest_fire.CONFIDENCE_LOW)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_LOW,
                                                    forest_fire.QUALITY_MEDIUM),
                         forest_fire.CONFIDENCE_LOW)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_LOW,
                                                    forest_fire.QUALITY_HIGH),
                         forest_fire.CONFIDENCE_LOW)
        # Mediuma probability
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_MEDIUM,
                                                    forest_fire.QUALITY_LOW),
                         forest_fire.CONFIDENCE_LOW)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_MEDIUM,
                                                    forest_fire.QUALITY_MEDIUM),
                         forest_fire.CONFIDENCE_NOMINAL)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_MEDIUM,
                                                    forest_fire.QUALITY_HIGH),
                         forest_fire.CONFIDENCE_NOMINAL)
        # High probability
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_HIGH,
                                                    forest_fire.QUALITY_LOW),
                         forest_fire.CONFIDENCE_NOMINAL)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_HIGH,
                                                    forest_fire.QUALITY_MEDIUM),
                         forest_fire.CONFIDENCE_HIGH)
        self.assertEqual(forest_fire.get_confidence(forest_fire.PROBABILITY_HIGH,
                                                    forest_fire.QUALITY_HIGH),
                         forest_fire.CONFIDENCE_HIGH)


def read_sat_data(fname, config):
    """Read test satellite data and put it in dictionary with names in the
    given config"""
    data = {}
    with np.load(fname) as fid:
        mask = fid['mask']
        data[config['vis_chan_name']] = np.ma.masked_where(mask, fid['vis'])
        data[config['nir_chan_name']] = np.ma.masked_where(mask, fid['nir'])
        data[config['mir_chan_name']] = np.ma.masked_where(mask, fid['mir'])
        data[config['ir1_chan_name']] = np.ma.masked_where(mask, fid['ir1'])
        data[config['ir2_chan_name']] = np.ma.masked_where(mask, fid['ir2'])
        data[config['sol_za_name']] = np.ma.masked_where(mask, fid['sol_za'])
        data[config['sat_za_name']] = np.ma.masked_where(mask, fid['sat_za'])
        data[config['rel_az_name']] = np.ma.masked_where(mask, fid['rel_az'])
        data[config['lat_name']] = np.ma.masked_where(mask, fid['lat'])
        data[config['lon_name']] = np.ma.masked_where(mask, fid['lon'])
        data['mask'] = mask

    return data


def suite():
    """The suite for test_utils
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestForestFire))

    return mysuite

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
