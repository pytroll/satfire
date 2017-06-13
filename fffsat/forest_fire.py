#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

import logging

import numpy as np

from fffsat import utils


class ForestFire(object):

    """Class for creating forest fire hot spots based on algorithm by
    Planck et. al."""

    logger = logging.get_logger("ForestFire")

    def __init__(self, config):
        # Configuration dictionary for ForestFire class
        self.config = config
        # Unprojected satpy scene with all the required channels, angles and
        # coordinates
        self.data = None
        # Common mask for all the datasets in self.data
        # All invalid pixels are set as True.  After processing the locations
        # marked as False are the valid forest fires.
        self.mask = None
        # Probability of fire
        self.probability = None
        # Quality of the hot spot retrieval
        self.quality = None
        # Cloud mask
        self.cloud_mask = None

    def run(self, msg=None, sat_fname=None, cma_fname=None):
        """Run everything"""
        if msg is not None:
            sat_fname, cma_fname = utils.get_filenames_from_msg(msg)
        if sat_fname is None:
            self.logger.critical("No satellite data in message")
            return False
        self.data = utils.read_sat_data(sat_fname,
                                        self.config["channels_to_load"])
        if cma_fname is not None:
            self.cloud_mask = utils.read_cma(cma_fname)

        self.mask = self.data[self.config["NIR_CHAN_NAME"]].mask.copy()

        self.mask_data()

    def save(self):
        """Save forest fires"""
        pass

    def clean(self):
        """Cleanup after processing."""
        self.data = None
        self.mask = None
        self.probability = None
        self.quality = None
        self.cloud_mask = None

    def apply_mask(self, mask):
        """Apply given mask to the product mask"""
        self.mask |= mask

    def resample_aux(self, data, lons, lats):
        """Resample auxiliary data to swath"""
        pass

    def mask_data(self):
        """Create and apply all masks"""
        for func_name in self.config["mask_functions"]:
            func_conf = self.config["mask_functions"][func_name]
            read_func = getattr(self, func_name)
            mask = read_func(func_conf)
            self.apply_mask(mask)

    # Static masks read from a file and resampled to match the swath

    def land_cover_mask(self):
        """Read and resample land cover exclusion mask"""
        mask, lons, lats = \
            utils.read_land_cover(self.config["land_cover_mask"]["mask_file"])
        mask = self.resample_aux(mask, lons, lats)
        return mask

    # Masking based on data in satellite projection, either external
    # or calculated from reflectances/BTs

    def snow_mask(self, config):
        """Read and resample snow exclusion mask"""
        # Read from NWC SAF?
        mask, lons, lats = utils.read_snow_mask(config["mask_file"])
        mask = self.resample_aux(mask, lons, lats)
        return mask

    def cloud_mask(self):
        """Get exclusion mask"""
        if self.cloud_mask is not None:
            return self.cloud_mask
        else:
            self.logger.warning("NWC SAF cloud mask not available")
            return self.create_cloud_mask()

    def water_mask(self):
        """Create water mask"""
        # eq. 1 from Planck et. al.
        # maybe not use this?
        vis = self.data[self.config["vis_chan_name"]]
        nir = self.data[self.config["nir_chan_name"]]
        threshold = self.config["water_mask"]["threshold"]
        mean_vis_nir = (vis + nir) / 2.
        std_vis_nir = np.std(np.dstack((vis.data, nir.data)), axis=2)
        mask = ((mean_vis_nir ** 2 / std_vis_nir) < threshold) & (vis > nir)
        return mask

    def sun_glint_mask(self):
        """Create Sun glint mask"""
        # eq. 5 - 8 from Planck et. al.
        mask = None
        return mask

    def fcv_mask(self):
        """Calculate fraction of vegetation exclusion mask"""
        # eq. 9 and 10 from Planck et.al. 2017
        ch1 = self.data[self.config["vis_chan_name"]]
        ch2 = self.data[self.config["nir_chan_name"]]
        ndvi = (ch2 - ch1) / (ch2 + ch1)
        ndvi_min = np.min(ndvi)
        ndvi_max = np.max(ndvi)
        fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
        mask = fvc < self.config["fcv_mask"]["threshold"]
        return mask

    def swath_edge_mask(self):
        """Create mask for the swath edges"""
        sza = self.data["sensor_zenith_angle"]
        threshold = self.config["swath_edge_mask"]["threshold"]
        mask = sza > threshold
        return mask

    def create_cloud_mask(self):
        """Create cloud mask from satellite data."""
        # Planck et.al. 2017 eq. 2 - 4
        vis = self.data[self.config["vis_chan_name"]]
        nir = self.data[self.config["nir_chan_name"]]
        mir = self.data[self.config["mir_chan_name"]]
        ir1 = self.data[self.config["ir1_chan_name"]]
        ir2 = self.data[self.config["ir2_chan_name"]]

        w_mir = np.exp((310. - mir) / 20.)
        w_delta_ch = np.exp((mir - (ir1 + ir2) / 2. - 14.) / 14.)
        mean_vis_nir = (vis + nir) / 2.

        cloud_th = self.config["cloud_mask"]["threshold"]
        clouds = (w_mir * w_delta_ch * mean_vis_nir) > cloud_th

        return clouds


def select_candidates(prob_level):
    """Select fire candidates based on the probability level"""
    pass


def calc_bg_nir_ir_mean():
    """Calculate mean difference between NIR and IR channels for the valid
    background pixels.  Used for day & night cases."""
    pass


def calc_bg_nir_ir_abs_dev():
    """Calculate mean absolute deviation of NIR and IR channel difference
    for the valid background pixels.  Used for day & night cases
    """
    # eq. 11 in Planck et. al.
    pass


def calc_bg_ir_mean():
    """Calculate mean of the valid background pixels of IR channel.  Used
    for only the night cases."""
    pass


def calc_bg_ir_abs_dev():
    """Calculate mean absolute deviation of IR channel for the valid
    background pixels.  Used only for night cases."""
    pass


def check_which_are_fires():
    """Check which of the candidate pixels are fires"""
    # eq. 12 (day) and 13 (night) of Planck et. al.
    pass


def calc_quality():
    """Calculate quality of the fire pixel by checking how far is the
    closest invalid pixel."""
    # table 4 of Planck et. al.
    pass
