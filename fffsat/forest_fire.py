#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

import numpy as np

from fffsat import utils


class ForestFire(object):

    """Class for creating forest fire hot spots based on algorithm by
    Planck et. al."""

    def __init__(self, config):
        self.config = config
        self.data = None
        self.mask = None
        self.msg = None

    def run(self, msg):
        """Run everything"""
        self.msg = msg
        self.data = utils.read_sat_data(msg)
        self.mask = self.data[self.config["NIR_CHAN_NAME"]].mask.copy()

        self.mask_data()

    def save(self):
        """Save forest fires"""
        pass

    def clean(self):
        """Cleanup after processing."""
        self.data = None
        self.mask = None
        self.msg = None

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

    def land_cover_mask(self, config):
        """Read and resample land cover exclusion mask"""
        mask, lons, lats = utils.read_land_cover(config["mask_file"])
        mask = self.resample_aux(mask, lons, lats)
        return mask

    def snow_mask(self, config):
        """Read and resample snow exclusion mask"""
        mask, lons, lats = utils.read_snow_mask(config["mask_file"])
        mask = self.resample_aux(mask, lons, lats)
        return mask

    # Masking based on data in satellite projection, either external
    # or calculated from reflectances/BTs

    def cloud_mask(self, config):
        """Read cloud exclusion mask"""
        # Get cloud mask filename from the message
        fname = None  # msg[""]
        mask = utils.read_cloud_mask(fname)
        return mask

    def water_mask(self, config):
        """Create water mask"""
        # eq. 1 from Planck et. al.
        # maybe not use this?
        mask = None
        return mask

    def sun_glint_mask(self, config):
        """Create Sun glint mask"""
        # eq. 5 - 8 from Planck et. al.
        mask = None
        return mask

    def fcv_mask(self, config):
        """Calculate fraction of vegetation exclusion mask"""
        # eq. 9 and 10 from Planck et. al
        ch1 = self.data[self.config["VIS_CHAN_NAME"]]
        ch2 = self.data[self.config["NIR_CHAN_NAME"]]
        ndvi = (ch2 - ch1) / (ch2 + ch1)
        ndvi_min = np.min(ndvi)
        ndvi_max = np.max(ndvi)
        fvc = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min)) ** 2
        mask = fvc < config["fvc_threshold"]
        return mask

    def swath_edge_mask():
        """Create mask for the swath edges"""
        mask = None
        return mask


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
