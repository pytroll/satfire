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

QUALITY_NOT_FIRE = 0
QUALITY_UNKNOWN = 1
QUALITY_LOW = 2
QUALITY_MEDIUM = 3
QUALITY_HIGH = 4


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
        # Cloud mask
        self.cloud_mask = None
        # Result of fire mapping
        self.fires = {}

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

        # Initial mask
        self.mask = self.data[self.config["NIR_CHAN_NAME"]].mask.copy()
        # Apply all masks
        self.mask_data()
        # Find hotspots
        self.find_hotspots()

    def save(self):
        """Save forest fires"""
        pass

    def clean(self):
        """Cleanup after processing."""
        self.data = None
        self.mask = None
        self.cloud_mask = None
        self.fires = {}

    def apply_mask(self, mask):
        """Apply given mask to the product mask"""
        self.mask |= mask

    def resample_aux(self, data, lons, lats):
        """Resample auxiliary data to swath"""
        pass

    def mask_data(self):
        """Create and apply all masks"""
        for func_name in self.config["mask_functions"]:
            read_func = getattr(self, func_name)
            mask = read_func()
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

    def snow_mask(self):
        """Read and resample snow exclusion mask"""
        # Read from NWC SAF?
        mask, lons, lats = utils.read_snow_mask(self.config["mask_file"])
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
        sat_za = np.radians(self.data[self.config["sat_za_name"]])
        sun_za = np.radians(self.data[self.config["sol_za_name"]])
        rel_aa = np.radians(self.data[self.config["rel_az_name"]])
        nir = self.data[self.config["nir_chan_name"]]

        angle_th1 = \
            np.radians(self.config["sun_glint_mask"]["angle_threshold_1"])
        angle_th2 = \
            np.radians(self.config["sun_glint_mask"]["angle_threshold_2"])
        nir_refl_th = self.config["sun_glint_mask"]["nir_refl_threshold"]

        glint = np.arccos(np.cos(sat_za) * np.cos(sun_za) -
                          np.sin(sat_za) * np.sin(sun_za) * np.cos(rel_aa))
        mask = ((glint < angle_th1) |
                ((glint < angle_th2) & (nir > nir_refl_th)))
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
        sza = self.data[self.config["sat_za_name"]]
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

    def find_hotspots(self):
        """Find hotspots from unmasked pixels"""
        day_mask = (self.data[self.config["sol_za_name"]] <
                    self.config["sol_za_day_limit"])
        nir = self.data[self.config["nir_chan_name"]]
        ir1 = self.data[self.config["ir1_chan_name"]]
        delta_nir_ir = nir - ir1

        probs = self.config["probability_levels"]
        for lvl in probs:
            day_nir = probs[lvl]["day"]["temp_nir"]
            day_nir_ir = probs[lvl]["day"]["delta_nir_ir"]
            night_nir = probs[lvl]["night"]["temp_nir"]
            night_nir_ir = probs[lvl][""]["delta_nir_ir"]

            candidates = (
                # Day side
                (day_mask &
                 (nir > day_nir) &
                 (delta_nir_ir > day_nir_ir)) |
                # Night side
                (np.invert(day_mask) &
                 (nir > night_nir) &
                 (delta_nir_ir > night_nir_ir)) &
                # Global mask
                np.invert(self.mask))
            rows, cols = np.nonzeros(candidates)

            for i in range(len(rows)):
                quality = self.qualify_fires(rows[i], cols[i],
                                             is_day=day_mask[rows[i], cols[i]])
                self.fires[(rows[i], cols[i])] = {'quality': quality,
                                                  'probability': lvl}

    def qualify_fires(self, row, col, is_day=True):
        """Check if hotspot at [row, col] is a fire or not."""
        # Get valid background pixels for MIR and IR108 channels around
        # [row, col]
        mir_bg, ir1_bg, masked_dist = self.get_background(row, col)
        if mir_bg is None or ir1_bg is None:
            return QUALITY_UNKNOWN

        mir = self.data[self.config["mir_chan_name"]][row, col]
        ir1 = self.data[self.config["ir1_chan_name"]][row, col]
        diff_mir_ir1 = mir - ir1

        # Calculate statistics
        mean_diff_bg = np.mean(mir_bg - ir1_bg)
        mad_diff_bg = utils.mean_absolute_deviation(mir_bg - ir1_bg)
        mean_ir1_bg = np.mean(ir1_bg)
        mad_ir1_bg = utils.mean_absolute_deviation(ir1_bg)

        if is_day:
            if ((diff_mir_ir1 > mean_diff_bg + mad_diff_bg) and
                    (ir1 > mean_ir1_bg + mad_ir1_bg - 3.)):
                if masked_dist == 3:
                    return QUALITY_LOW
                elif masked_dist == 5:
                    return QUALITY_MEDIUM
                else:
                    return QUALITY_HIGH
            else:
                return QUALITY_NOT_FIRE
        else:
            if (diff_mir_ir1 > mean_diff_bg + mad_diff_bg):
                if masked_dist == 3:
                    return QUALITY_LOW
                elif masked_dist == 5:
                    return QUALITY_MEDIUM
                else:
                    return QUALITY_HIGH
            else:
                return QUALITY_NOT_FIRE

    def get_background(self, row, col):
        """Find background pixels around pixel location [row, col]."""
        pass
