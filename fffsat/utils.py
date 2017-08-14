#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
"""Utility functions for FFFsat"""

import os.path
import yaml
from collections import OrderedDict

import numpy as np
import h5py

from satpy import Scene
from trollsift import parse

PROBABILITY_LOW = 2
PROBABILITY_MEDIUM = 3
PROBABILITY_HIGH = 4

QUALITY_NOT_FIRE = 0
QUALITY_UNKNOWN = 1
QUALITY_LOW = 2
QUALITY_MEDIUM = 3
QUALITY_HIGH = 4

BOX_SIZE_TO_QUALITY = {3: QUALITY_LOW,
                       5: QUALITY_MEDIUM,
                       7: QUALITY_HIGH}

# Earth radius
R_EARTH = 6371.2200


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def read_config(fname):
    """Read YAML config"""
    with open(fname, 'r') as fid:
        config = ordered_load(fid)

    return config


def get_filenames_from_msg(msg, config):
    """Find filenames for satellite data and cloud mask data from the message.
    """
    sat_fname = None
    cma_fname = None
    for dset in msg.data['dataset']:
        fname = dset['uri']
        # Try matching to filename patterns given in config
        try:
            parse(config["data_fname_pattern"], os.path.basename(fname))
            sat_fname = fname
        except ValueError:
            pass
        try:
            parse(config["cloud_mask_fname_pattern"], os.path.basename(fname))
            cma_fname = fname
        except ValueError:
            pass

    return sat_fname, cma_fname


def read_sat_data(fname, channels):
    """Read satellite data"""
    if not isinstance(fname, (list, set, tuple)):
        fname = [fname, ]
    glbl = Scene(filenames=fname)
    glbl.load(channels)

    return glbl


def check_globcover(fname, idxs, lonlats, footprints, settings):
    """Check globcover mask."""
    lons, lats = lonlats
    along, across = footprints
    max_swath_width = settings['max_swath_width']
    with h5py.File(fname, 'r') as fid:
        for i in range(len(idxs)):
            # Skip already masked pixels
            if idxs[i] is True:
                continue
            # Get mask data that covers satellite footprint
            data = get_footprint_data(fid, along[i], across[i],
                                      (lons[i], lats[i]), max_swath_width)

            # Check all different areatypes
            for area_type in settings:
                # Check if the location should be masked
                mask = data == settings[area_type]['value']
                ratio = float(mask.sum()) / float(mask.size)
                if ratio > settings[area_type]['limit']:
                    idxs[i] = True
                    # No need to check the other areas if the pixel is masked
                    break

    return idxs


def get_footprint_data(fid, along, across, lonlat, max_swath_width):
    """Get mask data that covers the footprint centered at (lon, lat)"""
    # For now, use average of along and across footprint diameters
    # The difference for AVHRR is at most
    pass


def read_cloud_mask():
    """Read cloud mask"""
    pass


def read_snow_mask():
    """Read snow mask"""
    pass


def read_water_mask():
    """Read water exclusion mask"""
    pass


def mean_abs_deviation(data):
    """Calculate absolute mean deviation of *data*"""
    return np.sum(np.abs(data - np.mean(data))) / data.size


def get_idxs_around_location(row, col, side, remove_neighbours=False):
    """Get indices around given location in a side x side box.  Optionally remove
    neighbouring pixels."""
    y_start = row - (side - 1) / 2
    y_end = row + (side - 1) / 2 + 1
    y_idxs = np.arange(y_start, y_end)

    x_start = col - (side - 1) / 2
    x_end = col + (side - 1) / 2 + 1
    x_idxs = np.arange(x_start, x_end)

    y_idxs, x_idxs = np.meshgrid(y_idxs, x_idxs)

    mask = np.ones(y_idxs.shape, dtype=np.bool)
    if remove_neighbours is True:
        start = side / 2 - 1
        end = start + 3
        mask[start:end, start:end] = False
        y_idxs = y_idxs[mask]
        x_idxs = x_idxs[mask]
    # In any case, mask the centre pixel
    else:
        mask[side / 2, side / 2] = False
        y_idxs = y_idxs[mask]
        x_idxs = x_idxs[mask]

    return y_idxs.ravel(), x_idxs.ravel()


def check_static_masks(logger, func_names, lonlats, footprints):
    """Check static masks"""
    # Create placeholder for invalid row/col locations.  By default all
    # pixels are valid (== False)
    idxs = [False for row in lonlats[0]]

    # Run mask functions
    for func_name in func_names:
        try:
            func = vars()[func_name]
        except KeyError:
            logger.error("No such function: utils.%s", func_name)
            continue
        try:
            filename = func_names[func_name]['filename']
        except KeyError:
            logger.error("No reader for %s", func_name)
            continue
        try:
            settings = func_names[func_name]['settings']
        except KeyError:
            logger.warning("No settings for %s", func_name)
            settings = {}

        # Mask data
        idxs = func(filename, idxs, lonlats, footprints, settings)

    return idxs


def calc_footprint_size(sat_zens, ifov, sat_alt, max_swath_width):
    """Calculate approximate footprint sizes for the given satellite
    zenith angles.  Return sizes in along-track and across-track directions."""
    # Satellite co-zenith angles
    sat_co_zens = np.radians(180. - sat_zens)
    # Satellite orbital radius
    sat_radius = R_EARTH + sat_alt
    # Third term in the quadratic equation
    c__ = -sat_alt * sat_alt - 2 * sat_alt * R_EARTH
    # Distance from satellite to ground in the view direction
    dist = solve_quadratic(1., -2 * R_EARTH * np.cos(sat_co_zens), c__)
    # Footprint length in along-track direction
    along_lengths = ifov * dist

    # Satellite view angle
    sat_view = np.arcsin(R_EARTH * np.sin(sat_co_zens) / sat_radius)

    # Calculate along-view distances to both inner and outer edges of the IFOV
    # Third term in the quadratic equation is now positive
    c__ *= -1
    # Distance to inner edges of each IFOV
    sat_view -= ifov / 2.
    a_dist = solve_quadratic(1., -2 * sat_radius * np.cos(sat_view),
                             c__, limit=max_swath_width)
    # Distances to sub-satellite point along surface
    a_ranges = R_EARTH * np.arcsin(a_dist * np.sin(sat_view) / R_EARTH)
    # Distance to outer edges of each IFOV
    sat_view += ifov
    b_dist = solve_quadratic(1., -2 * sat_radius * np.cos(sat_view),
                             c__, limit=max_swath_width)
    b_ranges = R_EARTH * np.arcsin(a_dist * np.sin(sat_view) / R_EARTH)

    across_lengths = np.abs(a_ranges - b_ranges)

    return along_lengths, across_lengths


def solve_quadratic(a__, b__, c__, limit=None):
    """Solve quadratic equation"""
    discriminant = b__ * b__ - 4 * a__ * c__
    x_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
    x_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    x__ = x_1.copy()
    idxs = np.isnan(x_1)
    x__[idxs] = x_2[idxs]

    if limit is not None:
        idxs = x__ > limit
        x__[idxs] = x_2[idxs]

    return x__
