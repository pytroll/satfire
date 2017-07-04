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


def check_static_masks(logger, func_names, lats, lons, radii):
    """Check static masks"""
    # Create placeholder for invalid row/col locations.  By default all
    # pixels are valid (== False)
    idxs = [False for row in lats]

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

        # Mask data
        idxs = func(filename, idxs, lats, lons, radii)

    return idxs


def calc_footprint_size(sat_zens, ifov, sat_alt):
    """Calculate approximate footprint sizes for the given satellite
    zenith angles.  Return sizes in along-track and across-track directions."""
    # Satellite co-zenith angles
    sat_co_zens = np.radians(180. - sat_zens)
    # Third term in the quadratic equation is same in all cases
    c__ = -sat_alt * sat_alt - 2 * sat_alt * R_EARTH
    # Cosine of satellite co-zenith angles
    tmp = np.cos(sat_co_zens)
    # Distance from satellite to ground in the view direction
    look_dist = solve_quadratic(1., -2 * R_EARTH * tmp, c__)
    # Footprint length in along-track direction
    along_lengths = ifov * look_dist

    # Footprint length in across-track direction
    # Adjust angles by half of the IFOV
    tmp = np.cos(sat_co_zens - ifov / 2.)
    # Distance to "first" edge
    a_look_dist = solve_quadratic(1., -2 * R_EARTH * tmp, c__)

    # Adjust angles by half of the IFOV
    tmp = np.cos(sat_co_zens + ifov / 2.)
    # Distance to "second" edge
    b_look_dist = solve_quadratic(1., -2 * R_EARTH * tmp, c__)

    tmp = np.sin(sat_co_zens) / (sat_alt + R_EARTH)
    # Distances to sub-satellite point along surface
    a_ranges = R_EARTH * np.arcsin(a_look_dist * tmp)
    b_ranges = R_EARTH * np.arcsin(b_look_dist * tmp)

    across_lengths = np.abs(a_ranges - b_ranges)

    return along_lengths, across_lengths


def solve_quadratic(a__, b__, c__):
    """Solve quadratic equation"""
    discriminant = b__ * b__ - 4 * a__ * c__
    with np.errstate(invalid='ignore'):
        x_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
        x_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    x__ = x_1.copy()
    idxs = np.isnan(x_1)
    x__[idxs] = x_2[idxs]

    return x__
