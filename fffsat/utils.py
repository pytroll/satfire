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


def read_sat_data(msg, channels):
    """Read satellite data"""
    filenames = [itm["uri"] for itm in msg["collection"]]
    glbl = Scene(platform_name=msg["platform_name"],
                 sensor=msg["sensor"],
                 start_time=msg["start_time"],
                 end_time=msg["end_time"],
                 filenames=filenames)
    glbl.load(channels)

    return glbl


def read_land_cover():
    """Read land cover exclusion mask"""
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
