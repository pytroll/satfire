
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

from satpy import Scene
from trollsift import parse


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
