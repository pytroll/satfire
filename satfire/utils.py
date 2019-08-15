#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
"""Utility functions for Satfire"""

import logging
import os.path
import yaml
from collections import OrderedDict
import datetime as dt

import xarray as xr
import numpy as np
import h5py

from satpy import Scene
from trollsift import parse

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


def save_hdf5(fname, data):
    """Save *data* as YAML to *fname*."""
    import h5py
    with h5py.File(fname, 'w') as fid:
        for key in data:
            fid.create_group(str(key))
            grp = fid[str(key)]
            for key2 in data[key]:
                val = data[key][key2]
                if isinstance(val, dt.datetime):
                    val = [val.year, val.month, val.day,
                           val.hour, val.minute, val.second, val.microsecond]
                grp[key2] = val


def get_filenames_from_msg(msg, config):
    """Find filenames for satellite data and cloud mask data from the message.
    """
    cma_tag = config['cma_message_tag']
    sat_tag = config['sat_message_tag']

    try:
        sat_fname = msg.data['collection'][sat_tag]['dataset'][0]['uri']
    except (KeyError, IndexError):
        sat_fname = None

    try:
        cma_fname = msg.data['collection'][cma_tag]['dataset'][0]['uri']
    except (KeyError, IndexError):
        cma_fname = None

    return sat_fname, cma_fname


def read_sat_data(fname, channels, reader):
    """Read satellite data"""
    if not isinstance(fname, (list, set, tuple)):
        fname = [fname, ]
    glbl = Scene(filenames=fname, reader=reader)
    glbl.load(channels)

    # Convert channel data to Numpy arrays and colled metadata to
    # dictionary.  This is a temporary work-around for modern
    # Satpy.
    data = {}
    metadata = {}
    for chan in channels:
        logging.info("Loading %s", chan)
        try:
            data[chan] = np.array(glbl[chan])
            metadata[chan] = glbl[chan].attrs
        except KeyError:
            logging.error("Channel %s not available", chan)
            return None
    # Global metadata
    metadata.update(glbl.attrs)
    metadata["proc_time"] = dt.datetime.utcnow()

    return data, metadata


def read_cma(fname):
    """Read cloud mask data"""
    try:
        data, _ = read_sat_data(fname, ["cma", ], "nc_nwcsaf_pps")
        cma = data['cma']
    except ValueError:
        return None
    return cma != 0


def check_globcover(fname, idxs, lonlats, footprints, settings, metadata):
    """Check globcover mask."""
    lons, lats = lonlats
    along, across = footprints

    masked_num = idxs.sum()

    logging.info("Apply Globcover masking")
    with h5py.File(fname, 'r') as fid:
        # Read mask coordinates and mask data
        mask_lon_v = fid['longitudes'][()]
        mask_lat_v = fid['latitudes'][()]
        full_mask = fid['data'][()]
        mask_resolution = np.abs(mask_lon_v[1] - mask_lon_v[0])

        # Convert coordinates to 2D arrays
        mask_lon, mask_lat = np.meshgrid(mask_lon_v, mask_lat_v)

        mask_lon_min, mask_lon_max = np.min(mask_lon_v), np.max(mask_lon_v)
        mask_lat_min, mask_lat_max = np.min(mask_lat_v), np.max(mask_lat_v)

        # Remove locations outside the mask area
        out_idxs = ((lons > mask_lon_max) |
                    (lons < mask_lon_min) |
                    (lats > mask_lat_max) |
                    (lats < mask_lat_min))
        idxs[out_idxs] = True
        logging.info("%d candidates removed outside Globcover area",
                     idxs.sum() - masked_num)
        masked_num = idxs.sum()

        logging.info("Check landuse for %d candidates", sum(~idxs))
        for i in range(len(idxs)):
            # Skip already masked pixels
            if idxs[i] is True:
                continue

            # Reduce mask size
            close_idxs = get_close_idxs(mask_lon_v, mask_lat_v,
                                        mask_resolution,
                                        lons[i], lats[i])

            # Get mask data that covers satellite footprint
            max_radius = np.max((along[i], across[i])) / 2.
            metadata[i]['footprint_radius'] = max_radius
            metadata[i]['along_radius'] = along[i] / 2.
            metadata[i]['across_radius'] = across[i] / 2.
            data = get_footprint_data(full_mask[close_idxs],
                                      mask_lon[close_idxs],
                                      mask_lat[close_idxs],
                                      max_radius, lons[i], lats[i])
            # If there's no data, we are outside of the mask thus discard this
            # point
            if data.size == 0:
                idxs[i] = True
                continue

            # Check all different areatypes
            for area_type in settings:
                # Check if the location should be masked
                mask = data == settings[area_type]['value']
                ratio = float(mask.sum()) / float(mask.size)
                metadata[i]['landuse_fraction_' + area_type] = ratio
                if ratio > settings[area_type]['limit']:
                    idxs[i] = True

    logging.info("Removed %d candidates based on landuse",
                 idxs.sum() - masked_num)
    logging.info("Globcover masking completed.")

    return idxs, metadata


def get_close_idxs(mask_lons, mask_lats, mask_resolution, lon, lat):
    """Find close indexes"""
    lon_origin = mask_lons[0]
    lat_origin = mask_lats[0]
    lon_idx = int((lon - lon_origin) / mask_resolution)
    lat_idx = int((lat_origin - lat) / mask_resolution)

    # Return Numpy slice objects, 40x80 pixels around the nominal location
    return np.s_[lat_idx - 20:lat_idx + 20, lon_idx - 40:lon_idx + 40]


def get_footprint_data(data, mask_lon, mask_lat, max_radius, lon, lat):
    """Get mask data that covers the footprint centered at (lon, lat).
    Circular footprint is assumed.
    """
    # Calculate approximate distance from nominal footprint location to all
    # mask pixels.
    dists, _ = haversine(lon, lat, mask_lon, mask_lat,
                         calc_bearings=False)

    # Find pixels that are within max_radius from the nominal location
    idxs = dists <= max_radius

    # Get the mask data for the reduced area
    mask = data[idxs]

    return mask


def haversine(lon1, lat1, lon2, lat2, calc_bearings=False):
    """Calculate Haversine distance and bearings from (lon1, lat2) to
    (lon2, lat2).  Lon2/lat2 can be multidimensional arrays.  Lon1/lat1
    need to have the same dimension with lon1/lat1 or be scalar."""

    # Ensure coordinates are Numpy arrays
    lon1 = ensure_numpy(lon1, dtype=np.float32)
    lat1 = ensure_numpy(lat1, dtype=np.float32)
    lon2 = ensure_numpy(lon2, dtype=np.float32)
    lat2 = ensure_numpy(lat2, dtype=np.float32)

    # Convert coordinates to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a__ = np.sin(dlat / 2)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2)**2
    c__ = 2 * np.arcsin(np.sqrt(a__))

    if calc_bearings:
        dLon = lon2 - lon1
        y__ = np.sin(dLon) * np.cos(lat2)
        x__ = np.cos(lat1) * np.sin(lat2) - \
            np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
        bearings = np.degrees(np.arctan2(y__, x__))
        # Wrap negative angles to positives
        bearings[bearings < 0] += 360
    else:
        bearings = None

    return R_EARTH * c__, bearings


def ensure_numpy(itm, dtype=None):
    """Ensure the given item is an numpy array"""
    if isinstance(itm, (int, float)):
        itm = [itm]
    if not isinstance(itm, np.ndarray):
        itm = np.array(itm)
    else:
        if len(itm.shape) == 0:
            itm = np.array([itm])

    if dtype is not None:
        itm = itm.astype(dtype)

    return itm


def mean_abs_deviation(data):
    """Calculate absolute mean deviation of *data*"""
    return np.sum(np.abs(data - np.mean(data))) / data.size


def get_idxs_around_location(row, col, side, remove_neighbours=False):
    """Get indices around given location in a side x side box.  Optionally remove
    neighbouring pixels."""
    y_start = row - int((side - 1) / 2)
    y_end = row + int((side - 1) / 2) + 1
    y_idxs = np.arange(y_start, y_end)

    x_start = col - int((side - 1) / 2)
    x_end = col + int((side - 1) / 2) + 1
    x_idxs = np.arange(x_start, x_end)

    y_idxs, x_idxs = np.meshgrid(y_idxs, x_idxs)

    mask = np.ones(y_idxs.shape, dtype=np.bool)
    if remove_neighbours is True:
        start = int(side / 2 - 1)
        end = start + 3
        mask[start:end, start:end] = False
        y_idxs = y_idxs[mask]
        x_idxs = x_idxs[mask]
    # In any case, mask the centre pixel
    else:
        mask[int(side / 2), int(side / 2)] = False
        y_idxs = y_idxs[mask]
        x_idxs = x_idxs[mask]

    return y_idxs.ravel(), x_idxs.ravel()


def check_static_masks(logger, func_names, lonlats, footprints):
    """Check static masks"""
    # Create placeholder for invalid row/col locations.  By default all
    # pixels are valid (== False)
    idxs = np.array([False for row in lonlats[0]], dtype=np.bool)
    metadata = np.array([{} for i in lonlats[0]])

    # Run mask functions
    for func_name in func_names:
        try:
            func = globals()[func_name]
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
        idxs, metadata = func(filename, idxs, lonlats, footprints, settings,
                              metadata)

    return idxs, metadata


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
    b_ranges = R_EARTH * np.arcsin(b_dist * np.sin(sat_view) / R_EARTH)

    across_lengths = np.abs(a_ranges - b_ranges)

    return along_lengths, across_lengths


def solve_quadratic(a__, b__, c__, limit=None):
    """Solve quadratic equation"""
    discriminant = b__ * b__ - 4 * a__ * c__
    x_1 = (-b__ + np.sqrt(discriminant)) / (2 * a__)
    x_2 = (-b__ - np.sqrt(discriminant)) / (2 * a__)

    x__ = x_1.copy()
    idxs = np.isnan(x_1)
    # x__[idxs] = x_2[idxs]
    x__ = xr.where(idxs, x_2, x__)

    if limit is not None:
        idxs = x__ > limit
        # x__[idxs] = x_2[idxs]
        x__ = xr.where(idxs, x_2, x__)

    return x__
