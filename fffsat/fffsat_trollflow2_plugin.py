#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

"""FFFsat plugin for Trollflow2"""

import logging

from fffsat.forest_fire import ForestFire

LOGGER = logging.getLogger("FFFSat")


def forest_fire(job):

    LOGGER.info("Finding forest fires.")
    config = job["product_list"]
    fff = ForestFire(config)
    try:
        if fff.run(msg=job["input_mda"]):
            if "text_fname_pattern" in config:
                fff.save_text()
            if "hdf5_fname_pattern" in config:
                fff.save_hdf5()
        fff.clean()
    finally:
        if fff._pub is not None:
            fff._pub.stop()
