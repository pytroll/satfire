#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Panu Lahtinen / FMI
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

"""The main() script for FMI Forest Fire satellite software"""

import sys
import logging
import logging.config

from fffsat.utils import read_config
from fffsat.forest_fire import ForestFire


def main():
    config = read_config(sys.argv[1])
    sat_fname = sys.argv[2]
    if len(sys.argv) > 3:
        cma_fname = sys.argv[3]
    else:
        cma_fname = None
    logging.config.dictConfig(config['standalone_log_config'])
    fff = ForestFire(config)
    fff.run(sat_fname=sat_fname, cma_fname=cma_fname)
    fff.save()

if __name__ == "__main__":
    main()
