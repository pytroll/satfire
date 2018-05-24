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

from posttroll.message import Message
from fffsat.utils import read_config
from fffsat.forest_fire import ForestFire


def main():
    config = read_config(sys.argv[1])
    sat_fname = sys.argv[2]
    sat_tag = config['sat_message_tag']
    msg_dict = {'platform_name': 'NOAA-19',
                'sensor': 'avhrr-3',
                'collection': {sat_tag:
                               {'dataset': [{'uri': sat_fname}]}}}
    if len(sys.argv) > 3:
        cma_fname = sys.argv[3]
        cma_tag = config['cma_message_tag']
        msg_dict['collection'][cma_tag] = {'dataset': [{'uri': cma_fname}]}
    else:
        cma_fname = None
    logging.config.dictConfig(config['standalone_log_config'])
    fff = ForestFire(config)
    msg = Message('foo', 'bar', msg_dict)
    try:
        if fff.run(msg=msg, sat_fname=sat_fname, cma_fname=cma_fname):
            fff.save_text()
    finally:
        fff._pub.stop()

if __name__ == "__main__":
    main()
