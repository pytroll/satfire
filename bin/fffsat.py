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

import fffsat


def main():
    config = fffsat.utils.read_config(sys.argv[1])

    fff = fffsat.ForestFire(config)
    try:
        fff.run()
    except KeyboardInterrupt:
        fff.stop()

if __name__ == "__main__":
    main()
