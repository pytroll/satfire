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
    sat_fname = sys.argv[2]
    cma_fname = sys.argv[3]

    fff = fffsat.ForestFire(config)
    fff.run(sat_fname=sat_fname, cma_fname=cma_fname)
    fff.save()

if __name__ == "__main__":
    main()
