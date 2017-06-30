#!/usr/bin/env python

import zipfile
import urllib
import os.path
import sys

import pandas as pd
import h5py
import numpy as np
from libtiff import TIFF

GLOBCOVER_URL = "http://due.esrin.esa.int/files/Globcover2009_V2.3_Global_.zip"

GLOBCOVER_RESOLUTION = 1. / 360.
GLOBCOVER_NORTH_LAT_LIMIT = 90.
GLOBCOVER_SOUTH_LAT_LIMIT = -65.
GLOBCOVER_WEST_LON_LIMIT = 180.
GLOBCOVER_EAST_LON_LIMIT = -180.

COMPRESSION = "gzip"


def download_gc():
    """Download Globcover data"""
    out_fname = os.path.basename(GLOBCOVER_URL)
    if os.path.exists(out_fname):
        print("File already downloaded")
        return out_fname
    print("Downloading Globcover data from %s" % GLOBCOVER_URL)
    urllib.urlretrieve(GLOBCOVER_URL, out_fname)
    print("Download completed!")
    return os.path.basename(GLOBCOVER_URL)


def unzip(fname):
    """Extract the given file and return names of the extracted files"""
    print("Extracting file %s" % fname)
    with zipfile.ZipFile(fname, "r") as fid:
        fid.extractall()
        fnames = [f.filename for f in fid.filelist]
    print("Extracted files: %s" % ', '.join(fnames))

    return fnames


def read_tif(fname):
    """Read TIFF image from the given filename.  Return a memmap view of the
    image.
    """
    tif = TIFF.open(fname)
    img = tif.read_image()

    return img


def read_legend(fname):
    """Read legend from the given Excel file.  Return a 2-tuple
    (values, labels).
    """
    legend = pd.read_excel(fname)
    values = [legend['Value'][i] for i in range(len(legend))]
    labels = [str(''.join(ch_ for ch_ in legend['Label'][i] if ord(ch_) < 128))
              for i in range(len(legend))]
    # labels = [str(legend['Label'][i]) for i in range(len(legend))]

    return (values, labels)


def calc_lonlats(shape):
    """Calculate longitude and latitude vectors."""
    lat_shape, lon_shape = shape
    lons = np.linspace(GLOBCOVER_WEST_LON_LIMIT - GLOBCOVER_RESOLUTION / 2.,
                       GLOBCOVER_EAST_LON_LIMIT + GLOBCOVER_RESOLUTION / 2.,
                       lon_shape)
    lats = np.linspace(GLOBCOVER_NORTH_LAT_LIMIT - GLOBCOVER_RESOLUTION / 2.,
                       GLOBCOVER_SOUTH_LAT_LIMIT + GLOBCOVER_RESOLUTION / 2.,
                       lat_shape)

    return lons, lats


def save_to_hdf5(fname, data, lons, lats, legend):
    """Save data and data legend to HDF5 file."""
    print("Writing data to %s" % fname)
    with h5py.File(fname, 'w') as fid:
        fid.create_dataset('data', data=data, compression=COMPRESSION)
        fid['data'].attrs['description'] = "GLOBCOVER v2.3 data"
        fid['longitudes'] = lons
        fid['longitudes'].attrs['description'] = "Center of pixel longitudes"
        fid['latitudes'] = lats
        fid['latitudes'].attrs['description'] = "Center of pixel latitudes"
        fid['legend_values'] = legend[0]
        fid['legend_values'].attrs['description'] = "Values in data"
        fid['legend_labels'] = legend[1]
        fid['legend_labels'].attrs['description'] = \
            "Data labels associated with data values"

    print("Data saved!")


def main():
    "Main()"
    hdf5_fname = sys.argv[1]

    zip_fname = download_gc()
    fnames = unzip(zip_fname)
    for fname in fnames:
        if fname.endswith('.tif') and 'CLA' not in fname:
            data = read_tif(fname)
        if fname.endswith('.xls'):
            legend = read_legend(fname)
    lons, lats = calc_lonlats(data.shape)
    save_to_hdf5(hdf5_fname, data, lons, lats, legend)

    print("All done!")

if __name__ == "__main__":
    main()
