import os
import numpy as np
import xarray as xr
import pytest
import intake_io
from .fixtures import *


def test_round_trip_uncompressed(tmp_path):
    fpath = os.path.join(tmp_path, "uncompressed.tif")
    if os.path.exists(fpath):
        os.remove(fpath)

    for img0, shape, axes, spacing, units in random_images():
        if img0.dtype in (np.int8, np.int32, np.int64, np.uint32, np.uint64, np.float64):
            continue
        if "i" in axes:
            continue
        try:
            assert not os.path.exists(fpath)
            intake_io.imsave(img0, fpath, compress=False)
            assert os.path.exists(fpath)

            with intake_io.source.TifSource(fpath) as src:
                img1 = intake_io.imload(src)["image"]

            assert axes == intake_io.get_axes(img1)
            assert shape == img1.shape
            assert spacing == intake_io.get_spacing(img1)
            assert units == intake_io.get_spacing_units(img1)
            assert np.mean(img0.data) not in (0, 1)
            assert np.mean(img0.data) == np.mean(img1.data)
        finally:
            if os.path.exists(fpath):
                os.remove(fpath)


def test_round_trip_compressed(tmp_path):
    fpaths = {
        False: os.path.join(tmp_path, "uncompressed.tif"),
        True: os.path.join(tmp_path, "compressed.tif")
    }
    for fpath in fpaths.values():
        if os.path.exists(fpath):
            os.remove(fpath)

    for img0 in ramp_images():
        if img0.dtype in (np.int8, np.int32, np.int64, np.uint32, np.uint64, np.float64):
            continue
        if "i" in intake_io.get_axes(img0):
            continue
        try:
            for compress, fpath in fpaths.items():
                if os.path.exists(fpath):
                    os.remove(fpath)

                assert not os.path.exists(fpath)
                intake_io.imsave(img0, fpath, compress=compress)
                assert os.path.exists(fpath)

                with intake_io.source.TifSource(fpath) as src:
                    img1 = intake_io.imload(src)["image"]
                img2 = intake_io.imload(fpath)["image"]
                
                assert img0.shape == img1.shape == img2.shape
                assert img0.dims == img1.dims == img2.dims
                assert np.mean(img0.data) not in (0, 1)
                assert np.mean(img0.data) == np.mean(img1.data) == np.mean(img2.data)

            assert os.path.getsize(fpaths[True]) <= 0.9 * os.path.getsize(fpaths[False])
        finally:
            for fpath in fpaths.values():
                if os.path.exists(fpath):
                    os.remove(fpath)


def test_load_from_url():
    url = "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel.ome.tif"
    img = intake_io.imload(url)
    assert img["image"].shape == (3, 167, 439)
    assert img["image"].dtype == np.int8
    assert intake_io.get_axes(img) == "cyx"

    url = "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-4D-series.ome.tif"
    img = intake_io.imload(url)
    assert img["image"].shape == (7, 3, 5, 167, 439)
    assert img["image"].dtype == np.int8
    assert intake_io.get_axes(img) == "tczyx"
