import os
import numpy as np
import xarray as xr
import pytest
import intake_io
import requests


def test_loading(tmp_path):
    url = "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel.ome.tif"
    fpath = os.path.join(tmp_path, url.rsplit("/", 1)[-1])
    with open(fpath, "wb") as fh:
        fh.write(requests.get(url).content)
    img = intake_io.imload(fpath)
    assert img["image"].shape == (3, 167, 439)
    assert img["image"].dtype == np.int8
    assert intake_io.get_axes(img) == "cyx"

    url = "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/bioformats-artificial/multi-channel-4D-series.ome.tif"
    fpath = os.path.join(tmp_path, url.rsplit("/", 1)[-1])
    with open(fpath, "wb") as fh:
        fh.write(requests.get(url).content)
    img = intake_io.imload(fpath)
    assert img["image"].shape == (7, 3, 5, 167, 439)
    assert img["image"].dtype == np.int8
    assert intake_io.get_axes(img) == "tczyx"
