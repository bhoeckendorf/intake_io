import os
import numpy as np
import xarray as xr
import pytest
import intake_io
from .fixtures import *


def test_round_trip_3d(tmp_path, image_3d_random):
    fpath = os.path.join(tmp_path, "uncompressed.nrrd")
    img0 = image_3d_random

    assert not os.path.exists(fpath)
    intake_io.imsave(img0, fpath)
    assert os.path.exists(fpath)

    with intake_io.source.NrrdSource(fpath) as src:
        img1 = intake_io.imload(src)["image"]

    assert img0.shape == img1.shape
    assert np.mean(img0.data) not in (0, 1)
    assert np.mean(img0.data) == np.mean(img1.data)


def test_round_trip_3d_compression(tmp_path, image_3d_ramp):
    img0 = image_3d_ramp
    fpaths = {
        False: os.path.join(tmp_path, "uncompressed.nrrd"),
        True: os.path.join(tmp_path, "compressed.nrrd")
    }
    for fpath in fpaths.values():
        assert not os.path.exists(fpath)

    for compress, fpath in fpaths.items():
        intake_io.imsave(img0, fpath, compress=compress)
        assert os.path.exists(fpath)

        with intake_io.source.NrrdSource(fpath) as src:
            img1 = intake_io.imload(src)["image"]
        img2 = intake_io.imload(fpath)["image"]
        
        assert img0.shape == img1.shape == img2.shape
        assert np.mean(img0.data) not in (0, 1)
        assert np.mean(img0.data) == np.mean(img1.data) == np.mean(img2.data)

    assert os.path.getsize(fpaths[True]) <= 0.75 * os.path.getsize(fpaths[False])
