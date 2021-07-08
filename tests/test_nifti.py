import os
import numpy as np
import xarray as xr
import pytest
import intake_io
from .fixtures import *


def test_round_trip_nifti_version_1(tmp_path):
    fpath = os.path.join(tmp_path, "compressed.nii.gz")
    if os.path.exists(fpath):
        os.remove(fpath)

    for img0, shape, axes, spacing, units in random_images():
        if axes != "zyx":
            continue
        try:
            assert not os.path.exists(fpath)
            intake_io.imsave(img0, fpath, nifti_version=1)
            assert os.path.exists(fpath)

            with intake_io.source.NiftiSource(fpath) as src:
                img1 = intake_io.imload(src)["image"]

            assert axes == intake_io.get_axes(img1)
            assert shape == img1.shape
            #assert spacing == intake_io.get_spacing(img1)
            np.testing.assert_array_almost_equal(spacing, intake_io.get_spacing(img1))
            assert units == intake_io.get_spacing_units(img1)
            assert np.mean(img0.data) not in (0, 1)
            assert np.mean(img0.data) == np.mean(img1.data)
        finally:
            if os.path.exists(fpath):
                os.remove(fpath)


def test_round_trip_nifti_version_2(tmp_path):
    fpath = os.path.join(tmp_path, "compressed.nii.gz")
    if os.path.exists(fpath):
        os.remove(fpath)

    for img0, shape, axes, spacing, units in random_images():
        if axes != "zyx":
            continue
        try:
            assert not os.path.exists(fpath)
            intake_io.imsave(img0, fpath, nifti_version=2)
            assert os.path.exists(fpath)

            with intake_io.source.NiftiSource(fpath) as src:
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


def test_load_from_url():
    urls = [
        "https://nifti.nimh.nih.gov/nifti-1/data/avg152T1_LR_nifti.nii.gz",
        "https://nifti.nimh.nih.gov/pub/dist/data/nifti2/avg152T1_LR_nifti2.nii.gz"
    ]
    for url in urls:
        img = intake_io.imload(url)
        assert img["image"].shape == (91, 109, 91)
        if "nifti-1" in url:
            assert img["image"].dtype == np.uint8
        else:
            assert img["image"].dtype == np.float32
        assert intake_io.get_axes(img) == "zyx"
        assert intake_io.get_spacing(img) == (2.0, 2.0, 2.0)
        assert intake_io.get_spacing_units(img) == ("mm", "mm", "mm")
