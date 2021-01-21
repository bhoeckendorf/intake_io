import numpy as np
import xarray as xr
import pytest
import intake_io


@pytest.fixture(params=[np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
def image_2d_ramp(request):
    dtype = request.param
    shape = (128, 256)
    try:
        info = np.iinfo(dtype)
    except ValueError:
        info = np.finfo(dtype)
    image = np.linspace(info.min, info.max, np.prod(shape), dtype=dtype).reshape(shape)
    yield intake_io.to_xarray(image)


@pytest.fixture(params=[np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
def image_3d_ramp(request):
    dtype = request.param
    shape = (8, 128, 256)
    try:
        info = np.iinfo(dtype)
        image = np.linspace(info.min, info.max, np.prod(shape), dtype=dtype).reshape(shape)
    except ValueError:
        info = np.finfo(dtype)
        image = np.linspace(-10000, +10000, np.prod(shape), dtype=dtype).reshape(shape)
    yield intake_io.to_xarray(image)


@pytest.fixture(params=[np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
def image_2d_random(request):
    dtype = request.param
    shape = (128, 256)
    try:
        info = np.iinfo(dtype)
        image = np.random.randint(info.min, info.max, shape, dtype)
    except ValueError:
        info = np.finfo(dtype)
        image = np.random.rand(*shape).astype(dtype)
        image *= info.max
        image -= np.mean(image) / 2
    yield intake_io.to_xarray(image)


@pytest.fixture(params=[np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64])
def image_3d_random(request):
    dtype = request.param
    shape = (8, 128, 256)
    try:
        info = np.iinfo(dtype)
        image = np.random.randint(info.min, info.max, shape, dtype)
    except ValueError:
        info = np.finfo(dtype)
        image = np.random.rand(*shape).astype(dtype)
        image *= info.max
        image -= np.mean(image) / 2
    yield intake_io.to_xarray(image)
