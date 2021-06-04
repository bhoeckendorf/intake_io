import numpy as np
import xarray as xr
import pytest
from intake_io import to_xarray


def _random_image(dtype, shape, **kwargs):
    try:
        info = np.iinfo(dtype)
        image = np.random.randint(info.min, info.max, shape, dtype)
    except ValueError:
        info = np.finfo(dtype)
        image = np.random.rand(*shape).astype(dtype)
        image *= info.max
        image -= np.mean(image) / 2
    return to_xarray(image, **kwargs)


def _ramp_image(dtype, shape, **kwargs):
    try:
        info = np.iinfo(dtype)
        image = np.linspace(info.min, info.max, np.prod(shape), dtype=dtype).reshape(shape)
    except ValueError:
        info = np.finfo(dtype)
        image = np.linspace(-10000, +20000, np.prod(shape), dtype=dtype).reshape(shape)
    return to_xarray(image, **kwargs)


def random_images():
    dtypes_all = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64)
    dtypes_common = (np.uint8, np.uint16, np.float32)

    shape_axes = [
        [(128, 256), "yx", (0.15, 0.05), None,   None, None, None, (None, None)],
        [(32, 128, 256), "zyx", (0.15, 1.56, 0.05), ("mm",),   None, None, None, ("mm", "mm", "mm")],
        [(8, 32, 128, 256), "czyx", (0.15, 1.56, 0.05), ("nm",),   None, None, None, ("nm", "nm", "nm")],
        [(8, 32, 128, 256), "zcyx", (0.15, 1.56, 0.05), {"z": "cm", "x": "nm"},   (32, 8, 128, 256), "czyx", None, ("cm", None, "nm")],
        [(8, 32, 128, 256), "zcxy", (0.15, 1.56, 0.05), ("cm",),   (32, 8, 256, 128), "czyx", (0.15, 0.05, 1.56), ("cm", "cm", "cm")],
        [(8, 32, 128, 256), "zcxy", {"y": 0.15, "x": 1.56, "z": 0.05}, None,   (32, 8, 256, 128), "czyx", (0.05, 0.15, 1.56), (None, None, None)],
        [(8, 32, 128, 256), "zcxy", {"x": 1.56, "z": 0.05}, None,   (32, 8, 256, 128), "czyx", (0.05, None, 1.56), (None, None, None)],
        [(8, 32, 128, 256), "tzyx", None, ("nm", "mm", "cm"),   None, None, (None, None, None, None), (None, None, None, None)],
        [(8, 32, 128, 256), "ztyx", None, ("nm", "mm", "cm"),   (32, 8, 128, 256), "tzyx", (None, None, None, None), (None, None, None, None)],
        [(2, 8, 32, 128, 256), "tczyx", None, None,   None, None, (None, None, None, None), (None, None, None, None)],
        [(2, 8, 32, 128, 256), "zctxy", (0.34, 0.56, 0.78), ("nm", "ms", "mm", "cm"),   (32, 8, 2, 256, 128), "tczyx", (0.34, None, 0.78, 0.56), ("ms", None, "cm", "mm")],
        [(4, 2, 8, 32, 128, 256), "itczyx", None, None,   None, None, (None, None, None, None), (None, None, None, None)],
        [(4, 2, 8, 32, 128, 256), "zcitxy", None, None,   (8, 32, 2, 4, 256, 128), "itczyx", (None, None, None, None), (None, None, None, None)],
    ]
    for shape, axes, spacing, units, output_shape, output_axes, output_spacing, output_units in shape_axes:
        if output_shape is None:
            output_shape = shape
        if output_axes is None:
            output_axes = axes
        if output_spacing is None:
            output_spacing = spacing
        if output_units is None:
            output_units = units
        dtypes = dtypes_all if len(shape) <= 3 else dtypes_common
        for dtype in dtypes:
            yield _random_image(dtype, shape, axes=axes, spacing=spacing, spacing_units=units), output_shape, output_axes, output_spacing, output_units


def ramp_images():
    dtypes_all = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64)
    dtypes_common = (np.uint8, np.uint16, np.float32)
    for shape in [(32, 128, 256)]:
        dtypes = dtypes_all if len(shape) <= 3 else dtypes_common
        for dtype in dtypes:
            yield _ramp_image(dtype, shape)
