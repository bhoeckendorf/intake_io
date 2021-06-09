import re
from typing import Any

import numpy as np
import tifffile
import xarray as xr

from .base import ImageSource, Schema
from .bioformats import _parse_ome_metadata
from ..util import _reorder_axes, get_axes, get_spacing, get_spacing_units


class TifSource(ImageSource):
    """Intake source for TIF files.

    Attributes:
        uri (str): URI (e.g. file system path or URL)
    """

    container = "ndarray"
    name = "tif"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, **kwargs):
        """
        Arguments:
            uri (str): URI (e.g. file system path or URL)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(**kwargs)
        self.uri = uri
        self._file = None

    def _get_schema(self) -> Schema:
        if self._file is None:
            self._file = tifffile.TiffFile(self.uri)

        # TODO: Support loading multiple series.
        if len(self._file.series) > 1:
            raise ValueError(
                f"Loading TIF files containing multiple series is currently unsupported. {self.uri} contains "
                f"{len(self._file.series)} series.")

        fileheader = {k: getattr(self._file, f"{k}_metadata") for k in self._file.flags}
        for flag in ("imagej", "ome"):
            if flag not in fileheader:
                metadata = getattr(self._file, f"{flag}_metadata", None)
                if metadata is not None:
                    fileheader[flag] = metadata

        if "ome" in fileheader:
            ome = _parse_ome_metadata(fileheader["ome"])
            shape = self._set_shape_metadata(ome["axes"], ome["shape"], ome["spacing"], ome["spacing_units"],
                                             ome["coords"])
            self._set_fileheader(ome["fileheader"])
            return Schema(
                dtype=ome["dtype"],
                shape=shape,
                npartitions=sum(len(i) for i in self._file.series),
                chunks=None
            )

        series = self._file.series[0]
        axes = series.axes.lower()
        shape = dict(zip(axes, series.shape))
        spacing = {}
        spacing_units = {}

        # # parse shape
        # for tag, axis in zip((257, 256), "yx"):
        #     try:
        #         v = series[0].tags[tag].value
        #     except KeyError:
        #         continue
        #     assert isinstance(v, int)
        #     shape[axis] = v

        # parse pixel spacing
        for ax in "yx":
            try:
                v = series[0].tags[f"{ax.upper()}Resolution"].value
            except KeyError:
                continue
            if isinstance(v, tuple):
                assert len(v) == 2
                v = v[1] / v[0]
            spacing[ax] = v

        try:
            v = int(series[0].tags["ResolutionUnit"].value)
            factors = {
                2: 254e2,  # inch
                3: 1e4,    # cm
                4: 1e3,    # mm
            }
            for ax in spacing.keys():
                if ax in "yx":
                    spacing[ax] *= factors[v]
                    spacing_units[ax] = "\u03BCm"
        except KeyError:
            pass

        if "imagej" in fileheader:
            metadata = fileheader["imagej"]

            # parse shape
            for field, ax in zip(("frames", "channels", "slices"), "tcz"):
                try:
                    shape[ax] = int(metadata[field])
                except KeyError:
                    continue

            # parse pixel spacing
            for field, ax in zip(("finterval", "spacing"), "tz"):
                try:
                    spacing[ax] = float(metadata[field])
                except KeyError:
                    continue

            # parse pixel spacing units            
            for field, ax in zip(["tunit", "zunit", "yunit", "unit"], "tzyx"):
                try:
                    spacing_units[ax] = metadata[field]
                except KeyError:
                    continue

            axes = "".join(i for i in "itzcyx" if i in shape)

        shape = self._set_shape_metadata(axes, shape, spacing, spacing_units)
        self._set_fileheader(fileheader)
        return Schema(
            dtype=self._file.series[0][0].dtype,
            shape=shape,
            npartitions=sum(len(i) for i in self._file.series),
            chunks=None
        )

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._reorder_axes(self._file.asarray())

    def _get_partition(self, i: int) -> np.ndarray:
        return self._reorder_axes(self._file.series[0][0].asarray())

    def _close(self):
        if self._file is not None:
            self._file.close()


def save_tif(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_tif(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".tif", f".{k}.tif", uri, flags=re.IGNORECASE)
                # TODO: handle .tiff, .ome.tif etc
                save_tif(image[k], uri_key, compress)
        return

    # TODO: Support saving multiple series.
    if "i" in image.coords and len(image.coords["i"]) > 1:
        raise ValueError(
            f"Saving multiple series in one TIF file is currently unsupported. {uri} would contain {len(image.coords['i'])} series.")

    axes = get_axes(image)
    args = {}
    args["resolution"] = tuple(1. / (get_spacing(image, i) or 1.) for i in "xy")

    # ImageJ doesn't support all dtypes
    if image.dtype not in (np.int8, np.int32, np.int64, np.uint32, np.uint64, np.float64):
        args["imagej"] = True
        args["metadata"] = {
            "axes": "".join(i for i in "itzcyx" if i in axes).upper(),
            "images": np.prod(image.shape[:-2]),
            "frames": len(image.coords["t"]) if "t" in image.coords else 1,
            "channels": len(image.coords["c"]) if "c" in image.coords else 1,
            "slices": len(image.coords["z"]) if "z" in image.coords else 1,
            "finterval": get_spacing(image, "t") or 0.0,
            "spacing": get_spacing(image, "z") or 1.0,
            "unit": get_spacing_units(image, "x") or "pixel",
            "yunit": get_spacing_units(image, "y") or "pixel",
            "zunit": get_spacing_units(image, "z") or "pixel",
            "tunit": get_spacing_units(image, "t") or "sec",
            "hyperstack": True,
            "mode": "composite"
        }

    pixels = _reorder_axes(image.data, axes, "itzcyx")
    tifffile.imsave(uri, pixels, **args)
