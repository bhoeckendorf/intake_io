import re
from typing import Any

import nrrd
import numpy as np
import xarray as xr

from .base import ImageSource, Schema
from ..util import get_axes, get_spacing, get_spacing_units, to_numpy


class NrrdSource(ImageSource):
    """Intake source for NRRD files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "nrrd"
    version = "0.0.1"
    partition_access = False

    def __init__(self, uri: str, **kwargs):
        """
        Arguments:
            uri (str): URI (file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(**kwargs)
        self.uri = uri

    def _get_schema(self) -> Schema:
        header = dict(nrrd.read_header(self.uri, {"channels": "quoted string list"}))

        shape = tuple(header["sizes"])[::-1]
        try:
            axes = "".join(header["labels"])[::-1]
        except KeyError:
            axes = get_axes(shape)
        unit_axes = "".join(i for i in axes if i in "tzyx")

        try:
            spacing = tuple(header["spacings"])[::-1]
        except KeyError:
            try:
                spacing = header["space directions"]
                spacing = tuple([spacing[d, d] for d in range(spacing.shape[0])])[::-1]
            except KeyError:
                spacing = ()
        if len(spacing) == len(unit_axes):
            spacing = {a: s for a, s in zip(unit_axes, spacing)}
        else:
            spacing = {}

        try:
            units = tuple(header["units"])[::-1]
        except KeyError:
            try:
                units = tuple(header["space units"])[::-1]
            except KeyError:
                units = ()
        if len(units) == len(unit_axes):
            units = {a: s for a, s in zip(unit_axes, units)}
        else:
            units = {}

        coords = {"c": header["channels"]} if header.get("channels") else None

        shape = self._set_shape_metadata(axes, shape, spacing, units, coords)
        self._set_fileheader(header)
        return Schema(
            dtype=np.dtype(header["type"]),
            shape=shape,
            npartitions=1,
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        # with open(self.url, "rb") as f:
        #    return nrrd.read_data(self._header, fh=f, filename=self.url, index_order="C")
        return self._reorder_axes(nrrd.read(self.uri, {"channels": "quoted string list"}, "C")[0])

    def _close(self):
        pass


def save_nrrd(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_nrrd(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".nrrd", f".{k}.nrrd", uri, flags=re.IGNORECASE)
                save_nrrd(image[k], uri_key, compress)
        return

    axes = list(get_axes(image))[::-1]
    kinds = {"i": "list", "t": "time", "c": "list"}
    header = {
        "space dimension": sum(i in axes for i in "tzyx"),
        "sizes": image.shape[::-1],
        "labels": axes,
        "kinds": [kinds.get(i) or "space" for i in axes],
        "encoding": "gzip" if compress else "raw"
    }

    header["spacings"] = []
    header["units"] = []
    for ax in [i for i in axes if i in "tzyx"]:
        header["spacings"].append(get_spacing(image, ax) or np.NaN)
        header["units"].append(get_spacing_units(image, ax) or "")
    header["space units"] = header["units"]

    space = np.eye(sum(ax in axes for ax in "zyx"))
    for i, s in enumerate(get_spacing(image, ax) or 1.0 for ax in [i for i in axes if i in "zyx"]):
        space[i, i] = s
    header["space directions"] = space

    if "c" in image.coords:
        header["channels"] = tuple(map(str, image.coords["c"].data))

    nrrd.write(uri, to_numpy(image), header, compression_level=4, index_order="C",
               custom_field_map={"channels": "quoted string list"})
