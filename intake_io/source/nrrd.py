from typing import Any, Optional

import nrrd
import numpy as np

from .base import ImageSource, Schema
from ..util import get_axes, get_spacing, get_spacing_units, partition_gen, to_numpy


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
        super().__init__(uri, **kwargs)

    def _get_schema(self) -> Schema:
        self._header = nrrd.read_header(self.open(), {"channels": "quoted string list"})

        shape = tuple(self._header["sizes"])[::-1]
        try:
            axes = "".join(self._header["labels"])[::-1]
        except KeyError:
            axes = get_axes(shape)
        unit_axes = "".join(i for i in axes if i in "tzyx")

        try:
            spacing = tuple(self._header["spacings"])[::-1]
        except KeyError:
            try:
                spacing = self._header["space directions"]
                spacing = tuple([spacing[d, d] for d in range(spacing.shape[0])])[::-1]
            except KeyError:
                spacing = ()
        if len(spacing) == len(unit_axes):
            spacing = {a: s for a, s in zip(unit_axes, spacing)}
        else:
            spacing = {}

        try:
            units = tuple(self._header["units"])[::-1]
        except KeyError:
            try:
                units = tuple(self._header["space units"])[::-1]
            except KeyError:
                units = ()
        if len(units) == len(unit_axes):
            units = {a: s for a, s in zip(unit_axes, units)}
        else:
            units = {}

        coords = {"c": self._header["channels"]} if self._header.get("channels") else None

        shape = self._set_shape_metadata(axes, shape, spacing, units, coords)
        self._set_fileheader(dict(self._header))

        try:
            dtype = np.dtype(self._header["type"])
        except TypeError:
            dtype = {
                "char": np.int8,
                "short": np.int16,
                "int": np.int32,
                "long": np.int64,

                "signed char": np.int8,
                "signed short": np.int16,
                "signed int": np.int32,
                "signed long": np.int64,

                "unsigned char": np.uint8,
                "unsigned short": np.uint16,
                "unsigned int": np.uint32,
                "unsigned long": np.uint64,

                "uchar": np.uint8,
                "ushort": np.uint16,
                "uint": np.uint32,
                "ulong": np.uint64,

                "float": np.float32,
                "double": np.float64
            }[self._header["type"]]

        return Schema(
            dtype=dtype,
            shape=shape,
            npartitions=1,
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        # return self._reorder_axes(nrrd.read_data(self._header, reader, index_order="C"))
        return self._reorder_axes(nrrd.read(self.uri, index_order="C")[0])


def save_nrrd(
        image: Any,
        uri: str,
        compress: bool = True,
        partition: Optional[str] = None,

        # Format-specific kwargs
        compression_type: str = "gzip",
        compression_level: int = 4
):
    if partition is None:
        partition = "itczyx"
    if not compress:
        compression_type = "raw"

    for img, _uri in partition_gen(image, partition, uri):
        axes = list(get_axes(img))[::-1]
        kinds = {"i": "list", "t": "time", "c": "list"}
        header = {
            "space dimension": sum(i in axes for i in "tzyx"),
            "sizes": img.shape[::-1],
            "labels": axes,
            "kinds": [kinds.get(i) or "space" for i in axes],
            "encoding": compression_type
        }

        header["spacings"] = []
        header["units"] = []
        for ax in [i for i in axes if i in "tzyx"]:
            header["spacings"].append(get_spacing(img, ax) or np.NaN)
            header["units"].append(get_spacing_units(image, ax) or "")
        header["space units"] = header["units"]

        space = np.eye(sum(ax in axes for ax in "zyx"))
        for i, s in enumerate(get_spacing(img, ax) or 1.0 for ax in [i for i in axes if i in "zyx"]):
            space[i, i] = s
        header["space directions"] = space

        if "c" in image.coords:
            header["channels"] = tuple(map(str, img.coords["c"].data))

        nrrd.write(_uri, to_numpy(img), header, compression_level=compression_level, index_order="C",
                   custom_field_map={"channels": "quoted string list"})
