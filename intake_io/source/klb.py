from typing import Any, Optional, Tuple

import numpy as np
import pyklb as klb

from .base import ImageSource, Schema
from ..util import get_axes, get_spacing, partition_gen, to_numpy


class KlbSource(ImageSource):
    """Intake source for KLB files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "klb"
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
        header = klb.readheader(self.uri)
        shape = dict(zip("tczyx", header["imagesize_tczyx"]))
        spacing = dict(zip("tczyx", header["pixelspacing_tczyx"]))
        axes = "".join(ax for ax, s in shape.items() if s > 1)
        for ax in [ax for ax in shape.keys() if ax not in axes]:
            shape.pop(ax)
            spacing.pop(ax)
        units = {ax: "s" if ax == "t" else "μm" for ax in spacing.keys()}

        shape = self._set_shape_metadata(axes, shape, spacing, units)
        self._set_fileheader(header)
        return Schema(
            dtype=np.dtype(header["datatype"]),
            shape=shape,
            npartitions=1,
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        return self._reorder_axes(klb.readfull(self.uri).squeeze())


def save_klb(
        image: Any,
        uri: str,
        compress: bool = True,
        partition: Optional[str] = None,

        # Format-specific kwargs
        block_shape: Tuple[int, ...] = None,
        compression_type: str = "bzip2"
):
    if partition is None:
        partition = "tczyx"
    if not compress:
        compression_type = "none"
    if block_shape is not None:
        block_shape = block_shape[::-1]

    for img, _uri in partition_gen(image, partition, uri):
        axes = get_axes(img)
        shape = [img.shape[axes.index(ax)] if ax in axes else 1 for ax in "tczyx"]
        spacing = [get_spacing(img, ax) or 1.0 for ax in "tczyx"]
        # TODO: Convert spacing units to match convention.
        klb.writefull(to_numpy(img).reshape(shape), _uri, pixelspacing_tczyx=spacing, blocksize_xyzct=block_shape, compression=compression_type)
