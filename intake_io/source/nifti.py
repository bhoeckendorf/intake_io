import os
import re
from functools import cached_property
from gzip import GzipFile
from typing import Any, Optional

import nibabel as nib
import numpy as np

from .base import ImageSource, Schema
from ..util import get_axes, get_spacing, get_spacing_units, partition_gen


class NiftiSource(ImageSource):
    """Intake source for NIFTI files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "nifti"
    version = "0.0.1"
    partition_access = False

    def __init__(self, uri: str, **kwargs):
        """
        Arguments:
            uri (str): URI (file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(uri, **kwargs)
        self._streams = []

    def open(self):
        if len(self._streams) == 0:
            self._streams.append(super().open())
            if os.path.splitext(self.uri)[-1].lower() == ".gz" and not isinstance(self._streams[0], GzipFile):
                self._streams.append(GzipFile(fileobj=self._streams[0], mode="rb"))
        return self._streams[-1]

    @cached_property
    def _nifti_version(self):
        version = None
        x = np.frombuffer(self.open().peek(4), dtype=np.int32, count=1)
        for byteorder in ("<", ">"):
            i = int(x.newbyteorder(byteorder))
            if i == 348:
                version = 1
            elif i == 540:
                version = 2
        self._streams[-1].seek(0)
        return version

    def _get_schema(self) -> Schema:
        if self._nifti_version == 1:
            header = nib.Nifti1Header.from_fileobj(self.open())
        else:
            header = nib.Nifti2Header.from_fileobj(self.open())

        dtype = np.dtype(header.get_data_dtype())
        shape = tuple(header.get_data_shape())
        axes = get_axes(shape)[::-1]
        spacing = tuple(header["pixdim"][1:len(shape) + 1])
        spacing_units = {}
        for ax in axes:
            if ax in "zyx":
                spacing_units[ax] = header.get_xyzt_units()[0]
            elif ax == "t":
                spacing_units[ax] = header.get_xyzt_units()[1]

        # clean file header
        header = dict(header)
        for k, v in header.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind == "S":
                    v = str(v)
                    try:
                        v = re.search(r"b'(.+)'", v, re.DOTALL).groups()[0]
                    except AttributeError:
                        pass
                    if v == "b''":
                        v = ""
                    header[k] = v
                elif v.size == 1:
                    if v.dtype.kind in ("i", "u"):
                        header[k] = int(v)
                    elif v.dtype.kind == "f":
                        header[k] = float(v)

        shape = self._set_shape_metadata(axes, shape, spacing, spacing_units)
        self._set_fileheader(header)
        return Schema(
            dtype=dtype,
            shape=shape,
            npartitions=1,
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        self._streams[-1].seek(0)
        if self._nifti_version == 2:
            fh = nib.Nifti2Image.from_bytes(self.open().read())
        else:
            fh = nib.Nifti1Image.from_bytes(self.open().read())
        out = self._reorder_axes(fh.dataobj.get_unscaled())
        fh.uncache()
        return out

    def _close(self):
        for stream in self._streams[::-1]:
            if stream is not None:
                stream.close()


def save_nifti(
        image: Any,
        uri: str,
        compress: bool = True,
        partition: Optional[str] = None,

        # Format-specific kwargs
        nifti_version: int = 2
):
    if partition is None:
        partition = "xyz"

    for img, _uri in partition_gen(image, partition, uri):
        if nifti_version == 1:
            header = nib.Nifti1Header()
        else:
            header = nib.Nifti2Header()

        header.set_data_shape(img.shape[::-1])
        header.set_data_dtype(img.dtype)
        header.set_xyzt_units(xyz=get_spacing_units(image, "x") or None, t=get_spacing_units(image, "t") or None)

        affine = np.eye(img.ndim + 1, img.ndim + 1)
        for i, ax in enumerate(get_axes(img)):
            affine[i, i] = get_spacing(img, ax) or 1.0

        if nifti_version == 1:
            ni = nib.Nifti1Image(
                img.data,
                affine=affine,
                header=header)
        else:
            ni = nib.Nifti2Image(
                img.data,
                affine=affine,
                header=header)

        nib.save(ni, _uri)
