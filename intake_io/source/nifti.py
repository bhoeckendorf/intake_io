import re
from gzip import GzipFile
from io import BytesIO
from typing import Any, Optional, Union

import nibabel
import numpy as np
import xarray as xr
from nibabel import Nifti1Image

from .base import ImageSource, Schema
from ..util import get_axes, get_spacing, to_numpy


class NiftiSource(ImageSource):
    """Intake source for NIFTI files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "nifti"
    version = "0.0.1"
    partition_access = False

    def __init__(self, uri: Optional[str], stream: Optional[Union[BytesIO, GzipFile]] = None, **kwargs):
        """
        Arguments:
            uri (str): URI (file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(**kwargs)
        self.uri = uri
        self._filehandle = None if stream is None else Nifti1Image.from_bytes(stream.read())

    def _get_schema(self) -> Schema:
        if self._filehandle is None:
            self._filehandle = nibabel.load(self.uri)
        header = self._filehandle.header

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
        out = self._reorder_axes(self._filehandle.dataobj.get_unscaled())
        self._filehandle.uncache()
        return out

    def _close(self):
        if self._filehandle is not None:
            self._filehandle.uncache()


def save_nifti(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_nifti(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".nii.gz", f".{k}.nii.gz", uri, flags=re.IGNORECASE)
                save_nifti(image[k], uri_key, compress)
        return

    h = nibabel.Nifti2Header()
    h.set_data_shape(image.shape[::-1])
    h.set_data_dtype(image.dtype)
    h.get("pixdim")[1:1 + image.ndim] = get_spacing(image)[::-1]
    ni = nibabel.Nifti2Image(
        to_numpy(image).T,
        affine=np.eye(image.ndim + 1, image.ndim + 1),
        header=h)
    nibabel.save(ni, uri)
