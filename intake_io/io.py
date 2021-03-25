import numpy as np
import xarray as xr
import intake
import re
from typing import Any, Optional
from .util import *
from .autodetect import *
from .source.imageio import save_tif
from .source.klb import save_klb
from .source.nifti import save_nifti
from .source.nrrd import save_nrrd


def imload(uri: str, partition: Any = None, metadata_only: bool = False) -> xr.Dataset:
    if isinstance(uri, intake.DataSource):
        if metadata_only:
            return uri.discover()
        return xr.Dataset({"image": to_xarray(uri, partition=partition)})
    elif isinstance(uri, intake.catalog.entry.CatalogEntry):
        return imload(uri.get(), partition, metadata_only)
    with autodetect(uri) as src:
        return imload(src, partition, metadata_only)
    
def imload_with_coords(uri: str, coords: dict, partition: Any = None) -> xr.Dataset:
    img = imload(uri, partition=partition)
    for key in coords.keys():
        if coords[key] is not None:
            img.coords[key] = np.arange(len(img[key])) * coords[key]
        else:
            img.coords[key] = np.arange(len(img[key]))
    return img


def imsave(image: Any, uri: str, compress: bool = True):
    luri = uri.lower()
    ext = os.path.splitext(luri)[-1]

    if len(ext) == 0:
        ext = ".zarr"

    if ext == ".nrrd":
        save_nrrd(image, uri, compress)
    elif ext == ".zarr":
        save_zarr(image, uri, compress)
    elif ext == ".tif":
        save_tif(image, uri, compress)
    elif luri.endswith(".nii.gz"):
        save_nifti(image, uri, compress)
    elif ext == ".klb":
        save_klb(image, uri, compress)
    else:
        raise NotImplementedError(f"intake_io.imsave(...) with file extension '{ext}'")


def save_zarr(image: Any, uri: str, compress: bool):
    image = image.chunk({"i": 1})
    if compress:
        compressor = zarr.Blosc(cname="zstd", clevel=4)
        encoding = {k: {"compressor": compressor} for k in image.keys()}
    else:
        encoding = {}
    image.to_zarr(uri, consolidated=True, encoding=encoding)
