import numpy as np
import xarray as xr
import intake
import re
import zarr
from typing import Any, Optional
from .util import *
from .autodetect import *
from .source.tif import save_tif
from .source.klb import save_klb
from .source.nifti import save_nifti
from .source.nrrd import save_nrrd

from .source.klb import save_klb as _save_klb
from .source.nifti import save_nifti as _save_nifti
from .source.nrrd import save_nrrd as _save_nrrd
from .source.tif import save_tif as _save_tif
from .util import to_xarray as _to_xarray

def imload(uri: str, partition: Any = None, metadata_only: bool = False, **kwargs) -> xr.Dataset:
    """
    Load image, autodetect source type.

    :param uri:
    :param partition:
        Load only a partition of the image (TODO: Support list of partitions)
    :param metadata_only:
        Load metadata only
    :param kwargs:
        Additional arguments passed to the source constructor. Notable fields supported by all subclasses of
        ImageSource are:

        output_axis_order (Optional[str]): deviate from the default "itczyx" output axis ordering, or None to use
        original axis ordering of the data.

        metadata (Optional[Dict[str, Any]]): add or overrule metadata fields, for instance:
            metadata={
                "axes": "tcxy",                       # specify input axis order
                "spacing": {"t": 0.3},                # overrule spacing
                "spacing_units": {"t": "s"},          # overrule spacing unit
                "coords": {"c": ["nuclei", "actin"]}  # use specific channel names
            }

        TODO: Document catalog mechanism to manage external metadata and refer to it here.
    :return:
    """
    if isinstance(uri, intake.DataSource):
        if metadata_only:
            return uri.discover()
        out = _to_xarray(uri, partition=partition)
        if not isinstance(out, xr.Dataset):
            out = xr.Dataset({"image": out})
        out.attrs["metadata"] = {}
        try:
            out.attrs["metadata"]["spacing_units"] = out["image"].attrs["metadata"]["spacing_units"]
        except KeyError:
            pass
        return out
    elif isinstance(uri, intake.catalog.entry.CatalogEntry):
        return imload(uri.get(), partition, metadata_only)
    with autodetect(uri, **kwargs) as src:
        return imload(src, partition, metadata_only)


def imsave(image: Any, uri: str, compress: bool = True):
    luri = uri.lower()
    ext = os.path.splitext(luri)[-1]

    if len(ext) == 0:
        ext = ".zarr"

    if ext == ".nrrd":
        _save_nrrd(image, uri, compress, partition)
    elif ext == ".zarr":
        _save_zarr(image, uri, compress)
    elif ext == ".tif":
        _save_tif(image, uri, compress, partition)
    elif ext == ".klb":
        _save_klb(image, uri, compress, partition)
    else:
        raise NotImplementedError(f"intake_io.imsave(...) with file extension '{ext}'")


def _save_zarr(image: Any, uri: str, compress: Optional[bool] = None):
    if compress is None:
        compress = True

    # image = image.chunk({"i": 1})
    if compress:
        compressor = zarr.Blosc(cname="zstd", clevel=4)
        encoding = {k: {"compressor": compressor} for k in image.keys()}
    else:
        encoding = {}
    image.to_zarr(uri, consolidated=True, encoding=encoding)
