from intake.source.zarr import ZarrArraySource
from typing import Any, Optional

import zarr

from .base import ImageSource


class XrZarrSource(ImageSource, ZarrArraySource):
    """Intake source for zarr and xr.zarr files.

    The parameters dtype and shape will be determined from the first
    file, if not given.

    Parameters
    ----------
    urlpath : str
        Location of data file(s), possibly including protocol
        information
    storage_options : dict
        Passed on to storage backend for remote files
    component : str or None
        If None, assume the URL points to an array. If given, assume
        the URL points to a group, and descend the group to find the
        array at this location in the hierarchy.
    kwargs : passed on to dask.array.from_zarr
    """
    container = "ndarray"
    name = "xr_zarr"
    version = "0.0.1"
    partition_access = False

    def __init__(self, uri: str, **kwargs):
        """
        Arguments:
            uri (str): URI (file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(uri, **kwargs)


def save_zarr(
        image: Any,
        uri: str,
        compress: bool = True,
        partition: Optional[str] = None,

        # Format-specific kwargs
        compression_type: str = "zstd",
        compression_level: int = 4
):
    # image = image.chunk({"i": 1})
    if compress:
        compressor = zarr.Blosc(cname=compression_type, clevel=compression_level)
        encoding = {k: {"compressor": compressor} for k in image.keys()}
    else:
        encoding = {}
    image.to_zarr(uri, consolidated=True, encoding=encoding)
