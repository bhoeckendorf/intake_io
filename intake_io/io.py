import os
from typing import Any, Optional

import intake
import xarray as xr
import zarr

from .autodetect import autodetect as _autodetect
from .source.klb import save_klb as _save_klb
from .source.nifti import save_nifti as _save_nifti
from .source.nrrd import save_nrrd as _save_nrrd
from .source.tif import save_tif as _save_tif
from .util import to_xarray as _to_xarray


def imload(uri: str, partition: Optional[Any] = None, metadata_only: bool = False, **kwargs) -> xr.Dataset:
    """
    Load image, autodetect source type.

    :param str uri:

    :param Optional[Any] partition:
        Load only a partition of the image

    :param bool metadata_only:
        Load metadata only

    :param kwargs:
        Additional arguments passed to the source constructor. Notable fields supported by all subclasses of
        :class:`intake_io.source.base.ImageSource` are:

        * **output_axis_order** (`Optional[str]`) --
          Deviate from default "itczyx" output axis ordering, or `None` to use original axis ordering of the data
        * **metadata** (`Optional[Dict[str, Any]]`) --
          Add or overrule metadata fields, for instance:

          .. code-block:: python

            metadata={
                "axes": "tcxy",                      # specify input axis order
                "spacing": {"t": 0.3},               # overrule spacing
                "spacing_units": {"t": "ms"},        # overrule spacing unit
                "coords": {"c": ["nuclei", "actin"]} # use specific channel names
            }

    :return: image
    :rtype: :class:`xarray.Dataset`
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
    with _autodetect(uri, **kwargs) as src:
        return imload(src, partition, metadata_only)


def imsave(image: Any, uri: str, compress: Optional[bool] = None, partition: Optional[str] = None, **kwargs):
    """
    Save image, autodetect format.

    Format is chosen based on URI extension. Defaults to .zarr if no extension is given.

    `image` is automatically split into multiple files if the output format doesn't support saving it in a single file. For instance, TIF hyperstacks support single arrays with up to
    `tzcyx` axes. An :class:`xarray.Dataset` with multiple variables (e.g. "image" and "segmentation")
    and extra axes (e.g. multiple image arrays along the `i`-axis) is saved as multiple files according to the following
    pattern (axes are separated by :code:`.`, key and value are separated by :code:`_`)::

        /path/to/filename.var_image.i_00.tif
        /path/to/filename.var_image.i_01.tif
        ...
        /path/to/filename.var_segmentation.i_00.tif
        /path/to/filename.var_segmentation.i_01.tif
        ...

    The indexes used in the file name are the actual :class:`xarray.DataArray` coordinates along their respective axes, e.g. if our `i` coordinates are :code:`[3, 10]` or :code:`["control", "treatment"]` then we'll get file names containing "i_03" or "i_treatment". The data variable key (e.g. "var_segmentation" in the example above) is omitted from the output file name if `image` doesn't have multiple data variables or if it isn't an :class:`xarray.Dataset`.
    
    The `partition` argument controls the partitioning scheme. By default, `image` is split only if needed, and into as few files as possible. In contrast, e.g. :code:`partition="tyx"` partitions the data into files containing only the given axes, in order (axes are reordered as needed), yielding e.g. the following file pattern::

        /path/to/filename.var_image.i_03.c_actin.tif
        /path/to/filename.var_image.i_03.c_nuclei.tif
        /path/to/filename.var_image.i_21.c_actin.tif
        /path/to/filename.var_image.i_21.c_nuclei.tif
        ...
        /path/to/filename.var_segmentation.i_03.c_actin.tif
        /path/to/filename.var_segmentation.i_03.c_nuclei.tif
        /path/to/filename.var_segmentation.i_21.c_actin.tif
        /path/to/filename.var_segmentation.i_21.c_nuclei.tif
        ...

    See `File patterns <https://intake-io.readthedocs.io/en/latest/filepatterns.html>`_ section of this documentation for help with loading these file patterns.
    
    :param Any image:

    :param str uri:

    :param Optional[bool] compress:
        Whether or not to compress the data. Default value `None` defers decision to backend. Where applicable,
        compression algorithm and parameters are chosen as a trade-off of compatibility, compression ratio and I/O
        performance. If compression is requested but unsupported, the data is saved uncompressed and no exception is
        raised.

    :param Optional[str] partition:
        Partition the data as needed into multiple files, each containing the given axes, in the given order. By default, as few files as possible are created.
    """
    luri = uri.lower()
    ext = os.path.splitext(luri)[-1]

    if compress is not None:
        assert "compress" not in kwargs
        kwargs["compress"] = compress

    if len(ext) == 0:
        ext = ".zarr"

    if ext == ".nrrd":
        _save_nrrd(image, uri, partition=partition, **kwargs)
    elif ext == ".zarr":
        _save_zarr(image, uri, partition=partition, **kwargs)
    elif ext == ".tif":
        _save_tif(image, uri, partition=partition, **kwargs)
    elif luri.endswith(".nii.gz") or luri.endswith(".nii"):
        _save_nifti(image, uri, partition=partition, **kwargs)
    elif ext == ".klb":
        _save_klb(image, uri, partition=partition, **kwargs)
    else:
        raise NotImplementedError(f"intake_io.imsave(...) with file extension '{ext}'")


def _save_zarr(
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
