import numpy as np
import xarray as xr
import imageio
import intake
import pyklb as klb
import nrrd
import nibabel
import re
from typing import Any, Optional
from .util import *
from .autodetect import *


def imload(uri: str, partition: Any = None, metadata_only: bool = False) -> xr.Dataset:
    if isinstance(uri, intake.DataSource):
        if metadata_only:
            return uri.discover()
        return xr.Dataset({"image": to_xarray(uri, partition=partition)})
    elif isinstance(uri, intake.catalog.entry.CatalogEntry):
        return imload(uri.get(), partition, metadata_only)
    with autodetect(uri) as src:
        return imload(src, partition, metadata_only)


def imsave(image: Any, uri: str, compress: bool = True):
    luri = uri.lower()
    ext = os.path.splitext(luri)[-1]

    if len(ext) == 0:
        ext = ".zarr"

    if ext == ".nrrd":
        _save_nrrd(image, uri, compress)
    elif ext == ".zarr":
        _save_zarr(image, uri, compress)
    elif ext == ".tif":
        _save_tif(image, uri, compress)
    elif luri.endswith(".nii.gz"):
        _save_nifti(image, uri, compress)
    elif ext == ".klb":
        _save_klb(image, uri, compress)
    else:
        raise NotImplementedError(f"intake_io.imsave(...) with file extension '{ext}'")


def _save_nrrd(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            _save_nrrd(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".nrrd", f".{k}.nrrd", uri, flags=re.IGNORECASE)
                _save_nrrd(image[k], uri_key, compress)
        return

    header = dict(encoding="gzip" if compress else "raw")
    header["space dimension"] = image.ndim

    try:
        header["labels"] = list(image.attrs["axes"])[::-1]
        header["kinds"] = []
        for i in header["labels"]:
            if i in "zyx":
                header["kinds"].append("space")
            elif i == "c":
                header["kinds"].append("scalar")
            elif i == "t":
                header["kinds"].append("time")
            else:
                header["kinds"].append("none")
    except (AttributeError, KeyError, TypeError):
        pass

    try:
        #header["units"] = list([image.attrs["spacing_units"][d] for d in header["labels"]])
        header["space units"] = list([image.attrs["spacing_units"][d] for d in header["labels"]])
    except (AttributeError, KeyError):
        pass

    spacing = get_spacing(image)
    if spacing is not None:
        #header["spacings"] = spacing[::-1]
        header["space directions"] = np.eye(len(spacing), len(spacing), dtype=np.float)
        for i, v in enumerate(spacing[::-1]):
            header["space directions"][i, i] = v

    try:
        header["channels"] = list(image.coords["c"])
    except KeyError:
        pass

    nrrd.write(uri, to_numpy(image), header, compression_level=4, index_order="C")


def _save_zarr(image: Any, uri: str, compress: bool):
    image = image.chunk({"i": 1})
    if compress:
        compressor = zarr.Blosc(cname="zstd", clevel=4)
        encoding = {k: {"compressor": compressor} for k in image.keys()}
    else:
        encoding = {}
    image.to_zarr(uri, consolidated=True, encoding=encoding)


def _save_tif(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            _save_tif(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".tif{1,2}", f".{k}.tif", uri, flags=re.IGNORECASE)
                _save_tif(image[k], uri_key, compress)
        return

    # todo: handle RGB
    mode = "v"
    if image.ndim == 2:
        mode = "i"
    with imageio.get_writer(uri, mode) as writer:
        if compress:
            writer.set_meta_data(dict(compress=4))
        writer.append_data(to_numpy(image))


def _save_nifti(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            _save_nifti(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".nii.gz", f".{k}.nii.gz", uri, flags=re.IGNORECASE)
                _save_nifti(image[k], uri_key, compress)
        return

    h = nibabel.Nifti2Header()
    h.set_data_shape(image.shape[::-1])
    h.set_data_dtype(image.dtype)
    h.get("pixdim")[1:1+image.ndim] = get_spacing(image)[::-1]
    ni = nibabel.Nifti2Image(
        to_numpy(image).T,
        affine=np.eye(image.ndim+1, image.ndim+1),
        header=h)
    nibabel.save(ni, uri)


def _save_klb(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            _save_klb(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".klb", f".{k}.klb", uri, flags=re.IGNORECASE)
                _save_klb(image[k], uri_key, compress)
        return

    assert image.ndim < 6

    shape_out = []
    spacing_out = []
    for i, d in enumerate("tczyx"):
        if not d in image.dims:
            if len(shape_out) == 0:
                continue
            shape_out.append(1)
            spacing_out.append(1.0)
        else:
            shape_out.append(image.dims[d])
            spacing_out.append(image.coords[d][1] - image.coords[d][0])

    klb.writefull(
        to_numpy(image).reshape(shape_out),
        uri,
        pixelspacing_tczyx = np.asarray(spacing_out, dtype=np.float32),
        compression = "bzip" if compress else "none"
    )
