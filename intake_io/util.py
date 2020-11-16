import os
import numpy as np
import dask.array as da
import xarray as xr
import copy
import intake
from typing import Any, Optional


def get_axes(image: Any) -> str:
    if isinstance(image, xr.DataArray):
        return "".join(image.dims)
    elif isinstance(image, xr.Dataset):
        ndim = len(image.dims)
        for k in image.keys():
            if image[k].ndim == ndim:
                return get_axes(image[k])
    elif isinstance(image, np.ndarray) or isinstance(image, da.Array):
        # attempt to detect RGB
        if image.ndim == 3 and (image.dtype == np.uint8 or image.dtype == np.float32):
            if image.shape[0] == 3:
                return "cyx"
            elif image.shape[-1] == 3:
                return "yxc"
        # assign according to default order
        return get_axes(image.ndim)
    elif isinstance(image, tuple):
        return get_axes(len(image))
    elif isinstance(image, int):
        return "itczyx"[-image:]
    raise NotImplementedError(f"intake_io.get_axes({type(image)})")


def get_spacing(image: Any) -> tuple:
    if isinstance(image, xr.DataArray):
        spacing = []
        for a in "tzyx":
            try:
                spacing.append(image.coords[a].values[1] - image.coords[a].values[0])
            except KeyError:
                continue
        return tuple(spacing)
    elif isinstance(image, xr.Dataset):
        ndim = len(image.dims)
        for k in image.keys():
            if image[k].ndim == ndim:
                return get_spacing(image[k])
    raise NotImplementedError(f"intake_io.get_spacing({type(image)})")


def to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, xr.DataArray):
        return image.data
    elif isinstance(image, xr.Dataset):
        if len(image) == 1:
            return image[list(image.keys())[0]].data
        raise ValueError(f"intake_io.to_numpy({type(image)}) argument has >1 data variables")
    raise NotImplementedError(f"intake_io.to_numpy({type(image)})")


def to_xarray(
        image: Any,
        spacing: Optional[tuple] = None,
        axes: Optional[str] = None,
        coords: Optional[dict] = None,
        spacing_units: Optional[tuple] = None,
        name: Optional[str] = None,
        attrs: Optional[dict] = None,
        partition: Any = None
        ) -> xr.DataArray:
    if isinstance(image, xr.DataArray):
        return image
    elif isinstance(image, xr.Dataset):
        if len(image) == 1:
            return image[list(image.keys())[0]]
        raise ValueError(f"intake_io.to_xarray({type(image)}, ...) argument has >1 data variables")
    elif isinstance(image, np.ndarray) or isinstance(image, da.Array):
        if axes is None:
            axes = get_axes(image)
        axes = axes[-image.ndim:]

        if coords is None:
            coords = {}
        else:
            coords = copy.deepcopy(coords)
        remove = []
        for k in coords.keys():
            if k not in axes:
                remove.append(k)
        for k in remove:
            coords.pop(k)

        if spacing is not None:
            spacing = spacing[-np.sum([a in axes for a in "tzyx"]):]
            for a, s in zip("tzyx"[-len(spacing):], spacing):
                if a not in coords:
                    coords[a] = np.arange(image.shape[axes.index(a)], dtype=np.float64) * s

        if len(coords) == 0:
            coords = None

        # xarray doesn't accept tuples as coords, convert to list if needed
        if coords is not None:
            for k, v in coords.items():
                if isinstance(v, tuple):
                    coords[k] = list(v)

        if spacing_units is not None:
            if attrs is None:
                attrs = {}
            if "spacing_units" not in attrs:
                attrs["spacing_units"] = spacing_units[-len(spacing):]

        return xr.DataArray(image, dims=list(axes), coords=coords, name=name, attrs=attrs)
    elif isinstance(image, intake.DataSource):
        image.discover()

        if axes is None:
            try:
                axes = image.metadata["axes"]
            except KeyError:
                axes = "itczyx"[-len(image.shape):]

        if spacing is None:
            try:
                spacing = image.metadata["spacing"]
            except KeyError:
                pass

        try:
            if coords is None:
                coords = image.metadata["coords"]
            else:
                coords = {**image.metadata["coords"], **coords}
        except KeyError:
            pass

        if partition is not None:
            if coords is not None:
                coords = copy.deepcopy(coords)
                try:
                    coords.pop(axes[0])
                except KeyError:
                    pass
            if axes[0] in "tzyx" and spacing is not None:
                spacing = spacing[1:]
            axes = axes[1:]
            img = to_xarray(image.read_partition(partition), spacing, axes, coords, spacing_units, name, attrs)
        else:
            img = to_xarray(image.read(), spacing, axes, coords, spacing_units, name, attrs)

        if "metadata" not in img.attrs:
            img.attrs["metadata"] = image.metadata
        else:
            img.attrs["metadata"] = {**image.metadata, **img.attrs["metadata"]}

        if "uri" not in img.attrs:
            try:
                img.attrs["uri"] = image.uri
            except KeyError:
                pass

        return img
    elif isinstance(image, intake.catalog.entry.CatalogEntry):
        return to_xarray(image.get(), spacing, axes, coords, spacing_units, name, attrs, partition)
    else:
        return to_xarray(to_numpy(image), get_spacing(image), axes, coords, spacing_units, name, attrs)


def clean_yaml(data: dict, rename_to: Optional[str] = None) -> dict:
    assert len(data["sources"].keys()) == 1
    name = list(data["sources"].keys())[0]
    imeta = data["sources"][name]["metadata"]
    for i in ("axes", "catalog_dir", "fileheader"):
        try:
            imeta.pop(i)
        except (KeyError, TypeError):
            pass
    try:
        for d in imeta["coords"]:
            v = imeta["coords"][d]
            if not isinstance(v, list):
                imeta["coords"][d] = list(v)
    except (KeyError, TypeError):
        pass
    try:
        if imeta["spacing"] is not None and not isinstance(imeta["spacing"], list):
            imeta["spacing"] = list(imeta["spacing"])
    except KeyError:
        pass
    if rename_to is not None:
        assert len(data["sources"]) == 1 and isinstance(rename_to, str) and not str.isnumeric(rename_to[0])
        data["sources"][rename_to] = data["sources"].pop(name)
    return data
