import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import dask.array as da
import intake
import numpy as np
import xarray as xr


def get_axes(x: Any) -> str:
    """
    Get axis order given image, shape or number of dimensions.

    .. list-table:: Supported axes

       * - Name
         - Dimension
       * - i
         - image index
       * - t
         - time
       * - c
         - channel
       * - z
         - z
       * - y
         - y
       * - x
         - x

    Uses metadata where available (xarray data structures), otherwise assumes itczyx-ordering (intake_io default) using
    the last ndim axes. The only exception are 2D RGB images, which may be cyx or yxc.

    :param image:
        One of the following types:
            * :class:`xarray.Dataset`
            * :class:`xarray.DataArray`
            * :class:`numpy.ndarray`
            * :class:`dask.Array`
            * :class:`tuple` (shape)
            * :class:`int` (number of dimensions)
    :return:
        Axis order, e.g. "tzyx"
    """
    AXES = "itczyx"

    if isinstance(x, xr.DataArray):
        if len(x.shape) == 0:
            raise ValueError(f"Image has no shape.")
        out = ""
        for i in map(str, x.dims):
            if len(i) != 1 or i not in AXES:
                raise ValueError(f"Unknown axis '{i}', supports '{AXES}'.")
            out += i
        return out

    elif isinstance(x, xr.Dataset):
        ndim = len(x.dims)
        for img in x.values():
            if img.ndim == ndim:
                return get_axes(img)
        raise ValueError(f"Dataset claims {ndim} dimensions but no data variable has that many dimensions.")

    elif isinstance(x, np.ndarray) or isinstance(x, da.Array):
        # attempt to detect RGB
        if x.ndim == 3 and any(np.issubdtype(x.dtype, i) for i in (np.uint8, np.float32)):
            if x.shape[0] == 3:
                return "cyx"
            elif x.shape[-1] == 3:
                return "yxc"
        # else defer
        return get_axes(x.shape)

    elif isinstance(x, tuple):
        # attempt to detect RGB
        if len(x) == 3:
            if x[0] == 3:
                return "cyx"
            elif x[-1] == 3:
                return "yxc"
        # else defer
        return get_axes(len(x))

    elif isinstance(x, int):
        if 1 <= x <= len(AXES):
            return AXES[-x:]
        raise ValueError(f"Found {x} axes, supports 1-{len(AXES)} axes.")

    raise NotImplementedError(f"intake_io.get_axes({type(x)})")


def get_spacing(image: Union[xr.Dataset, xr.DataArray]) -> Tuple[float, ...]:
    """
    Get pixel spacing of image.

    Returns 1 for any axis that lacks spacing metadata. Units are intended to be seconds and microns by convention.

    :param image:
        Image
    :return:
        Pixel spacing in "tzyx" order. Singleton dimensions may be dropped.
    """
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
        for img in image.values():
            if img.ndim == ndim:
                return get_spacing(img)
        raise ValueError(f"Dataset claims {ndim} dimensions but no data variable has that many dimensions.")

    raise NotImplementedError(f"intake_io.get_spacing({type(image)})")


def to_numpy(image: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> np.ndarray:
    """
    Get pixel array of image.

    :param image:
        Image
    :return:
        Array
    """
    if isinstance(image, np.ndarray):
        return image

    elif isinstance(image, xr.DataArray):
        return image.data

    elif isinstance(image, xr.Dataset):
        if len(image) == 1:
            return image[list(image.keys())[0]].data
        raise ValueError(f"Dataset has {len(image)} data variables, please specify which one to use.")

    raise NotImplementedError(f"intake_io.to_numpy({type(image)})")


def to_xarray(
        image: Any,
        spacing: Optional[Tuple[float, ...]] = None,
        axes: Optional[str] = None,
        coords: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        partition: Optional[Any] = None
) -> xr.DataArray:
    """
    Combine inputs to xarray.DataArray.

    Preserve metadata where available and possible.

    :param image:
        The image, dimension order should be itczyx, singleton dimensions may be dropped.
    :param spacing:
        Time resolution and pixel spacing in tzyx order, seconds and microns. Singleton dimensions may be dropped.
    :param axes:
        The image axes, e.g. "tyz".
    :param coords:
        Axis coordinates for axes not covered by the spacing parameter, or to overwrite the spacing parameter.
    :param name:
        Name
    :param attrs:
        Additional metadata
    :param partition:
        Partition
    :return:
        Image and any metadata
    """
    if isinstance(image, xr.DataArray):
        return image

    elif isinstance(image, xr.Dataset):
        if len(image) == 1:
            return image[list(image.keys())[0]]
        raise ValueError(f"Dataset has {len(image)} data variables, please specify which one to use.")

    elif isinstance(image, np.ndarray) or isinstance(image, da.Array):
        if axes is None:
            axes = get_axes(image)
        else:
            if any(i not in "itczyx" for i in axes):
                raise ValueError(f"Unknown axis in '{axes}', supports 'itczyx'.")
            if any(axes.count(i) > 1 for i in axes):
                raise ValueError(f"Duplicate axis in '{axes}'.")
            if len(axes) != image.ndim:
                raise ValueError(f"{image.ndim} image dimensions, but {len(axes)} axes: '{axes}'.")

        if coords is None:
            coords = {}
        else:
            # make copy, drop unused coordinates
            coords = dict(filter(lambda x: x[0] in axes, copy.deepcopy(coords).items()))

        if spacing is not None:
            # add spacing info to coords, if needed
            for a, s in zip(list(filter(lambda x: x in "tzyx", axes))[-len(spacing):], spacing):
                if a not in coords:
                    coords[a] = np.arange(image.shape[axes.index(a)], dtype=np.float64) * s

        # in some circumstances, xarray rejects tuples as coords, so convert to list
        for k, v in coords.items():
            if isinstance(v, tuple):
                coords[k] = list(v)

        if len(coords) == 0:
            coords = None

        return xr.DataArray(image, dims=list(axes), coords=coords, name=name, attrs=attrs)

    elif isinstance(image, intake.DataSource):
        image.discover()

        if axes is None:
            try:
                axes = image.metadata["axes"]
            except KeyError:
                axes = get_axes(image.shape)

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
            if spacing is not None and axes[0] in "tzyx":
                spacing = spacing[1:]
            axes = axes[1:]
            img = to_xarray(image.read_partition(partition), spacing, axes, coords, name, attrs)
        else:
            img = to_xarray(image.read(), spacing, axes, coords, name, attrs)

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
        return to_xarray(image.get(), spacing, axes, coords, name, attrs, partition)

    return to_xarray(to_numpy(image), get_spacing(image), axes, coords, name, attrs)


def get_catalog(sources: Iterable[intake.source.DataSource], filepath: Optional[str] = None) -> str:
    """
    Get catalog yml of data sources.

    :param sources:
        Data sources
    :param filepath:
         File path to save catalog to, or None if saving the result to file is not intended.
    :return:
        The yml text
    """
    yml = [sources[0].yaml(), *[i.yaml().replace("sources:\n", "") for i in sources[1:]]]
    yml = "\n".join(yml)
    if filepath is not None:
        if os.path.exists(filepath):
            raise FileExistsError(filepath)
        with open(filepath, "w") as f:
            f.write(yml)
    return yml


def clean_yaml(data: Dict[str, Any], rename_to: Optional[str] = None) -> Dict[str, Any]:
    """
    Utility function to clean YAML representation of intake source.

    Used during creation of YAML catalog from list of sources.

    :param data:
        Dict[str, Any]
    :param rename_to:
        Optional[str]
    :return:
        Cleaned dict
    """
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
