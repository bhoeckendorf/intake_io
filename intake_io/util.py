import os
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple, Union, Generator

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


def get_spacing(
        image: Union[xr.Dataset, xr.DataArray],
        axes: Optional[str] = None
) -> Union[Tuple[Optional[float], ...], Optional[float]]:
    """
    Get pixel spacing of image or specific image axes.

    The return value for an axis without spacing metadata is either `None` or `1.0`, depending on what format the image
    was loaded from. `None` is preferred.

    :param image:
        Image
    :param axes:
        Specific axes and ordering, or `None` for all axes in image dim order.
    :return:
        Pixel spacing as tuple, or directly if a single axis is requested.
    """
    if isinstance(image, xr.DataArray):
        if axes is None:
            axes = "".join(i for i in get_axes(image) if i in "tzyx")
            is_specific_axes = False
        else:
            is_specific_axes = True
        spacing = tuple(
            image.coords[ax].values[1] - image.coords[ax].values[0] if ax in image.coords else None for ax in axes)
        if is_specific_axes and len(axes) == 1:
            return spacing[0]
        return spacing

    elif isinstance(image, xr.Dataset):
        ndim = len(image.dims)
        for img in image.values():
            if img.ndim == ndim:
                return get_spacing(img, axes)
        raise ValueError(f"Dataset claims {ndim} dimensions but no data variable has that many dimensions.")

    raise NotImplementedError(f"intake_io.get_spacing({type(image)})")


def get_spacing_units(
        image: Union[xr.Dataset, xr.DataArray],
        axes: Optional[str] = None
) -> Union[Tuple[Optional[str], ...], Optional[str]]:
    """
    Get pixel spacing unit of image or specific image axes.

    The return value for an axis without spacing metadata is `None`.

    :param image:
        Image
    :param axes:
        Specific axes and ordering, or `None` for all axes in image dim order.
    :return:
        Pixel spacing units as tuple, or directly if a single axis is requested.
    """
    if not any(isinstance(image, i) for i in (xr.Dataset, xr.DataArray)):
        raise NotImplementedError(f"intake_io.get_spacing_units({type(image)})")
    if axes is None:
        axes = "".join(i for i in get_axes(image) if i in "tzyx")
        is_specific_axes = False
    else:
        is_specific_axes = True
    units = []
    for ax in axes:
        try:
            units.append(image.attrs["metadata"]["spacing_units"][ax])
        except KeyError:
            units.append(None)
    if is_specific_axes and len(axes) == 1:
        return units[0]
    return tuple(units)


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
        spacing: Union[Dict[str, Optional[float]], Tuple[Optional[float], ...], None] = None,
        axes: Optional[str] = None,
        coords: Optional[Dict[str, Any]] = None,
        spacing_units: Union[Dict[str, Optional[str]], Tuple[Optional[str], ...], None] = None,
        name: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        partition: Optional[Any] = None
) -> xr.DataArray:
    """
    Combine inputs to xarray.DataArray.

    Preserve metadata where available and possible.

    :param image:
        The image.
    :param spacing:
        Time resolution and pixel spacing, as dict (axes are keys) or tuple (in image axes order, ignoring the
        discontinuous axes "c" and "i"). Axes without pixel spacing metadata can be omitted (dict) or set to `None`
        (dict and tuple).
    :param axes:
        The image axes, e.g. "tyz".
    :param coords:
        Axis coordinates for axes not covered by the spacing parameter (e.g. channel names), or to overwrite the
        spacing parameter with e.g. non-uniform spacing values.
    :param spacing_units:
        Time resolution and pixel spacing units, as dict (axes are keys) or tuple (in image axes order, ignoring the
        discontinuous axes "c" and "i"). Axes without pixel spacing metadata can be omitted (dict) or set to `None`
        (dict and tuple).
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
            coords = dict(i for i in deepcopy(coords).items() if i[0] in axes)

        # add spacing info to coords, if needed
        if spacing is None:
            spacing = {}
        if spacing_units is None:
            spacing_units = {}
        spacing, spacing_units = _get_spacing_dicts(axes, spacing, spacing_units)
        for a, s in spacing.items():
            if a not in coords and s is not None:
                coords[a] = np.arange(image.shape[axes.index(a)], dtype=np.float64) * s

        # in some circumstances, xarray rejects tuples as coords, so convert to list
        for k, v in coords.items():
            if isinstance(v, tuple):
                coords[k] = list(v)

        if len(coords) == 0:
            coords = None

        if attrs is None:
            attrs = {"metadata": {"spacing_units": spacing_units}}
        else:
            if "metadata" not in attrs:
                attrs["metadata"] = {"spacing_units": spacing_units}
            else:
                attrs["metadata"].update({"spacing_units": spacing_units})

        return xr.DataArray(image, dims=list(axes), coords=coords, name=name, attrs=attrs)

    elif isinstance(image, intake.DataSource):
        return image.to_xarray()

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


def _get_spacing_dicts(
        axes: str,
        spacing: Union[Dict[str, Optional[float]], Tuple[Optional[float], ...]],
        units: Union[Dict[str, Optional[str]], Tuple[Optional[str], ...]]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    unit_axes = "".join(i for i in axes if i in "tzyx")

    # Normalize spacing
    if not isinstance(spacing, dict):
        spacing = {a: s for a, s in zip(unit_axes[-len(spacing):], spacing)}
    else:
        spacing = deepcopy(spacing)
    for ax in list(spacing.keys()):
        if ax not in unit_axes or spacing[ax] is None or np.isnan(spacing[ax]):
            spacing.pop(ax)

    # Normalize units
    if not isinstance(units, dict):
        units = {a: s for a, s in zip(unit_axes[-len(units):], units)}
    else:
        units = deepcopy(units)
    for ax in list(units.keys()):
        if ax not in spacing:
            units.pop(ax)
        elif units[ax] in (None, "", " ", "pixel", "pix", "px", "sec"):
            units.pop(ax)
            if spacing[ax] in (0.0, 1.0):
                spacing.pop(ax)
        elif "\\" in units[ax]:
            units[ax] = units[ax].encode("latin-1", "backslashreplace").decode("unicode-escape")

    # Fill missing spatial units
    unit = None
    for ax in [i for i in units.keys() if i in "zyx"]:
        if unit is None:
            unit = units[ax]
        elif units[ax] != unit:
            unit = None
            break
    if unit is not None:
        for ax in [i for i in "zyx" if i in axes]:
            if ax in spacing and ax not in units:
                units[ax] = unit

    return spacing, units


def _reorder_axes(array: np.ndarray, axes_source: str, axes_target: Optional[str] = None) -> np.ndarray:
    if axes_target is None:
        axes_target = "itczyx"
    if len(axes_target) > len(axes_source):
        axes_target = "".join(i for i in axes_target if i in axes_source)

    if axes_source == axes_target:
        return array

    for i in (axes_source, axes_target):
        if not len(set(i)) == len(i):
            raise ValueError(f"Duplicate axis in {i}.")

    if any(i not in axes_source for i in axes_target):
        raise ValueError(f"Requested axis order '{axes_target}' can't be satisfied by data '{axes_source}'.")
    if any(i not in axes_target for i in axes_source):
        raise ValueError(f"Requested axis order '{axes_target}' is insufficient for data '{axes_source}'.")

    try:
        reorder = [axes_source.index(i) for i in axes_target[-len(axes_source):]]
    except ValueError:
        raise ValueError(f"Can't reorder axes from '{axes_source}' to '{axes_target[-len(axes_source):]}'."
                         " This is usually caused by loading a data partition and asking for an axis order that would"
                         " require to intercalate axes from outside the loaded partition."
                         f" In this case, the loaded data is '{axes_source}' and the requested order"
                         f" is '{axes_target}'.")

    if len(reorder) < 2 or all(reorder[i] + 1 == reorder[i + 1] for i in range(len(reorder) - 1)):
        return array
    return np.ascontiguousarray(array.transpose(*reorder))


def _reorder_xaxes(image, axes_target: Optional[str] = None) -> np.ndarray:
    if axes_target is None:
        axes_target = "itczyx"
    axes_source = get_axes(image)
    if len(axes_target) > len(axes_source):
        axes_target = "".join(i for i in axes_target if i in axes_source)

    if axes_source == axes_target:
        return image

    if isinstance(image, xr.Dataset):
        out = xr.Dataset({k: _reorder_xaxes(v, axes_target) for k, v in image.items()})
        out.attrs = image.attrs
    else:
        out = xr.DataArray(
            _reorder_axes(image.data, axes_source, axes_target),
            image.coords,
            list(axes_target),
            image.name,
            image.attrs,
        )
    return out


def partition_gen(
        image: Union[xr.DataArray, xr.Dataset],
        inner_axes: str,
        uri: str,
        multikey: bool = False,
        sep_outer: str = ".",
        sep_inner: str = "_"
) -> Generator[Tuple[Union[xr.DataArray, xr.Dataset], str], None, None]:
    axes = get_axes(image)
    outer_axes = "".join(i for i in axes if i not in inner_axes)

    uri_base, uri_ext = os.path.splitext(uri)
    int_paddings = [j for j in outer_axes if np.all([np.issubdtype(i.dtype, np.integer) for i in image.coords[j].data])]
    int_paddings = {i: len(str(np.max(image.coords[i].data))) for i in int_paddings}

    if isinstance(image, xr.Dataset) and not multikey:
        for key, img in image.items():
            if len(image.keys()) > 1:
                _uri = os.path.splitext(uri)
                _uri = f"{_uri[0]}{sep_outer}var{sep_inner}{key}{_uri[1]}"
            else:
                _uri = uri
            for out in partition_gen(img, inner_axes, _uri, multikey, sep_outer, sep_inner):
                yield out
        return

    if isinstance(image, xr.Dataset):
        for img in image.values():
            if img.ndim == len(axes):
                arr = img.data
    else:
        arr = image.data

    with np.nditer(arr, flags=("multi_index",), op_flags=("readonly",), op_axes=[[axes.index(i) for i in outer_axes],]) as ndit:
        for it in ndit:
            ix = dict(zip(outer_axes, ndit.multi_index))
            ix = {ax: image.coords[ax].data[ix] for ax, ix in ix.items()}
            out = _reorder_xaxes(image.sel(**ix), inner_axes)

            uri_out = [uri_base]
            for ax in outer_axes:
                if ax in int_paddings:
                    uri_out.append(f"{ax.upper()}{sep_inner}{str(ix[ax]).rjust(int_paddings[ax], '0')}")
                else:
                    uri_out.append(f"{ax.upper()}{sep_inner}{ix[ax]}")

            yield out, sep_outer.join(uri_out) + uri_ext


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
