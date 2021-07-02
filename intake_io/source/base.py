from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import fsspec
import numpy as np
from intake.source.base import DataSource, Schema
from yaml import dump as _dump

from ..util import _get_spacing_dicts, _reorder_axes, get_axes, to_xarray


class ImageSource(DataSource):

    def __init__(self, uri: str, *args, output_axis_order: Optional[str] = "itczyx", **kwargs):
        super().__init__(*args, **kwargs)
        self.uri = uri
        self._output_axis_order = output_axis_order
        if self._output_axis_order is not None and len(set(self._output_axis_order)) != len(self._output_axis_order):
            raise ValueError(f"Duplicate axis in {self._output_axis_order}.")

    def open(self):
        try:
            return self.__file
        except AttributeError:
            self.__file = fsspec.open(self.uri).open()
            return self.__file

    def to_xarray(self, partition=None):
        self._load_metadata()
        axes = self.metadata.get("axes") or get_axes(self.shape)
        spacing = self.metadata.get("spacing") or {}
        spacing_units = self.metadata.get("spacing_units") or {}
        coords = self.metadata.get("coords") or {}

        if partition is not None:
            if spacing is not None and axes[0] in "tzyx":
                spacing = spacing[1:]
            axes = axes[1:]
            img = to_xarray(self.read_partition(partition), spacing, axes, coords, spacing_units)
        else:
            img = to_xarray(self.read(), spacing, axes, coords, spacing_units)

        img.attrs["metadata"] = self.metadata
        try:
            img.attrs["uri"] = self.uri
        except AttributeError:
            pass

        return img

    def _set_shape_metadata(
            self,
            axes: str,
            shape: Union[Dict[str, int], Tuple[int, ...]],
            spacing: Union[Dict[str, float], Tuple[float, ...]],
            spacing_units: Dict[str, str],
            coords: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, ...]:
        if len(set(axes)) != len(axes):
            raise ValueError(f"Duplicate axis in {axes}.")

        if isinstance(shape, dict):
            if any(ax not in shape for ax in axes):
                raise ValueError(f"Axes and shape mismatch: {axes}, {shape}.")
            self.metadata["original_shape"] = tuple(shape[ax] for ax in axes)
        else:
            if len(axes) != len(shape):
                raise ValueError(f"Axes and shape mismatch: {axes}, {shape}.")
            self.metadata["original_shape"] = shape
            shape = dict(zip(axes, shape))
        self.metadata["original_axes"] = axes

        if not isinstance(spacing, dict):
            spacing = dict(zip([i for i in axes if i in "tzyx"][-len(spacing):], spacing))

        # reassign axes if needed, don't reorder
        if self.metadata.get("axes"):
            _axes = axes
            axes = self.metadata["axes"]

            if len(set(axes)) != len(axes):
                raise ValueError(f"Duplicate axis in {axes}.")

            if len(axes) != len(_axes):
                raise ValueError(f"Nr of specified and original axes mismatch: {axes}, {_axes}.")

            _shape = deepcopy(shape)
            shape = {n: _shape[o] for o, n in zip(_axes, axes)}

            _spacing = deepcopy(spacing)
            spacing = {n: _spacing[o] for o, n in zip(_axes, axes) if o in _spacing}

            _spacing_units = deepcopy(spacing_units)
            spacing_units = {n: _spacing_units[o] for o, n in zip(_axes, axes) if o in _spacing_units}

            if coords is not None:
                _coords = deepcopy(coords)
                coords = {n: _coords[o] for o, n in zip(_axes, axes) if o in _coords}

        self.metadata["axes"] = axes if self._output_axis_order is None else "".join(
            i for i in self._output_axis_order if i in axes)

        spacing.update(self.metadata.get("spacing") or {})
        spacing_units.update(self.metadata.get("spacing_units") or {})
        spacing, spacing_units = _get_spacing_dicts(axes, spacing, spacing_units)
        self.metadata["spacing"] = spacing
        self.metadata["spacing_units"] = spacing_units

        if coords is not None:
            coords.update(self.metadata.get("coords") or {})
            for k in list(coords.keys()):
                if coords[k] is None or len(coords[k]) == 0:
                    coords.pop(k)
            self.metadata["coords"] = coords

        return tuple(shape[k] for k in self.metadata["axes"])

    def _close(self):
        try:
            self.__file.close()
        except AttributeError:
            pass

    def _set_fileheader(self, header: Dict[Any, Any]):
        metadata = self.metadata.get("fileheader") or {}
        self.metadata["fileheader"] = {**header, **metadata}

    def _reorder_axes(self, array: np.ndarray, axes_source: Optional[str] = None) -> np.ndarray:
        if axes_source is None:
            axes_source = self.metadata["original_axes"][-array.ndim:]
        return _reorder_axes(array, axes_source, self.metadata["axes"])

    def _yaml(self, rename_to: Optional[str] = None) -> Dict[str, Any]:
        out = deepcopy(super()._yaml())

        metadata = out["sources"][self.name]["metadata"]
        for k in ("spacing", "spacing_units"):
            try:
                metadata[k] = list(metadata[k])
            except (KeyError, TypeError):
                pass
        try:
            metadata.pop("fileheader")
        except KeyError:
            pass

        if rename_to is not None:
            out["sources"][rename_to] = out["sources"].pop(self.name)

        return out

    def yaml(self, rename_to: Optional[str] = None) -> str:
        data = self._yaml(rename_to)
        return _dump(data, default_flow_style=False)
