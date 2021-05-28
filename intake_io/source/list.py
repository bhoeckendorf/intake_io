import os
import numpy as np
import intake
import natsort
from ..autodetect import autodetect
from ..util import get_axes, clean_yaml
from typing import Optional, Union


class ListSource(intake.source.base.DataSource):
    """Intake source for a list of URIs and intake sources.

    Attributes:
        items (list): items
        axis (str): axis to concatenate items along
    """

    container = "ndarray"
    name = "list"
    version = "0.0.1"
    partition_access = True

    def __init__(self, items: list, axis: Optional[str] = None, as_float32 = False, metadata: Optional[dict] = None):
        """
        Arguments:
            items (list): items
            axis (str, default='z'): axis to concatenate items along
            as_float32 (bool, default=False): convert items to float32 pixel type
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.items = items        
        self.axis = axis
        self.as_float32 = as_float32
        self.uri = None

    def _get_schema(self) -> intake.source.base.Schema:
        if isinstance(self.items[0], intake.source.base.DataSource):
            metadata = self.items[0].discover()
            try:
                self.uri = self.items[0].uri
            except AttributeError:
                pass
        else:
            with autodetect(self.items[0]) as src:
                metadata = src.discover()
            self.uri = self.items[0]

        schema = intake.source.base.Schema(
            #datashape=metadata["datashape"],
            shape=metadata["shape"],
            dtype=metadata["dtype"],
            npartitions=len(self.items),
            chunks=None,
            extra_metadata=metadata["metadata"]
        )
        schema["extra_metadata"].update(self.metadata)

        if schema["npartitions"] > 1:
            schema["shape"] = (schema["npartitions"], *schema["shape"])
            schema["datashape"] = schema["shape"]
            try:
                schema["extra_metadata"]["spacing"] = (float(schema["extra_metadata"]["fileheader"]["SliceThickness"]), *schema["extra_metadata"]["spacing"])
            except Exception:
                try:
                    schema["extra_metadata"]["spacing"] = (float(schema["extra_metadata"]["fileheader"]["OME"]["Image"]["Pixels"]["@PhysicalSizeZ"]), *schema["extra_metadata"]["spacing"])
                except Exception:
                    pass
        axes = schema["extra_metadata"]["axes"]
        if axes is None:
            axes = get_axes(metadata["shape"])
        axes = "itczyx"[-len(schema["shape"]):-len(axes)] + axes
        schema["extra_metadata"]["axes"] = axes
        if self.as_float32:
            schema["datatype"] = np.float32
        return schema

    def _get_partition(self, i: int) -> np.ndarray:
        if isinstance(self.items[i], intake.source.base.DataSource):
            out = self.items[i].read()
        else:
            with autodetect(self.items[i]) as src:
                out = src.read()
        if self.as_float32 and out.dtype != np.float32:
            return out.astype(np.float32)
        return out

    def read_partition(self, i: Union[int, str]) -> np.ndarray:
        self._load_metadata()
        if isinstance(i, str):
            i = self.metadata["coords"][self.metadata["axes"][0]].index(i)
        return self._get_partition(i)

    def read(self) -> np.ndarray:
        self._load_metadata()
        out = np.zeros(self.shape, np.float32 if self.as_float32 else self.as_type)
        for i in range(out.shape[0]):
            out[i] = self.read_partition(i)
        return out

    def _close(self):
        pass

    def sort_items(self):
        if self.items is None:
            self._load_metadata()
        self.items = natsort.natsorted(self.items, alg=natsort.IGNORECASE)

    def yaml(self, rename=None) -> str:
        from yaml import dump
        data = self._yaml()
        data = clean_yaml(data, rename)
        return dump(data, default_flow_style=False)
