import os
import numpy as np
import intake
import natsort
from ..autodetect import autodetect
from ..util import get_axes
from typing import Optional


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

    def __init__(self, items: list, axis: str = "z", metadata: Optional[dict] = None):
        """
        Arguments:
            items (list): items
            axis (str, default='z'): axis to concatenate items along
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.items = items        
        self.axis = axis
        self.uri = None

    def _get_schema(self) -> intake.source.base.Schema:
        if isinstance(self.items[0], intake.source.base.DataSource):
            metadata = self.items[0].discover()
            self.uri = self.items[0].uri
        else:
            with autodetect(self.items[0]) as src:
                metadata = src.discover()
            self.uri = self.items[0]

        schema = intake.source.base.Schema(
            datashape=metadata["datashape"],
            shape=metadata["shape"],
            dtype=metadata["dtype"],
            npartitions=len(self.items),
            chunks=None,
            extra_metadata=metadata["metadata"]
        )

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
        schema["extra_metadata"]["axes"] = self.axis + axes
        return schema

    def _get_partition(self, i: int) -> np.ndarray:
        if isinstance(self.items[i], intake.source.base.DataSource):
            return self.items[i].read()
        with autodetect(self.items[i]) as src:
            return src.read()

    def read(self) -> np.ndarray:
        self._load_metadata()
        out = np.zeros(self.shape, self.dtype)
        for i in range(out.shape[0]):
            out[i] = self.read_partition(i)
        return out

    def _close(self):
        pass

    def sort_items(self):
        if self.items is None:
            self._load_metadata()
        self.items = natsort.natsorted(self.items, alg=natsort.IGNORECASE)
