from typing import Optional, Union

import natsort
import numpy as np

from .base import ImageSource, Schema
from ..autodetect import autodetect
from ..util import get_axes

from concurrent.futures import ThreadPoolExecutor


class ListSource(ImageSource):
    """Intake source for a list of URIs and intake sources.

    Attributes:
        items (list): items
        axis (str): axis to concatenate items along
    """

    container = "ndarray"
    name = "list"
    version = "0.0.1"
    partition_access = True

    def __init__(self, items: list, axis: Optional[str] = None, as_float32=False, num_workers=6, **kwargs):
        """
        Arguments:
            items (list): items
            axis (str, default='z'): axis to concatenate items along
            as_float32 (bool, default=False): convert items to float32 pixel type
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(None, **kwargs)
        self.items = items
        self.axis = axis
        self.as_float32 = as_float32
        self.num_workers = num_workers
        self.uri = None

    def _get_schema(self) -> Schema:
        if isinstance(self.items[0], ImageSource):
            metadata = self.items[0].discover()
            try:
                self.uri = self.items[0].uri
            except AttributeError:
                pass
        else:
            with autodetect(self.items[0]) as src:
                metadata = src.discover()
            self.uri = self.items[0]

        shape = self._set_shape_metadata(
            self.axis + metadata["metadata"]["original_axes"],
            tuple([len(self.items), *metadata["metadata"]["original_shape"]]),
            metadata["metadata"]["spacing"], metadata["metadata"]["spacing_units"])
        self._set_fileheader(metadata["metadata"]["fileheader"])
        return Schema(
            dtype=np.float32 if self.as_float32 else np.dtype(metadata["dtype"]),
            shape=shape,
            npartitions=len(self.items),
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        if isinstance(self.items[i], ImageSource):
            out = self.items[i].read()
        else:
            with autodetect(self.items[i], output_axis_order=self.metadata["original_axes"][1:]) as src:
                out = src.read()
        if self.as_float32 and out.dtype != np.float32:
            return out.astype(np.float32)
        return out

    def read_partition(self, i: Union[int, str]) -> np.ndarray:
        self._load_metadata()
        if isinstance(i, str):
            i = self.metadata["coords"][self.metadata["axes"][0]].index(i)
        return self._reorder_axes(self._get_partition(i))

    def read(self) -> np.ndarray:
        self._load_metadata()
        out = np.zeros(self.metadata["original_shape"], np.float32 if self.as_float32 else self.dtype)
        def _task(i):
            out[i] = self._get_partition(i)
        with ThreadPoolExecutor(self.num_workers) as executor:
            executor.map(_task, range(len(self.items)))
        return self._reorder_axes(out)

    def sort_items(self):
        if self.items is None:
            self._load_metadata()
        self.items = natsort.natsorted(self.items, alg=natsort.IGNORECASE)
