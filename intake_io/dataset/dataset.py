from functools import cached_property
from typing import Any, Tuple, Dict, Union

import intake
import numpy as np
import pandas as pd
from intake.catalog import Catalog

from .util import *
from .. import io


class IntakeDataset:

    def __init__(self, catalog: Union[str, Catalog], loader=None):
        self._loader = loader
        if self._loader is None:
            self._loader = lambda src, partition: dict(data=io.imload(src, partition=partition))
        self._items = []
        self._num_partitions = []
        self._cumsum_partitions = []
        if isinstance(catalog, Catalog):
            self._parse_catalog(catalog)
        else:
            catalog = intake.open_catalog(catalog)
            self._parse_catalog(catalog)
            catalog.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return self._cumsum_partitions[-1]

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item_ix, partition_ix = self._get_item_partition_ixs(i)
        if self._num_partitions[item_ix] > 1:
            return {**dict(sample_index=i), **self._loader(self._items[item_ix], partition_ix)}
        else:
            return {**dict(sample_index=i), **self._loader(self._items[item_ix], None)}

    def close(self):
        for item in self._items:
            item.close()

    def _parse_catalog(self, catalog: Catalog):
        self._items.extend(getattr(catalog, i) for i in list(catalog))

        # get nr of partitions per item
        for item in self._items:
            item.discover()
            if "i" in item.metadata["axes"]:
                assert item.metadata["axes"].index("i") == 0
                assert item.npartitions == item.shape[0]
                self._num_partitions.append(item.npartitions)
            else:
                self._num_partitions.append(1)

        # cumsum partitions to map linear indices to catalog items & their partitions
        self._cumsum_partitions = np.cumsum(self._num_partitions)

    def _get_item_partition_ixs(self, i: int) -> Tuple[int, int]:
        if i < 0 or i >= len(self):
            raise IndexError("Index out of range.")
        item_ix = int(np.argwhere(self._cumsum_partitions > i)[0])
        partition_ix = i - (self._cumsum_partitions[item_ix - 1] if item_ix > 0 else 0)
        return item_ix, partition_ix

    def get_dtype(self, i: int) -> np.dtype:
        item_ix, _ = self._get_item_partition_ixs(i)
        return self._items[item_ix].dtype

    def get_shape(self, i: int) -> Tuple[int, ...]:
        item_ix, _ = self._get_item_partition_ixs(i)
        shape = self._items[item_ix].shape
        return shape[self._num_partitions[item_ix] - 1:]

    @cached_property
    def median_shape(self) -> Tuple[int, ...]:
        shapes = [self.get_shape(i) for i in range(self.__len__())]
        ndims = np.asarray([len(i) for i in shapes])
        assert np.all(ndims == ndims[0])
        median = np.median(np.asarray(shapes), axis=0)
        return tuple(map(int, np.round(median)))

    def get_spacing(self, i: int) -> Tuple[float, ...]:
        item_ix, _ = self._get_item_partition_ixs(i)
        return self._items[item_ix].metadata["spacing"]

    @cached_property
    def median_spacing(self) -> Tuple[float, ...]:
        spacings = [self.get_spacing(i) for i in range(self.__len__())]
        ndims = np.asarray([len(i) for i in spacings])
        assert np.all(ndims == ndims[0])
        return tuple(np.median(np.asarray(spacings), axis=0))

    def get_metadata(self, i: int) -> Dict[str, Any]:
        item_ix, _ = self._get_item_partition_ixs(i)
        return self._items[item_ix].discover()


class CategorizedDataset(IntakeDataset):

    def __init__(self, catalog: Union[str, Catalog], get_category, loader=None):
        super().__init__(catalog, loader)
        self._get_category = get_category

    def __getitem__(self, i: int) -> Dict[str, Any]:
        data = super().__getitem__(i)
        data["category"] = self.get_category(i)
        return data

    def get_category(self, i: int) -> str:
        return self._get_category(self.get_metadata(i))

    @cached_property
    def all_categories(self) -> pd.DataFrame:
        return get_categories(self)


class AnnotatedDataset(IntakeDataset):

    def __init__(self, catalog: Union[str, Catalog], get_annotations, loader=None):
        super().__init__(catalog, loader)
        self._get_annotations = get_annotations

    def __getitem__(self, i: int) -> Dict[str, Any]:
        data = super().__getitem__(i)
        data["annotations"] = self.get_annotations(i)
        return data

    def get_annotations(self, i: int) -> pd.DataFrame:
        return self._get_annotations(self.get_metadata(i))

    def get_category_name_column(self, columns) -> str:
        cols = list(filter(lambda x: "category" in x.lower(), columns))
        if len(cols) > 1:
            cols = list(filter(lambda x: "category" in x.lower(), cols))
        assert len(cols) == 1
        return cols[0]

    @cached_property
    def all_categories(self) -> pd.DataFrame:
        return get_categories(self)


class MappedDataset:

    def __init__(self, data: Any, mapper: Any):
        self._data = data
        self._mapper = mapper

    def __getattr__(self, item):
        if item == "__getitem__":
            return getattr(self, item)
        return getattr(self._data, item)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self._mapper(self._data[i])
