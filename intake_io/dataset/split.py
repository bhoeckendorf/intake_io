from functools import cached_property
from typing import Any, Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from .dataset import Dataset
from .util import get_categories


class SubsetDataset(Dataset):

    def __init__(self, data: Any, ixs: Tuple[int, ...]):
        super().__init__()
        self._data = data
        self._ixs = ixs

    def __getattr__(self, item):
        if item == "__getitem__":
            return getattr(self, item)
        return getattr(self._data, item)

    def __len__(self):
        return len(self._ixs)

    def _load_index(self, i: int) -> Dict[str, Any]:
        return self._data[self._ixs[i]]

    @cached_property
    def subset_categories(self) -> pd.DataFrame:
        return get_categories(self)


def shuffle_split(data, ratio: float = 0.2) -> Dict[str, List[Dict[str, Any]]]:
    out = dict(
        trn=(),
        val=()
    )

    if len(data) == 0:
        return out

    def is_valid():
        for key, name in (("trn", "train"), ("val", "validation")):
            for ix in range(len(data)):
                try:
                    if data.get_metadata(ix)[f"must_be_in_{name}_set"] and ix not in out[key]:
                        return False
                except KeyError:
                    pass
        return True

    while len(out["trn"]) == 0 or not is_valid():
        for trn_ixs, val_ixs in ShuffleSplit(n_splits=1, test_size=ratio).split(np.arange(len(data))):
            for k, ixs in zip(("trn", "val"), (trn_ixs, val_ixs)):
                out[k] = tuple(map(int, ixs))

    for k in out.keys():
        out[k] = SubsetDataset(data, out[k])
    return out
