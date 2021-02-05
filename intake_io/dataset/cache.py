from typing import Any, Dict


class CachedDataset:

    def __init__(self, data: Any):
        self._data = data
        self._cache = {}

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
        if i not in self._cache:
            self._cache[i] = self._data[i]
        return self._cache[i]
