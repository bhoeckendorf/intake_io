import lmdb
from .serialization import serialize as _serialize, deserialize as _deserialize


class CachedDataset:

    def __init__(self, data, cache_dir, map_size_gb=1, **kwargs):
        self.data = data
        self.cache_dir = cache_dir
        self._cache = lmdb.open(path=self.cache_dir, map_size=map_size_gb * 1024**3, **kwargs)
        self.transform = None

    def __getattr__(self, x):
        try:
            return self.__getattribute__(x)
        except AttributeError:
            return self.data.__getattr__(x)

    def __getitem__(self, x):
        k = str(x).encode()

        with self._cache.begin(buffers=True) as txn:
            y = txn.get(k, None)

        if y is None:
            y = self.data[x]
            with self._cache.begin(buffers=True, write=True) as txn:
                txn.put(k, self.serialize(y))
        else:
            y = self.deserialize(y)

        if self.transform is not None:
            y = self.transform(y)
        return y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.data)

    def close(self):
        self._cache.close()

    def serialize(self, x):
        return _serialize(x, None)

    def deserialize(self, x):
        return _deserialize(x, None)
