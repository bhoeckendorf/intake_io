from copy import deepcopy
import lmdb
from .serialization import serialize as _serialize, deserialize as _deserialize


class CachedDataset:

    def __init__(self, data, cache_dir, map_size_gb=1, transform=None, flush_every_n=1, **kwargs):
        self.data = data
        self.cache_dir = cache_dir
        self.map_size_gb = map_size_gb
        self._kwargs = deepcopy(kwargs)
        self.transform = transform
        self.flush_every_n = flush_every_n
        self._setup()

    def _setup(self):
        self._cache = lmdb.open(path=self.cache_dir, map_size=self.map_size_gb * 1024**3, **self._kwargs)
        self._write_txn_ = None

    def __copy__(self):
        self.flush()
        return CachedDataset(deepcopy(self.data), self.cache_dir, self.map_size_gb, deepcopy(self.transform), self.flush_every_n, **deepcopy(self._kwargs))

    def __deepcopy__(self, *args):
        return self.__copy__()

    def __getstate__(self):
        self.flush()
        return {k: getattr(self, k) for k in (
            "data", "cache_dir", "map_size_gb", "_kwargs", "transform", "flush_every_n"
        )}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._setup()

    # def __getattr__(self, x):
    #     try:
    #         return self.__getattribute__(x)
    #     except AttributeError:
    #         return self.data.__getattr__(x)

    def __getitem__(self, x):
        k = str(x).encode()
        y = self.get(k)
        if y is None:
            y = self.data[x]
            self.put(k, y)
        if self.transform is not None:
            y = self.transform(y)
        return y

    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self[i]

    def get(self, k):
        with self._cache.begin(buffers=True, write=False) as txn:
            y = txn.get(k, None)
            if y is not None:
                y = self._deserialize(y)
        return y

    def _write_txn(self, buffers=True, write=True):
        if self._write_txn_ is None:
            self._write_txn_ = self._cache.begin(buffers=buffers, write=write)
            self._n = 0
        return self._write_txn_

    def put(self, k, v):
        self._write_txn().put(k, self._serialize(v))
        self._n += 1
        if self._n >= self.flush_every_n:
            self.flush()

    def flush(self):
        if self._write_txn_ is None:
            return
        self._write_txn_.commit()
        self._n = 0
        self._write_txn_ = None

    def __len__(self):
        return len(self.data)

    def close(self):
        self.flush()
        self._cache.close()

    def _serialize(self, x):
        return _serialize(x, None)

    def _deserialize(self, x):
        return _deserialize(x, None)
