from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from torch.utils.data import get_worker_info
import zmq

from .cache import CachedDataset
from .serialization import deserialize as _deserialize, serialize as _serialize


class _RemoteDataset:

    def __init__(self, hostname, port, key, data_name, transform=None):
        self.hostname = hostname
        self.port = port
        self._key = key.encode()
        self.data_name = data_name
        self.transform = transform
        self._len = None
        self._socket = None
        self._worker_ids = set()

    def _setup(self):
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.hostname}:{self.port}")

    def __copy__(self):
        return _RemoteDataset(self.hostname, self.port, self._key.decode("utf8"), self.data_name, deepcopy(self.transform))

    def __deepcopy__(self, *args):
        return self.__copy__()

    def __getstate__(self):
        out = {k: getattr(self, k) for k in (
            "hostname", "port", "_key", "data_name", "transform", "_len", "_worker_ids"
        )}
        out["_socket"] = None
        return out

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getitem__(self, x):
        winfo = get_worker_info()
        wid = None if winfo is None else winfo.id
        if self._socket is not None and not wid in self._worker_ids:
            self._socket.close()
            self._socket = None

        if self._socket is None:
            self._setup()

        socket = self._socket
        socket.send_serialized((x, self.data_name), self.serialize)
        y = socket.recv_serialized(self.deserialize)
        if isinstance(y, Exception):
            raise y
        elif x != "__len__":
            return y

        if self.transform is not None:
            y = self.transform(y)
        return y

    def __len__(self):
        if self._len is None:
            self._len = self["__len__"]
        return self._len

    def deserialize(self, x):
        return _deserialize(x, self._key)

    def serialize(self, x):
        return _serialize(x, self._key, maxsize=50 * 1024 ** 2, cname="zlib", clevel=4)


class CachedRemoteDataset(CachedDataset):
    # TODO: Extra overhead on first load, item is first fetched and deserialized, then serialized again to be cached.
    # Possibly intercept and store item before deserialization, or perhaps utilize the separate serializations by
    # optimizing separate serialization parameters for transport and caching.

    def __init__(self, hostname, port, key, data_name, cache_dir, map_size_gb=1, **kwargs):
        super().__init__(_RemoteDataset(hostname, port, key, data_name), cache_dir, map_size_gb=map_size_gb, **kwargs)


class DatasetServer:

    def __init__(self, datasets, port, key, num_workers=4):
        context = zmq.Context().instance()

        frontend = context.socket(zmq.ROUTER)
        frontend.bind(f"tcp://*:{port}")

        backend = context.socket(zmq.DEALER)
        backend.bind("ipc://backend.ipc")

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i in range(num_workers):
                pool.submit(DatasetWorker, i, datasets, key)
            try:
                zmq.proxy(frontend, backend)
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Stopping ...")
            finally:
                frontend.close()
                backend.close()
                context.term()


class DatasetWorker:

    def __init__(self, i, datasets, key):
        self._key = key.encode()
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.connect("ipc://backend.ipc")

        print(f"DatasetWorker {i} started")

        try:
            while True:
                try:
                    x, name = socket.recv_serialized(self.deserialize)
                    if x == "__stop__":
                        break
                    elif x == "__len__":
                        y = len(datasets[name])
                    else:
                        y = datasets[name][x]
                except Exception as ex:
                    y = ex
                socket.send_serialized(y, self.serialize)
        finally:
            socket.close()
            context.term()

    def deserialize(self, x):
        return _deserialize(x, self._key)

    def serialize(self, x):
        return _serialize(x, self._key, maxsize=50 * 1024 ** 2, cname="zlib", clevel=4)


def start_server(datasets, *args, **kwargs):
    return DatasetServer(datasets, *args, **kwargs)
