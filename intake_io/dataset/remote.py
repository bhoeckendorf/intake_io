from concurrent.futures import ProcessPoolExecutor

import zmq

from .cache import CachedDataset
from .serialization import deserialize as _deserialize, serialize as _serialize


class _RemoteDataset:

    def __init__(self, hostname, port, key):
        self._key = key.encode()
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{hostname}:{port}")
        self._len = None

    def __getitem__(self, x):
        self.socket.send_serialized(x, self.serialize)
        y = self.socket.recv_serialized(self.deserialize)
        if isinstance(y, Exception):
            raise y
        return y

    def __len__(self):
        if self._len is None:
            self._len = self["__len__"]
        return self._len

    def deserialize(self, x):
        return _deserialize(x, self._key)

    def serialize(self, x):
        return _serialize(x, self._key, maxsize=50 * 1024 ** 2, cname="zlib", clevel=4)


class RemoteDataset(CachedDataset):
    # TODO: Extra overhead on first load, item is first fetched and deserialized, then serialized again to be cached.
    # Possibly intercept and store item before deserialization, or perhaps utilize the separate serializations by
    # optimizing separate serialization parameters for transport and caching.

    def __init__(self, hostname, port, key, cache_dir):
        super().__init__(_RemoteDataset(hostname, port, key), cache_dir)


class DatasetServer:

    def __init__(self, data, port, key, backend_port=None, num_workers=4):
        if backend_port is None:
            backend_port = port + 1

        context = zmq.Context().instance()

        frontend = context.socket(zmq.ROUTER)
        frontend.bind(f"tcp://*:{port}")

        backend = context.socket(zmq.DEALER)
        backend.bind(f"tcp://*:{backend_port}")

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for _ in range(num_workers):
                pool.submit(DatasetWorker, data, "localhost", backend_port, key)
            zmq.proxy(frontend, backend)


class DatasetWorker:

    def __init__(self, data, hostname, port, key):
        self._key = key.encode()
        socket = zmq.Context.instance().socket(zmq.REP)
        socket.connect(f"tcp://{hostname}:{port}")

        print("DatasetWorker started")

        try:
            while True:
                try:
                    x = socket.recv_serialized(self.deserialize)
                    if x == "__stop__":
                        break
                    elif x == "__len__":
                        y = len(data)
                    else:
                        y = data[x]
                except Exception as ex:
                    y = ex
                socket.send_serialized(y, self.serialize)
        finally:
            socket.disconnect()
            socket.close()

    def deserialize(self, x):
        return _deserialize(x, self._key)

    def serialize(self, x):
        return _serialize(x, self._key, maxsize=50 * 1024 ** 2, cname="zlib", clevel=4)


def start_server(data, *args, **kwargs):
    return DatasetServer(data, *args, **kwargs)
