import os
import re
from gzip import GzipFile
from io import BytesIO

import flywheel
import numpy as np

from .base import ImageSource, Schema
from .nifti import NiftiSource


class FlywheelSource(ImageSource):
    container = "ndarray"
    name = "flywheel"
    version = "0.0.1"
    partition_access = False

    _apikeys = {}
    _clients = {}

    @staticmethod
    def set_apikey(hostname: str, key: str):
        _apikeys[hostname] = key

    def __init__(self, uri: str, **kwargs):
        super().__init__(**kwargs)
        self.uri = uri
        try:
            parts = re.search(r"flywheel://(.+)/(.+)/(.+)", self.uri).groups()
            self._hostname, self._container_id, self._filename = parts
        except AttributeError:
            try:
                self._container_id, self._filename = re.search(r"flywheel:/{1,2}(.+)/(.+)", self.uri).groups()
                self._hostname = "flywheel.stjude.org"
                self.uri = f"flywheel://{self._hostname}/{self._container_id}/{self._filename}"
            except AttributeError:
                raise ValueError(f"Invalid URI: {self.uri}")
        self._kwargs = kwargs
        self._internal = None

    def _get_schema(self) -> Schema:
        try:
            client = self._clients[self._hostname]
        except KeyError:
            try:
                self._clients[self._hostname] = flywheel.Client(f"{self._hostname}:{self._apikeys[self._hostname]}")
                client = self._clients[self._hostname]
            except KeyError:
                raise ValueError(
                    f"API key for {self._hostname} not set. Call FlywheelSource.set_apikey('{self._hostname}', key) once.")

        if self._internal is None:
            if any(self._filename.lower().endswith(i) for i in (".nii.gz", ".nii")):
                stream = BytesIO(client.get(self._container_id).read_file(self._filename))
                if os.path.splitext(self._filename)[-1].lower() == ".gz":
                    stream = GzipFile(fileobj=stream)
                self._internal = NiftiSource(self.uri, stream, **self._kwargs)
            else:
                raise ValueError(f"Direct loading is currently only supported for .nii[.gz] files. This is '{self.uri}'")

        metadata = self._internal.discover()
        self.metadata.update(metadata["metadata"])
        return Schema(
            dtype=metadata["dtype"],
            shape=metadata["shape"],
            npartitions=metadata["npartitions"],
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        return self._internal.read_partition(i)

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._internal.read()

    def _close(self):
        if self._internal is not None:
            self._internal.close()
