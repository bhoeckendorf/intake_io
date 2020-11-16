import os
import numpy as np
import flywheel
import intake
from typing import Optional
from ..autodetect import autodetect
from ..util import clean_yaml


class FlywheelSource(intake.source.base.DataSource):
    container = "ndarray"
    name = "flywheel"
    version = "0.0.1"
    partition_access = False
    client = None
    cache_dir = None

    @staticmethod
    def set_apikey(key: str):
        if FlywheelSource.client is None:
            FlywheelSource.client = flywheel.Client(key)

    @staticmethod
    def set_cache_dir(dir: str):
        if FlywheelSource.cache_dir is None:
            FlywheelSource.cache_dir = dir

    def __init__(self, acquisition_id: str, file_name: str, metadata: Optional[dict] = None):
        super().__init__(metadata=metadata)
        self.acquisition_id = acquisition_id
        self.file_name = file_name
        self.file_id = None
        self._fpath = None  # location of file in cache_dir
        self._internal = None  # handle to file
        self._acquisition = None  # handle to database
        self._cache_dir = cache_dir

    def _get_schema(self) -> intake.source.base.Schema:
        if self._internal is None:
            self._acquisition = self.client.get(self.acquisition_id)
            for file in self._acquisition.files:
                if file.name == self.file_name:
                    self.file_id = file.id
                    break
            if self.file_id is None:
                raise FileNotFoundError(f"Could not resolve flywheel entry: acquisition id: {self.acquisition_id}, file: {self.file_name}")
            ext = ".".join(self.file_name.split(".")[1:])
            self._fpath = os.path.join(self._cache_dir, f"flywheel_fileid_{self.file_id}.{ext}")
            if not os.path.exists(self._fpath):
                self._acquisition.download_file(self.file_name, self._fpath)
            self._internal = autodetect(self._fpath)
        self._internal._load_metadata()
        schema = self._internal._get_schema()
        schema["extra_metadata"]["acquisition"] = dict(self._acquisition)
        for i in ("subject", "session"):
            schema["extra_metadata"][i] = dict(FlywheelImageSource.client.get(schema["extra_metadata"]["acquisition"]["parents"][i]))
        return schema

    def _get_partition(self, i: int) -> np.ndarray:
        return self._internal.read_partition(i)

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._internal.read()

    def _close(self):
        if self._internal is not None:
            self._internal.close()

    def yaml(self, with_plugin=False, rename=None) -> str:
        from yaml import dump
        data = self._yaml(with_plugin)
        data = clean_yaml(data, rename)
        name = list(data["sources"].keys())[0]
        for i in ("coords", "spacing"):
            try:
                data["sources"][name]["metadata"].pop(i)
            except KeyError:
                pass
        keys = list(data["sources"][name]["metadata"].keys())
        for k in keys:
            data["sources"][name]["metadata"].pop(k)
        return dump(data, default_flow_style=False)
