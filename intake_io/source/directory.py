import os
import numpy as np
import intake
import natsort
from .list import *
from typing import Optional


class DirSource(intake.source.base.DataSource):
    """Intake source for flat file system directories.

    Attributes:
        uri (str): URI (file system path)
        extension (str): file name extension to look for
        axis (str): axis to concatenate files along
    """

    container = "ndarray"
    name = "dir"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, extension: Optional[str] = None, axis: str = "z", metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (file system path)
            extension (str, optional): file name extension to look for
            axis (str, default='z'): axis ('t', 'c', 'z') to concatenate files along
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri
        self.extension = extension
        self.axis = axis
        self._internal = None

    def _get_schema(self) -> intake.source.base.Schema:
        if self._internal is None:
            fpaths = []
            if self.extension is None:
                for fname in natsort.natsorted(os.listdir(self.uri), alg=natsort.IGNORECASE):
                    if not fname.startswith("."):
                        fpaths.append(fname)
            else:
                for fname in natsort.natsorted(os.listdir(self.uri), alg=natsort.IGNORECASE):
                    if not fname.startswith(".") and fname.lower().endswith(self.extension):
                        fpaths.append(fname)
            fpaths = [os.path.join(self.uri, i) for i in fpaths]
            self._internal = ListSource(fpaths, axis=self.axis)
        return self._internal._get_schema()

    def _get_partition(self, i: int) -> np.ndarray:
        return self._internal._get_partition(i)

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._internal.read()

    def _close(self):
        if self._internal is not None:
            self._internal.close()
