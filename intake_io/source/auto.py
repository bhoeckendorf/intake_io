import numpy as np
import intake
from ..autodetect import autodetect
from typing import Optional


class AutoSource(intake.source.base.DataSource):
    """Intake source for NRRD files.

    Attributes:
        uri (str): URI (e.g. URL or file system path)
    """

    container = "ndarray"
    name = "auto"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (e.g. URL or file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri
        self._internal = None

    def _get_schema(self) -> intake.source.base.Schema:
        if self._internal is None:
            self._internal = autodetect(self.uri)
        return self._internal._get_schema()

    def _get_partition(self, i) -> np.ndarray:
        return self._internal.get_partition(i)

    def _close(self):
        if self._internal is not None:
            self._internal.close()
