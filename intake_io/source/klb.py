import numpy as np
import pyklb as klb
import intake
from typing import Optional


class KlbSource(intake.source.base.DataSource):
    """Intake source for KLB files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "klb"
    version = "0.0.1"
    partition_access = False

    def __init__(self, uri: str, metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (file system path)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri

    def _get_schema(self) -> intake.source.base.Schema:
        h = klb.readheader(self.uri)

        shape = h["imagesize_tczyx"]
        n_leading_ones = 0
        for i in shape:
            if i < 2:
                n_leading_ones += 1
            else:
                break
        shape = shape[n_leading_ones:]

        return intake.source.base.Schema(
            datashape=tuple(shape),
            shape=tuple(shape),
            dtype=np.dtype(h["datatype"]),
            npartitions=1,
            chunks=None,
            extra_metadata=dict(
                axes="tczyx"[-len(shape):],
                spacing=h["pixelspacing_tczyx"][-len(shape):],
                spacing_units=None,
                coords=None,
                fileheader=h
            )
        )

    def _get_partition(self, i: int) -> np.ndarray:
        return klb.readfull(self.uri).squeeze()

    def _close(self):
        pass
