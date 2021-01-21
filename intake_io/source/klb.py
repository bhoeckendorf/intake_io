import numpy as np
import pyklb as klb
import intake
from typing import Any, Optional
from ..util import *


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


def save_klb(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_klb(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".klb", f".{k}.klb", uri, flags=re.IGNORECASE)
                save_klb(image[k], uri_key, compress)
        return

    assert image.ndim < 6

    shape_out = []
    spacing_out = []
    for i, d in enumerate("tczyx"):
        if not d in image.dims:
            if len(shape_out) == 0:
                continue
            shape_out.append(1)
            spacing_out.append(1.0)
        else:
            shape_out.append(image.dims[d])
            spacing_out.append(image.coords[d][1] - image.coords[d][0])

    klb.writefull(
        to_numpy(image).reshape(shape_out),
        uri,
        pixelspacing_tczyx = np.asarray(spacing_out, dtype=np.float32),
        compression = "bzip" if compress else "none"
    )
