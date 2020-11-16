import numpy as np
import nrrd
import intake
from typing import Optional


class NrrdSource(intake.source.base.DataSource):
    """Intake source for NRRD files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "nrrd"
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
        self._custom_fields = dict(channels="quoted string list")

    def _get_schema(self) -> intake.source.base.Schema:
        h = nrrd.read_header(self.uri, self._custom_fields)

        try:
            axes = "".join(h["labels"][::-1])
        except KeyError:
            axes = None

        try:
            spacing = tuple(h["spacings"][::-1])
        except KeyError:
            try:
                spacing = h["space directions"]
                spacing = tuple([spacing[d,d] for d in range(spacing.shape[0])][::-1])
            except KeyError:
                spacing = None

        try:
            units = h["units"]
        except KeyError:
            units = None
        try:
            sunits = h["space units"]
            if units is None:
                units = sunits
            else:
                assert np.all([i == j for i,j in zip(units, sunits)])
        except KeyError:
            pass
        if units is not None:
            if axes is not None:
                assert len(units) == len(axes)
                units = {d: u for d, u in zip(axes, units)}

        try:
            coords["c"] = h["channels"]
        except KeyError:
            coords = None

        return intake.source.base.Schema(
            datashape=tuple(h["sizes"][::-1]),
            shape=tuple(h["sizes"][::-1]),
            dtype=np.dtype(h["type"]),
            npartitions=1,
            chunks=None,
            extra_metadata=dict(
                axes=axes,
                spacing=spacing,
                spacing_units=units,
                coords=coords,
                fileheader=dict(h)
            )
        )

    def _get_partition(self, i: int) -> np.ndarray:
        #with open(self.url, "rb") as f:
        #    return nrrd.read_data(self._header, fh=f, filename=self.url, index_order="C")
        return nrrd.read(self.uri, self._custom_fields, index_order="C")[0]

    def _close(self):
        pass
