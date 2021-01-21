import numpy as np
import xarray as xr
from typing import Any, Optional
import nrrd
import intake
from ..util import *


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


def save_nrrd(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_nrrd(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".nrrd", f".{k}.nrrd", uri, flags=re.IGNORECASE)
                save_nrrd(image[k], uri_key, compress)
        return

    header = dict(encoding="gzip" if compress else "raw")
    header["space dimension"] = image.ndim

    try:
        header["labels"] = list(image.attrs["axes"])[::-1]
        header["kinds"] = []
        for i in header["labels"]:
            if i in "zyx":
                header["kinds"].append("space")
            elif i == "c":
                header["kinds"].append("scalar")
            elif i == "t":
                header["kinds"].append("time")
            else:
                header["kinds"].append("none")
    except (AttributeError, KeyError, TypeError):
        pass

    try:
        #header["units"] = list([image.attrs["spacing_units"][d] for d in header["labels"]])
        header["space units"] = list([image.attrs["spacing_units"][d] for d in header["labels"]])
    except (AttributeError, KeyError):
        pass

    spacing = get_spacing(image)
    if spacing is not None:
        #header["spacings"] = spacing[::-1]
        header["space directions"] = np.eye(len(spacing), len(spacing), dtype=np.float)
        for i, v in enumerate(spacing[::-1]):
            header["space directions"][i, i] = v

    try:
        header["channels"] = list(image.coords["c"])
    except KeyError:
        pass

    nrrd.write(uri, to_numpy(image), header, compression_level=4, index_order="C")
