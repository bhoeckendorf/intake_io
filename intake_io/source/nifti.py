import re
import numpy as np
import nibabel
import intake
from typing import Optional

# ToDos:
# Reversing the axes order to zyx invalidates the affine matrix in the header.
# There may be additional meta data in the header object attributes/functions.

class NiftiSource(intake.source.base.DataSource):
    """Intake source for NIFTI files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "nifti"
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
        self._filehandle = None

    def _get_schema(self) -> intake.source.base.Schema:
        if self._filehandle is None:
            self._filehandle = nibabel.load(self.uri)
        h = self._filehandle.header

        axes = None
        shape = h.get_data_shape()
        spacing = h["pixdim"][1:len(shape)+1]
        spacing_units = {}
        if axes is not None:
            for d in axes:
                if d in "zyx":
                    spacing_units[d] = h.get_xyzt_units()[0]
                elif d == "t":
                    spacing_units[d] = h.get_xyzt_units()[1]
        if len(spacing_units) == 0:
            spacing_units = None

        # clean file header
        header = dict(h)
        for k, v in header.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind == "S":
                    v = str(v)
                    try:
                        v = re.search(r"b'(.+)'", v, re.DOTALL).groups()[0]
                    except AttributeError:
                        pass
                    if v == "b''":
                        v = ""
                    header[k] = v
                elif v.size == 1:
                    if v.dtype.kind in ("i", "u"):
                        header[k] = int(v)
                    elif v.dtype.kind == "f":
                        header[k] = float(v)

        return intake.source.base.Schema(
            datashape=shape[::-1],
            shape=shape[::-1],
            dtype=np.dtype(h.get_data_dtype()),
            npartitions=1,
            chunks=None,
            extra_metadata=dict(
                axes=axes,
                spacing=tuple(spacing[::-1]),
                spacing_units=None,
                coords=None,
                fileheader=header
            )
        )

    def _get_partition(self, i: int) -> np.ndarray:
        out = self._filehandle.dataobj.get_unscaled().T
        self._filehandle.uncache()
        return out

    def _close(self):
        pass
