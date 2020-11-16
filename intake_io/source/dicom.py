import numpy as np
import io
import zipfile
import pydicom
import intake
from typing import Optional

# ToDos:
# Reversing the axes order to zyx invalidates the affine matrix in the header.
# There may be additional meta data in the header object attributes/functions.
# Currently assuming 2D since no 3D example file available.
# Pixel type deduction is missing.

class DicomSource(intake.source.base.DataSource):
    """Intake source for DICOM files.

    Attributes:
        uri (str): URI (file system path)
    """

    container = "ndarray"
    name = "dicom"
    version = "0.0.1"
    partition_access = False

    @staticmethod
    def _parse_header(handle) -> dict:
        header = {}
        for attr in dir(handle):
            if attr.startswith("_"):
                break
            header[attr] = getattr(handle, attr)
        return header

    @staticmethod
    def _static_get_schema(handle) -> intake.source.base.Schema:
        header = DicomSource._parse_header(handle)
        spacing = tuple(float(i) for i in handle.PixelSpacing[::-1])
        assert len(spacing) == 2

        return intake.source.base.Schema(
            datashape=(handle.Columns, handle.Rows),
            shape=(handle.Columns, handle.Rows),
            dtype=np.int16,
            npartitions=1,
            chunks=None,
            extra_metadata=dict(
                axes=None,
                spacing=spacing,
                spacing_units=None,
                coords=None,
                fileheader=header
            )
        )

    def __init__(self, uri: str, metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (e.g. file system path or URL)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri

    def _get_schema(self) -> intake.source.base.Schema:
        h = pydicom.dcmread(self.uri, stop_before_pixels=True)
        return DicomSource._static_get_schema(h)

    def _get_partition(self, i: int) -> np.ndarray:
        return pydicom.dcmread(self.uri, force=True).pixel_array.T

    def _close(self):
        pass


class DicomZipSource(intake.source.base.DataSource):
    """Intake source for zipped DICOM file series.

    Attributes:
        uri (str): URI (file system path)
        files (list): Files in archive
    """

    container = "ndarray"
    name = "dicomzip"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, order_by: str = "z", metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (file system path)
            order_by (str, detault='z'): Order files by z ('z') or t ('t')
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri
        self._order_by = order_by
        self._zipfile = None
        self.files = []
        self._headers = None

    def _read(self, i: int, **kwargs) -> np.ndarray:
        with self._zipfile.open(self.files[i]) as f:
            return pydicom.dcmread(io.BytesIO(f.read()), **kwargs)

    def _parse_headers_and_order_files(self):
        self._headers = [DicomSource._parse_header(self._read(i, stop_before_pixels=True)) for i in range(len(self.files))]
        field = "AcquisitionTime" if self._order_by == "t" else "SliceLocation"
        order = np.argsort([float(i[field]) for i in self._headers])
        self._headers = [self._headers[i] for i in order]
        self.files = [self.files[i] for i in order]

    def _get_schema(self) -> intake.source.base.Schema:
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.uri)
        for file in self._zipfile.namelist():
            if os.path.splitext(file)[-1].lower() in (".dicom", ".dcm"):
                self.files.append(file)
        self._parse_headers_and_order_files()

        h = self._read(0, stop_before_pixels=True)
        schema = DicomSource._static_get_schema(h)
        schema["npartitions"] = len(self.files)
        if schema["npartitions"] > 1:
            schema["shape"] = (schema["npartitions"], *schema["shape"])
            schema["datashape"] = schema["shape"]
            schema["extra_metadata"]["spacing"] = (float(schema["extra_metadata"]["fileheader"]["SliceThickness"]), *schema["extra_metadata"]["spacing"])
            schema["extra_metadata"]["fileheaders"] = self._headers
        return schema

    def _get_partition(self, i: int) -> np.ndarray:
        #h = self._read(i, stop_before_pixels=True)
        #s = _get_intake_schema(h)
        return self._read(i, force=True).pixel_array.T

    def read(self) -> np.ndarray:
        self._load_metadata()
        out = np.zeros(self.shape, self.dtype)
        for i in range(out.shape[0]):
            out[i] = self._get_partition(i)
        return out

    def _close(self):
        if self._zipfile is not None:
            self._zipfile.close()
