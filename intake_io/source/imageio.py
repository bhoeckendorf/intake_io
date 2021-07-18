import numpy as np
import imageio
import intake
from typing import Any, Optional
from .bioformats import BioformatsSource
from ..util import *


class ImageIOSource(intake.source.base.DataSource):
    """Intake source using imageio as backend.

    Attributes:
        uri (str): URI (e.g. file system path or URL)
    """

    container = "ndarray"
    name = "imageio"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, metadata: Optional[dict] = None):
        """
        Arguments:
            uri (str): URI (e.g. file system path or URL)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(metadata=metadata)
        self.uri = uri
        self._reader = None

    def _get_schema(self) -> intake.source.base.Schema:
        if self._reader is None:
            self._reader = imageio.get_reader(self.uri)
        im = self._reader.get_data(0)
        assert im.ndim == 2 or im.ndim == 3 and im.shape[-1] == 3
        #if im.shape[-1] == 3:
        #    self._img = np.transpose(self._img, (2,0,1))

        fileheader = self._reader.get_meta_data()
        shape = {k: 1 for k in "itczyx"}
        spacing = {k: None for k in "tzyx"}
        if "is_ome" in fileheader and fileheader["is_ome"]:
            s = BioformatsSource._static_get_schema(fileheader["description"])
            s.npartitions = self._reader.get_length()
            return s
        elif "is_imagej" in fileheader and fileheader["is_imagej"]:
            metadata = dict()
            for l in fileheader["is_imagej"].split():
                k, v = l.split("=")
                metadata[k] = v
            fileheader["is_imagej"] = metadata
            for field, axis in zip(["frames", "channels", "slices"], "tcz"):
                try:
                    shape[axis] = int(metadata[field])
                except KeyError:
                    continue
            for field, axis in zip(["finterval", "spacing"], "tz"):
                try:
                    spacing[axis] = float(metadata[field])
                except KeyError:
                    continue

        for a, s in zip("yx", im.shape):
            shape[a] = s
        axes = "".join(k if shape[k] > 1 else "" for k in "itczyx")
        shape = tuple(filter(lambda x: x > 1, (shape[k] for k in "itczyx")))
        spacing = tuple(filter(lambda x: x is not None, (spacing[k] for k in "tzyx")))
        if len(spacing) == 0:
            spacing = None

        return intake.source.base.Schema(
            datashape=shape,
            shape=shape,
            dtype=im.dtype,
            npartitions=self._reader.get_length(),
            chunks=None,
            extra_metadata=dict(
                axes=axes,
                spacing=spacing,
                spacing_units=None,
                coords=None,
                fileheader=fileheader
            )
        )

    def read(self) -> np.ndarray:
        self._load_metadata()
        if self.npartitions == 1:
            return self.read_partition(0)
        out = np.zeros(self.shape, self.dtype)
        for i in range(out.shape[0]):
            out[i] = self.read_partition(i)
        return out

    def _get_partition(self, i: int) -> np.ndarray:
        out = self._reader.get_data(i)
        if out.ndim == 3 and out.shape[-1] == 3:
            out = out.transpose(2, 0, 1)
        return out

    def _close(self):
        if self._reader is not None:
            self._reader.close()


def save_tif(image: Any, uri: str, compress: bool):
    if isinstance(image, xr.Dataset):
        if len(image) == 1:
            save_tif(image[list(image.keys())[0]], uri, compress)
        else:
            for k in image.keys():
                uri_key = re.sub(".tif{1,2}", f".{k}.tif", uri, flags=re.IGNORECASE)
                save_tif(image[k], uri_key, compress)
        return

    # todo: handle RGB
    mode = "v"
    if image.ndim == 2:
        mode = "i"
    with imageio.get_writer(uri, mode=mode) as writer:
        if compress:
            writer.set_meta_data(dict(compress=4))
        writer.append_data(to_numpy(image))
