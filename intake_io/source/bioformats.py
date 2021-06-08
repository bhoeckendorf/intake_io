import numpy as np
import bioformats
import xmltodict
from .base import ImageSource, Schema


def _parse_ome_metadata(xml: str):
    xml = xmltodict.parse(xml)
    pixmeta = xml["OME"]["Image"]["Pixels"]

    axes = pixmeta["@DimensionOrder"][::-1].lower()
    shape = {ax: int(pixmeta[f"@Size{ax.upper()}"]) for ax in axes}

    axes = "".join(ax for ax in axes if shape[ax] > 1)
    for ax in list(shape.keys()):
        if ax not in axes:
            shape.pop(ax)

    spacing = {}
    for ax in axes:
        try:
            spacing[ax] = float(pixmeta[f"@PhysicalSize{ax.upper()}"])
        except KeyError:
            pass

    spacing_units = {}
    for ax in axes:
        try:
            spacing_units[ax] = pixmeta[f"@PhysicalSize{ax.upper()}Unit"]
        except KeyError:
            pass

    # parse channel names
    try:
        channels = tuple(i["@Name"] for i in pixmeta["Channel"])
        coords = dict(c=channels)
    except (KeyError, TypeError):
        coords = {}

    return {
        "dtype": np.dtype(xml["OME"]["Image"]["Pixels"]["@Type"]),
        "axes": axes,
        "shape": shape,
        "spacing": spacing,
        "spacing_units": spacing_units,
        "coords": coords,
        "fileheader": xml
    }


class BioformatsSource(ImageSource):
    """Intake source using Bioformats as backend.

    Attributes:
        uri (str): URI (e.g. file system path or URL)
    """

    container = "ndarray"
    name = "bioformats"
    version = "0.0.1"
    partition_access = True

    def __init__(self, uri: str, **kwargs):
        """
        Arguments:
            uri (str): URI (e.g. file system path or URL)
            metadata (dict, optional): Extra metadata, handed over to intake
        """
        super().__init__(**kwargs)
        self.uri = uri

    def _get_schema(self) -> Schema:
        # Parse metadata, start a JVM if needed.
        # This JVM must continue to run until javabridge is no longer needed,
        # since javabridge does't support starting another one. Current
        # solution is to start a JVM upon first use of bioformats, and never
        # stop it.
        try:
            xml = bioformats.get_omexml_metadata(self.uri)
        except AttributeError:
            bioformats.javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
            xml = bioformats.get_omexml_metadata(self.uri)
        xml = _parse_ome_metadata(xml)
        shape = self._set_shape_metadata(xml["axes"], xml["shape"], xml["spacing"], xml["spacing_units"], xml["coords"])
        self._set_fileheader(xml["fileheader"])
        return Schema(
            dtype=xml["dtype"],
            shape=shape,
            npartitions=1,
            chunks=None
        )

    def _get_partition(self, i: int) -> np.ndarray:
        all_axes = "tczyx"
        axes = self.metadata["axes"]
        mnmx = np.zeros([len(all_axes), 2], np.uint64)
        mnmx[:, 1] = [self.shape[axes.find(d)] if d in axes else 1 for d in all_axes]
        partitioned_axis = all_axes.find(axes[0])
        mnmx[partitioned_axis, :] = i
        mnmx[partitioned_axis, 1] += 1
        out = np.zeros(np.max(mnmx, 1) - np.min(mnmx, 1), self.dtype)
        with bioformats.ImageReader(self.uri) as reader:
            for it, t in enumerate(range(*mnmx[0, :])):
                for ic, c in enumerate(range(*mnmx[1, :])):
                    for iz, z in enumerate(range(*mnmx[2, :])):
                        out[it, ic, iz] = reader.read(c=c, z=z, t=t, rescale=False)
        return out.squeeze()

    def read(self) -> np.ndarray:
        self._load_metadata()
        out = np.zeros(self.metadata["original_shape"], self.dtype)
        if out.ndim == 2:
            pass
        else:
            with bioformats.ImageReader(self.uri) as reader:
                # it = np.nditer(out,
                #    flags=("c_index", "multi_index"),
                #    op_flags=("writeonly",),
                #    op_axes=(list(range(out.ndim-2)),))
                # while not it.finished:
                #    ix = dict(zip(self.metadata["axes"][:len(it.multi_index)], it.multi_index))
                #    out[it.multi_index] = reader.read(ix.get("c"), ix.get("z") or 0, ix.get("t") or 0, rescale=False)
                #    it.iternext()

                with np.nditer(out, flags=("multi_index",), op_flags=("readonly",),
                               op_axes=(list(range(out.ndim - 2)),)) as ndit:
                    for it in ndit:
                        ix = dict(zip(self.metadata["original_axes"][:len(ndit.multi_index)], ndit.multi_index))
                        # it[...] = reader.read(ix.get("c"), ix.get("z") or 0, ix.get("t") or 0, rescale=False)
                        out[ndit.multi_index] = reader.read(ix.get("c"), ix.get("z") or 0, ix.get("t") or 0,
                                                            rescale=False)
        return self._reorder_axes(out)

    def _close(self):
        pass
