import numpy as np
import bioformats
import xmltodict
import intake
from ..util import clean_yaml
from typing import Optional


class BioformatsSource(intake.source.base.DataSource):
    """Intake source using Bioformats as backend.

    Attributes:
        uri (str): URI (e.g. file system path or URL)
    """

    container = "ndarray"
    name = "bioformats"
    version = "0.0.1"
    partition_access = True

    @staticmethod
    def _static_get_schema(xml: str) -> intake.source.base.Schema:
        h = xmltodict.parse(xml)
        pixmeta = h["OME"]["Image"]["Pixels"]

        # check for unknown axes
        for d in pixmeta["@DimensionOrder"]:
            assert d in "TCZYX"

        # parse axes, shape and spacing
        axes = ""
        shape = []
        spacing = []
        for d in "TCZYX":
            size = int(pixmeta[f"@Size{d}"])
            if size > 1:
                axes += d.lower()
                shape.append(size)
                if d in "ZYX":
                    try:
                        spacing.append(float(pixmeta[f"@PhysicalSize{d}"]))
                    except Exception:
                        break
        shape = tuple(shape)
        if len(spacing) != np.sum([d in axes for d in "zyx"]):
            spacing = None
        else:
            spacing = tuple(spacing)

        # parse channel names
        try:
            channels = tuple(i["@Name"] for i in pixmeta["Channel"])
            assert len(channels) == 1 and "c" not in axes or len(channels) == shape[axes.find("c")]
            coords = dict(c=channels)
        except (KeyError, TypeError):
            coords = None

        return intake.source.base.Schema(
            datashape=shape,
            shape=shape,
            dtype=np.dtype(h["OME"]["Image"]["Pixels"]["@Type"]),
            npartitions=shape[0],
            chunks=None,
            extra_metadata=dict(
                axes=axes,
                spacing=spacing,
                spacing_units=None,
                coords=coords,
                fileheader=h
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
        # Parse metadata, start a JVM if needed.
        # This JVM must continue to run until javabridge is no longer needed,
        # since javabridge does't support starting another one. Current
        # solution is to start a JVM upon first use of bioformats, and never
        # stop it.
        try:
            return BioformatsSource._static_get_schema(bioformats.get_omexml_metadata(self.uri))
        except AttributeError:
            bioformats.javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
            return BioformatsSource._static_get_schema(bioformats.get_omexml_metadata(self.uri))

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
        axes = self.metadata["axes"]
        shape = [self.shape[axes.find(d)] if d in axes else 1 for d in "tczyx"]
        out = np.zeros(shape, self.dtype)
        with bioformats.ImageReader(self.uri) as reader:
            for t in range(out.shape[0]):
                for c in range(out.shape[1]):
                    for z in range(out.shape[2]):
                        out[t, c, z] = reader.read(c=c, z=z, t=t, rescale=False)
        return out.squeeze()

    def _close(self):
        pass

    def yaml(self, with_plugin=False, rename=None) -> str:
        from yaml import dump
        data = self._yaml(with_plugin)
        data = clean_yaml(data, rename)
        name = list(data["sources"].keys())[0]
        for i in ("coords", "spacing"):
            try:
                data["sources"][name]["metadata"].pop(i)
            except KeyError:
                pass
        return dump(data, default_flow_style=False)
