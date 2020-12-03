import os
import re
import numpy as np
import pandas as pd
import intake
import natsort
from functools import lru_cache
from typing import Optional
from ..util import get_axes
from ..autodetect import autodetect


class FilePattern:
    axes_all = "itczyx"
    axes_type_any = "ic"  # axes that are allowed non-int dtype

    def __init__(
            self,
            folder: str,
            axis_tags: dict,
            extensions: list,
            include_filters: list = [],
            exclude_filters: list = [],
            ixs: Optional[list] = None
            ):
        self.folder = folder
        self.axis_tags = axis_tags
        self.extensions = extensions
        self.include_filters = include_filters
        self.exclude_filters = exclude_filters
        self.ixs = ixs

    @property
    def axes_inner(self) -> str:
        try:
            out = self.get_file_metadata()["metadata"]["axes"]
            if out is None:
                return get_axes(self.shape_inner)
            else:
                return out
        except KeyError:
            return get_axes(self.shape_inner)

    @property
    @lru_cache(maxsize=16)
    def axes_outer(self) -> str:
        for a, tag in self.axis_tags.items():
            if a not in self.axes_all:
                raise ValueError(f"Unknown axis '{a}', valid axes are '{self.axes_all}'")
            if not isinstance(tag, re.Pattern):
                self.axis_tags[a] = re.compile(f"{tag}(\\d+)", re.IGNORECASE)
        return "".join([a if a in self.axis_tags else "" for a in self.axes_all])

    @property
    def axes(self) -> str:
        assert all([a not in self.axes_inner for a in self.axes_outer])
        return self.axes_outer + self.axes_inner

    @property
    def coords_inner(self) -> dict:
        try:
            out = self.get_file_metadata()["metadata"]["coords"]
            if out is None:
                return {}
        except KeyError:
            return {}

    @property
    def coords_outer(self) -> dict:
        self.shape_outer  # declares, defines, caches result
        return self._coords_outer

    @property
    def coords(self) -> dict:
        assert all([k not in self.coords_inner for k in self.coords_outer.keys()])
        return {**self.coords_inner, **self.coords_outer}

    @property
    def dtype(self):
        return self.get_file_metadata()["dtype"]

    @property
    @lru_cache(maxsize=16)
    def files(self) -> pd.DataFrame:
        files = []
        for root, _, fnames in os.walk(self.folder):
            for fname in fnames:
                if fname.startswith(".") or os.path.splitext(fname)[-1].lower() not in self.extensions:
                    continue
                fpath = os.path.join(root, fname)
                if any([i not in fpath for i in self.include_filters]) or any([i in fpath for i in self.exclude_filters]):
                    continue
                coords = self._get_coordinates(fpath)
                if coords is None:
                    continue
                if self.ixs is not None and coords[0] not in self.ixs:
                    continue
                if not os.path.isfile(fpath):
                    continue
                files.append([*coords, fpath])

        if len(files) == 0:
            raise ValueError("No files found that match the given pattern.")

        files = natsort.natsorted(files, key=lambda x: x[:-1], alg=natsort.IGNORECASE)
        files = pd.DataFrame(files, columns=[*self.axes_outer, "file"])

        # convert non-int axes to int if possible for all files
        for a in self.axes_outer:
            try:
                files[a] = files[a].astype(np.integer)
            except ValueError:
                continue

        return files

    def _get_coordinates(self, file: str) -> Optional[list]:
        # take file path, axes (e.g. "icz"), tags (e.g. {"i":"image", "c":"ch", "z":"z"}),
        # return axis coordinates of file path, or None if not all tags are found
        coordinates = []
        for a in self.axes_outer:
            coords = re.findall(self.axis_tags[a], file)
            if len(coords) == 0:
                break

            # all axes other then i, c must have integer coordinates
            try:
                coords = [int(i) for i in coords]
            except ValueError:
                if a in self.axes_type_any:
                    pass
                else:
                    break

            if not all(i == coords[0] for i in coords[1:]):
                raise ValueError(f"Mismatched duplicate axis tags for axis '{a}', tag '{self.axis_tags[a]}' in file '{file}'")
            coordinates.append(coords[0])
        if len(coordinates) != len(self.axes_outer):
            return None
        return coordinates

    def get_shape_groups(self) -> list:
        groups = {}
        for ix, df in self.files.groupby(self.axes_outer[0]):
            try:
                groups[df.shape[0]].append(ix)
            except KeyError:
                groups[df.shape[0]] = [ix]
        return list(groups.values())

    @property
    def shape_inner(self) -> tuple:
        return self.get_file_metadata()["shape"]

    @property
    @lru_cache(maxsize=16)
    def shape_outer(self) -> tuple:
        if len(self.get_shape_groups()) > 1:
            raise ValueError("Multiple shape groups")

        shape_outer = []
        self._coords_outer = {}
        for a in self.axes_outer:
            n = self.files[a].unique()
            shape_outer.append(len(n))
            # keep coordinates as metadata if axis is not 0-based, consecutive numeric
            # zyx are always treated as 0-based, but must be consecutive
            if np.issubdtype(n.dtype, np.integer) and n[-1] - n[0] + 1 == len(n) and (n[0] == 0 or a in "zyx"):
                continue
            if a in "zyx":
                raise ValueError(f"{a}-axis must be consecutive")
            self._coords_outer[a] = list(n)

        if np.prod(shape_outer) != self.files.shape[0]:
            raise ValueError("Prod != len(files)")
        return tuple(shape_outer)

    @property
    def shape(self) -> tuple:
        return tuple([*self.shape_outer, *self.shape_inner])

    @lru_cache(maxsize=16)
    def get_file_metadata(self):
        file = self.files.iloc[0]["file"]
        with autodetect(file) as src:
            return src.discover()

    def _get_rows(self, ix: tuple):
        rows = self.files[self.files[self.axes_outer[0]] == ix[0]]
        for a, i in zip(self.axes_outer[1:len(ix)-1], ix[1:]):
            rows = rows[rows[a] == i]
        return rows

    def _load_file(self, i):
        if not isinstance(i, str):
            row = self._get_rows(i)
            assert row.shape[0] == 1
            i = row["file"]
        with autodetect(i) as src:
            return src.read()

    def load_partition(self, i):
        ix = self.files[self.axes_outer[0]].unique()[i]
        rows = self._get_rows(ix)
        assert rows.shape[0] == np.prod(self.shape_outer[1:])
        out = np.zeros((rows.shape[0], *self.shape_inner), self.dtype)
        for j, file in enumerate(rows["file"]):
            out[j] = self._load_file(file)
        return out.reshape(self.shape[1:])

    def load(self):
        out = np.zeros((self.files.shape[0], *self.shape_inner), self.dtype)
        for i, file in enumerate(self.files["file"]):
            out[i] = self._load_file(file)
        return out.reshape(self.shape)


class FilePatternSource(intake.source.base.DataSource):
    container = "ndarray"
    name = "filepattern"
    version = "0.0.1"
    partition_access = True

    @staticmethod
    def get(folder, axis_tags, extensions, include_filters=[], exclude_filters=[], metadata=None):
        srcs = []
        files = FilePattern(folder, axis_tags, extensions, include_filters, exclude_filters)
        groups = files.get_shape_groups()
        if len(groups) > 1:
            for ixs in groups:
                src = FilePatternSource(folder, axis_tags, extensions, include_filters, exclude_filters, ixs, copy.deepcopy(metadata))
                srcs.append(src)
        else:
            src = FilePatternSource(folder, axis_tags, extensions, include_filters, exclude_filters, None, metadata)
            srcs.append(src)
        return srcs

    def __init__(self, folder, axis_tags, extensions, include_filters=[], exclude_filters=[], ixs=None, metadata=None):
        super().__init__(metadata=metadata)
        self.uri = folder
        self._files = FilePattern(folder, axis_tags, extensions, include_filters, exclude_filters, ixs)

    def _get_schema(self) -> intake.source.base.Schema:
        schema = intake.source.base.Schema(
            datashape=self._files.shape,
            shape=self._files.shape,
            dtype=self._files.dtype,
            npartitions=self._files.shape[0],
            chunks=None,
            extra_metadata=self._files.get_file_metadata()["metadata"]
        )
        schema["extra_metadata"]["axes"] = self._files.axes
        try:
            schema["extra_metadata"]["coords"].update(self._files.coords)
        except (AttributeError, KeyError):
            schema["extra_metadata"]["coords"] = self._files.coords
        
        # ad-hoc for Anna
        try:
            spacing_z = float(schema["extra_metadata"]["fileheader"]["OME"]["Image"]["Pixels"]["@PhysicalSizeZ"])
            schema["extra_metadata"]["spacing"] = tuple([spacing_z, *schema["extra_metadata"]["spacing"]])
        except Exception:
            pass
        
        return schema

    def _get_partition(self, i) -> np.ndarray:
        return self._files.load_partition(i)

    def read(self) -> np.ndarray:
        self._load_metadata()
        return self._files.load()

    def _close(self):
        pass

    def _yaml(self, with_plugin=False) -> str:
        data = super()._yaml(with_plugin)
        try:
            data["sources"]["filepattern"]["metadata"].pop("fileheader")
        except (TypeError, KeyError):
            pass
        try:
            data["sources"]["filepattern"]["metadata"].pop("axes")
        except (TypeError, KeyError):
            pass
        try:
            data["sources"]["filepattern"]["metadata"]["coords"].pop("i")
        except (TypeError, KeyError):
            pass
        args = data["sources"]["filepattern"]["args"]
        args.pop("fpaths")
        args.pop("metadata")
        for d in args["axis_tags"]:
            pattern = args["axis_tags"][d]
            if isinstance(pattern, re.Pattern):
                args["axis_tags"][d] = pattern.pattern
            if args["axis_tags"][d].endswith(r"(\d+)"):
                args["axis_tags"][d] = args["axis_tags"][d][:-5]
        return data
