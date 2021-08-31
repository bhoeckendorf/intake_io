from fsspec import AbstractFileSystem
from fsspec.callbacks import _DEFAULT_CALLBACK
import io
import natsort
import flywheel

class FlywheelFileSystem(AbstractFileSystem):

    cachable = True
    _cached = False
    protocol = "flywheel"
    async_impl = False
    root_marker = "/"

    def __init__(self, hostname, apikey, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hostname = hostname
        self._client = flywheel.Client(f"{self._hostname}:{apikey.split(':')[-1]}")

    def _strip_hostname(self, x):
        x = self._strip_protocol(x)
        if x.startswith(self._hostname):
            return x[len(self._hostname):].lstrip(self.sep)
        return x

    def mkdir(self, path, create_parents=True, **kwargs):
        raise NotImplementedError

    def makedirs(self, path, exist_ok=False):
        raise NotImplementedError

    def rmdir(self, path):
        raise NotImplementedError

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_hostname(path).rstrip(self.sep).lstrip(self.root_marker)
        if len(path.strip()) == 0:
            node = None
            items = [i.id for i in self._client.groups()]
        else:
            try:
                parent, file = path.rsplit(self.sep, 1)
            except ValueError:
                parent = path
                file = None
            if file in ("analyses", "files"):
                # List analyses and files if path ends on "/analyses" or "/files"
                node = self._client.lookup(parent)
                items = getattr(node, file)
            else:
                node = self._client.lookup(path)
                items = getattr(node, node.child_types[0])
                try:
                    items = items()
                except TypeError:
                    pass
        
        # List analysis files only on ".../analyses/name/files"
        if node is not None and self._type(node) == "analysis" and not file == "files":
            items = list(filter(lambda x: not self.isfile(x), items))

        try:
            items.sort(key=lambda x: x.timestamp)
        except (AttributeError, TypeError):
            items = natsort.natsorted(items, key=self._ls_name, alg=natsort.IGNORECASE)
        
        # Add "analyses" and "files" entries if needed, to top of list, after sorting.
        for field in ("analyses", "files")[::-1]:
            try:
                if len(getattr(node, field)) > 0 and not (len(getattr(node, field)) == 1 and getattr(node, field)[0] == "files") and file != field:
                    items.insert(0, field)
            except AttributeError:
                continue

        paths = [self.root_marker + path + self.sep + i for i in map(self._ls_name, items)]
        if not detail:
            return paths
        else:
            items = list(map(self.info, items))
            for i, n in zip(items, paths):
                i["name"] = n

        return items

    def _ls_name(self, x):
        if not isinstance(x, str) and self._type(x) == "group":
            return x.id
        for field in ("label", "name"):
            try:
                return getattr(x, field)
            except AttributeError:
                continue
        return x

    def walk(self, path, maxdepth=None, **kwargs):
        full_dirs = {}
        dirs = {}
        files = {}

        try:
            detail = kwargs.pop("detail") or False
        except KeyError:
            detail = False

        for item in self.ls(path, detail=True, **kwargs):
            pathname = item["name"]
            itemname = pathname.rstrip(self.sep).rsplit(self.sep, 1)[-1]
            if not self.isfile(item) and pathname != path:
                if itemname in ("analyses", "files"):
                    item = {}
                full_dirs[pathname] = item
                dirs[itemname] = item
            elif pathname == path:
                files[""] = item
            else:
                files[itemname] = item

        if detail:
            yield path, dirs, files
        else:
            yield path, list(dirs), list(files)

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                return

        for d in full_dirs:
            yield from self.walk(d, maxdepth=maxdepth, detail=detail, **kwargs)

    def info(self, path, **kwargs):
        out = {}
        if not isinstance(path, str):
            node = path
            out["name"] = [self._ls_name(node)]

            parent = node
            while hasattr(parent, "parent"):
                if self._type(parent) == "analysis":
                    out["name"].insert(0, "analyses")
                elif self._type(parent) == "file":
                    out["name"].insert(0, "files")
                parent = self._client.get(getattr(parent, "parent")["id"])
                out["name"].insert(0, self._ls_name(parent))
            try:
                parents = parent["parents"]
                for field in ("acquisition", "session", "subject", "project", "group"):
                    id = parents.get(field) or None
                    if id is not None:
                        out["name"].insert(0, self._ls_name(self._client.get(id)))
            except KeyError:
                pass

            out["name"] = self.sep.join(out["name"])
            out["type"] = self._type(node)
        else:
            out["name"] = self._strip_hostname(path).rstrip(self.sep)
            out["type"] = self._type(out["name"])
            node = self._client.lookup(out["name"])
        out["size"] = self.size(node)
        out["created"] = self.created(node)
        out["modified"] = self.modified(node)
        out["data"] = node
        return out
    
    def _type(self, path):
        if self.isfile(path):
            return "file"

        if isinstance(path, str):
            path = self._strip_hostname(path).rstrip(self.sep).split(self.sep)
            if path[-1] in ("analyses", "files"):
                return "directory"
            if len(path) > 1 and path[-2] == "analyses":
                return "analysis"
            if len(path) == 1:
                return "group"
            elif len(path) == 2:
                return "project"
            elif len(path) == 3:
                return "subject"
            elif len(path) == 4:
                return "session"
            elif len(path) == 5:
                return "acquisition"
            else:
                raise ValueError(f'Unknown type at path "{self.sep.join(path)}"')
        else:
            kind = str(type(path)).lower()
            for i in ("group", "project", "subject", "session", "acquisition", "analysis"):
                if i in kind:
                    return i

        raise ValueError(f'Unknown type "{type(path)}".')

    def size(self, path):
        if not isinstance(path, str):
            return path.get("size") or None
        if path.rstrip(self.sep).rsplit(self.sep, 1)[-1] in ("analyses", "files"):
            return None
        return self.size(self.info(path))

    def isdir(self, path):
        return not self.isfile(path)

    def isfile(self, path):
        if not isinstance(path, str):
            return "file" in str(type(path)).lower()
        try:
            return path.rstrip(self.sep).rsplit(self.sep, 2)[-2] == "files"
        except IndexError:
            return False

    def cat_file(self, path, start=None, end=None, **kwargs):
        raise NotImplementedError

    def pipe_file(self, path, value, **kwargs):
        raise NotImplementedError

    def get_file(self, rpath, lpath, **kwargs):
        _rpath, fname = rpath.rsplit(self.sep, 1)
        info = self.info(_rpath)
        while "files" not in info["data"]:
            _rpath = rpath.rsplit(self.sep, 1)[0]
            info = self.info(_rpath)
        info["data"].download_file(fname, lpath)

    def get(self, rpath, lpath, recursive=False, callback=_DEFAULT_CALLBACK, **kwargs):
        self.get_file(rpath, lpath, **kwargs)

    def put_file(self, lpath, rpath, **kwargs):
        raise NotImplementedError

    def head(self, path, size=1024):
        raise NotImplementedError

    def tail(self, path, size=1024):
        raise NotImplementedError

    def cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    def expand_path(self, path, recursive=False, maxdepth=None):
        raise NotImplementedError

    def rm_file(self, path):
        raise NotImplementedError

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path.rstrip("/"))
        if "/" in path:
            parent = path.rsplit("/", 1)[0].lstrip(cls.root_marker)
            return cls.root_marker + parent
        else:
            return cls.root_marker

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        # return AbstractBufferedFile(
        #     self,
        #     path,
        #     mode,
        #     block_size,
        #     autocommit,
        #     cache_options=cache_options,
        #     **kwargs,
        # )
        container, file = path.split("/files/")
        container = self.info(container)
        file = io.BytesIO(container["data"].read_file(file))
        return io.BufferedRandom(file)

    def open(self, path, mode="rb", block_size=None, cache_options=None, **kwargs):
        path = self._strip_hostname(path)
        if "b" not in mode:
            mode = mode.replace("t", "") + "b"

            text_kwargs = {
                k: kwargs.pop(k)
                for k in ["encoding", "errors", "newline"]
                if k in kwargs
            }
            return io.TextIOWrapper(
                self.open(path, mode, block_size, **kwargs), **text_kwargs
            )
        else:
            ac = kwargs.pop("autocommit", not self._intrans)
            f = self._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=ac,
                cache_options=cache_options,
                **kwargs,
            )
            if not ac and "r" not in mode:
                self.transaction.files.append(f)
            return f

    def touch(self, path, truncate=True, **kwargs):
        raise NotImplementedError

    def created(self, path):
        if not isinstance(path, str):
            return path.get("created") or None
        return self.info(path).get("created") or None

    def modified(self, path):
        if not isinstance(path, str):
            return path.get("modified") or None
        return self.info(path).get("modified") or None
