from fsspec import AbstractFileSystem
import re
import io
import requests

class RenderFileSystem(AbstractFileSystem):

    # For use with https://github.com/saalfeldlab/render
    # Very basic data access only

    cachable = True
    _cached = False
    protocol = "render"
    async_impl = False
    root_marker = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        raise NotImplementedError

    def walk(self, path, maxdepth=None, **kwargs):
        raise NotImplementedError

    def info(self, path, **kwargs):
        raise NotImplementedError

    def size(self, path):
        raise NotImplementedError

    def isdir(self, path):
        raise NotImplementedError

    def isfile(self, path):
        raise NotImplementedError

    def cat_file(self, path, start=None, end=None, **kwargs):
        raise NotImplementedError

    def pipe_file(self, path, value, **kwargs):
        raise NotImplementedError

    def get_file(self, rpath, lpath, **kwargs):
        raise NotImplementedError

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

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        # e.g. render://bioimg-wkst06.stjude.sjcrh.local/owner/project/stack/c0_z1002_y3389_x10193_h2048_w2048_s1.tif
        host, owner, project, stack, file = self._strip_protocol(path).split(self.sep)
        try:
            c, z, y, x, h, w, s, f = re.match(r"c(.*)_z(\d+)_y(\d+)_x(\d+)_h(\d+)_w(\d+)_s(.*)\.(.*)", file).groups()
        except AttributeError:
            z, y, x, h, w, s, f = re.match(r"z(\d+)_y(\d+)_x(\d+)_h(\d+)_w(\d+)_s(.*)\.(.*)", file).groups()
            c = None
        f = f.lower()
        if f == "tif":
            f += "f"
        
        uri = f"http://{host}:8080/render-ws/v1/owner/{owner}/project/{project}/stack/{stack}/z/{z}/box/{x},{y},{w},{h},{s}/{f}-image"
        if c is not None:
            uri += f"?channels={c}"
        rq = requests.get(uri, stream=True)
        return io.BufferedRandom(io.BytesIO(rq.content))

    def touch(self, path, truncate=True, **kwargs):
        raise NotImplementedError

    def created(self, path):
        raise NotImplementedError

    def modified(self, path):
        raise NotImplementedError
