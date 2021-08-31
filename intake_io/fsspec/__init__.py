import importlib
import fsspec

if importlib.util.find_spec("flywheel") is not None:
    fsspec.register_implementation("flywheel", "intake_io.fsspec.flywheel.FlywheelFileSystem")

fsspec.register_implementation("render", "intake_io.fsspec.render.RenderFileSystem")
