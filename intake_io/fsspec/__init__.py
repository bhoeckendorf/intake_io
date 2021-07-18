import importlib
import fsspec

if importlib.find_loader("flywheel") is not None:
    fsspec.register_implementation("flywheel", "intake_io.fsspec.flywheel.FlywheelFileSystem")
