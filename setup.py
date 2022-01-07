from setuptools import setup, find_packages

setup(
    name="intake_io",
    version="0.0.1",
    description="",
    long_description="",
    author="Burkhard Hoeckendorf",
    author_email="burkhard.hoeckendorf@pm.me",
    url="https://github.com/bhoeckendorf/intake_io",
    license="MIT",
    packages=find_packages(exclude=("tests*", "docs*")),
    install_requires=[
        "aiohttp",
        "bioformats",
        "dask",
        "imageio",
        "intake",
        "natsort",
        "nibabel",
        "numpy",
        "pandas",
        "pydicom",
        "pytest",
        "pynrrd",
        "requests",
        "tifffile",
        "xarray",
        "xmltodict",
        "zarr"
      ],
    extras_require={
        "all": [
            "flywheel-sdk",
            "git+https://github.com/bhoeckendorf/pyklb.git@skbuild"
        ]
    },
    tests_require=["pytest"],
    entry_points={
        "intakedrivers": [
            "auto = intake_io.source.AutoSource",
            "bioformats = intake_io.source.BioformatsSource",
            "dicom = intake_io.source.DicomSource",
            "dicomzip = intake_io.source.DicomZipSource",
            "directory = intake_io.source.DirSource",
            "filepattern = intake_io.source.FilePatternSource",
            "flywheel = intake_io.source.FlywheelSource",
            "imageio = intake_io.source.ImageIOSource",
            "klb = intake_io.source.KlbSource",
            "list = intake_io.source.ListSource",
            "nifti = intake_io.source.NiftiSource",
            "nrrd = intake_io.source.NrrdSource"
        ]
    }
)
