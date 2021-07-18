# Installation

intake_io doesn't have versions, conda or pip packages currently, although all of this is intended to change. Below 
are the current installation and updating procedures.

## conda

```sh
conda install -y -c conda-forge dask imagecodecs imageio intake intake-xarray python-javabridge mrcfile natsort 
nibabel numcodecs pandas pydicom pynrrd tifffile xarray xmltodict zarr

pip install git+https://github.com/bhoeckendorf/pyklb.git@skbuild git+https://github.com/bhoeckendorf/intake_io.git
```

## pip

```sh
pip install git+https://github.com/bhoeckendorf/pyklb.git@skbuild git+https://github.com/bhoeckendorf/intake_io.git
```

## Updating
The following command updates intake_io to current master branch without touching any other packages:

```sh
pip install --no-deps --force-reinstall --no-cache-dir git+https://github.com/bhoeckendorf/intake_io.git
```
