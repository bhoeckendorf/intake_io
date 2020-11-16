# [intake_io](https://github.com/bhoeckendorf/intake_io)

Image-focused IO for [Python](https://www.python.org) using [intake](https://intake.readthedocs.io/en/latest/) as data loading backend and [xarray](http://xarray.pydata.org/en/stable/) as target.

## Why intake?

- Consistent metadata handling, incl. loading of metadata only.
- YAML [catalogs](https://intake.readthedocs.io/en/latest/catalog.html) to maintain data collections from heterogeneous sources, e.g. for machine learning.
- Ability to save [additional arbitrary metadata](https://intake.readthedocs.io/en/latest/catalog.html#metadata) with YAML catalog entries. Supplements metadata of image header (e.g. provide channel names or set pixel spacing) and is saved with the data instead of in analysis scripts. Also useful to add ground truth annotations or specify where to find them.
- Consistent interface to load partial data, if supported by underlying data structure.
- [Dask](https://dask.org)-[friendly.](https://intake.readthedocs.io/en/latest/quickstart.html?highlight=dask#working-with-dask)
- See more in the [intake documentation](https://intake.readthedocs.io/en/latest/).

## Why xarray?

[Xarray](http://xarray.pydata.org/en/stable/why-xarray.html) elegantly keeps related images (e.g. image and segmentation masks) and metadata in a single data structure.

## Drivers

The following drivers are implemented in this project, additional drivers (e.g. zarr) are available [elsewhere](https://intake.readthedocs.io/en/latest/plugin-directory.html). Please note that some overlap exists between the drivers. Most of this is on purpose. For instance, imageio can load nrrd files, but limitations with metadata handling prompted the use of a dedicated pynrrd-based driver instead.

### Primary

| Name | Description |
| - | - |
| [Bio-Formats](https://www.openmicroscopy.org/bio-formats/) | most  image [formats](https://docs.openmicroscopy.org/bio-formats/6.5.1/supported-formats.html) used in microscopy, incl. many that are proprietary |
| [DICOM](https://github.com/pydicom/pydicom) | .dicom, .dcm, .dicom.zip, .dcm.zip |
| [imageio](https://github.com/imageio/imageio) | most standard image [formats](https://imageio.readthedocs.io/en/stable/formats.html) |
| [KLB](https://github.com/bhoeckendorf/pyklb) | .klb |
| [NIFTI](https://github.com/nipy/nibabel) | .nii, .nii.gz |
| [NRRD](https://github.com/mhe/pynrrd) | .nrrd |

### Meta

These drivers attempt to automatically detect the correct primary driver to use.
| Name | Description |
| - | - |
| Directory | concatenate image files in a directory along given axis |
| File pattern | specify file pattern to load image files into multi-dimensional array |
| [Flywheel](https://flywheel.io) | commercial HIPAA and GDPR compliant image database |
| List | concatenate list of files or intake sources along given axis |

## Installation

### conda

Execute the following in your conda environment of choice:
```sh
$ conda install -y -c conda-forge dask imagecodecs imageio intake intake-xarray javabridge mrcfile natsort nibabel numcodecs pandas pydicom pynrrd tifffile xarray xmltodict zarr
```
Then continue installing intake_io with pip as described in the following section.

### pip

Execute the following in your virtualenv/conda environment of choice:
```sh
$ pip install flywheel-sdk python-bioformats git+https://github.com/bhoeckendorf/pyklb.git@skbuild git+https://github.com/bhoeckendorf/intake_io.git
```
