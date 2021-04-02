# Data sources

The following drivers are implemented in this project, additional drivers (e.g. zarr) are available [elsewhere](https://intake.readthedocs.io/en/latest/plugin-directory.html). Please note that some overlap exists between the drivers. Most of this is on purpose. For instance, imageio can load nrrd files, but limitations with metadata handling prompted the use of a dedicated pynrrd-based driver instead.

## Primary

| Name | Description |
| - | - |
| [Bio-Formats](https://www.openmicroscopy.org/bio-formats/) | most  image [formats](https://docs.openmicroscopy.org/bio-formats/6.5.1/supported-formats.html) used in microscopy, incl. many that are proprietary |
| [DICOM](https://github.com/pydicom/pydicom) | .dicom, .dcm, .dicom.zip, .dcm.zip |
| [imageio](https://github.com/imageio/imageio) | most standard image [formats](https://imageio.readthedocs.io/en/stable/formats.html) |
| [KLB](https://github.com/bhoeckendorf/pyklb) | .klb |
| [NIFTI](https://github.com/nipy/nibabel) | .nii, .nii.gz |
| [NRRD](https://github.com/mhe/pynrrd) | .nrrd |

## Meta

These drivers attempt to automatically detect the correct primary driver to use.
| Name | Description |
| - | - |
| Directory | concatenate image files in a directory along given axis |
| File pattern | specify file pattern to load image files into multi-dimensional array |
| [Flywheel](https://flywheel.io) | commercial image database, HIPAA and GDPR compliant |
| List | concatenate list of files or intake sources along given axis |
