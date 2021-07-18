# Data sources

The following drivers are implemented in this project, additional drivers (e.g. zarr) are available [elsewhere](https://intake.readthedocs.io/en/latest/plugin-directory.html). Please note that some overlap exists between the drivers. Most of this is on purpose. For instance, imageio can load nrrd files, but limitations with metadata handling prompted the use of a dedicated pynrrd-based driver instead.

## Primary

| Name | File extension/description | Docs |
| - | - | - |
| [Bio-Formats](https://www.openmicroscopy.org/bio-formats/) | most  image [formats](https://docs.openmicroscopy.org/bio-formats/6.5.1/supported-formats.html) used in microscopy, incl. many that are proprietary | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/bioformats/index.html#intake_io.source.bioformats.BioformatsSource) |
| [DICOM](https://github.com/pydicom/pydicom) | .dicom, .dcm, .dicom.zip, .dcm.zip | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/dicom/index.html#intake_io.source.dicom.DicomSource) |
| [imageio](https://github.com/imageio/imageio) | most standard image [formats](https://imageio.readthedocs.io/en/stable/formats.html) | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/imageio/index.html#intake_io.source.imageio.ImageioSource) |
| [KLB](https://github.com/bhoeckendorf/pyklb) | .klb | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/klb/index.html#intake_io.source.klb.KlbSource) |
| [NIFTI](https://github.com/nipy/nibabel) | .nii, .nii.gz | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/nifti/index.html#intake_io.source.nifti.NiftiSource) |
| [NRRD](https://github.com/mhe/pynrrd) | .nrrd | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/nrrd/index.html#intake_io.source.nrrd.NrrdSource) |
| [TIF](https://github.com/cgohlke/tifffile) | tif, tiff, ome.tif, ome.tiff | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/tif/index.html#intake_io.source.tif.TifSource) |

## Meta

These drivers attempt to automatically detect the correct primary driver to use. When the data is partitioned across multiple files of different formats, a combination of sources is used.

| Name | Description | Docs |
| - | - | - |
| Directory | concatenate image files in a directory along given axis  | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/dir/index.html#intake_io.source.dir.DirSource) |
| File pattern | specify file pattern to load image files into multi-dimensional array  | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/filepattern/index.html#intake_io.source.filepattern.FilePatternSource) |
| [Flywheel](https://flywheel.io) | commercial image database, HIPAA and GDPR compliant  | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/flywheel/index.html#intake_io.source.flywheel.FlywheelSource) |
| List | concatenate list of files or intake sources along given axis  | [link](https://intake-io.readthedocs.io/en/latest/autoapi/intake_io/source/list/index.html#intake_io.source.list.ListSource) |

## Compatibility

The basic formats and protocols (e.g. local file, http, ftp, ...) supported by each source are listed in the source's documentation, as linked above. Additionally, below is a list of commonly used software with an incomplete list of image export/import formats and methods with intake_io.

### [ANTs](https://github.com/ANTsX/ANTs)

ANTs prefers .nrrd and .nii.gz formats for images and pixel spacing metadata. intake_io loads and saves these formats in a manner that is compatible with ANTs.

### [ilastik](https://www.ilastik.org)

Saving images as TIF hyperstacks (see ImageJ/Fiji section below) before loading them into ilastik is one of the most reliable methods to import complex images with axis metadata into ilastik. Other image formats supported by ilastik will work too, but may require manually specifying the axis order after import.

### [ImageJ](https://imagej.net)/[Fiji](https://fiji.sc)

intake_io attempts to save TIF files according to ImageJ's TIF hyperstack specification, including axes and spacing metadata. However, multi-image arrays (axis "i") are unsuported by the hyperstack specification. Furthermore, the following pixel data types are unsupported by ImageJ's TIF loader: int8, u/int32, u/int64, float64.
