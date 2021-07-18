import os

import intake

from . import source


def autodetect(uri: str, **kwargs) -> intake.source.DataSource:
    """
    Autodetect intake source given URI.

    Keyword arguments are passed to the source constructor.

    If no other source is more suitable, it returns an instance of :class:`intake_io.source.ImageIOSource`, which uses
    `imageio <https://github.com/imageio/imageio>`_.

    This function doesn't check whether the data can actually be loaded.

    :param uri:
        URI (e.g. file system path or URL)
    :param kwargs:
        Arguments passed to the source constructor
    :return:
        Data source
    """
    luri = uri.lower()
    lext = os.path.splitext(luri)[-1]
    if lext == ".nrrd":
        return source.NrrdSource(uri, **kwargs)
    elif lext in (".tif", ".tiff"):
        return source.TifSource(uri, **kwargs)
    elif luri.endswith(".nii.gz") or lext == ".nii":
        return source.NiftiSource(uri, **kwargs)
    elif lext in (".dicom", ".dcm"):
        return source.DicomSource(uri, **kwargs)
    elif luri.endswith(".dicom.zip") or luri.endswith(".dcm.zip"):
        return source.DicomZipSource(uri, **kwargs)
    elif lext == ".klb":
        return source.KlbSource(uri, **kwargs)
    elif luri.endswith(".ome.tif") or luri.endswith(".ome.tiff") \
            or lext not in (".tif", ".tiff", ".png", ".jpg", ".gif", ".mp4"):
        return source.BioformatsSource(uri, **kwargs)
    else:
        return source.ImageIOSource(uri, **kwargs)
