import os
import intake
from . import source


def autodetect(uri: str) -> intake.source.base.DataSource:
    """
    Autodetect intake source given URI.

    Arguments:
        uri (str): URI (e.g. file system path or URL)

    Returns:
        An intake source. If no other source is more suitable, it is of type
        intake_io.source.ImageIOSource, which uses imageio.
        This function doesn't check whether the data can actually be loaded.
    """
    luri = uri.lower()
    lext = os.path.splitext(luri)[-1]
    if lext == ".nrrd":
        return source.NrrdSource(uri)
    elif luri.endswith(".nii.gz") or lext == ".nii":
        return source.NiftiSource(uri)
    elif lext in (".dicom", ".dcm"):
        return source.DicomSource(uri)
    elif luri.endswith(".dicom.zip") or luri.endswith(".dcm.zip"):
        return source.DicomZipSource(uri)
    elif lext == ".klb":
        return source.KlbSource(uri)
    elif lext in (".png", ".jpg", ".gif", ".mp4"):
        return source.ImageIOSource(uri)
    else:
        return source.BioformatsSource(uri)
