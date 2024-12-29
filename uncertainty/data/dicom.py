"""
Set of methods to load and save DICOM files
"""

import os
from datetime import date
from typing import Iterable, Iterator, Optional

import numpy as np
import pydicom as dicom
import rt_utils
import toolz as tz
import toolz.curried as curried
from loguru import logger
from tqdm import tqdm

from .. import constants as c
from ..utils import (
    curry,
    generate_full_paths,
    list_files,
    logger_wraps,
    merge_with_reduce,
    rename_key,
    star,
    starfilter,
    transform_nth,
)
from .datatypes import MaskDict, PatientScan


# ============ Helper functions ============
@logger_wraps()
def _flip_array(array: np.ndarray) -> np.ndarray:
    """
    Flip on width and depth axes given array of shape (H, W, D)

    Assuming LPS orientation where L axis is right to left, flipping
    along width and depth axes makes the width increase from left to right
    and depth from top to bottom.
    """
    return tz.pipe(
        array,
        curry(np.flip)(axis=-1),  # Flip depth to be top-down in ascending order
        curry(np.flip)(axis=1),  # Flip width from right->left to left->right
    )


@curry
def _dicom_type_is(dicom: dicom.Dataset, uid: str) -> bool:
    # Some dicom files do not have SOPClassUID...
    return hasattr(dicom, "SOPClassUID") and dicom.SOPClassUID == uid


def _standardise_roi_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


@logger_wraps()
def _dicom_slice_order(dicom_file: dicom.Dataset) -> np.ndarray:
    """
    Return slice order of input DICOM slice within its volume

    Computes the projection dot(IPP, cross(IOP, IOP)), where IPP is the Image
    Position (Patient) and IOP is the Image Orientation (Patient). See
    https://blog.redbrickai.com/blog-posts/introduction-to-dicom-coordinate
    """
    return tz.pipe(
        np.array(dicom_file.ImageOrientationPatient),
        lambda iop: np.cross(iop[:3], iop[3:]),
        lambda normal: np.dot(np.array(dicom_file.ImagePositionPatient), normal),
    )


@logger_wraps()
def _get_uniform_spacing(
    dicom_files: Iterable[dicom.Dataset],
) -> Optional[tuple[float, float, float]]:
    """
    Return spacings from `dicom_files` if they are the same across all DICOM files, else None
    """
    # Check that each spacing in each axis is the same
    is_uniform = lambda spacings: all(
        [
            all([spacing == spacings[:, axis][0] for spacing in spacings[:, axis]])
            for axis in range(spacings.shape[1])
        ]
    )
    return tz.pipe(
        dicom_files,
        curried.map(
            lambda dicom_file: [float(spacing) for spacing in dicom_file.PixelSpacing]
            + [float(dicom_file.SliceThickness)],
        ),
        list,
        lambda spacings: (
            tuple(spacings[0]) if spacings and is_uniform(np.array(spacings)) else None
        ),
    )  # type: ignore


@logger_wraps()
def _get_dicom_slices(dicom_path: str) -> Iterator[dicom.Dataset]:
    """
    Return all DICOM files in `dicom_path` containing .dcm files
    """
    return tz.pipe(
        dicom_path,
        list_files,
        curried.filter(lambda fname: fname.endswith(".dcm")),
        curried.map(lambda fname: dicom.dcmread(fname, force=True)),
    )  # type: ignore


@logger_wraps()
def _get_ct_image_slices(dicom_path: str) -> Iterable[dicom.Dataset]:
    """
    Return all CT image slices from `dicom_path` in slice order
    """
    return tz.pipe(
        dicom_path,
        _get_dicom_slices,
        curried.filter(_dicom_type_is(uid=c.CT_IMAGE)),
        # Some slice are not part of the volume (have thickness = 0)
        curried.filter(lambda dicom_file: float(dicom_file.SliceThickness) > 0),
        curried.sorted(key=_dicom_slice_order),
    )  # type: ignore


@logger_wraps()
@logger.catch()
@curry
def _load_roi_mask(
    name: str,
    rt_struct: rt_utils.RTStruct,
) -> Optional[np.ndarray]:
    """
    Return ROI mask given `name` in `rt_struct`, else None if exception occurs

    Just a functional wrapper around rt_struct.get_roi_mask_by_name
    """
    return rt_struct.get_roi_mask_by_name(name)


@logger_wraps()
def _load_rt_structs(dicom_path: str) -> Iterator[rt_utils.RTStruct]:
    """
    Create list of RTStructBuilder from DICOM RT struct file in `dicom_path`
    """
    return tz.pipe(
        dicom_path,
        list_files,
        curried.filter(lambda path: path.endswith(".dcm")),
        curried.filter(
            lambda path: _dicom_type_is(
                dicom.dcmread(path, force=True), c.RT_STRUCTURE_SET
            )
        ),
        curried.map(
            lambda rt_struct_path: (
                rt_utils.RTStructBuilder.create_from(
                    dicom_series_path=dicom_path,
                    rt_struct_path=rt_struct_path,
                )
            )
        ),
    )  # type: ignore


# ============ Main functions ============


@logger_wraps(level="INFO")
@curry
def load_volume(dicom_path: str) -> Optional[np.ndarray]:
    """
    Load 3D volume of shape (H, W, D) from DICOM files in `dicom_path`

    Returned volume will have each 2D slice's width increase from left to
    right and height from top to bottom. The depth of the volume will
    increase from top to bottom (head to feet). The intensity values are also
    in Hounsfield units (HU).

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files
    """
    dicom_slices = list(_get_ct_image_slices(dicom_path))
    intercepts = [float(d_slice.RescaleIntercept) for d_slice in dicom_slices]
    slopes = [float(d_slice.RescaleSlope) for d_slice in dicom_slices]

    return tz.pipe(
        dicom_path,
        _get_ct_image_slices,
        curried.map(lambda d_slice: d_slice.pixel_array),
        # Convert to HD scale
        lambda slices: [
            (slice_ * slope) + intercept
            for slice_, intercept, slope in zip(slices, intercepts, slopes)
        ],
        list,
        # put depth on last axis
        lambda slices: np.stack(slices, axis=-1) if slices else None,
        lambda volume: _flip_array(volume) if volume is not None else None,
    )  # type: ignore


@curried.excepts(Exception, handler=logger.exception)  # type: ignore
@logger_wraps(level="INFO")
@curry
def load_mask(dicom_path: str) -> Optional[MaskDict]:
    """
    Load masks of shape (H, W, D) from a folder of DICOM files in `dicom_path`

    Returned mask will have each 2D slice's width increase from left to
    right and height from top to bottom. The depth of the volume will
    increase from top to bottom (head to feet).

    **NOTE**: May not be the same shape as the volume.

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files including the RT struct file
    """
    rt_struct = list(_load_rt_structs(dicom_path))
    if rt_struct == []:
        raise ValueError(f"No RT struct file found in {dicom_path}")
    elif len(rt_struct) != 1:
        logger.warning(
            f"Multiple RT struct files found! Using the first one at {rt_struct[0]}..."
        )

    roi_names = rt_struct[0].get_roi_names()
    # catch() decorator don't work with curried functions >:(, rewrap here
    load_roi_mask = tz.pipe(
        _load_roi_mask(rt_struct=rt_struct[0]),
        logger.catch(),
    )

    return tz.pipe(
        (roi_names, roi_names),
        star(zip),
        curried.map(transform_nth(1, load_roi_mask)),
        curried.map(tuple),
        starfilter(lambda name, mask: mask is not None),
        curried.map(transform_nth(1, _flip_array)),
        curried.map(transform_nth(0, _standardise_roi_name)),
        dict,
    )  # type: ignore


@logger_wraps(level="INFO")
@logger.catch()
@curry
def load_patient_scan(dicom_path: str) -> Optional[PatientScan]:
    """
    Load PatientScan from directory of DICOM files in `dicom_path`

    Patient scan is a dictionary containing the volume, mask, and metadata
    of the scan. The volume and mask will have shape (width, height, depth)
    where each 2D slice's width increases from left to right and height from
    top to bottom. The depth of the arrays will increase from top to bottom
    (head to feet). The volume is in Hounsfield units (HU).

    None is returned if no organ masks are found. Use `load_volume` and
    `load_mask` if only one of the volume or mask is available.

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files
    """
    dicom_slices = list(_get_ct_image_slices(dicom_path))
    spacings = _get_uniform_spacing(dicom_slices)
    if not dicom_slices:
        raise ValueError(f"No DICOM files found in {dicom_path}")
    if not spacings:
        raise ValueError(f"Failed to load DICOM at {dicom_path}: non-uniform spacing")

    d_file = dicom_slices[0]  # Get one dicom file to extract PatientID
    volume = load_volume(dicom_path)
    mask = load_mask(dicom_path)

    if volume is None:
        raise ValueError(f"Failed to load volume from {dicom_path}")
    if not mask:
        raise ValueError(f"Failed to load mask from {dicom_path}")

    return {
        "patient_id": d_file.PatientID,
        "volume": volume,
        "masks": mask,
        "dimension_original": (d_file.Rows, d_file.Columns, len(dicom_slices)),
        "spacings": spacings,
        "modality": d_file.Modality,
        "manufacturer": d_file.Manufacturer,
        "scanner": d_file.ManufacturerModelName,
        "study_date": date(
            int(d_file.StudyDate[:-4]),
            int(d_file.StudyDate[4:6]),
            int(d_file.StudyDate[6:]),
        ),
        "organ_ordering": list(mask.keys()),
    }  # type: ignore


@logger_wraps(level="INFO")
@curry
def load_all_volumes(dicom_collection_path: str) -> Iterator[np.ndarray]:
    """
    Load 3D volumes from folders of DICOM files in `dicom_collection_path`

    Returned volumes will have each 2D slice's width increase from left to
    right and height from top to bottom. The depth of the volumes will
    increase from top to bottom (head to feet). The intensity values are also
    in Hounsfield units (HU).

    Parameters
    ----------
    dicom_collection_path : str
        Path to the directory containing directories of DICOM files
    """
    return tz.pipe(
        dicom_collection_path,
        generate_full_paths(path_generator=os.listdir),
        curried.map(load_volume),
        curried.filter(lambda volume: volume is not None),
    )  # type: ignore


@logger_wraps(level="INFO")
@curry
def load_all_masks(dicom_collection_path: str) -> Iterator[MaskDict]:
    """
    Load dictionary of masks from folders of DICOM files in dicom_collection_path

    Returned masks will have each 2D slice's width increase from left to
    right and height from top to bottom. The depth of the masks will
    increase from top to bottom (head to feet).

    **NOTE**: May not be the same shape as the volume.

    Parameters
    ----------
    dicom_collection_path : str
        Path to the directory containing directories of DICOM files including the RT struct file
    """
    return tz.pipe(
        dicom_collection_path,
        generate_full_paths(path_generator=os.listdir),
        curried.map(load_mask),
        curried.filter(lambda mask: mask is not None),
    )  # type: ignore


@logger_wraps(level="INFO")
@curry
def load_all_patient_scans(dicom_collection_path: str) -> Iterator[PatientScan]:
    """
    Load PatientScans from folders of DICOM files in `dicom_collection_path`

    Patient scan is a dictionary containing the volume, mask, and metadata
    of the scan. The volume and mask will have shape (width, height, depth)
    where each 2D slice's width increases from left to right and height from
    top to bottom. The depth of the arrays will increase from top to bottom
    (head to feet). The volume is in Hounsfield units (HU).

    Parameters
    ----------
    dicom_collection_path : str
        Path to the directory containing folders of DICOM files
    """
    return tz.pipe(
        dicom_collection_path,
        generate_full_paths(path_generator=os.listdir),
        curried.map(load_patient_scan),
        curried.filter(lambda scan: scan is not None),
    )  # type: ignore


def load_roi_names(dicom_dir: str) -> Iterator[list[str]]:
    """
    Return a list of all ROI names for each folders of DICOM files in given directory

    Parameters
    ----------
    dicom_dir : str
        Path to the directory containing folders of DICOM files
    """
    return tz.pipe(
        dicom_dir,
        generate_full_paths(path_generator=os.listdir),
        curried.map(_load_rt_structs),
        curried.map(list),
        curried.filter(lambda lst: lst != []),
        curried.map(curried.get(0)),
        curried.map(lambda rt_struct: rt_struct.get_roi_names()),
    )  # type: ignore


def purge_dicom_dir(dicom_dir: str, prog_bar: bool = True) -> None:
    """
    Remove all .dcm files that are not part of a CT image series or RT struct

    WARNING: This searches through all subdirectories as well!

    This is necessary as rt_utils.RTStructBuilder.create_from may try to
    use e.g. array dimensions in dose plans which will be different from actual
    array dimensions in the CT image series -> hence an array broadcasting error.
    See https://github.com/qurit/rt-utils/issues/62

    Parameters
    ----------
    dicom_dir : str
        Path to the directory containing DICOM files
    prog_bar : bool, optional
        Whether to show a progress bar, by default True
    """
    tz.pipe(
        dicom_dir,
        list_files,
        lambda files: tqdm(
            files, desc="Purging DICOM files", disable=not prog_bar, total=len(files)
        ),
        curried.filter(lambda path: path.endswith(".dcm")),
        curried.filter(
            lambda path: (
                not _dicom_type_is(dicom.dcmread(path, force=True), c.CT_IMAGE)
                and not _dicom_type_is(
                    dicom.dcmread(path, force=True), c.RT_STRUCTURE_SET
                )
            )
        ),
        curried.map(os.remove),
        list,
    )


@logger_wraps(level="INFO")
def compute_dataset_stats(
    dataset: Iterable[PatientScan],
) -> dict[str, dict[str, np.ndarray | set[str]]]:
    """
    Return dictionary of statistics from patient scans in `dataset`

    Parameters
    ----------
    dataset : Iterable[PatientScan]
        List of patient scans

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary containing the keys:
        - "dimension_original": Mean dimensions of the original volumes
          when they were loaded
        - "dimension_actual": Mean dimensions of the volumes
        - "spacings": Mean spacings of the volumes
        - "manufacturer": Set of manufacturers for the scanners
        - "scanner": Set of scanner names
    """
    return tz.pipe(
        dataset,
        curried.map(
            curried.keyfilter(
                lambda k: k
                in [
                    "volume",
                    "spacings",
                    "dimension_original",
                    "manufacturer",
                    "scanner",
                ]
            )
        ),
        curried.map(curried.update_in(keys=["volume"], func=lambda v: v.shape)),
        merge_with_reduce(
            func=lambda x, y: [x] + [y] if not isinstance(x, list) else x + [y]
        ),
        curried.update_in(keys=["spacings"], func=lambda s: np.array(s).mean(axis=0)),
        curried.update_in(
            keys=["dimension_original"], func=lambda d: np.array(d).mean(axis=0)
        ),
        curried.update_in(keys=["volume"], func=lambda v: np.array(v).mean(axis=0)),
        curried.update_in(keys=["manufacturer"], func=set),
        curried.update_in(keys=["scanner"], func=set),
        rename_key("volume", "dimension_actual"),
    )  # type: ignore
