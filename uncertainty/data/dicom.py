"""
Set of methods to load and save DICOM files
"""

import itertools
import os
from typing import Generator, Iterable, Optional

import numpy as np
import rt_utils
import pydicom as dicom
import toolz as tz
import toolz.curried as curried
from tqdm import tqdm
from loguru import logger
from fn import _

from .preprocessing import make_isotropic
from .datatypes import Mask, PatientScan
from ..utils.common import conditional, unpack_args, yield_val, apply_if_truthy
from ..utils.path import list_files, generate_full_paths
from ..utils.wrappers import curry
from ..utils.logging import logger_wraps
from ..utils.parallel import pmap
from ..common import constants as c

# ============ Helper functions ============


@curry
def _dicom_type_is(uid, d):
    return d.SOPClassUID == uid


def _standardise_roi_name(name):
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
        lambda iop: np.cross(iop[0:3], iop[3:]),
        lambda normal: np.dot(np.array(dicom_file.ImagePositionPatient), normal),
    )


@logger_wraps()
def _most_common_shape(
    dicom_files: Iterable[dicom.Dataset],
) -> Optional[tuple[int, int]]:
    """
    Most common shape of DICOM files in dicom_files

    None is returned if all DICOM files have unique shapes
    """
    return tz.pipe(
        dicom_files,
        curried.map(lambda f: (f.Rows, f.Columns)),
        tz.frequencies,
        lambda freq_dict: (
            max(freq_dict, key=freq_dict.get)
            # Return None if all DICOM files have unique shapes
            if (not all(val == 1 for val in freq_dict.values()))
            else None
        ),
    )


@curry
def _filter_by_most_common_shape(
    dicom_files: Iterable[dicom.Dataset],
) -> Iterable[dicom.Dataset]:
    """
    Filter list of DICOM files by most common shape if exist, else raise error

    Moved to a separate function as _most_common_shape can only be called once on
    an iterator
    """
    return tz.pipe(
        dicom_files,
        itertools.tee,  # Split into two iterators where one calculates shape
        unpack_args(lambda it1, it2: (_most_common_shape(it1), it2)),
        unpack_args(
            lambda shape, it: (
                dicom_file
                for dicom_file in it
                if (dicom_file.Rows, dicom_file.Columns) == shape
            )
        ),
    )


@logger_wraps()
def _get_uniform_spacing(
    dicom_files: Iterable[dicom.Dataset],
) -> Optional[tuple[float, float, float]]:
    """
    Return spacings from dicom files if they are uniform, else None
    """
    is_uniform = lambda spacings: all(
        [
            # Check that each spacing in each axis is the same
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
    )


@tz.memoize
def _get_dicom_slices(dicom_path: str) -> Iterable[dicom.Dataset]:
    """
    Return all DICOM files in dicom_path in slice order, filter by type and thickness
    """
    return tz.pipe(
        dicom_path,
        list_files,
        pmap(dicom.dcmread),
        curried.filter(_dicom_type_is(c.CT_IMAGE)),
        # Some slice are not part of the volume (have thickness = "0")
        curried.filter(lambda dicom_file: float(dicom_file.SliceThickness) > 0),
        curried.sorted(key=_dicom_slice_order),
    )


@logger_wraps()
@curry
def _load_roi_mask(
    rt_struct: rt_utils.RTStructBuilder,
    name: str,
) -> Optional[tuple[str, Generator[np.ndarray, None, None]]]:
    """
    Wrapper to get_roi_mask_by_name, delay execution and return None if exception is raised
    """
    try:
        return name, rt_struct.get_roi_mask_by_name(name)
    except AttributeError as e:
        logger.warning(
            f"[WARNING]: Failed to load {name} ROI mask for {rt_struct.ds.PatientID}, ROI name present but ContourSequence is missing.",
        )
        return None
    except Exception as e:
        logger.warning(
            f"[WARNING]: Failed to load {name} ROI mask for {rt_struct.ds.PatientID}. \t{e}",
        )
        return None


@logger_wraps()
def _load_rt_struct(dicom_path: str) -> Optional[rt_utils.RTStructBuilder]:
    """
    Create RTStructBuilder from DICOM RT struct file in dicom_path

    None is returned if no RT struct file is found
    """
    return tz.pipe(
        dicom_path,
        list_files,
        curried.filter(
            lambda path: _dicom_type_is(c.RT_STRUCTURE_SET)(dicom.dcmread(path))
        ),
        list,
        apply_if_truthy(
            lambda rt_struct_paths: (
                rt_utils.RTStructBuilder.create_from(
                    dicom_series_path=dicom_path,
                    rt_struct_path=rt_struct_paths[0],
                )
            )
        ),
    )


@logger_wraps()
@curry
def _preprocess_volume(
    array: np.array,
    spacings: tuple[float, float, float],
    intercept_slopes: Iterable[tuple[float, float]],
    method: str,
):
    """
    Preprocess volume to have isotropic spacing and HU scale
    """
    return tz.pipe(
        array,
        lambda arr: np.moveaxis(arr, -1, 0),
        lambda arr: [
            arr_slice * intercept_slope[1] + intercept_slope[0]
            for arr_slice, intercept_slope in zip(arr, intercept_slopes)
        ],
        np.stack,
        lambda arr: np.moveaxis(arr, 0, -1),
        make_isotropic(spacings, method=method),
    )


@logger_wraps()
@curry
def _preprocess_mask(
    name_mask_pairs: tuple[str, np.ndarray], dicom_path: str
) -> Optional[tuple[str, np.ndarray]]:
    _make_mask_isotropic = unpack_args(
        lambda name, mask, spacings: (
            name,
            # Delay, make_isotropic is very slow
            yield_val(make_isotropic, spacings, mask, method="nearest"),
        )
    )

    return tz.pipe(
        dicom_path,
        _get_dicom_slices,
        _get_uniform_spacing,
        # (name, mask, spacings)
        lambda spacings: conditional(
            spacings is not None,
            [(name, mask, spacings) for name, mask in name_mask_pairs],
        ),
        # If spacings is None, return None
        lambda name_mask_spacing_lst: (
            map(_make_mask_isotropic, name_mask_spacing_lst)
            if name_mask_spacing_lst and name_mask_spacing_lst[0][2] is not None
            else None
        ),
    )


# ============ Main functions ============
@logger_wraps(level="INFO")
@curry
def load_patient_scan(
    dicom_path: str, method: str = "linear", preprocess: bool = True
) -> Optional[PatientScan]:
    """
    Load PatientScan from directory of DICOM files in dicom_path

    Preprocessing involves interpolating the volume and masks to have isotropic spacing,
    mapping the volume pixel values to Hounsfield units (HU), and standardising
    the ROI names in the masks to be snake_case. None is returned if no DICOM files
    are found.

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files
    method : str, optional
        Interpolation method for volume, by default "linear", see `scipy.interpolate`
    preprocess : bool, optional
        Whether to preprocess the volume and mask, by default True. WARNING: will
        take significantly longer to load if set to True
    """
    empty_lst_if_none = lambda val: conditional(val == [None], [], val)

    return tz.pipe(
        dicom_path,
        list_files,
        pmap(dicom.dcmread),
        # Get one dicom file to extract PatientID
        lambda dicom_files: next(dicom_files, None),
        apply_if_truthy(
            lambda dicom_file: (
                PatientScan(
                    dicom_file.PatientID,
                    yield_val(load_volume, dicom_path, method, preprocess),
                    empty_lst_if_none([load_mask(dicom_path, preprocess)]),
                )
            )
        ),
    )


@logger_wraps(level="INFO")
@curry
def load_patient_scans(
    dicom_collection_path: str, method: str = "linear", preprocess: bool = True
) -> Iterable[PatientScan]:
    """
    Load PatientScans from folders of DICOM files in dicom_collection_path

    Preprocessing involves interpolating the volume and masks to have isotropic spacing,
    mapping the volume pixel values to Hounsfield units (HU), and standardising
    the ROI names in the masks to be in snake_case. None is returned if no DICOM
    files are found.

    Parameters
    ----------
    dicom_collection_path : str
        Path to the directory containing folders of DICOM files
    method : str, optional
        Interpolation method for volume, by default "linear", see `scipy.interpolate`
    preprocess : bool, optional
        Whether to preprocess the volume and mask, by default True. WARNING: will
        take significantly longer to load if set to True
    """
    return tz.pipe(
        dicom_collection_path,
        generate_full_paths(path_generator=os.listdir),
        pmap(load_patient_scan(method=method, preprocess=preprocess)),
    )


@logger_wraps(level="INFO")
@curry
def load_volume(
    dicom_path: str, method: str = "linear", preprocess: bool = True
) -> Optional[np.array]:
    """
    Load 3D isotropic volume in range (0, 1) from DICOM files in dicom_path

    Preprocessing involves interpolating the volume to have isotropic spacing and
    mapping the pixel values to Hounsfield units (HU)

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files
    method : str, optional
        Interpolation method for volume, by default "linear", see `scipy.interpolate`
    preprocess : bool, optional
        Whether to preprocess the volume and mask, by default True. WARNING: will
        take significantly longer to load if set to True
    """

    def stack_volume(files):
        return tz.pipe(
            [dicom_file.pixel_array for dicom_file in files],
            apply_if_truthy(
                lambda slices: np.moveaxis(np.stack(slices), 0, -1),
            ),
        )

    return tz.pipe(
        dicom_path,
        _get_dicom_slices,
        lambda slices: itertools.tee(slices, 3),
        unpack_args(
            lambda it1, it2, it3: (
                stack_volume(it1),
                _get_uniform_spacing(it2),
                [
                    (
                        float(d_slice.RescaleIntercept),
                        float(d_slice.RescaleSlope),
                    )
                    for d_slice in it3
                ],
            )
        ),
        unpack_args(
            lambda volume, spacings, intercept_slopes: (
                (
                    _preprocess_volume(
                        volume, spacings, intercept_slopes, method=method
                    )
                    if preprocess
                    else volume
                )
                if spacings is not None
                else None
            )
        ),
    )


@logger_wraps(level="INFO")
@curry
def load_all_volumes(
    dicom_collection_path: str, method: str = "linear", preprocess: bool = True
) -> Generator[np.ndarray, None, None]:
    """
    Load 3D volumes from folders of DICOM files in dicom_collection_path
    """
    return tz.pipe(
        dicom_collection_path,
        generate_full_paths(path_generator=os.listdir),
        pmap(load_volume(method=method, preprocess=preprocess)),
    )


@logger_wraps(level="INFO")
@curry
def load_mask(dicom_path: str, preprocess: bool = True) -> Optional[Mask]:
    """
    Load organ-Mask pair from one observer from a folder of DICOM files in dicom_path

    Preprocessing involves interpolating the mask to have isotropic spacing
    and standardising the ROI names to be in snake_case. `None` is
    returned if no RT struct file is found.

    Parameters
    ----------
    dicom_path : str
        Path to the directory containing DICOM files including the RT struct file
    preprocess : bool, optional
        Whether to preprocess the mask using information stored in the DICOM files,
        by default True. WARNING: will take significantly longer to load if set to True.
    """
    rt_struct = _load_rt_struct(dicom_path)

    if rt_struct is None:
        return None

    return tz.pipe(
        rt_struct.get_roi_names(),
        curried.map(_load_roi_mask(rt_struct)),
        # (roi_name, mask_generator) pairs, only successful masks are kept
        curried.filter(_),  # not None
        curried.map(
            unpack_args(lambda name, mask: (_standardise_roi_name(name), mask))
        ),
        conditional(preprocess, _preprocess_mask(dicom_path=dicom_path), tz.identity),
        lambda name_mask_lst: Mask(dict(name_mask_lst) if name_mask_lst else {}),
    )


@logger_wraps(level="INFO")
@curry
def load_all_masks(
    dicom_collection_path: str, preprocess: bool = True
) -> Generator[Mask, None, None]:
    """
    Load Mask list from folders of DICOM files in dicom_collection_path
    """
    return tz.pipe(
        generate_full_paths(dicom_collection_path, os.listdir),
        pmap(load_mask(preprocess=preprocess)),
    )


@logger_wraps(level="INFO")
@curry
def save_dicom_scans_to_h5py(dicom_path: str, save_dir: str, preprocess=True) -> None:
    """
    Save PatientScans to .h5 files in save_dir from folders of DICOM files in dicom_path
    """

    return tz.pipe(
        dicom_path,
        load_patient_scans(preprocess=preprocess),
        curried.filter(_.masks.get_organ_names()),
        curried.map(_.save_h5py(save_dir)),
        tqdm,
        list,
        lambda x: None,
    )
