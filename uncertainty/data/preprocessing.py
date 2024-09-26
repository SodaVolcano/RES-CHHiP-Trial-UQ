"""
Collection of functions to preprocess numpy arrays
"""

from typing import Iterable, Literal, Optional

import numpy as np
import SimpleITK as sitk
import toolz as tz
from toolz import curried
import torchio as tio

from uncertainty.utils.parallel import pmap

from .. import constants as c
from ..utils.common import call_method, conditional
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry
from .patient_scan import PatientScan
from ..config import Configuration, configuration
from ..constants import BODY_THRESH, HU_RANGE, ORGAN_MATCHES
from .utils import to_torchio_subject, from_torchio_subject
from .mask import get_organ_names, masks_as_array


@logger_wraps()
@curry
def map_interval(
    from_range: tuple[np.number, np.number],
    to_range: tuple[np.number, np.number],
    array: np.ndarray,
) -> np.ndarray:
    """
    Map values in an array in range from_range to to_range
    """
    return tz.pipe(
        array,
        lambda a: np.array(
            (a - from_range[0]) / float(from_range[1] - from_range[0]), dtype=float
        ),
        lambda arr_scaled: to_range[0] + (arr_scaled * (to_range[1] - to_range[0])),
    )


@logger_wraps()
def z_score_scale(array: np.ndarray) -> np.ndarray:
    """
    Z-score normalise array to have mean 0 and standard deviation 1

    Calculated as (x - mean) / std
    """
    return (array - array.mean()) / array.std()


@logger_wraps()
@curry
def make_isotropic(
    spacings: Iterable[np.number],
    array: np.ndarray,
    method: Literal["nearest", "linear", "b_spline", "gaussian"] = "linear",
) -> np.ndarray:
    """
    Return an isotropic array with uniformly 1 unit of spacing between coordinates

    Array can ONLY be **2D** or **3D**

    Parameters
    ----------
    spacings : Iterable[npt.Number]
        Spacing of coordinate points for each axis
    array : npt.NDArray[Any, npt.Number]
        Either a **2D** or **3D** array to interpolate
    method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    npt.NDArray[Any, npt.Number]
        Interpolated array on an isotropic grid
    """
    interpolators = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "b_spline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
    }
    new_size = [
        int(round(old_size * old_spacing))
        for old_size, old_spacing, in zip(array.shape, spacings)
    ]

    return tz.pipe(
        array,
        # sitk don't work with bool datatypes in mask array
        lambda arr: arr.astype(np.float32),
        # sitk moves (H, W, D) to (D, W, H) >:( move axis here so img is (H, W, D)
        lambda arr: np.moveaxis(arr, 1, 0),  # width to first axis
        lambda arr: conditional(
            len(arr.shape) == 3, np.moveaxis(arr, -1, 0), arr  # depth to first axis
        ),
        lambda arr: sitk.GetImageFromArray(arr),
        call_method("SetOrigin", (0, 0, 0)),
        call_method("SetSpacing", spacings),
        lambda img: sitk.Resample(
            img,
            new_size,
            sitk.Transform(),
            interpolators[method],
            img.GetOrigin(),
            (1, 1, 1),  # new spacing
            img.GetDirection(),
            0,
            img.GetPixelID(),
        ),
        lambda img: sitk.GetArrayFromImage(img),
        # arr is (D, W, H) move back to (H, W, D)
        lambda arr: conditional(
            len(arr.shape) == 3, np.moveaxis(arr, 0, -1), arr  # depth to last axis
        ),
        lambda arr: np.moveaxis(arr, 1, 0),  # height to first axis
    )


@logger_wraps()
@curry
def filter_roi(
    roi_names: list[str],
    keep_list: list[str] = c.ROI_KEEP_LIST,
    exclusion_list: list[str] = c.ROI_EXCLUSION_LIST,
) -> list[str]:
    """
    Filter out ROIs based on keep and exclusion lists

    Parameters
    ----------
    roi_names : list[str]
        Array of ROI names
    keep_list : list[str]
        List of substrings to keep
    exclusion_list : list[str]
        List of substrings to exclude

    Returns
    -------
    list[str]
        Array of ROI names not in exclusion list and containing any substring in keep list
    """

    def is_numeric(name):
        try:
            float(name)
            return True
        except ValueError:
            return False

    def not_excluded(name):
        return not any(
            exclude in name for exclude in exclusion_list
        ) and not is_numeric(name)

    return tz.pipe(
        roi_names,
        curried.filter(not_excluded),
        curried.filter(lambda name: any(keep in name for keep in keep_list)),
        list,
    )


@logger_wraps()
@curry
def find_organ_roi(organ: str, roi_lst: list[str]) -> Optional[str]:
    """
    Find a unique ROI name in the list that contains the organ names (possibly multiple)

    If more than one ROI name correspond to the organ, the shortest name is chosen.
    If multiple ROI names have the shortest length, the first one in alphabetical
    order is chosen. None is returned if no ROI name is found.
    """
    return tz.pipe(
        roi_lst,
        sorted,
        lambda lst: [name for name in lst if name in c.ORGAN_MATCHES[organ]],
        # encase [min(...)] in list as we will use lst[0] later
        lambda names: [min(names, key=len)] if len(names) > 1 else names,
        lambda name: None if name == [] else name[0],
    )  # type: ignore


@logger_wraps(level="INFO")
@curry
def preprocess_data(
    scan: PatientScan,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess a PatientScan object into (volume, masks) pairs of shape (C, H, W, D)

    Mask for multiple organs are stacked along the first dimension to have shape
    (organ, height, width, depth). Mask is `None` if not all organs are present.
    """

    def preprocess_volume(scan: PatientScan) -> np.ndarray:
        return tz.pipe(
            scan.volume,
            lambda vol: np.expand_dims(  # to (channel, height, width, depth)
                vol, axis=0
            ),
            lambda vol: np.astype(vol, np.float32),  # sitk can't work with float16
        )  # type: ignore

    def preprocess_mask(scan: PatientScan) -> Optional[np.ndarray]:
        """
        Returns a mask with all organs present, or None if not all organs are present
        """
        names = tz.pipe(
            get_organ_names(scan.masks[""]),
            filter_roi,
            lambda mask_names: [
                find_organ_roi(organ, mask_names) for organ in ORGAN_MATCHES
            ],
            curried.filter(lambda m: m is not None),
            list,
        )

        # If not all organs are present, return None
        if len(names) != len(ORGAN_MATCHES):
            return None

        return tz.pipe(
            scan.masks[""],
            masks_as_array(organ_ordering=names),
            lambda arr: np.moveaxis(arr, -1, 0),  # to (organ, height, width, depth)
            lambda mask: np.astype(mask, np.float32),  # sitk can't work with float16
        )  # type: ignore

    return tz.juxt(preprocess_volume, preprocess_mask)(scan)


@logger_wraps(level="INFO")
@curry
def preprocess_data_configurable(
    volume_mask: tuple[np.ndarray, np.ndarray], config: Configuration = configuration()
):
    """
    Preprocess data according to the configuration
    """
    BODY_MASK = volume_mask[0] > BODY_THRESH
    shape = volume_mask[0].shape[1:]

    def torchio_crop_or_pad(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """crop/pad array centered using the target mask to at least patch_size"""
        return tz.pipe(
            (arr, BODY_MASK),
            to_torchio_subject,
            tio.CropOrPad(
                (
                    max(dim, patch_dim)
                    for dim, patch_dim in zip(shape, config["patch_size"])
                ),  # type: ignore
                padding_mode=np.min(arr),
                mask_name="mask",
            ),
            from_torchio_subject,
            lambda x: x[0].numpy(),
        )

    return tz.pipe(
        volume_mask,
        curried.map(torchio_crop_or_pad),
        tuple,
        lambda vol_mask: (z_score_scale(vol_mask[0]), vol_mask[1]),
    )


@logger_wraps(level="INFO")
@curry
def preprocess_dataset(
    dataset: Iterable[PatientScan | None],
    n_workers: int = 1,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Preprocess a dataset of PatientScan objects into (volume, masks) pairs

    Mask for multiple organs are stacked along the last dimension to have
    shape (height, width, depth, n_organs). An instance is filtered out if
    not all organs are present.

    Parameters
    ----------
    dataset : Iterable[PatientScan | None]
        Dataset of PatientScan objects
    n_workers : int
        Number of parallel processes to use, set to 1 to disable, by default 1
    """
    mapper = pmap(n_workers=n_workers) if n_workers > 1 else curried.map
    return tz.pipe(
        dataset,
        curried.filter(lambda x: x is not None),
        mapper(preprocess_data),
        curried.filter(lambda x: x[1] is not None),
    )  # type: ignore
